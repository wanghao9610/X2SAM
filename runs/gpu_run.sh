#!/usr/bin/env bash
# set -x

#######################################################################
#                          PART 1  Environment                         #
#######################################################################
# Logformat
log_time=$(date "+%Y-%m-%d %H:%M:%S")
log_format="[$log_time] [INFO] [$0]"

# CUDA
echo -e "$log_format Nvidia-smi : \n$(nvidia-smi)"
echo -e "$log_format Cuda : \n$(nvcc -V)"
echo -e "$log_format Conda : \n$(which python)"

#######################################################################
#                          PART 2  Project Config                      #
#######################################################################
# Directory
PROJ_HOME=${PROJ_HOME:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}
code_name="x2sam"
code_dir="$PROJ_HOME/$code_name"
wksp_dir="/workdir/code"
data_dir="$PROJ_HOME/datas"
init_dir="$PROJ_HOME/inits"
work_dir="$PROJ_HOME/wkdrs"
exec_dir=$HJOB_DIR/exec
hostfile=$HJOB_DIR/hostfile
export ROOT_DIR="$PROJ_HOME/"
export DATA_DIR="$data_dir/"
export INIT_DIR="$init_dir/"
export WORK_DIR="$work_dir/"

export LMUData="$data_dir/LMUData"
export HF_HOME="$init_dir/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_DATASET_TIMEOUT=120
export USE_TRITON_KERNEL=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=eth0

# export DEBUG_MODE=true  # enable for debugging
# export DEBUG_ITEM=16  # debug item index
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Model Config
deepspeed_config="x2sam/configs/deepspeed/deepspeed_zero2.json"
config_file="${1:-x2sam/x2sam/configs/x2sam/s3_train/x2sam_qwen3_vl_4b_sam2.1_hiera_large_m2f_e1_gpu32_s3_lora.py}"
read -r -a modes <<< "${2:-train segeval vlmeval visualize}"
prefix="${3:-}"
suffix="${4:-}"
stage="$(basename "$(dirname "$config_file")")"
vlm_name="x2sam-qwen3vl-4b-sam2-hiera-large-lora"
model_name="$(basename "$config_file" .py)"
work_dir="$work_dir/$stage/${prefix:+$prefix/}$model_name$suffix"
lscp_file="$work_dir/last_checkpoint"

# Distributed Training Config
num_nodes=${NUM_NODES:-1}
node_rank=${NODE_RANK:-0}
gpu_per_node=${GPU_PER_NODE:-$(nvidia-smi -L | wc -l)}
master_addr=${MASTER_ADDR:-"127.0.0.1"}
master_port=${MASTER_PORT:-29510}

# #######################################################################
#                          PART 3  Run Config                          #
# #######################################################################

# Run
for mode in "${modes[@]}"
do  
    echo -e "$log_format Mode: $mode."
    time=$(date "+%Y%m%d-%H%M%S")
    [ "$node_rank" -ne 0 ] && sleep 30
    if [ "$mode" = "train" ] && [ ! -d "$work_dir" ] && [ "$node_rank" = 0 ]; then
        mkdir -p "$work_dir"
        cp -rf "$(realpath "$0")" "$code_dir" "$work_dir"
        find "$work_dir/$code_name" -name "*.crc" -type f -delete
        find "$work_dir/$code_name" -type f -exec chmod 664 {} +
        find "$work_dir/$code_name" -type d -exec chmod 775 {} +
    fi
    if [ -d "$work_dir" ]; then
        code_dir="$work_dir/$code_name"
        cp "$(realpath "$0")" "$work_dir"
    fi
    cd "$code_dir"
    export CODE_DIR="$code_dir/"
    echo -e "$log_format code_dir: $code_dir"

    [ -f "$config_file" ] || config_file="${config_file#$code_name/}"
    [ -f "$config_file" ] || { echo -e "$log_format Config file not found: $config_file" >&2; exit 1; }
    
    # mode: train
    trained_flag=0
    if [ "$mode" = "train" ]; then
        PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --nnodes="$num_nodes" --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$gpu_per_node" \
            "$code_dir/x2sam/tools/train.py" \
            "$config_file" \
            --work-dir "$work_dir" \
            --resume auto \
            --launcher pytorch \
            --deepspeed "$deepspeed_config" \
            --seed 1024 | { [ "$node_rank" = "0" ] && tee "$work_dir/train-${time}.log" || cat; }
    fi
    if [ -f "$lscp_file" ] || [[ ! " ${modes[*]} " =~ " train " ]]; then
        trained_flag=1
    fi
    # mode: eval
    if [ "$mode" = "segeval" ]; then
        if [ "$node_rank" = 0 ] && [ -f "$work_dir/last_checkpoint" ]; then
            echo -e "$log_format Converting $model_name to PT format."
            PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH \
                python "$code_dir/x2sam/tools/model_tools/ds_zero_to_pytorch.py" \
                --work-dir \
                "$work_dir"
        fi
        PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --nnodes="$num_nodes" --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$gpu_per_node" \
            "$code_dir/x2sam/tools/eval.py" \
            "$config_file" \
            --launcher pytorch \
            --work-dir "$work_dir" \
            --seed 0 \
            --pth_model latest --rerun | { [ "$node_rank" = "0" ] && tee "$work_dir/segeval-${time}.log" || cat; }
    fi
    # mode: vlmeval
    if [ "$mode" = "vlmeval" ] && [ "$trained_flag" = 1 ]; then
        if [ "$node_rank" = 0 ] && [ ! -d "$work_dir/pytorch_model" ]; then
            echo -e "$log_format Converting $model_name to HF format."
            PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH \
                python "$code_dir/x2sam/tools/model_tools/pth_to_hf.py" \
                "$code_dir/$config_file" \
                "$work_dir" \
                --pth_model latest \
                --save-format huggingface
            rm "$init_dir/$vlm_name"
            ln -s "$work_dir/pytorch_model" "$init_dir/$vlm_name"
        fi
        if [ -d "$work_dir/pytorch_model" ]; then
            echo -e "$log_format Evaluating VLM: $model_name."
            [ "$node_rank" -ne 0 ] && sleep 30
            PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                torchrun --nnodes="$num_nodes" --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$gpu_per_node" \
                "$code_dir/x2sam/evaluation/vlmeval/run.py" \
                --data MME MMBench_DEV_EN SEEDBench_IMG POPE GQA_TestDev_Balanced AI2D_TEST ScienceQA_VAL \
                --model "$vlm_name" \
                --work-dir "$work_dir/vlmeval_image_results" | { [ "$node_rank" = "0" ] && tee "$work_dir/vlmeval_image-${time}.log" || cat; }

            PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                torchrun --nnodes="$num_nodes" --node_rank="$node_rank" --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$gpu_per_node" \
                "$code_dir/x2sam/evaluation/vlmeval/run.py" \
                --data Video-MME_64frame MVBench_64frame MLVU_64frame LongVideoBench_64frame \
                --verbose \
                --model "$vlm_name" \
                --work-dir "$work_dir/vlmeval_video_results" | { [ "$node_rank" = "0" ] && tee "$work_dir/vlmeval_video-${time}.log" || cat; }
        fi
    fi
    # mode: visualize
    if [ "$mode" = "visualize" ]; then
        echo -e "$log_format Visualizing $model_name."
        PYTHONPATH="$(realpath "$code_dir")":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python "$code_dir/x2sam/tools/visualize.py" \
            "$config_file" \
            --work-dir "$work_dir" \
            --pth_model latest \
            --concat-aux-img \
            --max-samples 400 | { [ "$node_rank" = "0" ] && tee "$work_dir/visualize-${time}.log" || cat; }
    fi
done