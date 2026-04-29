from copy import deepcopy
from os import getenv

import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW

from x2sam.dataset import ImageSamDataset, ImgGenSegDataset
from x2sam.dataset.collate_fns import x2sam_collate_fn
from x2sam.dataset.process_fns import img_genseg_postprocess_fn, process_map_fn_factory
from x2sam.dataset.processors import Sam2ImageProcessor
from x2sam.dataset.samplers import LengthGroupedSampler
from x2sam.engine.hooks import DatasetInfoHook, ModelInfoHook, PTCheckpointHook
from x2sam.engine.runner import TrainLoop
from x2sam.evaluation.evaluators import ImgGenSegEvaluator
from x2sam.model import X2SamModel
from x2sam.model.segmentors import XSegmentor
from x2sam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from x2sam.model.segmentors.sam2 import Sam2Model
from x2sam.utils.visualize import Visualizer

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Directories
code_dir = getenv("CODE_DIR", "./x2sam/")
data_dir = getenv("DATA_DIR", "./datas/")
init_dir = getenv("INIT_DIR", "./inits/")
work_dir = getenv("WORK_DIR", "./wkdrs/")

# Model
mask_encoder_name_or_path = init_dir + "sam2.1-hiera-large"
mask_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

# Data
img_sam_data_root = data_dir + "img_sam/"
img_genseg_data_root = data_dir + "img_genseg/"

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 0.01  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Logging
logging_interval = 10

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# TODO: add special tokens via import from x2sam.utils
use_binary_cls = True

extra_image_processor = dict(
    type=Sam2ImageProcessor.from_pretrained,
    pretrained_model_name_or_path=mask_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=X2SamModel,
    freeze_mask_encoder=True,
    use_activation_checkpointing=False,
    postprocess_fn=img_genseg_postprocess_fn,
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=Sam2Model.from_pretrained,
            pretrained_model_name_or_path=mask_encoder_name_or_path,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=mask_decoder_name_or_path,
                use_backbone=False,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                feature_channels=[256, 256, 256],
                num_feature_levels=3,
                attn_implementation="compiled",
                head_cls_type="linear",
                num_labels=1,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        use_prompt_encoder=False,
        use_memory=False,
        init_decoder=True,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
sam11m_img_sam_dataset = dict(
    type=ImageSamDataset,
    data_path=img_sam_data_root + "sam11m_img_sam.json",
    image_folder=img_sam_data_root + "SA1B",
    extra_image_processor=extra_image_processor,
    num_sample=float("inf"),
    num_ann=100,
    task_name="img_sam",
    data_name="sam11m_img_sam",
    expand2square=False,
    use_binary_cls=use_binary_cls,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=sam11m_img_sam_dataset,
    persistent_workers=False,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=x2sam_collate_fn),
)

val_datasets = [
    dict(
        type=ImgGenSegDataset,
        data_path=img_genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=img_genseg_data_root + "coco2017/val2017",
        pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_val2017",
        sem_segmap_folder=img_genseg_data_root + "coco2017/panoptic_semseg_val2017",
        task_name="img_genseg",
        data_name="coco_img_genseg_panoptic",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=img_genseg_postprocess_fn, task_name="panoptic_genseg"),
        extra_image_processor=extra_image_processor,
        expand2square=True,
        use_binary_cls=use_binary_cls,
    ),
    dict(
        type=ImgGenSegDataset,
        data_path=img_genseg_data_root + "coco2017/annotations/panoptic_val2017.json",
        image_folder=img_genseg_data_root + "coco2017/val2017",
        pan_segmap_folder=img_genseg_data_root + "coco2017/panoptic_val2017",
        sem_segmap_folder=img_genseg_data_root + "coco2017/panoptic_semseg_val2017",
        task_name="img_genseg",
        data_name="coco_img_genseg_panoptic",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=img_genseg_postprocess_fn, task_name="semantic_genseg"),
        extra_image_processor=extra_image_processor,
        expand2square=True,
        use_binary_cls=use_binary_cls,
    ),
    dict(
        type=ImgGenSegDataset,
        data_path=img_genseg_data_root + "coco2017/annotations/instances_val2017.json",
        image_folder=img_genseg_data_root + "coco2017/val2017",
        task_name="img_genseg",
        data_name="coco_img_genseg_instance",
        data_mode="eval",
        postprocess_fn=dict(type=process_map_fn_factory, fn=img_genseg_postprocess_fn, task_name="instance_genseg"),
        extra_image_processor=extra_image_processor,
        expand2square=True,
        use_binary_cls=use_binary_cls,
    ),
]

val_evaluators = [
    dict(
        type=ImgGenSegEvaluator,
        data_name="coco_img_genseg_panoptic",
        distributed=True,
    ),
    dict(
        type=ImgGenSegEvaluator,
        data_name="img_genseg_semantic",
        distributed=True,
    ),
    dict(
        type=ImgGenSegEvaluator,
        data_name="coco_img_genseg_instance",
        distributed=True,
    ),
]

vis_datasets = val_datasets

vis_datasets = deepcopy(val_datasets)
for dataset in vis_datasets:
    if dataset["task_name"] in ["img_genseg", "img_ovseg", "img_vgdseg", "img_intseg", "vid_genseg"]:
        dataset["postprocess_fn"]["threshold"] = 0.5  # type: ignore

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    constructor="CustomOptimWrapperConstructor",
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, norm_type=2, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
    paramwise_cfg=dict(
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# set visualizer
visualizer = dict(
    type=Visualizer,
    scale=1.0,
    font_size_scale=1.0,
)

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "connector", "segmentor.encoder", "segmentor.pixel_decoder", "segmentor.decoder"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook),
    dict(type=PTCheckpointHook),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)

find_unused_parameters = True
