# :bookmark_tabs: Datasets

## Structure

```
inits
├── huggingface
├── Qwen3-VL-4B-Instruct
├── X2SAM
├── mask2former-swin-large-coco-panoptic
└── sam2.1-hiera-large
```

## HFD Downloader Setting

We provide a custom downloader [`hfd`](../srcs/tools/hfd.sh) for downloading models, you can use it to download models from Hugging Face.

```bash
chmod +x $PROJ_HOME/srcs/tools/hfd.sh
alias hfd="$PROJ_HOME/srcs/tools/hfd.sh"
```

## X2SAM

```bash
hfd hao9610/X2SAM --tools aria2c -x 8 --save_dir $PROJ_HOME/inits
```

## Qwen3VL

```bash
hfd Qwen/Qwen3-VL-4B-Instruct --tools aria2c -x 8 --save_dir $PROJ_HOME/inits
```

## SAM2.1

```bash
hfd facebook/sam2.1-hiera-large --tools aria2c -x 8 --save_dir $PROJ_HOME/inits
```

## Mask2Former

```bash
hfd facebook/mask2former-swin-large-coco-panoptic --tools aria2c -x 8 --save_dir $PROJ_HOME/inits
```