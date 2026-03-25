# :bookmark_tabs: Datasets

## File Structure

```
datas
├── img_chat
│   └── llava
│       ├── LLaVA-CC3M-Pretrain-595K
│       ├── llava_images
│       ├── LLaVA-Instruct-150K
│       └── LLaVA-Pretrain
├── img_gcgseg
│   └── grand_f
│       ├── annotations
│       └── images
├── img_genseg
│   └── coco
│       ├── annotations
│       ├── panoptic_semseg_train2017
│       ├── panoptic_semseg_val2017
│       ├── panoptic_train2017
│       ├── panoptic_val2017
│       ├── stuff_train2017_pixelmaps
│       ├── stuff_val2017_pixelmaps
│       ├── test2017
│       ├── train2014
│       ├── train2017
│       ├── val2014
│       └── val2017
├── img_intseg
│   └── coco_int
│       ├── annotations
│       └── coco2017
├── img_ovseg
│   ├── ade20k
│   │   ├── ade20k_panoptic_train
│   │   ├── ade20k_panoptic_val
│   │   ├── annotations
│   │   ├── annotations_detectron2
│   │   ├── annotations_instance
│   │   └── images
│   └── pascal_ctx
│       ├── annotations_ctx459
│       ├── annotations_ctx59
│       ├── images
│       └── labels
├── img_reaseg
│   └── lisa
│       ├── explanatory
│       ├── test
│       ├── train
│       └── val
├── img_refseg
│   ├── annotations
│   ├── grefcoco
│   ├── images
│   │   ├── saiapr_tc-12
│   │   ├── train2014
│   │   └── val2014
│   ├── refclef
│   ├── refcoco
│   ├── refcocog
│   ├── refcocop
│   └── train2014
├── img_sam
│   └── SA1B
├── img_vgdseg
│   └── coco_vgd
│       ├── annotations
│       └── coco2017
├── LMUData
│   ├── datasets--lmms-lab--Video-MME
│   │   ├── subtitle
│   │   ├── video
│   │   └── videomme
│   ├── datasets--lmms-lab--VideoMMMU
│   │   ├── Adaptation
│   │   ├── Comprehension
│   │   ├── images
│   │   ├── Perception
│   │   └── videos
│   ├── datasets--longvideobench--LongVideoBench
│   │   ├── subtitles
│   │   └── videos
│   ├── datasets--MLVU--MVLU
│   │   ├── MLVU
│   │   └── MLVU_Test
│   ├── datasets--opencompass--MMBench-Video
│   │   ├── video
│   │   └── video_pkl
│   ├── datasets--OpenGVLab--MVBench
│   │   ├── json
│   │   └── video
│   └── images
│       ├── AI2D_TEST
│       ├── GQA_TestDev_Balanced
│       ├── LongVideoBench
│       ├── MLVU_MCQ
│       ├── MLVU_OpenEnded
│       ├── MMBench
│       ├── MMBench_V11
│       ├── MMBench-Video
│       ├── MME
│       ├── MVBench
│       ├── POPE
│       ├── ScienceQA_TEST
│       ├── ScienceQA_VAL
│       ├── SEEDBench_IMG
│       ├── Video-MME
│       └── VideoMMMU
├── vid_chat
│   └── VideoChatGPT
│       └── ActivityNet
├── vid_gcgseg
│   └── VideoGLaMM
│       ├── anet_gcg
│       ├── burst
│       ├── hcstvg_gcg
│       ├── mevis
│       ├── mevis_gcg
│       ├── processed
│       ├── video_gcg
│       ├── vidstg
│       ├── vidstg_gcg
│       ├── youtube_rvos2021
│       ├── youtube_vis2019
│       ├── ytvis
│       ├── ytvos_gcg
│       └── ziped
├── vid_genseg
│   ├── VIPSeg_720P
│   │   ├── images
│   │   ├── panomasks
│   │   └── panomasksRGB
│   ├── VSPW_480p
│   │   ├── annotations
│   │   ├── data
│   │   ├── images
│   │   └── semmasks
│   ├── youtube_vis2019
│   │   ├── annotations
│   │   ├── train
│   │   └── valid
│   └── youtube_vis2021
│       ├── test
│       ├── train
│       └── valid
├── vid_objseg
│   └── youtube_vos2019
│       ├── annotations
│       ├── test
│       ├── train
│       └── valid
├── vid_ovseg
│   └── youtube_vis2021
│       ├── annotations
│       ├── train
│       └── valid
├── vid_reaseg
│   ├── ReasonVOS
│   │   ├── Annotations
│   │   └── JPEGImages
│   └── ReVOS
│       ├── annotations
│       └── JPEGImages
├── vid_refseg
│   ├── davis_rvos2017
│   │   ├── Annotations
│   │   ├── davis_text_annotations
│   │   ├── JPEGImages
│   │   └── meta_expressions
│   └── youtube_rvos2021
│       ├── Annotations
│       ├── meta_expressions
│       ├── train
│       └── valid
└── vid_vgdseg
    ├── vipseg_vgd
    │   ├── annotations
    │   └── VIPSeg_720P
    └── ytvis_vgd
        ├── annotations
        └── youtube_vis2019
            ├── annotations
            ├── train
            └── valid
```

## Image Segmentation Datasets

### 1. Image Generic Segmentation Datasets

* COCO Dataset for Image Generic Segmentation (Semantic, Instance, Panoptic)

    Please refer to the following steps to download and process COCO dataset.
    ```bash
    cd $root_dir
    mkdir -p datas/img_genseg/coco2017
    export temp_data_dir=$root_dir/datas/img_genseg
    # download coco2017 dataset
    wget http://images.cocodataset.org/zips/train2017.zip -O $temp_data_dir/coco2017/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip -O $temp_data_dir/coco2017/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $temp_data_dir/coco2017/annotations_trainval2017.zip
    wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip -O $temp_data_dir/coco2017/panoptic_annotations_trainval2017.zip

    # unzip dataset and remove zip files
    unzip $temp_data_dir/coco2017/train2017.zip -d $temp_data_dir/coco2017
    unzip $temp_data_dir/coco2017/val2017.zip -d $temp_data_dir/coco2017
    unzip $temp_data_dir/coco2017/annotations_trainval2017.zip -d $temp_data_dir/coco2017
    unzip $temp_data_dir/coco2017/panoptic_annotations_trainval2017.zip -d $temp_data_dir/coco2017
    unzip $temp_data_dir/coco2017/annotations/panoptic_train2017.zip -d $temp_data_dir/coco2017
    unzip $temp_data_dir/coco2017/annotations/panoptic_val2017.zip -d $temp_data_dir/coco2017
    rm $temp_data_dir/coco2017/train2017.zip $temp_data_dir/coco2017/val2017.zip $temp_data_dir/coco2017/annotations_trainval2017.zip $temp_data_dir/coco2017/panoptic_annotations_trainval2017.zip $temp_data_dir/coco2017/annotations/panoptic_train2017.zip $temp_data_dir/coco2017/annotations/panoptic_val2017.zip

    # download coco2014 images
    mkdir -p datas/img_genseg/coco2014
    wget http://images.cocodataset.org/zips/train2014.zip -O $temp_data_dir/coco2014/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip -O $temp_data_dir/coco2014/val2014.zip
    # unzip dataset
    unzip $temp_data_dir/coco2014/train2014.zip -d $temp_data_dir/coco2014
    unzip $temp_data_dir/coco2014/val2014.zip -d $temp_data_dir/coco2014
    rm $temp_data_dir/coco2014/train2014.zip $temp_data_dir/coco2014/val2014.zip

    unset temp_data_dir
    ```

### 2. Image Open-Vocabulary Segmentation Datasets

* ADE20K Dataset for Image Open-Vocabulary Segmentation

    Please refer to the following steps to download and process ADE20K dataset.
    ```bash
    cd $root_dir
    mkdir -p datas/img_ovseg/ade20k
    export temp_data_dir=$root_dir/datas/img_ovseg/ade20k
    # download dataset
    wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O $temp_data_dir/ADEChallengeData2016.zip
    # unzip dataset and rename the folder
    unzip $temp_data_dir/ADEChallengeData2016.zip -d $temp_data_dir
    mv $temp_data_dir/ADEChallengeData2016 $temp_data_dir/ade20k
    # remove zip file
    rm $temp_data_dir/ADEChallengeData2016.zip
    # convert dataset
    python $root_dir/xsam/xsam/tools/dataset_tools/prepare_ade20k_panoptic.py
    python $root_dir/xsam/xsam/tools/dataset_tools/prepare_ade20k_semantic.py
    python $root_dir/xsam/xsam/tools/dataset_tools/prepare_ade20k_instance.py

    unset temp_data_dir
    ```

### 3. Image Referring Segmentation Datasets

* RefCOCO/+/g Datasets for Image Referring Segmentation

    Please refer to the following steps to download and process RefCOCO/+/g datasets.
    ```bash
    cd $root_dir
    mkdir -p datas/img_refseg
    mkdir -p datas/img_refseg/images
    export temp_data_dir=$root_dir/datas/img_refseg
    # download dataset
    wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip -O $temp_data_dir/refcoco.zip
    wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip -O $temp_data_dir/refcoco+.zip
    wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip -O $temp_data_dir/refcocog.zip
    # unzip dataset
    unzip $temp_data_dir/refclef.zip -d $temp_data_dir
    unzip $temp_data_dir/refcoco.zip -d $temp_data_dir
    unzip $temp_data_dir/refcoco+.zip -d $temp_data_dir
    unzip $temp_data_dir/refcocog.zip -d $temp_data_dir
    rm $temp_data_dir/refclef.zip $temp_data_dir/refcoco.zip $temp_data_dir/refcoco+.zip $temp_data_dir/refcocog.zip    
    unset temp_data_dir

    # softlink coco2014 images
    ln -s $root_dir/datas/img_genseg/coco2014 $temp_data_dir/images/coco2014

    unset temp_data_dir
    ```
* GRefCOCO Datasets for Image Referring Segmentation

### 4. Image Reasoning Segmentation Datasets

* Lisa Dataset for Image Reasoning Segmentation

    Please refer to the [Lisa Dataset](https://github.com/JIA-Lab-research/LISA) to [download the dataset](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy), then refer to the following steps to process the dataset.

    ```bash
    cd $root_dir
    mkdir -p datas/img_reaseg/lisa
    export temp_data_dir=$root_dir/datas/img_reaseg/lisa
    mkdir -p $temp_data_dir/explanatory
    # suppose you have downloaded the dataset and put them in $temp_data_dir as below structure
    # img_reaseg
    # └── lisa
    #     ├── train.zip
    #     ├── val.zip
    #     ├── test.zip
    #     └── explanatory
    #         └── train.json

    # unzip dataset
    unzip $temp_data_dir/train.zip -d $temp_data_dir
    unzip $temp_data_dir/val.zip -d $temp_data_dir
    unzip $temp_data_dir/test.zip -d $temp_data_dir
    mv $temp_data_dir/train.json $temp_data_dir/explanatory/train.json
    rm $temp_data_dir/train.zip $temp_data_dir/val.zip $temp_data_dir/test.zip

    unset temp_data_dir
    ```

### 5. Image GCG Segmentation Datasets

* GranD-f Dataset for Image GCG Segmentation
    Download the [Dataset](https://drive.usercontent.google.com/download?id=1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_&export=download&authuser=0&confirm=t&uuid=bb3fe3db-b08c-48f0-9280-2e56c0910987&at=AN8xHooqlXNOUCiIJYVHFMBLtmVn%3A1754293785835)(GranDf_HA_images.zip) from Google Drive and put it in $root_dir/datas/img_gcgseg.
    ```bash
    cd $root_dir
    mkdir -p datas/img_gcgseg/grand_f/images
    export temp_data_dir=$root_dir/datas/img_gcgseg
    # download dataset
    hfd MBZUAI/GranD-f --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv GranD-f $temp_data_dir/annotations
    # unzip dataset
    unzip $temp_data_dir/GranD-f_HA_images.zip -d $temp_data_dir/images
    rm $temp_data_dir/GranD-f_HA_images.zip

    # download flickr30k images
    wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip -O $temp_data_dir/flickr30k-images.zip
    unzip $temp_data_dir/flickr30k-images.zip -d $temp_data_dir/images
    mv $temp_data_dir/images/flickr30k-images $temp_data_dir/images/flickr30k
    rm $temp_data_dir/flickr30k-images.zip

    # softlink coco2017 and coco2014 images
    ln -s $root_dir/datas/img_genseg/coco2017 $temp_data_dir/images/coco2017
    ln -s $root_dir/datas/img_genseg/coco2014 $temp_data_dir/images/coco2014

    unset temp_data_dir
    ```

### 6. Image Interactive Segmentation Datasets

* COCO-Interactive Dataset for Image Interactive Segmentation

    Please refer to the [COCO-Interactive Dataset](https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0) to [download the dataset](https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0) (PSALM_data.zip), then refer to the following steps to process the dataset.
    
    ```bash
    cd $root_dir
    mkdir -p datas/img_intseg/coco_int
    export temp_data_dir=$root_dir/datas/img_intseg/coco_int
    # download dataset
    wget https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0 -O $temp_data_dir/PSALM_data.zip
    # unzip dataset
    unzip $temp_data_dir/PSALM_data.zip -d $temp_data_dir
    # unzip dataset
    unzip $temp_data_dir/PSALM_data.zip -d $temp_data_dir
    mv $temp_data_dir/PSALM_data/coco_interactive_train_psalm.json $temp_data_dir/PSALM_data/coco_interactive_val_psalm.json $temp_data_dir/annotations
    ln -s $root_dir/datas/img_genseg/coco2017 $temp_data_dir/coco2017
    rm -rf $temp_data_dir/PSALM_data $temp_data_dir/PSALM_data.zip

    unset temp_data_dir
    ```

### 7. Image VGD Segmentation Datasets

* COCO-VGD Dataset for Image VGD Segmentation
    
    Please refer to the [COCO-VGD Dataset](https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations) to [download the dataset](https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations) (vgdseg_annotations), then refer to the following steps to process the dataset.
    
    ```bash
    cd $root_dir
    mkdir -p datas/img_vgdseg/coco_vgd
    export temp_data_dir=$root_dir/datas/img_vgdseg/coco_vgd
    mkdir -p $temp_data_dir/images
    # download dataset
    wget https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations -O $temp_data_dir/vgdseg_annotations.zip
    # unzip dataset
    # unzip dataset
    unzip $temp_data_dir/vgd_seg_annotations.zip -d $temp_data_dir
    mv $temp_data_dir/vgd_annotations $temp_data_dir/annotations
    ln -s $root_dir/datas/img_genseg/coco2017 $temp_data_dir/coco2017
    rm $temp_data_dir/vgd_seg_annotations.zip

    unset temp_data_dir
    ```

## Video Segmentation Datasets

### 1. Video Generic Segmentation Datasets
* VIPSeg Dataset for Video Panoptic Segmentation

    Please refer to the [VIPSeg Dataset](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/blob/main/README.md) to [download](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/blob/main/README.md#vipseg-dataset-a-large-scale-video-panoptic-segmentation-dataset) the dataset then [change VIPSeg to 720P and COCO format](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/blob/main/README.md#change-vipseg-to-720p-and-coco-format).

* VSPW Dataset for Video Semantic Segmentation

    Please refer to the [VSPW Dataset](https://github.com/VSPW-dataset/VSPW-dataset-download/README.md) to [download](https://github.com/VSPW-dataset/VSPW-dataset-download?tab=readme-ov-file#vspw-dataset-download) VSPW 480P dataset.

* YouTube-VIS 2019 Dataset for Video Instance Segmentation

    Please refer to the [YouTube-VIS 2019 Dataset](https://codalab.lisn.upsaclay.fr/competitions/6064#participate-get_data) to [download the video frames](https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC)(train_all_frames_zip, valid_all_frames_zip) and [download the annotations](https://drive.google.com/drive/folders/17Cc4PLu3YvKB0xfczElGBcqpqpaYz9Fx)(instances_train_subset.json, instances_val_sub.json). Then refer to the following steps to process the dataset.

    NOTE: `train_all_frames_zip` is only available in [Baidu Pan](https://pan.baidu.com/s/1x4bQ0AuyshS7-ZmE9I0FnQ)(access code: uu4q).
    ```bash
    cd $root_dir
    mkdir -p datas/vid_genseg/youtube_vis2019
    export temp_data_dir=$root_dir/datas/vid_genseg/youtube_vis2019
    cd $temp_data_dir/train_all_frames_zip
    7z x valid_all_frames.7z.001
    cd $temp_data_dir/valid_all_frames_zip
    7z x valid_all_frames.7z.001
    mkdir -p $temp_data_dir/train
    mkdir -p $temp_data_dir/valid
    mv train_all_frames/JPEGImages $temp_data_dir/train
    mv valid_all_frames/JPEGImages $temp_data_dir/valid
    mv instances_train_sub.json $temp_data_dir/train/instances.json
    mv instances_val_sub.json $temp_data_dir/valid/instances.json

    unset temp_data_dir
    ```

### 2. Video Open-Vocabulary Segmentation Datasets
* YouTube-VIS 2021 Dataset for Video Open-Vocabulary Segmentation

    Please refer to the [YouTube-VIS 2021 Dataset](https://codalab.lisn.upsaclay.fr/competitions/7680#participate-get_data) to [download the dataset](https://drive.google.com/drive/folders/1RAc7ETOeeV5nT2nbKMxG7QpjmgC4QHHF), then unzip them into $root_dir/datas/vid_ovseg/youtube_vis2021.

### 3. Video Referring Segmentation Datasets

* Youtube-RefVOS 2021 Dataset for Video Referring Segmentation

    Please refer to the [Youtube-RefVOS 2021 Dataset](https://competitions.codalab.org/competitions/29139#participate-get_data) to [download the dataset](https://drive.google.com/drive/folders/1J45ubR8Y24wQ6dzKOTkfpd9GS_F9A2kb), then unzip them into $root_dir/datas/vid_refseg/youtube_rvos2021.

* DAVIS-RefVOS 2017 Dataset for Video Referring Segmentation

    Please refer to the [DAVIS 2017 Dataset](https://competitions.codalab.org/competitions/29139#participate-get_data) to [download the dataset](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and [download the referring annotations](https://www.mpi-inf.mpg.de/fileadmin/inf/d2/khoreva/davis_text_annotations.zip), then unzip them into $root_dir/datas/vid_refseg/davis_rvos2017.

### 4. Video Reasoning Segmentation Datasets

* ReVOS Dataset for Video Reasoning Segmentation

### 5. Video GCG Segmentation Datasets

### 6. Video Object Segmentation Datasets

### 7. Video VGD Segmentation Datasets

## Image Chat & Video Chat Datasets

