# :bookmark_tabs: Datasets

## Structure

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
│       │   ├── train
│       │   └── val_test
│       └── images
│           ├── coco2014 -> ../../../img_genseg/coco2014
│           ├── coco2017 -> ../../../img_genseg/coco2017
│           ├── flickr30k
│           └── GranDf_HA_images
├── img_genseg
│   └── coco2017
│       ├── annotations
│       ├── panoptic_semseg_train2017
│       ├── panoptic_semseg_val2017
│       ├── panoptic_train2017
│       ├── panoptic_val2017
│       ├── stuff_train2017_pixelmaps
│       ├── stuff_val2017_pixelmaps
│       ├── test2017
│       ├── train2014 -> ../coco2014/train2014
│       ├── train2017
│       ├── val2014 -> ../coco2014/val2014
│       └── val2017
├── img_intseg
│   └── coco_int
│       ├── annotations
│       └── coco2017 -> ../../img_genseg/coco2017
├── img_ovseg
│   └── ade20k
│       ├── ade20k_instance_catid_mapping.txt
│       ├── ade20k_instance_imgCatIds.json
│       ├── ade20k_instance_train.json
│       ├── ade20k_instance_val.json
│       ├── ade20k_panoptic_train
│       ├── ade20k_panoptic_train.json
│       ├── ade20k_panoptic_val
│       ├── ade20k_panoptic_val.json
│       ├── annotations
│       ├── annotations_detectron2
│       ├── annotations_instance
│       ├── images
│       ├── objectInfo150.txt
│       └── sceneCategories.txt
├── img_reaseg
│   └── lisa
│       ├── explanatory
│       ├── test
│       ├── train
│       └── val
├── img_refseg
│   └── refcocos
│       ├── annotations
│       ├── grefcoco
│       ├── images
│       │   └── train2014 -> ../../../img_genseg/coco2014/train2014
│       ├── refclef
│       ├── refcoco
│       ├── refcoco+
│       ├── refcocog
│       └── refcocop -> refcoco+
├── img_vgdseg
│   └── coco_vgd
│       ├── annotations
│       └── coco2017 -> ../../img_genseg/coco2017
├── LMUData
│   ├── AI2D_TEST.tsv
│   ├── datasets--lmms-lab--Video-MME
│   │   ├── subtitle
│   │   ├── video
│   │   ├── videomme
│   │   └── Video-MME.tsv
│   ├── datasets--lmms-lab--VideoMMMU
│   │   ├── Adaptation
│   │   ├── images
│   │   ├── Perception
│   │   ├── VideoMMMU.tsv
│   │   └── videos
│   ├── datasets--longvideobench--LongVideoBench
│   │   ├── LongVideoBench.tsv
│   │   ├── lvb_test_wo_gt.json
│   │   ├── lvb_val.json
│   │   ├── subtitles
│   │   └── videos
│   ├── datasets--MLVU--MVLU
│   │   ├── MLVU
│   │   ├── MLVU_MCQ.tsv
│   │   ├── MLVU_OpenEnded.tsv
│   │   └── MLVU_Test
│   ├── datasets--opencompass--MMBench-Video
│   │   ├── MMBench-Video_a.json
│   │   ├── MMBench-Video_q.json
│   │   ├── MMBench-Video.tsv
│   │   ├── README.md
│   │   ├── video
│   │   └── video_pkl
│   ├── datasets--OpenGVLab--MVBench
│   │   ├── json
│   │   ├── MVBench.tsv
│   │   └── video
│   ├── GQA_TestDev_Balanced.tsv
│   ├── images
│   │   ├── AI2D_TEST
│   │   ├── GQA_TestDev_Balanced
│   │   ├── LongVideoBench
│   │   ├── MLVU_MCQ
│   │   ├── MLVU_OpenEnded
│   │   ├── MMBench
│   │   ├── MMBench_V11
│   │   ├── MMBench-Video
│   │   ├── MME
│   │   ├── MVBench
│   │   ├── POPE
│   │   ├── ScienceQA_TEST
│   │   ├── ScienceQA_VAL
│   │   ├── SEEDBench_IMG
│   │   ├── Video-MME
│   │   └── VideoMMMU
│   ├── MMBench_DEV_EN.tsv
│   ├── MMBench_DEV_EN_V11.tsv
│   ├── MME.tsv
│   ├── POPE_local.tsv
│   ├── POPE.tsv
│   ├── ScienceQA_TEST.tsv
│   ├── ScienceQA_VAL.tsv
│   └── SEEDBench_IMG.tsv
├── vid_chat
│   └── video_chatgpt
│       ├── ActivityNet
│       │   └── train
│       │       ├── v_00Dk03Jr70M.mp4
│       │       ├── ...
│       │       └── v_zzz_3yWpTXo.mp4
│       ├── videochatgpt_train.json
│       └── VideoInstruct100K.json
├── vid_gcgseg
│   └── video_glamm
│       ├── anet_gcg
│       ├── burst
│       ├── hcstvg_gcg
│       ├── mevis
│       ├── mevis_gcg
│       ├── processed
│       ├── video_gcg
│       ├── vidstg_gcg
│       ├── ytrvos21
│       ├── ytvis19
│       └── ytvos_gcg
├── vid_genseg
│   ├── vipseg_720p
│   │   ├── images
│   │   ├── panomasks
│   │   ├── panomasksRGB
│   │   ├── panoptic_gt_VIPSeg.json
│   │   ├── panoptic_gt_VIPSeg_train.json
│   │   ├── panoptic_gt_VIPSeg_val.json
│   │   ├── panoVIPSeg_categories.json
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── val.txt
│   ├── vspw_480p
│   │   ├── annotations
│   │   ├── data.txt
│   │   ├── images
│   │   ├── label_num_dic_final.json
│   │   ├── semmasks
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── val.txt
│   └── ytvis19
│       ├── train
│       └── valid
├── vid_objseg
│   └── ytvos19
│       ├── annotations
│       ├── test
│       ├── train
│       └── valid
├── vid_ovseg
│   └── ytvis21
│       ├── test
│       ├── train
│       └── valid
├── vid_reaseg
│   ├── reason_vos
│   │   ├── Annotations
│   │   ├── JPEGImages
│   │   └── meta_expressions.json
│   └── revos
│       ├── annotations
│       └── JPEGImages
├── vid_refseg
│   ├── davis17
│   │   ├── davis_supervised
│   │   ├── davis_text_annotations
│   │   ├── davis_unsupervised
│   │   ├── meta_expressions
│   │   ├── vid_refseg_davis17_train.json
│   │   └── vid_refseg_davis17_val.json
│   └── ytrvos21
│       ├── meta_expressions
│       ├── train
│       └── valid
└── vid_vgdseg
    ├── vipseg_vgd
    │   ├── annotations
    │   └── vipseg_720p
    └── ytvis_vgd
        ├── annotations
        └── ytvis19
```

## HFD Downloader Setting

We provide a custom downloader [`hfd`](../srcs/tools/hfd.sh) for downloading datasets from Hugging Face. You can use it to download datasets from Hugging Face.

```bash
chmod +x $PROJ_HOME/srcs/tools/hfd.sh
alias hfd="$PROJ_HOME/srcs/tools/hfd.sh"
```

## Image Segmentation Datasets

### 1. Image Generic Segmentation Datasets

* COCO Dataset for Image Generic Segmentation (Semantic, Instance, Panoptic)

    Please refer to the following steps to download and process COCO dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_genseg/coco2017
    export temp_data_dir=$PROJ_HOME/datas/img_genseg
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
    cd $PROJ_HOME
    mkdir -p datas/img_ovseg
    export temp_data_dir=$PROJ_HOME/datas/img_ovseg
    # download dataset
    wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O $temp_data_dir/ADEChallengeData2016.zip
    # unzip dataset and rename the folder
    unzip $temp_data_dir/ADEChallengeData2016.zip -d $temp_data_dir
    mv $temp_data_dir/ADEChallengeData2016 $temp_data_dir/ade20k
    # remove zip file
    rm $temp_data_dir/ADEChallengeData2016.zip
    # convert dataset
    python $PROJ_HOME/x2sam/x2sam/tools/dataset_tools/prepare_ade20k_panoptic.py
    python $PROJ_HOME/x2sam/x2sam/tools/dataset_tools/prepare_ade20k_semantic.py
    python $PROJ_HOME/x2sam/x2sam/tools/dataset_tools/prepare_ade20k_instance.py

    unset temp_data_dir
    ```

### 3. Image Referring Segmentation Datasets

* RefCOCO/+/g Datasets for Image Referring Segmentation

    Please refer to the following steps to download and process RefCOCO/+/g datasets.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_refseg/refcocos/images
    export temp_data_dir=$PROJ_HOME/datas/img_refseg/refcocos
    # download dataset
    wget https://web.archive.org/web/20220413011631/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip -O $temp_data_dir/refclef.zip
    wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip -O $temp_data_dir/refcoco.zip
    wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip -O $temp_data_dir/refcoco+.zip
    wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip -O $temp_data_dir/refcocog.zip
    # unzip dataset
    unzip $temp_data_dir/refclef.zip -d $temp_data_dir
    unzip $temp_data_dir/refcoco.zip -d $temp_data_dir
    unzip $temp_data_dir/refcoco+.zip -d $temp_data_dir
    unzip $temp_data_dir/refcocog.zip -d $temp_data_dir
    rm $temp_data_dir/refclef.zip $temp_data_dir/refcoco.zip $temp_data_dir/refcoco+.zip $temp_data_dir/refcocog.zip

    # softlink coco2014 images
    ln -s $PROJ_HOME/datas/img_genseg/coco2014/train2014 $temp_data_dir/images/train2014

    unset temp_data_dir
    ```
* gRefCOCO Datasets for Image Referring Segmentation

    Please refer to the following steps to download and process gRefCOCO datasets.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_refseg/refcocos/grefcoco
    export temp_data_dir=$PROJ_HOME/datas/img_refseg/refcocos/grefcoco
    cd $temp_data_dir
    hfd gRefCOCO/gRefCOCO --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv $temp_data_dir/gRefCOCO/* $temp_data_dir
    rm -rf $temp_data_dir/gRefCOCO

    unset temp_data_dir
    ```

### 4. Image Reasoning Segmentation Datasets

* Lisa Dataset for Image Reasoning Segmentation

    Please refer to the [Lisa Dataset](https://github.com/JIA-Lab-research/LISA) to [download the dataset](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy), then refer to the following steps to process the dataset.

    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_reaseg/lisa
    export temp_data_dir=$PROJ_HOME/datas/img_reaseg/lisa
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
    Download the [Dataset](https://drive.usercontent.google.com/download?id=1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_&export=download&authuser=0&confirm=t&uuid=bb3fe3db-b08c-48f0-9280-2e56c0910987&at=AN8xHooqlXNOUCiIJYVHFMBLtmVn%3A1754293785835)(GranDf_HA_images.zip) from Google Drive and put it in `$PROJ_HOME/datas/img_gcgseg/grand_f`.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_gcgseg/grand_f/images
    export temp_data_dir=$PROJ_HOME/datas/img_gcgseg/grand_f
    # download dataset
    hfd MBZUAI/GranD-f --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv $temp_data_dir/GranD-f $temp_data_dir/annotations
    # unzip dataset
    unzip $temp_data_dir/GranD-f_HA_images.zip -d $temp_data_dir/images
    rm $temp_data_dir/GranD-f_HA_images.zip

    # download flickr30k images
    wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip -O $temp_data_dir/flickr30k-images.zip
    unzip $temp_data_dir/flickr30k-images.zip -d $temp_data_dir/images
    mkdir -p $temp_data_dir/images/flickr30k/images
    mv $temp_data_dir/images/flickr30k-images $temp_data_dir/images/flickr30k/images/train
    rm $temp_data_dir/flickr30k-images.zip

    # softlink coco2017 and coco2014 images
    ln -s $PROJ_HOME/datas/img_genseg/coco2017 $temp_data_dir/images/coco2017
    ln -s $PROJ_HOME/datas/img_genseg/coco2014 $temp_data_dir/images/coco2014

    unset temp_data_dir
    ```

### 6. Image Interactive Segmentation Datasets

* COCO-Interactive Dataset for Image Interactive Segmentation

    Please refer to the [COCO-Interactive Dataset](https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0) to [download the dataset](https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0) (PSALM_data.zip), then refer to the following steps to process the dataset.
    
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_intseg/coco_int
    export temp_data_dir=$PROJ_HOME/datas/img_intseg/coco_int
    mkdir -p $temp_data_dir/annotations
    # download dataset
    wget https://drive.usercontent.google.com/download?id=1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3&export=download&authuser=0 -O $temp_data_dir/PSALM_data.zip
    # unzip dataset
    unzip $temp_data_dir/PSALM_data.zip -d $temp_data_dir
    mv $temp_data_dir/PSALM_data/coco_interactive_train_psalm.json $temp_data_dir/PSALM_data/coco_interactive_val_psalm.json $temp_data_dir/annotations
    ln -s $PROJ_HOME/datas/img_genseg/coco2017 $temp_data_dir/coco2017
    rm -rf $temp_data_dir/PSALM_data $temp_data_dir/PSALM_data.zip

    unset temp_data_dir
    ```

### 7. Image VGD Segmentation Datasets

* COCO-VGD Dataset for Image VGD Segmentation
    
    Please refer to the [COCO-VGD Dataset](https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations) to [download the dataset](https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations) (vgdseg_annotations), then refer to the following steps to process the dataset.
    
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_vgdseg/coco_vgd
    export temp_data_dir=$PROJ_HOME/datas/img_vgdseg/coco_vgd
    mkdir -p $temp_data_dir/annotations
    # download dataset
    wget https://huggingface.co/hao9610/X-SAM/tree/main/vgdseg_annotations -O $temp_data_dir/vgdseg_annotations.zip
    # unzip dataset
    unzip $temp_data_dir/vgdseg_annotations.zip -d $temp_data_dir
    mv $temp_data_dir/vgdseg_annotations/* $temp_data_dir/annotations
    ln -s $PROJ_HOME/datas/img_genseg/coco2017 $temp_data_dir/coco2017
    rm -rf $temp_data_dir/vgdseg_annotations $temp_data_dir/vgdseg_annotations.zip

    unset temp_data_dir
    ```

## Video Segmentation Datasets

### 1. Video Generic Segmentation Datasets
* VIPSeg Dataset for Video Panoptic Segmentation

    Please refer to the [VIPSeg Dataset](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/blob/main/README.md) to [download](https://drive.google.com/file/d/1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ) the dataset, then refer to the following steps to process the dataset.
    ```bash
    TODO
    cd $PROJ_HOME
    mkdir -p datas/vid_genseg/vipseg_720p
    export temp_data_dir=$PROJ_HOME/datas/vid_genseg/vipseg_720p
    # download dataset
    wget https://drive.google.com/file/d/1B13QUiE82xf7N6nVHclb4ErN-Zuai-sZ -O $temp_data_dir/vipseg.zip
    # unzip dataset
    unzip $temp_data_dir/vipseg.zip -d $temp_data_dir
    rm $temp_data_dir/vipseg.zip

    unset temp_data_dir
    ```

* VSPW Dataset for Video Semantic Segmentation

    Please refer to the [VSPW Dataset](https://github.com/VSPW-dataset/VSPW-dataset-download/README.md) to [download](https://github.com/VSPW-dataset/VSPW-dataset-download?tab=readme-ov-file#vspw-dataset-download) VSPW 480P dataset, then refer to the following steps to process the dataset.
    ```bash
    TODO
    cd $PROJ_HOME
    mkdir -p datas/vid_genseg/vspw_480p
    export temp_data_dir=$PROJ_HOME/datas/vid_genseg/vspw_480p
    # download dataset
    wget https://github.com/VSPW-dataset/VSPW-dataset-download?tab=readme-ov-file#vspw-dataset-download -O $temp_data_dir/vspw.zip
    # unzip dataset
    unzip $temp_data_dir/vspw.zip -d $temp_data_dir
    rm $temp_data_dir/vspw.zip

    unset temp_data_dir
    ```

* YouTube-VIS 2019 Dataset for Video Instance Segmentation

    Please refer to the [YouTube-VIS 2019 Dataset](https://codalab.lisn.upsaclay.fr/competitions/6064#participate-get_data) to [download the video frames](https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC)(train_all_frames_zip, valid_all_frames_zip) and [download the annotations](https://drive.google.com/drive/folders/17Cc4PLu3YvKB0xfczElGBcqpqpaYz9Fx)(instances_train_subset.json, instances_val_sub.json). Then refer to the following steps to process the dataset.

    NOTE: `train_all_frames_zip` is only available in [Baidu Pan](https://pan.baidu.com/s/1x4bQ0AuyshS7-ZmE9I0FnQ)(access code: uu4q).
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_genseg/ytvis19
    export temp_data_dir=$PROJ_HOME/datas/vid_genseg/ytvis19
    cd $temp_data_dir/train_all_frames_zip
    7z x train_all_frames.7z.001
    cd $temp_data_dir/valid_all_frames_zip
    7z x valid_all_frames.7z.001
    mkdir -p $temp_data_dir/train
    mkdir -p $temp_data_dir/valid
    mv $temp_data_dir/train_all_frames_zip/train_all_frames/JPEGImages $temp_data_dir/train
    mv $temp_data_dir/valid_all_frames_zip/valid_all_frames/JPEGImages $temp_data_dir/valid
    mv $temp_data_dir/instances_train_sub.json $temp_data_dir/train/train.json
    mv $temp_data_dir/instances_val_sub.json $temp_data_dir/valid/valid.json

    unset temp_data_dir
    ```

### 2. Video Open-Vocabulary Segmentation Datasets
* YouTube-VIS 2021 Dataset for Video Open-Vocabulary Segmentation

    Please refer to the [YouTube-VIS 2021 Dataset](https://codalab.lisn.upsaclay.fr/competitions/7680#participate-get_data) to [download the dataset](https://drive.google.com/drive/folders/1RAc7ETOeeV5nT2nbKMxG7QpjmgC4QHHF), then refer to the following steps to process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_ovseg/ytvis21
    export temp_data_dir=$PROJ_HOME/datas/vid_ovseg/ytvis21
    unzip train.zip -d $temp_data_dir/train
    unzip val.zip -d $temp_data_dir/valid
    unzip test.zip -d $temp_data_dir/test

    unset temp_data_dir
    ```

### 3. Video Referring Segmentation Datasets

* Youtube-RefVOS 2021 Dataset for Video Referring Segmentation

    Please refer to the [Youtube-RefVOS 2021 Dataset](https://competitions.codalab.org/competitions/29139#participate-get_data) to [download the dataset](https://drive.google.com/drive/folders/1J45ubR8Y24wQ6dzKOTkfpd9GS_F9A2kb), then refer to the following steps to process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_refseg/ytrvos21
    export temp_data_dir=$PROJ_HOME/datas/vid_refseg/ytrvos21
    unzip train.zip -d $temp_data_dir/train
    unzip val.zip -d $temp_data_dir/valid
    unzip meta_expressions.zip -d $temp_data_dir/meta_expressions

    unset temp_data_dir
    ```

* DAVIS-RefVOS 2017 Dataset for Video Referring Segmentation

    Please refer to the [DAVIS 2017 Dataset](https://competitions.codalab.org/competitions/29139#participate-get_data) to [download the dataset](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and [download the referring annotations](https://www.mpi-inf.mpg.de/fileadmin/inf/d2/khoreva/davis_text_annotations.zip), then refer to the following steps to process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_refseg/davis17
    export temp_data_dir=$PROJ_HOME/datas/vid_refseg/davis17
    unzip DAVIS-2017-trainval-480p.zip -d $temp_data_dir/DAVIS-2017-trainval-480p
    unzip davis_text_annotations.zip -d $temp_data_dir/davis_text_annotations
    mv $temp_data_dir/DAVIS-2017-trainval-480p/DAVIS/* $temp_data_dir
    rm -rf $temp_data_dir/DAVIS-2017-trainval-480p

    unset temp_data_dir
    ```

### 4. Video Reasoning Segmentation Datasets

* ReasonVOS Dataset for Video Reasoning Segmentation

    Please refer to the ReasonVOS dataset source to download the dataset, then organize it with the paths used by the training config.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_reaseg/reason_vos
    export temp_data_dir=$PROJ_HOME/datas/vid_reaseg/reason_vos
    # put JPEGImages, Annotations, and meta_expressions.json under $temp_data_dir

    unset temp_data_dir
    ```

* ReVOS Dataset for Video Reasoning Segmentation

    Please refer to the [ReVOS Dataset](https://github.com/cilinyan/VISA) to [download the dataset](https://mailsjlueducn-my.sharepoint.com/:f:/g/personal/yancl9918_mails_jlu_edu_cn/Ek3rFeIbNZtAv8kxVxr5n6sB6g3kbIThTscrq8cNt0zvgA?e=ZeuVzH), then refer to the following steps to process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_reaseg/revos
    export temp_data_dir=$PROJ_HOME/datas/vid_reaseg/revos
    mkdir -p $temp_data_dir/annotations
    # download dataset and put them in $temp_data_dir
    unzip JPEGImages.zip -d $temp_data_dir/JPEGImages
    mv *.json $temp_data_dir/annotations

    unset temp_data_dir
    ```

### 5. Video GCG Segmentation Datasets

* VideoGLaMM Dataset for Video GCG Segmentation

    Please refer to the [VideoGLaMM Dataset](https://github.com/mbzuai-oryx/VideoGLaMM) to [download the dataset](https://github.com/mbzuai-oryx/VideoGLaMM/blob/main/Dataset.md), then refer to the following steps to process the dataset.
    
    `NOTE`: As the original donwload link is not available for long time downloading, we download them and upload it to [Baidu Pan](https://pan.baidu.com/s/1_mQOqdI6j67R6MG8Xc3ZZA)(access code: xsam).
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_gcgseg/video_glamm
    export temp_data_dir=$PROJ_HOME/datas/vid_gcgseg/video_glamm

    # anet_gcg
    cd $temp_data_dir
    unzip activitynet_entities_gcg.zip -d $temp_data_dir
    mv activitynet_entities_gcg $temp_data_dir/anet_gcg

    # mevis_gcg
    cd $temp_data_dir
    unzip mevis_gcg.zip -d $temp_data_dir
    hfd FudanCVL/MeViS --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv MeViS $temp_data_dir/mevis   # download mevis dataset
    unzip hcstvg_gcg.zip -d $temp_data_dir
    unzip ytvos_gcg.zip -d $temp_data_dir
    ln -s $PROJ_HOME/datas/vid_refseg/ytrvos21 $temp_data_dir/ytrvos21

    # video_gcg
    cd $temp_data_dir
    unzip burst_ytvis_gcg.zip -d $temp_data_dir
    ln -s $PROJ_HOME/datas/vid_genseg/ytvis19 $temp_data_dir/video_gcg/yt19
    cd video_gcg
    mkdir -p $temp_data_dir/video_gcg/burst
    hfd chengyenhsieh/TAO-Amodal --hf_username YOUR_NAME --hf_token YOUR_TOKEN --tool aria2c -x 16 --save_dir ./ --dataset --include frames     # download burst dataset
    mv TAO-Amodal/frames/* $temp_data_dir/video_gcg/burst
    rm -rf TAO-Amodal
    find . -maxdepth 1 -name "$temp_data_dir/video_gcg/burst/train/*.zip" -print0 | xargs -0 -P $(nproc) -I {} unzip -q {} -d $temp_data_dir/video_gcg/burst/train
    find . -maxdepth 1 -name "$temp_data_dir/video_gcg/burst/val/*.zip" -print0 | xargs -0 -P $(nproc) -I {} unzip -q {} -d $temp_data_dir/video_gcg/burst/val
    rm -rf $temp_data_dir/video_gcg/burst/train/*.zip
    rm -rf $temp_data_dir/video_gcg/burst/val/*.zip
    
    # vidstg_gcg
    unzip vidstg_gcg.zip -d $temp_data_dir
    unzip videoGLaMM_processed.zip -d $temp_data_dir
    mv $temp_data_dir/processed $temp_data_dir

    # hcstvg_gcg
    cd $temp_data_dir
    unzip hcstvg_gcg.zip -d $temp_data_dir

    unset temp_data_dir
    ```

### 6. Video Object Segmentation Datasets

* YouTube-VOS 2019 Dataset for Video Object Segmentation

    Please refer to the [YouTube-VOS 2019 Dataset](https://competitions.codalab.org/competitions/29139#participate-get_data) to [download the dataset](https://drive.google.com/drive/folders/1XwjQ-eysmOb7JdmJAwfVOBZX-aMbHccC), then refer to the following steps to process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_objseg/ytvos19
    export temp_data_dir=$PROJ_HOME/datas/vid_objseg/ytvos19
    unzip train.zip -d $temp_data_dir/train
    unzip val.zip -d $temp_data_dir/valid

    unset temp_data_dir
    ```

### 7. Video VGD Segmentation Datasets

* VIPSeg-VGD Dataset for Video VGD Segmentation

    Please refer to the following steps to download and process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_vgdseg/vipseg_vgd/annotations
    export temp_data_dir=$PROJ_HOME/datas/vid_vgdseg/vipseg_vgd
    cd $temp_data_dir
    hfd hao9610/VideoVGD --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv VideoVGD/vid_vgdseg_vipseg*.json $temp_data_dir/annotations
    ln -s $PROJ_HOME/datas/vid_genseg/vipseg_720p $temp_data_dir/vipseg_720p

    unset temp_data_dir
    ```

* YTVIS-VGD Dataset for Video VGD Segmentation

    Please refer to the following steps to download and process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_vgdseg/ytvis_vgd
    export temp_data_dir=$PROJ_HOME/datas/vid_vgdseg/ytvis_vgd
    cd $temp_data_dir
    hfd hao9610/VideoVGD --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mkdir -p $temp_data_dir/annotations
    mv VideoVGD/vid_vgdseg_yt19*.json $temp_data_dir/annotations
    ln -s $PROJ_HOME/datas/vid_genseg/ytvis19 $temp_data_dir/ytvis19

    unset temp_data_dir
    ```

## Image Chat & Video Chat Datasets

* LLaVA-Instruct Dataset for Image Chat

    Please refer to the [LLaVA Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) to download the dataset, then refer to the following steps to process the dataset.
    ```bash
    cd $PROJ_HOME
    mkdir -p datas/img_chat/llava
    export temp_data_dir=$PROJ_HOME/datas/img_chat/llava
    hfd liuhaotian/LLaVA-Instruct-150K --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    hfd liuhaotian/LLaVA-Pretrain --tools aria2c -x 8 --save_dir $temp_data_dir --dataset

    mkdir $temp_data_dir/llava_images
    # Please prepare the GQA, OCR_VQA, TEXTVQA, VG datasets and put them in $temp_data_dir as below structure
    # llava_images
    # ├── coco
    # ├── gqa
    # ├── ocr_vqa
    # ├── textvqa
    # └── vg
    
    # COCO Dataset
    ln -s $PROJ_HOME/datas/img_genseg/coco2017 $temp_data_dir/llava_images/coco

    # GQA Dataset
    cd $temp_data_dir/llava_images
    mkdir $temp_data_dir/llava_images/gqa
    cd $temp_data_dir/llava_images/gqa
    wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
    unzip images.zip
    rm images.zip

    # OCR_VQA Dataset
    cd $temp_data_dir/llava_images
    hfd ej2/llava-ocr-vqa --tools aria2c -x 8 --save_dir $temp_data_dir/llava_images --dataset
    tar -xvf $temp_data_dir/llava_images/llava-ocr-vqa/ocr_vqa.tar -C $temp_data_dir/llava_images/ocr_vqa

    # TEXTVQA Dataset
    cd $temp_data_dir/llava_images
    wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
    unzip train_val_images.zip -d $temp_data_dir/llava_images/textvqa/train_images
    rm train_val_images.zip

    # VG Dataset
    cd $temp_data_dir/llava_images
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O $temp_data_dir/llava_images/vg/images.zip
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O $temp_data_dir/llava_images/vg/images2.zip
    unzip $temp_data_dir/llava_images/vg/images.zip -d $temp_data_dir/llava_images/vg
    unzip $temp_data_dir/llava_images/vg/images2.zip -d $temp_data_dir/llava_images/vg
    rm $temp_data_dir/llava_images/vg/images.zip
    rm $temp_data_dir/llava_images/vg/images2.zip

    unset temp_data_dir
    ```

* VideoInstruct100K Dataset for Video Chat

    Please refer to the [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) to [download the dataset](https://mbzuaiac-my.sharepoint.com/personal/hanoona_bangalath_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhanoona%5Fbangalath%5Fmbzuai%5Fac%5Fae%2FDocuments%2FVideo%2DChatGPT%2FData%5FCode%5FModel%5FRelease%2FData%2Ftraining%5Fvideos&viewid=7813d070%2D5dd9%2D4b3b%2D873e%2De519f40b7340), then refer to the following steps to process the dataset.

    ```bash
    cd $PROJ_HOME
    mkdir -p datas/vid_chat/video_chatgpt
    export temp_data_dir=$PROJ_HOME/datas/vid_chat/video_chatgpt
    hfd MBZUAI/VideoInstruct-100K --tools aria2c -x 8 --save_dir $temp_data_dir --dataset
    mv VideoInstruct-100K/VideoInstruct100K.json $temp_data_dir
    rm -rf $temp_data_dir/VideoInstruct-100K
    mkdir -p $temp_data_dir/ActivityNet/train
    find training_videos -name "*.tar" -print0 | xargs -0 -P $(nproc) -I {} tar -xvf {} -C $temp_data_dir/ActivityNet/train/

    unset temp_data_dir
    ```

* Image Chat & Video Chat Benchmark Datasets

    `VLMEvalKit` will automatically download the image chat and video chat benchmark datasets for evaluation.
