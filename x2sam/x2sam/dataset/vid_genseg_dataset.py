import copy
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from panopticapi.utils import rgb2id
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from x2sam.structures import BoxMode
from x2sam.utils import comm
from x2sam.utils.logging import print_log
from x2sam.utils.palette import get_palette

from .utils.catalog import MetadataCatalog
from .utils.format import format_cat_name
from .utils.mask import decode_mask
from .utils.panoptic import IdGenerator
from .utils.ytvos import YTVOS
from .vid_base_dataset import VidBaseDataset

# invalid catids in VSPW val set
VSPW_VAL_INVALID_CATIDS = [252, 253]


class VidGenSegDataset(VidBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init__(
        self,
        *args,
        use_full_cat=False,
        use_variant_cat=False,
        caption_data_path=None,
        pan_segmap_folder=None,
        sem_segmap_folder=None,
        imgmap_suffix=".jpg",
        segmap_suffix=".png",
        label_shift=0,
        label_file=None,
        split_file=None,
        use_threads=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            use_full_cat=use_full_cat,
            use_variant_cat=use_variant_cat,
            caption_data_path=caption_data_path,
            pan_segmap_folder=pan_segmap_folder,
            sem_segmap_folder=sem_segmap_folder,
            imgmap_suffix=imgmap_suffix,
            segmap_suffix=segmap_suffix,
            label_shift=label_shift,
            label_file=label_file,
            split_file=split_file,
            use_threads=use_threads,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.use_full_cat = kwargs.get("use_full_cat", False)
        self.use_variant_cat = kwargs.get("use_variant_cat", False)
        self.caption_data_path = kwargs.get("caption_data_path", None)
        self.pan_segmap_folder = kwargs.get("pan_segmap_folder", None)
        self.sem_segmap_folder = kwargs.get("sem_segmap_folder", None)
        self.imgmap_suffix = kwargs.get("imgmap_suffix", ".jpg")
        self.segmap_suffix = kwargs.get("segmap_suffix", ".png")
        self.label_shift = kwargs.get("label_shift", 0)
        self.label_file = kwargs.get("label_file", None)
        self.split_file = kwargs.get("split_file", None)
        self.use_threads = kwargs.get("use_threads", True)

    def _set_instance_metadata(self, coco_data, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_colors = (
            [x["color"] for x in sorted(cats, key=lambda x: x["id"])]
            if "color" in cats[0]
            else get_palette("random", len(cats))
        )
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)}
        cat_id_to_name = {x["id"]: x["name"] for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        thing_cats = [x for x in cats]
        thing_cat_ids = [x["id"] for x in thing_cats]
        thing_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in thing_cat_ids
        }
        thing_cat_id_to_name = {thing_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in thing_cats}
        thing_cat_id_to_color = {
            thing_cat_id_to_contiguous_id[x["id"]]: cat_colors[thing_cat_id_to_contiguous_id[x["id"]]]
            for x in thing_cats
        }

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path if gt_json is None else gt_json,
            data_name=self.data_name,
            dataset_classes=cat_id_to_name,
            dataset_colors=cat_id_to_color,
            thing_classes=thing_cat_id_to_name,
            thing_colors=thing_cat_id_to_color,
            dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
            thing_dataset_id_to_contiguous_id=thing_cat_id_to_contiguous_id,
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            label_shift=self.label_shift,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _set_semantic_metadata(self, coco_data, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        cats = coco_data["categories"]
        for cat in cats:
            cat["id"] -= self.label_shift
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_colors = (
            [x["color"] for x in sorted(cats, key=lambda x: x["id"])]
            if "color" in cats[0]
            else get_palette("random", len(cats))
        )
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)}
        cat_id_to_name = {x["id"]: x["name"] for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        stuff_cats = [x for x in cats]
        stuff_cat_ids = [x["id"] for x in stuff_cats]
        stuff_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in stuff_cat_ids
        }
        stuff_cat_id_to_name = {stuff_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in stuff_cats}
        stuff_cat_id_to_color = {
            stuff_cat_id_to_contiguous_id[x["id"]]: cat_colors[stuff_cat_id_to_contiguous_id[x["id"]]]
            for x in stuff_cats
        }

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path if gt_json is None else gt_json,
            segmap_suffix=self.segmap_suffix,
            pan_segmap_folder=self.pan_segmap_folder,
            sem_segmap_folder=self.sem_segmap_folder,
            data_name=self.data_name,
            dataset_classes=cat_id_to_name,
            dataset_colors=cat_id_to_color,
            stuff_classes=stuff_cat_id_to_name,
            stuff_colors=stuff_cat_id_to_color,
            dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
            stuff_dataset_id_to_contiguous_id=stuff_cat_id_to_contiguous_id,
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            invalid_catids=VSPW_VAL_INVALID_CATIDS,
            background_label=self.background_label,
            label_shift=self.label_shift,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _set_panoptic_metadata(self, coco_data, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_colors = (
            [x["color"] for x in sorted(cats, key=lambda x: x["id"])]
            if "color" in cats[0]
            else get_palette("random", len(cats))
        )
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)}
        cat_id_to_name = {x["id"]: x["name"] for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        thing_cats = [x for x in cats if x.get("isthing", None) == 1]
        thing_cat_ids = [x["id"] for x in thing_cats]
        stuff_cats = [x for x in cats if x.get("isthing", None) == 0]
        stuff_cat_ids = [x["id"] for x in stuff_cats]
        thing_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in thing_cat_ids
        }
        stuff_cat_id_to_contiguous_id = {
            cat_id: cont_id for cont_id, cat_id in enumerate(cat_ids) if cat_id in stuff_cat_ids
        }
        thing_cat_id_to_name = {thing_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in thing_cats}
        stuff_cat_id_to_name = {stuff_cat_id_to_contiguous_id[x["id"]]: x["name"] for x in stuff_cats}
        thing_cat_id_to_color = {
            thing_cat_id_to_contiguous_id[x["id"]]: cat_colors[thing_cat_id_to_contiguous_id[x["id"]]]
            for x in thing_cats
        }
        stuff_cat_id_to_color = {
            stuff_cat_id_to_contiguous_id[x["id"]]: cat_colors[stuff_cat_id_to_contiguous_id[x["id"]]]
            for x in stuff_cats
        }
        id_generator = IdGenerator({cat["id"]: cat for cat in cats})

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path if gt_json is None else gt_json,
            segmap_suffix=self.segmap_suffix,
            pan_segmap_folder=self.pan_segmap_folder,
            sem_segmap_folder=self.sem_segmap_folder,
            data_name=self.data_name,
            dataset_classes=cat_id_to_name,
            dataset_colors=cat_id_to_color,
            thing_classes=thing_cat_id_to_name,
            thing_colors=thing_cat_id_to_color,
            stuff_classes=stuff_cat_id_to_name,
            stuff_colors=stuff_cat_id_to_color,
            dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
            thing_dataset_id_to_contiguous_id=thing_cat_id_to_contiguous_id,
            stuff_dataset_id_to_contiguous_id=stuff_cat_id_to_contiguous_id,
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            label_shift=self.label_shift,
            id_generator=id_generator,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _set_metadata(self, coco_data, **kwargs):
        if "semantic" in self.data_name:
            self._set_semantic_metadata(coco_data, **kwargs)
        elif "instance" in self.data_name:
            self._set_instance_metadata(coco_data, **kwargs)
        elif "panoptic" in self.data_name:
            self._set_panoptic_metadata(coco_data, **kwargs)
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

    def _sample_cats(self, cat_ids, anns):
        def _sample_cat_ids(cat_ids, num_class=10000):
            cat_ids = (
                random.sample(cat_ids, min(len(cat_ids), num_class))
                if self.use_random_cat or self.use_variant_cat
                else cat_ids
            )
            return cat_ids

        num_class = min(len(cat_ids), self.num_class)
        pos_cat_ids = sorted(set(ann["category_id"] for ann in anns))
        neg_cat_ids = sorted(set(cat_ids) - set(pos_cat_ids))
        if self.data_mode == "train" and self.use_variant_cat:
            if random.random() < 0.5:
                sampled_cat_ids = _sample_cat_ids(cat_ids, num_class)
            else:
                sample_neg_cat_ids = _sample_cat_ids(
                    neg_cat_ids, random.randint(0, max(num_class - len(pos_cat_ids), 0))
                )
                sampled_cat_ids = _sample_cat_ids(pos_cat_ids + sample_neg_cat_ids, num_class)
        elif self.data_mode == "train" and self.use_full_cat:
            sampled_cat_ids = _sample_cat_ids(cat_ids, num_class)
        elif self.data_mode == "train" and not self.use_full_cat:
            sampled_cat_ids = _sample_cat_ids(pos_cat_ids)
        else:
            sampled_cat_ids = cat_ids

        for ann in anns:
            ann["category_id"] = sampled_cat_ids.index(ann["category_id"])

        return sampled_cat_ids

    def _process_batch_data_worker(self, video_list):
        videos = []
        annotations = []
        ann_cnt = 0
        video_cnt = 0

        for video in video_list:
            try:
                file_names = sorted(os.listdir(osp.join(self.video_folder, video)))
                if not file_names:
                    continue
                width, height = Image.open(osp.join(self.video_folder, video, file_names[0])).size

                vid_frames_info = []
                semseg_map_paths = [
                    osp.join(
                        self.sem_segmap_folder,
                        video,
                        file_name.replace(self.imgmap_suffix, self.segmap_suffix),
                    )
                    for file_name in file_names
                ]

                for file_name, semseg_map_path in zip(file_names, semseg_map_paths):
                    semseg_map = Image.open(semseg_map_path)
                    semseg_map = np.array(semseg_map, dtype=np.int64)
                    if self.label_shift != 0:
                        # 0 and 255 are ignored values
                        semseg_map[semseg_map == 0] = self.ignore_value
                        semseg_map = semseg_map + self.label_shift
                        semseg_map[semseg_map == (self.ignore_value + self.label_shift)] = self.ignore_value
                    unique_cats = [
                        cat_id
                        for cat_id in np.unique(semseg_map).tolist()
                        if cat_id != self.ignore_value and cat_id not in VSPW_VAL_INVALID_CATIDS
                    ]
                    vid_frames_info.append(
                        {
                            "file_name": file_name,
                            "unique_cats": unique_cats,
                        }
                    )

                all_cat_ids = sorted(
                    set([cat_id for frame_data in vid_frames_info for cat_id in frame_data["unique_cats"]])
                )
                for cat_id in all_cat_ids:
                    annotations.append(
                        {
                            "id": ann_cnt,
                            "video_id": video_cnt,
                            "iscrowd": 0,
                            "category_id": cat_id,
                        }
                    )
                    ann_cnt += 1

                videos.append(
                    {
                        "id": video_cnt,
                        "file_names": [osp.join(video, fn) for fn in file_names],
                        "height": height,
                        "width": width,
                    }
                )
                video_cnt += 1
            except Exception as e:
                print_log(f"Error processing video {video}: {e}", logger="current")
                continue
        return videos, annotations

    def convert_vspw_to_coco_format(self):
        with open(self.label_file, "r") as f:
            cats = [{"id": int(v) + self.label_shift, "name": k} for k, v in json.load(f).items()]
        with open(self.split_file, "r") as f:
            video_list = sorted([v.strip() for v in f.readlines()])

        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(f"Using {num_workers} workers for processing videos", logger="current")

        batch_size = max(1, min(16, len(video_list) // num_workers))
        batches = [video_list[i : i + batch_size] for i in range(0, len(video_list), batch_size)]

        videos = []
        annotations = []
        ann_id = 0
        video_id = 0

        if self.use_threads:
            print_log(
                f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks",
                logger="current",
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._process_batch_data_worker, batch) for batch in batches]
                for future in tqdm(futures, desc="Loading VSPW dataset", ncols=80):
                    batch_videos, batch_annotations = future.result()

                    if not batch_videos:
                        continue

                    for batch_ann in batch_annotations:
                        batch_ann["id"] += ann_id
                        batch_ann["video_id"] += video_id

                    for vid in batch_videos:
                        vid["id"] += video_id

                    videos.extend(batch_videos)
                    annotations.extend(batch_annotations)

                    ann_id += len(batch_annotations)
                    video_id += len(batch_videos)
        else:
            with mp.Pool(num_workers) as pool:
                for batch_res in tqdm(
                    pool.imap(self._process_batch_data_worker, batches),
                    total=len(batches),
                    desc="Loading VSPW dataset",
                ):
                    batch_videos, batch_annotations = batch_res

                    if not batch_videos:
                        continue

                    for batch_ann in batch_annotations:
                        batch_ann["id"] += ann_id
                        batch_ann["video_id"] += video_id

                    for vid in batch_videos:
                        vid["id"] += video_id
                    videos.extend(batch_videos)
                    annotations.extend(batch_annotations)

                    ann_id += len(batch_annotations)
                    video_id += len(batch_videos)
        coco_data = {
            "categories": cats,
            "videos": videos,
            "annotations": annotations,
            "info": {"description": "vspw dataset"},
        }

        os.makedirs(osp.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f)

        return coco_data

    def _load_semantic_genseg_data(self, coco_data):
        # vspw dataset
        rets = []
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats if cat["id"] >= 0])
        cat_ids2names = {cat["id"]: format_cat_name(cat["name"]) for cat in cats}

        coco_api = YTVOS(dataset=coco_data)
        vid_ids = sorted(coco_api.getVidIds())
        for vid_id in vid_ids:
            _vid_info = coco_api.loadVids(vid_id)[0]
            ann_ids = coco_api.getAnnIds(vidIds=[vid_id])
            _image_files = sorted(_vid_info["file_names"])
            _anns = coco_api.loadAnns(ann_ids)
            caption = None

            if len(_anns) == 0:
                self.woann_cnt += 1
                continue

            anns = []
            image_files = []
            for i in range(len(_image_files)):
                img_anns = []
                for _ann in _anns:
                    if int(_ann.get("iscrowd", 0)) != 0:
                        continue

                    ann = {
                        "id": _ann["id"] * 100 + i,
                        "iscrowd": int(_ann.get("iscrowd", 0)),
                        "category_id": _ann["category_id"],
                    }
                    img_anns.append(ann)

                if len(img_anns) == 0:
                    continue
                image_files.append(_image_files[i])
                anns.append(img_anns)

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            # random sample cat_ids to shuffle the order
            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns)
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                self.woann_cnt += 1
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                cur_anns = copy.deepcopy(
                    sampled_anns[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_anns
                )
                # filter frames with no annotations
                cur_image_files = [image_file for image_file, ann in zip(cur_image_files, cur_anns) if len(ann) > 0]
                cur_anns = [ann for ann in cur_anns if len(ann) > 0]
                if len(cur_image_files) == 0 or len(cur_anns) == 0:
                    continue
                if len(cur_image_files) < 2 and self.num_frames > 1:
                    continue

                sampled_cat_ids = self._sample_cats(cat_ids, [_ann for ann in cur_anns for _ann in ann])
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

                vid_info = {
                    "video_id": _vid_info["id"],
                    "video_name": osp.dirname(cur_image_files[0]),
                    "file_names": cur_image_files,
                    "height": _vid_info["height"],
                    "width": _vid_info["width"],
                    "chunk_id": i,
                }

                rets.append(
                    {
                        "video_id": _vid_info["id"],
                        "image_files": cur_image_files,
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "caption": caption,
                        "annotations": cur_anns,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "video_info": vid_info,
                    }
                )

        return rets

    def _load_instance_genseg_data(self, coco_data):
        rets = []
        # youtube_vis dataset
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_ids2names = {cat["id"]: format_cat_name(cat["name"]) for cat in cats}
        coco_api = YTVOS(dataset=coco_data)
        vid_ids = sorted(coco_api.getVidIds())
        for vid_id in vid_ids:
            _vid_info = coco_api.loadVids(vid_id)[0]
            ann_ids = coco_api.getAnnIds(vidIds=[vid_id])
            _image_files = sorted(_vid_info["file_names"])
            _anns = coco_api.loadAnns(ann_ids)

            anns = []
            images = []
            image_id = 0
            for i in range(len(_image_files)):
                img_anns = []
                for _ann in _anns:
                    if int(_ann.get("iscrowd", 0)) != 0 or _ann["segmentations"][i] is None:
                        continue

                    segmentation = _ann["segmentations"][i]
                    if isinstance(segmentation, dict) and isinstance(segmentation["counts"], list):
                        segmentation = mask_utils.frPyObjects(segmentation, *segmentation["size"])
                        segmentation["counts"] = segmentation["counts"].decode("utf-8")

                    ann = {
                        "id": _ann["id"] * 100 + i,
                        "image_id": image_id,
                        "segmentation": segmentation,
                        "bbox": [0.0, 0.0, _vid_info["width"], _vid_info["height"]],  # placeholder
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "area": _ann["areas"][i],
                        "iscrowd": int(_ann.get("iscrowd", 0)),
                        "category_id": _ann["category_id"],
                    }
                    img_anns.append(ann)

                if len(img_anns) == 0:
                    continue
                images.append(
                    {
                        "id": image_id,
                        "file_name": _image_files[i],
                        "height": _vid_info["height"],
                        "width": _vid_info["width"],
                    }
                )
                anns.append(img_anns)
                image_id += 1

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            # random sample cat_ids to shuffle the order
            sampled_images, sampled_anns = self._sample_frames(images, anns)
            if len(sampled_images) == 0 or len(sampled_anns) == 0:
                self.woann_cnt += 1
                continue
            for i in range(max(math.ceil(len(sampled_images) / self.num_frames), 1)):
                cur_images = copy.deepcopy(
                    sampled_images[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_images
                )
                cur_anns = copy.deepcopy(
                    sampled_anns[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_anns
                )
                cur_image_files = [image["file_name"] for image in cur_images]
                # filter frames with no annotations
                cur_image_files = [image_file for image_file, ann in zip(cur_image_files, cur_anns) if len(ann) > 0]
                cur_anns = [ann for ann in cur_anns if len(ann) > 0]
                if len(cur_image_files) == 0 or len(cur_anns) == 0:
                    continue
                if len(cur_image_files) < 2 and self.num_frames > 1:
                    continue

                sampled_cat_ids = self._sample_cats(cat_ids, [_ann for ann in cur_anns for _ann in ann])
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

                vid_info = {
                    "video_id": vid_id,
                    "video_name": osp.dirname(cur_image_files[0]),
                    "images": cur_images,
                    "file_names": cur_image_files,
                    "height": _vid_info["height"],
                    "width": _vid_info["width"],
                    "chunk_id": i,
                }

                rets.append(
                    {
                        "video_id": _vid_info["id"],
                        "image_files": cur_image_files,
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "annotations": cur_anns,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "video_info": vid_info,
                    }
                )

        return rets

    def _load_panoptic_genseg_data(self, coco_data):
        rets = []
        cats = coco_data["categories"]
        cat_ids = sorted([cat["id"] for cat in cats])
        cat_ids2names = {cat["id"]: format_cat_name(cat["name"]) for cat in cats}
        # vipseg dataset
        coco_data["videos"] = sorted(coco_data["videos"], key=lambda x: x["video_id"])
        coco_data["annotations"] = sorted(coco_data["annotations"], key=lambda x: x["video_id"])

        for _vid_info, _ann_info in zip(coco_data["videos"], coco_data["annotations"]):
            vid_id = _vid_info["video_id"]
            _images = sorted(_vid_info["images"], key=lambda x: x["id"])
            _anns = sorted(_ann_info["annotations"], key=lambda x: x["image_id"])
            assert vid_id == _ann_info["video_id"]
            assert [image["id"] for image in _images] == [ann["image_id"] for ann in _anns]
            assert (
                len(set([image["height"] for image in _images])) == 1
                and len(set([image["width"] for image in _images])) == 1
            )

            _segments_info = [ann["segments_info"] for ann in _anns]
            sampled_images, sampled_segments_info = self._sample_frames(_images, _segments_info)
            if len(sampled_images) == 0 or len(sampled_segments_info) == 0:
                self.woann_cnt += 1
                continue
            for i in range(max(math.ceil(len(sampled_images) / self.num_frames), 1)):
                cur_images = copy.deepcopy(
                    sampled_images[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_images
                )
                cur_segments_info = copy.deepcopy(
                    sampled_segments_info[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_segments_info
                )
                # filter frames with no annotations
                cur_images = [image for image in cur_images if len(image) > 0]
                cur_segments_info = [segment_info for segment_info in cur_segments_info if len(segment_info) > 0]
                if len(cur_images) == 0 or len(cur_segments_info) == 0:
                    continue
                if len(cur_images) < 2 and self.num_frames > 1:
                    continue
                cur_image_files = [f"{vid_id}/{image['file_name']}" for image in cur_images]
                seg_map_path = [
                    image_file.replace(self.imgmap_suffix, self.segmap_suffix) for image_file in cur_image_files
                ]

                if len(cur_segments_info) == 0:
                    self.woann_cnt += 1
                    continue

                cur_anns = []
                for segment_info in cur_segments_info:
                    for info in segment_info:
                        cur_anns.append(info)

                # random sample cat_ids to shuffle the order
                sampled_cat_ids = self._sample_cats(cat_ids, cur_anns)
                sampled_cat_names = [cat_ids2names[cat_id] for cat_id in sampled_cat_ids]

                vid_info = {
                    "video_id": vid_id,
                    "video_name": vid_id,
                    "images": cur_images,
                    "file_names": cur_image_files,
                    "height": cur_images[0]["height"],
                    "width": cur_images[0]["width"],
                    "chunk_id": i,
                }

                rets.append(
                    {
                        "video_id": vid_id,
                        "image_files": cur_image_files,
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "seg_map": seg_map_path,
                        "segments_info": cur_segments_info,
                        "sampled_cats": sampled_cat_names,
                        "sampled_labels": sampled_cat_ids,
                        "video_info": vid_info,
                    }
                )

        return rets

    def _decode_semantic_genseg_data(self, data_dict):
        image_files = data_dict["image_files"]
        image_sizes = data_dict["image_sizes"]
        sampled_labels = data_dict["sampled_labels"]
        annotations = data_dict["annotations"]
        mask_labels = []
        class_labels = []

        for image_file, image_size, _anns in zip(image_files, image_sizes, annotations):
            _mask_labels = []
            _class_labels = []
            height, width = image_size
            semseg_map = Image.open(
                osp.join(self.sem_segmap_folder, image_file.replace(self.imgmap_suffix, self.segmap_suffix))
            )
            semseg_map = np.array(semseg_map, dtype=np.int64)
            if self.label_shift != 0:
                # 0 and 255 are ignored values
                semseg_map[semseg_map == 0] = self.ignore_value
                semseg_map = semseg_map + self.label_shift
                semseg_map[semseg_map == (self.ignore_value + self.label_shift)] = self.ignore_value
            ann_cat_ids = set([ann["category_id"] for ann in _anns])
            semseg_cat_ids = sorted(
                [
                    cat_id
                    for cat_id in np.unique(semseg_map).tolist()
                    if cat_id != self.ignore_value and cat_id not in VSPW_VAL_INVALID_CATIDS
                ]
            )
            for cat_id in semseg_cat_ids:
                assert sampled_labels.index(cat_id) in ann_cat_ids
                binary_mask = semseg_map == cat_id
                _mask_labels.append(binary_mask)
                _class_labels.append(sampled_labels.index(cat_id))

            if len(_mask_labels) == 0:
                _mask_labels = torch.zeros((0, height, width), dtype=torch.bool)
                _class_labels = torch.zeros((0,), dtype=torch.int64)
            else:
                _mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _mask_labels])
                _class_labels = torch.tensor(np.array(_class_labels), dtype=torch.int64)

            mask_labels.append(_mask_labels)
            class_labels.append(_class_labels)

        return mask_labels, class_labels

    def _decode_instance_genseg_data(self, data_dict):
        image_sizes = data_dict["image_sizes"]
        anns = data_dict["annotations"]
        mask_labels = []
        class_labels = []
        for _anns, image_size in zip(anns, image_sizes):
            _mask_labels = []
            _class_labels = []
            height, width = image_size
            for _ann in _anns:
                binary_mask = decode_mask(_ann["segmentation"], height, width)
                _mask_labels.append(binary_mask)
                _class_labels.append(_ann["category_id"])

            _mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _mask_labels])
            _class_labels = torch.tensor(np.array(_class_labels), dtype=torch.int64)
            mask_labels.append(_mask_labels)
            class_labels.append(_class_labels)

        return mask_labels, class_labels

    def _decode_panoptic_genseg_data(self, data_dict):
        segments_info = data_dict.get("segments_info", None)
        seg_map_path = data_dict.get("seg_map", None)
        if seg_map_path is None:
            height, width = data_dict["image_size"]
            mask_labels = torch.zeros((0, 0, height, width), dtype=torch.bool)
            class_labels = torch.zeros((0, 0), dtype=torch.int64)
        else:
            # TODO: upsample the seg_map to the same size as the image
            mask_labels = []
            class_labels = []
            for _seg_map_path, _segment_info in zip(seg_map_path, segments_info):
                seg_map = Image.open(osp.join(self.pan_segmap_folder, _seg_map_path)).convert("RGB")
                seg_map = rgb2id(np.array(seg_map))

                _mask_labels = []
                _class_labels = []
                for segment_info in _segment_info:
                    cat_id = segment_info["category_id"]
                    if not segment_info["iscrowd"]:
                        mask = seg_map == segment_info["id"]
                        _class_labels.append(cat_id)
                        _mask_labels.append(mask)
                if len(_mask_labels) == 0:
                    _mask_labels = torch.zeros((0, seg_map.shape[-2], seg_map.shape[-1]))
                    _class_labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    _mask_labels = torch.stack(
                        [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _mask_labels]
                    )
                    _class_labels = torch.tensor(np.array(_class_labels), dtype=torch.int64)

                mask_labels.append(_mask_labels)
                class_labels.append(_class_labels)

        return mask_labels, class_labels

    def _load_ann_data(self):
        if self.data_path is not None and osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                coco_data = json.load(f)
        else:
            assert "semantic" in self.data_name
            coco_data = self.convert_vspw_to_coco_format()

        if "semantic" in self.data_name:
            rets = self._load_semantic_genseg_data(coco_data)
        elif "instance" in self.data_name:
            rets = self._load_instance_genseg_data(coco_data)
        elif "panoptic" in self.data_name:
            rets = self._load_panoptic_genseg_data(coco_data)
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

        if self.data_mode == "eval" and "instance" in self.data_name:
            base_tmp = tempfile.gettempdir()
            cache_dir = osp.join(base_tmp, "x2sam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            print_log(f"Saving {self.data_name} gt_json to {cache_dir}...", logger="current")
            tmp_file = osp.join(cache_dir, f"{self.data_name}.json")
            if comm.is_main_process():
                with open(tmp_file, "w") as f:
                    json.dump(rets, f)
            comm.synchronize()
            self._set_metadata(coco_data, gt_json=tmp_file)
        else:
            self._set_metadata(coco_data)

        return rets

    def _decode_ann_data(self, data_dict):
        if "semantic" in self.data_name:
            mask_labels, class_labels = self._decode_semantic_genseg_data(data_dict)
        elif "instance" in self.data_name:
            mask_labels, class_labels = self._decode_instance_genseg_data(data_dict)
        elif "panoptic" in self.data_name:
            mask_labels, class_labels = self._decode_panoptic_genseg_data(data_dict)
        else:
            raise ValueError(f"Invalid dataset type: {self.data_name}")

        # padding mask_labels to the same length
        max_len = max(len(mask_label) for mask_label in mask_labels)
        mask_labels = torch.stack(
            [
                torch.cat(
                    [
                        mask_label,
                        torch.zeros(
                            (max_len - mask_label.shape[0], mask_label.shape[1], mask_label.shape[2]),
                            dtype=mask_label.dtype,
                        ),
                    ]
                )
                for mask_label in mask_labels
            ]
        )
        class_labels = torch.stack(
            [
                torch.cat(
                    [
                        class_label,
                        torch.ones(
                            (max_len - class_label.shape[0],),
                            dtype=class_label.dtype,
                        )
                        * self.ignore_label,
                    ]
                )
                for class_label in class_labels
            ]
        )

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict
