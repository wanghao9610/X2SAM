import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from x2sam.utils import comm
from x2sam.utils.logging import print_log

from .utils.catalog import MetadataCatalog
from .utils.mask import decode_mask, encode_mask
from .vid_vgdseg_dataset import VidVGDSegDataset

YTVOS_TRAIN_CATEGORIES = [
    "airplane",
    "ape",
    "bear",
    "bike",
    "bird",
    "boat",
    "bucket",
    "bus",
    "camel",
    "cat",
    "cow",
    "crocodile",
    "deer",
    "dog",
    "dolphin",
    "duck",
    "eagle",
    "earless_seal",
    "elephant",
    "fish",
    "fox",
    "frisbee",
    "frog",
    "giant_panda",
    "giraffe",
    "hand",
    "hat",
    "hedgehog",
    "horse",
    "knife",
    "leopard",
    "lion",
    "lizard",
    "monkey",
    "motorbike",
    "mouse",
    "others",
    "owl",
    "paddle",
    "parachute",
    "parrot",
    "penguin",
    "person",
    "plant",
    "rabbit",
    "raccoon",
    "sedan",
    "shark",
    "sheep",
    "sign",
    "skateboard",
    "snail",
    "snake",
    "snowboard",
    "squirrel",
    "surfboard",
    "tennis_racket",
    "tiger",
    "toilet",
    "train",
    "truck",
    "turtle",
    "umbrella",
    "whale",
    "zebra",
]

YTVOS_VALID_CATEGORIES = [
    "airplane",
    "ape",
    "backpack",
    "ball",
    "bear",
    "bike",
    "bird",
    "boat",
    "bottle",
    "box",
    "bucket",
    "bus",
    "butterfly",
    "camel",
    "camera",
    "cat",
    "chameleon",
    "cloth",
    "cow",
    "crocodile",
    "cup",
    "deer",
    "dog",
    "dolphin",
    "duck",
    "eagle",
    "earless_seal",
    "elephant",
    "eyeglasses",
    "fish",
    "flag",
    "fox",
    "frisbee",
    "frog",
    "giant_panda",
    "giraffe",
    "guitar",
    "hand",
    "handbag",
    "hat",
    "hedgehog",
    "horse",
    "jellyfish",
    "kangaroo",
    "knife",
    "leopard",
    "lion",
    "lizard",
    "microphone",
    "mirror",
    "monkey",
    "motorbike",
    "mouse",
    "others",
    "owl",
    "paddle",
    "parachute",
    "parrot",
    "penguin",
    "person",
    "plant",
    "rabbit",
    "raccoon",
    "ring",
    "rope",
    "sedan",
    "shark",
    "sheep",
    "shovel",
    "sign",
    "skateboard",
    "small_panda",
    "snail",
    "snake",
    "snowboard",
    "spider",
    "squirrel",
    "stuffed_toy",
    "surfboard",
    "table",
    "tennis_racket",
    "tiger",
    "tissue",
    "toilet",
    "train",
    "truck",
    "turtle",
    "umbrella",
    "watch",
    "whale",
    "zebra",
]

YTVOS_SEEN_CATEGORIES = set(YTVOS_TRAIN_CATEGORIES) & set(YTVOS_VALID_CATEGORIES)
YTVOS_UNSEEN_CATEGORIES = set(YTVOS_VALID_CATEGORIES) - set(YTVOS_TRAIN_CATEGORIES)
INVALID_CATEGORIES = {"gaint_panda": "giant_panda"}


class VidObjSegDataset(VidVGDSegDataset):
    def __init__(
        self,
        *args,
        imgmap_suffix=".jpg",
        segmap_suffix=".png",
        data_split="seen",
        meta_file=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            imgmap_suffix=imgmap_suffix,
            segmap_suffix=segmap_suffix,
            data_split=data_split,
            meta_file=meta_file,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.imgmap_suffix = kwargs.get("imgmap_suffix", ".jpg")
        self.segmap_suffix = kwargs.get("segmap_suffix", ".png")
        self.data_split = kwargs.get("data_split", "seen")
        self.meta_file = kwargs.get("meta_file", None)

        if self.data_split == "seen":
            self.cat_names = YTVOS_SEEN_CATEGORIES
        elif self.data_split == "unseen":
            self.cat_names = YTVOS_UNSEEN_CATEGORIES
        elif self.data_split == "all":
            self.cat_names = list(set(YTVOS_TRAIN_CATEGORIES + YTVOS_VALID_CATEGORIES))
        else:
            raise ValueError(f"Invalid data split: {self.data_split}")

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path if gt_json is None else gt_json,
            thing_classes=self.cats,
            data_name=self.data_name,
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _process_batch_data_worker(self, batch):
        """
        Worker for loading objseg annotations for a batch of videos.
        Returns a list of dicts:
            {
                "video_id": vid_id,
                "image_files": image_files,
                "height": height,
                "width": width,
                "anns": anns,
                "sampled_labels": sampled_labels,
            }
        """
        rets = []
        for vid_id in batch:
            try:
                meta_data = self.meta_data[vid_id]
                image_dir = osp.join(self.video_folder, vid_id)
                image_files = [
                    osp.join(vid_id, image_file)
                    for image_file in sorted(os.listdir(image_dir))
                    if image_file.endswith(self.imgmap_suffix)
                ]
                if len(image_files) == 0:
                    continue
                width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

                for i in range(max(math.ceil(len(meta_data["objects"]) / self.num_class), 1)):
                    cur_meta_data = (
                        copy.deepcopy(
                            dict(
                                itertools.islice(
                                    meta_data["objects"].items(), i * self.num_class, (i + 1) * self.num_class
                                )
                            )
                        )
                        if self.num_class > 0
                        else meta_data["objects"]
                    )
                    anns = [[] for _ in range(len(image_files))]
                    sampled_labels = []
                    for obj_id in sorted(cur_meta_data.keys()):
                        obj_category = (
                            INVALID_CATEGORIES[cur_meta_data[obj_id]["category"]]
                            if cur_meta_data[obj_id]["category"] in INVALID_CATEGORIES
                            else cur_meta_data[obj_id]["category"]
                        )
                        if obj_category not in self.cat_names:
                            continue
                        cat_id = self.cat_name_to_id[f"{vid_id}_{obj_id}"]
                        sampled_labels.append(cat_id)
                        obj_id_int = int(obj_id)
                        for frame_id, image_file in enumerate(image_files):
                            segmap_file = image_file.replace(self.imgmap_suffix, self.segmap_suffix)
                            segmap = Image.open(osp.join(self.gt_video_folder, segmap_file))
                            segmap = np.array(segmap)
                            binary_mask = (segmap == obj_id_int).astype(np.uint8)
                            if binary_mask.sum() == 0:
                                continue
                            anns[frame_id].append(
                                {
                                    "category_id": cat_id,
                                    "segmentation": encode_mask(binary_mask),
                                }
                            )

                    for image_anns in anns:
                        for ann in image_anns:
                            ann["category_id"] = sampled_labels.index(ann["category_id"])

                    rets.append(
                        {
                            "video_id": vid_id,
                            "image_files": image_files,
                            "height": height,
                            "width": width,
                            "anns": anns,
                            "sampled_labels": list(sampled_labels),
                            "sample_id": i,
                        }
                    )
            except Exception as e:
                print_log(f"Error processing video {vid_id}: {e}", logger="current")
                continue
        return rets

    def _load_ann_data(self):
        if self.meta_file is not None and osp.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                self.meta_data = json.load(f)["videos"]

        vid_ids = list(self.meta_data.keys())
        # fake categories for ytvis_vgdseg, each object as a category
        cats = sorted(
            [f"{vid_id}_{obj_id}" for vid_id in vid_ids for obj_id in sorted(self.meta_data[vid_id]["objects"].keys())]
        )
        self.cats = [{"id": cat_id, "name": cat_name} for cat_id, cat_name in enumerate(cats)]
        self.cat_ids = [cat_info["id"] for cat_info in self.cats]
        self.cat_id_to_name = {cat_id: cat_name for cat_id, cat_name in enumerate(cats)}
        self.cat_name_to_id = {cat_name: cat_id for cat_id, cat_name in enumerate(cats)}

        video_rets = None
        if self.data_path is not None and osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                video_rets = json.load(f)
        if video_rets is None or max([len(video_ret["sampled_labels"]) for video_ret in video_rets]) > self.num_class:
            num_workers = min(64, max(1, mp.cpu_count() - 10))
            print_log(f"Using {num_workers} workers for processing videos", logger="current")

            batch_size = max(1, min(16, len(vid_ids) // num_workers if len(vid_ids) > 0 else 1))
            batches = [vid_ids[i : i + batch_size] for i in range(0, len(vid_ids), batch_size)]

            video_rets = []
            if self.use_threads:
                print_log(
                    f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks",
                    logger="current",
                )
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(self._process_batch_data_worker, batch) for batch in batches]
                    for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                        batch_results = future.result()
                        if batch_results:
                            video_rets.extend(batch_results)
            else:
                with mp.Pool(num_workers) as pool:
                    for batch_results in tqdm(
                        pool.imap(self._process_batch_data_worker, batches),
                        total=len(batches),
                        desc=f"Loading {self.data_name} dataset",
                        ncols=80,
                    ):
                        if batch_results:
                            video_rets.extend(batch_results)

            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(video_rets, f)

        rets = []
        for video_ret in video_rets:
            if video_ret is None:
                continue
            (
                vid_id,
                image_files,
                height,
                width,
                anns,
                sampled_labels,
                sample_id,
            ) = (
                video_ret["video_id"],
                video_ret["image_files"],
                video_ret["height"],
                video_ret["width"],
                video_ret["anns"],
                video_ret["sampled_labels"],
                video_ret["sample_id"],
            )
            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns, self.num_frames)
            if len(sampled_image_files) == 0:
                self.woann_cnt += 1
                continue
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
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

                # get the index of the vprompt that contains the most categories instead of the first frame
                vprompt_index = (
                    self._get_vprompt_index(cur_anns)
                    if self.data_mode == "train" or self.vprompt_index == -1
                    else self.vprompt_index
                )

                vid_info = {
                    "video_id": vid_id,
                    "video_name": osp.dirname(cur_image_files[0]),
                    "file_names": cur_image_files,
                    "height": height,
                    "width": width,
                    "sample_id": sample_id,
                    "chunk_id": i,
                }
                assert self.num_class > 1 if self.data_mode == "train" else self.num_class == 1

                rets.append(
                    {
                        "video_id": vid_id,
                        "image_files": cur_image_files,
                        "image_sizes": [(height, width)] * len(cur_image_files),
                        "sampled_labels": sampled_labels,
                        "contiguous_labels": self.cat_ids,
                        "annotations": cur_anns,
                        "video_info": vid_info,
                        "vprompt_index": vprompt_index,
                    }
                )

        if self.data_mode == "eval":
            base_tmp = tempfile.gettempdir()
            cache_dir = osp.join(base_tmp, "x2sam_cache")
            os.makedirs(cache_dir, exist_ok=True)
            print_log(f"Saving {self.data_name} gt_json to {cache_dir}...", logger="current")
            tmp_file = osp.join(cache_dir, f"{self.data_name}.json")
            if comm.is_main_process():
                with open(tmp_file, "w") as f:
                    json.dump(rets, f)
            comm.synchronize()
            self._set_metadata(gt_json=tmp_file)
        else:
            self._set_metadata()

        return rets

    def _decode_ann_data(self, data_dict):
        image_sizes = data_dict["image_sizes"]
        annotations = data_dict["annotations"]
        vprompt_index = data_dict["vprompt_index"]

        mask_labels = []
        class_labels = []
        vprompt_masks = []
        vprompt_catids = sorted(set([ann["category_id"] for ann in annotations[vprompt_index]]))
        height, width = image_sizes[0]
        for image_anns in annotations:
            _mask_labels = []
            _class_labels = []
            _vprompt_masks = []
            for ann in image_anns:
                category_id = ann["category_id"]
                segmentation = ann["segmentation"]
                binary_mask = decode_mask(segmentation, height, width)
                if category_id not in vprompt_catids:
                    continue
                _mask_labels.append(binary_mask)
                _class_labels.append(category_id)
                _vprompt_masks.append(binary_mask)

            if len(_mask_labels) == 0:
                _mask_labels = torch.zeros((0, height, width))
                _class_labels = torch.zeros((0,), dtype=torch.int64)
                _vprompt_masks = torch.zeros((0, height, width))
            else:
                _mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _mask_labels])
                _class_labels = torch.tensor(np.array(_class_labels), dtype=torch.int64)
                _vprompt_masks = torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _vprompt_masks]
                )

            mask_labels.append(_mask_labels)
            class_labels.append(_class_labels)
            vprompt_masks.append(_vprompt_masks)

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
        # pad class_labels with background_label(-1) as placeholder
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
        vprompt_masks = vprompt_masks[vprompt_index]
        assert vprompt_masks.shape[0] > 0, f"vprompt_masks is all zeros, {vprompt_index}, {class_labels}"

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
                "vprompt_masks": vprompt_masks,
                "vprompt_index": vprompt_index,
            }
        )

        return data_dict
