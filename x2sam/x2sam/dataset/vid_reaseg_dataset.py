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
from pycocotools import mask as mask_utils
from tqdm import tqdm

from x2sam.utils import comm
from x2sam.utils.logging import print_log

from .utils.catalog import MetadataCatalog
from .utils.format import format_cat_name
from .utils.mask import decode_mask
from .vid_refseg_dataset import VidRefSegDataset


class VidReaSegDataset(VidRefSegDataset):
    def __init__(
        self,
        *args,
        mask_file=None,
        fg_mask_file=None,
        **kwargs,
    ):
        super().__init__(*args, mask_file=mask_file, fg_mask_file=fg_mask_file, **kwargs)

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.mask_file = kwargs.get("mask_file", None)
        self.fg_mask_file = kwargs.get("fg_mask_file", None)

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path if gt_json is None else gt_json,
            data_name=self.data_name,
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def _load_reasonvos_reaseg_data(self):
        # reason_vos dataset
        if self.meta_file is not None and osp.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                self.meta_data = json.load(f)["videos"]
        if self.exp_meta_file is not None and osp.exists(self.exp_meta_file):
            with open(self.exp_meta_file, "r") as f:
                self.exp_meta_data = json.load(f)["videos"]

        rets = []
        for vid_id in self.exp_meta_data.keys():
            meta_data = self.exp_meta_data[vid_id]
            exp_meta_data = self.exp_meta_data[vid_id]
            image_files = [osp.join(vid_id, f"{frame_id}{self.imgmap_suffix}") for frame_id in exp_meta_data["frames"]]
            sampled_image_files, _ = self._sample_frames(image_files)
            if len(sampled_image_files) == 0:
                self.woann_cnt += 1
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                if len(cur_image_files) == 0:
                    continue
                if len(cur_image_files) < 2 and self.num_frames > 1:
                    continue
                width, height = Image.open(osp.join(self.video_folder, cur_image_files[0])).size

                vid_info = {
                    "video_id": vid_id,
                    "video_name": vid_id,
                    "image_files": cur_image_files,
                    "height": height,
                    "width": width,
                }

                cur_anns = []
                for exp_id, exp_data in exp_meta_data["expressions"].items():
                    exp = exp_data["exp"]
                    object_id = exp_data["obj_id"]
                    object_name = meta_data["objects"][object_id]["category"]
                    cur_anns.append(
                        {
                            "category_id": int(exp_id),
                            "object_id": int(object_id),
                            "object_name": object_name,
                            "exp": exp,
                        }
                    )

                if self.data_mode != "train":
                    for ann in cur_anns:
                        rets.append(
                            {
                                "video_id": vid_id,
                                "image_files": cur_image_files,
                                "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                                "sampled_sents": [ann["exp"]],
                                "annotations": [ann],
                                "video_info": vid_info,
                            }
                        )
                else:
                    rets.append(
                        {
                            "video_id": vid_id,
                            "image_files": cur_image_files,
                            "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                            "sampled_sents": [ann["exp"] for ann in cur_anns],
                            "annotations": cur_anns,
                            "video_info": vid_info,
                        }
                    )
        return rets

    def process_batch_revos_data_worker(self, batch):
        """
        Worker for loading reaseg annotations for a batch of videos.
        Returns a list of tuples:
            (vid_id, image_files, height, width, anns, sampled_sents, sample_id)
        """
        rets = []
        for vid_id in batch:
            try:
                if vid_id not in self.exp_meta_data:
                    continue
                exp_meta_data = self.exp_meta_data[vid_id]
                image_files = [
                    osp.join(vid_id, f"{frame_id}{self.imgmap_suffix}") for frame_id in sorted(exp_meta_data["frames"])
                ]
                width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

                for i in range(max(math.ceil(len(exp_meta_data["expressions"]) / self.num_class), 1)):
                    _cur_exp_data = (
                        copy.deepcopy(
                            dict(
                                itertools.islice(
                                    exp_meta_data["expressions"].items(), i * self.num_class, (i + 1) * self.num_class
                                )
                            )
                        )
                        if self.num_class > 0
                        else exp_meta_data["expressions"]
                    )
                    sampled_sents = []
                    is_sentences = []
                    anns = [[] for _ in range(len(image_files))]

                    cur_exp_data = []
                    for exp_data in _cur_exp_data.values():
                        # type_id: 0: sentence, 1: phrase, but type_id is not right in the dataset, we use exp.endswith("?") to determine the type
                        if self.data_split == "reasoning" and exp_data["exp"].endswith("?"):
                            cur_exp_data.append(exp_data)
                        elif self.data_split == "referring" and not exp_data["exp"].endswith("?"):
                            cur_exp_data.append(exp_data)
                        elif self.data_split == "all":
                            cur_exp_data.append(exp_data)
                        else:
                            continue

                    if len(cur_exp_data) == 0:
                        continue

                    for cat_id, exp_data in enumerate(cur_exp_data):
                        exp = format_cat_name(exp_data["exp"])
                        ann_ids = exp_data["anno_id"]
                        sampled_sents.append(exp)
                        # here question as sentence, phrase as category name
                        is_sentences.append(exp_data["exp"].endswith("?"))
                        for frame_id in range(len(image_files)):
                            segmentation = []
                            for ann_id in ann_ids:
                                rle_masks = self.mask_data[str(ann_id)]
                                if rle_masks[frame_id] is not None:
                                    segmentation.append(rle_masks[frame_id])

                            if len(segmentation) == 0:
                                continue
                            segmentation = mask_utils.merge(
                                [
                                    (
                                        mask_utils.frPyObjects(seg, *seg["size"])
                                        if isinstance(seg["counts"], list)
                                        else seg
                                    )
                                    for seg in segmentation
                                ]
                            )
                            segmentation["counts"] = segmentation["counts"].decode("utf-8")
                            anns[frame_id].append(
                                {
                                    "exp": exp,
                                    "category_id": cat_id,
                                    "segmentation": segmentation,
                                }
                            )

                    rets.append(
                        {
                            "video_id": vid_id,
                            "image_files": image_files,
                            "height": height,
                            "width": width,
                            "anns": anns,
                            "sampled_sents": sampled_sents,
                            "is_sentences": is_sentences,
                            "sample_id": i,
                        }
                    )
            except Exception as e:
                print_log(f"Error processing video {vid_id}: {e}", logger="current")
                continue

        return rets

    def _load_revos_reaseg_data(self):
        # revos dataset
        video_rets = None
        if self.data_path is not None and osp.exists(self.data_path):
            video_rets = json.load(open(self.data_path, "r"))

        assert self.data_split in ["reasoning", "referring", "all"]
        if video_rets is None or max([len(video_ret["sampled_sents"]) for video_ret in video_rets]) > self.num_class:
            vid_ids = list(self.exp_meta_data.keys())
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
                    futures = [executor.submit(self.process_batch_revos_data_worker, batch) for batch in batches]
                    for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                        batch_results = future.result()
                        if batch_results:
                            video_rets.extend(batch_results)
            else:
                with mp.Pool(num_workers) as pool:
                    for batch_results in tqdm(
                        pool.imap(self.process_batch_revos_data_worker, batches),
                        total=len(batches),
                        desc=f"Loading {self.data_name} dataset",
                        ncols=80,
                    ):
                        if batch_results:
                            video_rets.extend(batch_results)

            comm.synchronize()
            if comm.is_main_process():
                os.makedirs(osp.dirname(self.data_path), exist_ok=True)
                tmp_path = self.data_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(video_rets, f)
                os.replace(tmp_path, self.data_path)

        rets = []
        for vid_ret in video_rets:
            if vid_ret is None:
                continue
            vid_id, image_files, height, width, anns, sampled_sents, is_sentences, sample_id = (
                vid_ret["video_id"],
                vid_ret["image_files"],
                vid_ret["height"],
                vid_ret["width"],
                vid_ret["anns"],
                vid_ret["sampled_sents"],
                vid_ret["is_sentences"],
                vid_ret["sample_id"],
            )
            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns, self.num_frames)
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                self.woann_cnt += 1
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                cur_anns = (
                    copy.deepcopy(sampled_anns[i * self.num_frames : (i + 1) * self.num_frames])
                    if self.num_frames > 0
                    else sampled_anns
                )
                cur_image_files = [image_file for image_file, ann in zip(cur_image_files, cur_anns) if len(ann) > 0]
                cur_anns = [ann for ann in cur_anns if len(ann) > 0]
                if len(cur_image_files) == 0 or len(cur_anns) == 0:
                    continue
                if len(cur_image_files) < 2 and self.num_frames > 1:
                    continue

                assert self.num_class > 1 if self.data_mode == "train" else self.num_class == 1

                vid_info = {
                    "video_id": vid_id,
                    "video_name": vid_id,
                    "file_names": cur_image_files,
                    "height": height,
                    "width": width,
                    "sample_id": sample_id,
                    "chunk_id": i,
                }
                rets.append(
                    {
                        "video_id": vid_id,
                        "image_files": cur_image_files,
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "sampled_sents": sampled_sents,
                        "is_sentences": is_sentences,
                        "annotations": cur_anns,
                        "video_info": vid_info,
                    }
                )

        return rets

    def _decode_reasonvos_reaseg_data(self, data_dict):
        image_files = data_dict["image_files"]
        image_sizes = data_dict["image_sizes"]
        annotations = data_dict["annotations"]

        mask_labels = []
        class_labels = []
        for image_file, image_size in zip(image_files, image_sizes):
            _mask_labels = []
            _class_labels = []
            height, width = image_size
            for ann in annotations:
                object_id = ann["object_id"]
                category_id = ann["category_id"]
                segmap_file = image_file.replace(self.imgmap_suffix, self.segmap_suffix)
                segmap = Image.open(osp.join(self.gt_video_folder, segmap_file))
                segmap = np.array(segmap)
                binary_mask = (segmap == object_id).astype(np.uint8)
                if binary_mask.sum() == 0:
                    continue
                _mask_labels.append(binary_mask)
                _class_labels.append(category_id)

            if len(_mask_labels) == 0:
                _mask_labels = torch.zeros((0, height, width))
                _class_labels = torch.zeros((0,), dtype=torch.int64)
            else:
                _mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _mask_labels])
                _class_labels = torch.tensor(np.array(_class_labels), dtype=torch.int64)

            mask_labels.append(_mask_labels)
            class_labels.append(_class_labels)

        return mask_labels, class_labels

    def _decode_revos_reaseg_data(self, data_dict):
        image_sizes = data_dict["image_sizes"]
        annotations = data_dict["annotations"]

        mask_labels = []
        class_labels = []
        height, width = image_sizes[0]
        for image_anns in annotations:
            _mask_labels = []
            _class_labels = []
            for ann in image_anns:
                category_id = ann["category_id"]
                segmentation = ann["segmentation"]
                binary_mask = decode_mask(segmentation, height, width)
                _mask_labels.append(binary_mask)
                _class_labels.append(category_id)

            if len(_mask_labels) == 0:
                _mask_labels = torch.zeros((0, height, width))
                _class_labels = torch.zeros((0,), dtype=torch.int64)
            else:
                _mask_labels = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in _mask_labels])
                _class_labels = torch.tensor(np.array(_class_labels), dtype=torch.int64)

            mask_labels.append(_mask_labels)
            class_labels.append(_class_labels)

        return mask_labels, class_labels

    def _load_ann_data(self):
        if self.meta_file is not None and osp.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                self.meta_data = json.load(f)["videos"]
        if self.exp_meta_file is not None and osp.exists(self.exp_meta_file):
            with open(self.exp_meta_file, "r") as f:
                self.exp_meta_data = json.load(f)["videos"]
        if self.mask_file is not None and osp.exists(self.mask_file):
            with open(self.mask_file, "r") as f:
                self.mask_data = json.load(f)
        if self.fg_mask_file is not None and osp.exists(self.fg_mask_file):
            with open(self.fg_mask_file, "r") as f:
                self.fg_mask_data = json.load(f)

        if "reason_vos" in self.exp_meta_file:
            rets = self._load_reasonvos_reaseg_data()
        elif "revos" in self.exp_meta_file:
            rets = self._load_revos_reaseg_data()
        else:
            raise ValueError(f"Invalid dataset: {self.exp_meta_file}")

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
        if "reason_vos" in self.exp_meta_file:
            mask_labels, class_labels = self._decode_reasonvos_reaseg_data(data_dict)
        elif "revos" in self.exp_meta_file:
            mask_labels, class_labels = self._decode_revos_reaseg_data(data_dict)
        else:
            raise ValueError(f"Invalid dataset: {self.exp_meta_file}")

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

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
            }
        )

        return data_dict
