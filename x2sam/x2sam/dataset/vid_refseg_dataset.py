import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from x2sam.utils import comm
from x2sam.utils.logging import print_log

from .utils.catalog import MetadataCatalog
from .utils.mask import decode_mask, encode_mask
from .vid_base_dataset import VidBaseDataset


class VidRefSegDataset(VidBaseDataset):
    def __init__(
        self,
        *args,
        data_split="train",
        meta_file=None,
        exp_meta_file=None,
        imgmap_suffix=".jpg",
        segmap_suffix=".png",
        use_threads=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_split=data_split,
            meta_file=meta_file,
            exp_meta_file=exp_meta_file,
            imgmap_suffix=imgmap_suffix,
            segmap_suffix=segmap_suffix,
            use_threads=use_threads,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.data_split = kwargs.get("data_split", "train")
        self.meta_file = kwargs.get("meta_file", None)
        self.exp_meta_file = kwargs.get("exp_meta_file", None)
        self.imgmap_suffix = kwargs.get("imgmap_suffix", ".jpg")
        self.segmap_suffix = kwargs.get("segmap_suffix", ".png")
        self.use_threads = kwargs.get("use_threads", True)

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

    def process_batch_yt_videos_worker(self, batch):
        """
        Worker for loading refseg annotations for a batch of videos.
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
                    osp.join(vid_id, f"{frame_id}{self.imgmap_suffix}") for frame_id in exp_meta_data["frames"]
                ]
                width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

                for i in range(max(math.ceil(len(exp_meta_data["expressions"]) / self.num_class), 1)):
                    cur_exp_data = (
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
                    anns = [[] for _ in range(len(image_files))]
                    for cat_id, (exp_id, exp_data) in enumerate(cur_exp_data.items()):
                        exp = exp_data["exp"]
                        obj_id = exp_data["obj_id"]
                        sampled_sents.append(exp)
                        for frame_id, image_file in enumerate(image_files):
                            segmap_file = image_file.replace(self.imgmap_suffix, self.segmap_suffix)
                            if "train" in self.data_path:
                                segmap = Image.open(osp.join(self.gt_video_folder, segmap_file))
                                segmap = np.array(segmap)
                                binary_mask = (segmap == int(obj_id)).astype(np.uint8)
                            elif "valid" in self.data_path:
                                # valid set is different from train set
                                segmap_file = osp.join(
                                    self.gt_video_folder,
                                    osp.dirname(segmap_file),
                                    exp_id,
                                    osp.basename(segmap_file),
                                )
                                if not osp.exists(segmap_file):
                                    print_log(f"Segmap file {segmap_file} not found", logger="current")
                                    continue
                                segmap = Image.open(segmap_file)
                                binary_mask = (np.array(segmap) / 255).astype(np.uint8)
                            else:
                                raise ValueError(f"Unsupported dataset: {self.data_name}")
                            if binary_mask.sum() == 0:
                                continue
                            if binary_mask.shape != (height, width):
                                # some semgmap in the valid set is not the same size as the image
                                binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            anns[frame_id].append(
                                {
                                    "exp": exp,
                                    "category_id": cat_id,
                                    "segmentation": encode_mask(binary_mask),
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
                            "sample_id": i,
                        }
                    )
            except Exception as e:
                print_log(f"Error processing video {vid_id}: {e}", logger="current")
                continue

        return rets

    def _load_yt_ann_data(self):
        video_rets = None
        if self.data_path is not None and osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                video_rets = json.load(f)
        if video_rets is None or max([len(video_ret["sampled_sents"]) for video_ret in video_rets]) > self.num_class:
            if self.exp_meta_file is not None and osp.exists(self.exp_meta_file):
                with open(self.exp_meta_file, "r") as f:
                    self.exp_meta_data = json.load(f)["videos"]

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
                    futures = [executor.submit(self.process_batch_yt_videos_worker, batch) for batch in batches]
                    for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                        batch_results = future.result()
                        if batch_results:
                            video_rets.extend(batch_results)
            else:
                with mp.Pool(num_workers) as pool:
                    for batch_results in tqdm(
                        pool.imap(self.process_batch_yt_videos_worker, batches),
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
            vid_id, image_files, height, width, anns, sampled_sents, sample_id = (
                video_ret["video_id"],
                video_ret["image_files"],
                video_ret["height"],
                video_ret["width"],
                video_ret["anns"],
                video_ret["sampled_sents"],
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

                vid_info = {
                    "video_id": vid_id,
                    "video_name": vid_id,
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
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "sampled_sents": sampled_sents,
                        "annotations": cur_anns,
                        "video_info": vid_info,
                    }
                )
        return rets

    def process_batch_davis_videos_worker(self, batch):
        """
        Worker for loading refseg annotations for a batch of videos.
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
                    osp.join(vid_id, f"{frame_id}{self.imgmap_suffix}") for frame_id in exp_meta_data["frames"]
                ]
                width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

                for i in range(max(math.ceil(len(exp_meta_data["expressions"]) / self.num_class), 1)):
                    cur_exp_data = (
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
                    anns = [[] for _ in range(len(image_files))]
                    for cat_id, exp_data in enumerate(cur_exp_data.values()):
                        exp = exp_data["exp"]
                        obj_id = exp_data["obj_id"]
                        sampled_sents.append(exp)
                        for frame_id, image_file in enumerate(image_files):
                            segmap_file = image_file.replace(self.imgmap_suffix, self.segmap_suffix)
                            segmap = Image.open(osp.join(self.gt_video_folder, segmap_file))
                            segmap = np.array(segmap)
                            binary_mask = (segmap == int(obj_id)).astype(np.uint8)
                            if binary_mask.sum() == 0:
                                continue
                            anns[frame_id].append(
                                {
                                    "exp": exp,
                                    "category_id": cat_id,
                                    "segmentation": encode_mask(binary_mask),
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
                            "sample_id": i,
                        }
                    )
            except Exception as e:
                print_log(f"Error processing video {vid_id}: {e}", logger="current")
                continue

        return rets

    def _convert_davis_to_yt_format(self):
        def _read_davis_text_annotations(text_ann_dir, ann_files):
            invalid_video_mapping = {
                "clasic-car": "classic-car",
                "dog-scale": "dogs-scaled",
                "motor-bike": "motorbike",
            }
            video_exps = defaultdict(dict)
            for ann_file in ann_files:
                with open(osp.join(text_ann_dir, ann_file[0]), "r", encoding=ann_file[1]) as f:
                    for line in f.readlines():
                        line = line.strip()
                        video_name, obj_id = line.split()[:2]
                        exp = " ".join(line.split()[2:])[1:-1]
                        video_name = invalid_video_mapping.get(video_name, video_name)
                        exp_id = len(video_exps[video_name]) if video_name in video_exps else 0
                        video_exps[video_name][exp_id] = {
                            "exp": exp,
                            "obj_id": obj_id,
                        }
            video_exps = {
                video_name: {
                    k: v[1] for k, v in enumerate(sorted(video_exp_dict.items(), key=lambda kv: int(kv[1]["obj_id"])))
                }
                for video_name, video_exp_dict in sorted(video_exps.items(), key=lambda kv: kv[0])
            }
            return video_exps

        text_ann_dir = osp.join(self.data_root, "davis_text_annotations")
        imageset_dir = osp.join(self.data_root, "ImageSets", "2017")
        video_dir = osp.join(self.data_root, "JPEGImages", "480p")
        ann_files = [
            ("Davis17_annot1_full_video.txt", "utf-8"),
            ("Davis17_annot1.txt", "utf-8"),
            ("Davis17_annot2_full_video.txt", "latin-1"),
            ("Davis17_annot2.txt", "latin-1"),
        ]
        video_exps = _read_davis_text_annotations(text_ann_dir, ann_files)
        with open(osp.join(imageset_dir, f"{self.data_split}.txt"), "r") as f:
            video_names = [line.strip() for line in f.readlines()]

        videos_dict = {}
        for video_name in video_names:
            if video_name not in video_exps:
                print_log(f"Video {video_name} not found in davis text annotations", logger="current")
                continue
            video_exp_list = video_exps[video_name]
            image_files = os.listdir(osp.join(video_dir, video_name))
            video_frames = sorted([image_file.split(".")[0] for image_file in image_files])
            videos_dict[video_name] = {
                "frames": video_frames,
                "expressions": video_exp_list,
            }

        os.makedirs(osp.dirname(self.exp_meta_file), exist_ok=True)
        with open(self.exp_meta_file, "w") as f:
            json.dump({"videos": videos_dict}, f)

    def _load_davis_ann_data(self):
        video_rets = None
        if self.data_path is not None and osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                video_rets = json.load(f)
        if video_rets is None or max([len(video_ret["sampled_sents"]) for video_ret in video_rets]) > self.num_class:
            assert self.exp_meta_file is not None, "Exp meta file is required"
            if not osp.exists(self.exp_meta_file):
                self._convert_davis_to_yt_format()

            if self.exp_meta_file is not None and osp.exists(self.exp_meta_file):
                with open(self.exp_meta_file, "r") as f:
                    self.exp_meta_data = json.load(f)["videos"]

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
                    futures = [executor.submit(self.process_batch_davis_videos_worker, batch) for batch in batches]
                    for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                        batch_results = future.result()
                        if batch_results:
                            video_rets.extend(batch_results)
            else:
                with mp.Pool(num_workers) as pool:
                    for batch_results in tqdm(
                        pool.imap(self.process_batch_davis_videos_worker, batches),
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
            vid_id, image_files, height, width, anns, sampled_sents, sample_id = (
                video_ret["video_id"],
                video_ret["image_files"],
                video_ret["height"],
                video_ret["width"],
                video_ret["anns"],
                video_ret["sampled_sents"],
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

                vid_info = {
                    "video_id": vid_id,
                    "video_name": vid_id,
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
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "sampled_sents": sampled_sents,
                        "annotations": cur_anns,
                        "video_info": vid_info,
                    }
                )
        return rets

    def _load_ann_data(self):
        if "yt" in self.data_name:
            rets = self._load_yt_ann_data()
        elif "davis" in self.data_name:
            rets = self._load_davis_ann_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.data_name}")

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

        mask_labels = []
        class_labels = []
        height, width = image_sizes[0]
        for image_anns in annotations:
            _mask_labels = []
            _class_labels = []
            for ann in image_anns:
                segmentation = ann["segmentation"]
                binary_mask = decode_mask(segmentation, height, width)
                category_id = ann["category_id"]
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
