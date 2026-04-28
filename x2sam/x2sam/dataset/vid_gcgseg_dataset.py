import copy
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from x2sam.utils.constants import DEFAULT_PEND_TOKEN, DEFAULT_PSTART_TOKEN
from x2sam.utils.logging import print_log

from .utils.catalog import MetadataCatalog
from .utils.image import expand2square
from .utils.mask import decode_mask, encode_mask
from .vid_base_dataset import VidBaseDataset


def get_phrase_and_obj_ids_from_caption(caption):
    # Pattern to match labels in square brackets and object IDs in parentheses
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(pattern, caption)

    # Prepare the results in a dictionary format
    results = [{"label": label, "object_ids": ids.split(", ")} for label, ids in matches]

    list_of_obj_ids = []
    labels = []
    # Print the results
    for result in results:
        list_of_obj_ids.append(result["object_ids"])
        labels.append(result["label"])
    return list_of_obj_ids, labels


def format_caption(text):
    # Pattern to match labels in square brackets and object IDs in parentheses
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    # Substitute matched labels with the desired format
    formatted_text = re.sub(pattern, rf"{DEFAULT_PSTART_TOKEN}\1{DEFAULT_PEND_TOKEN}", text)

    return formatted_text


class VidGCGSegDataset(VidBaseDataset):
    def __init__(
        self,
        *args,
        data_split="train",
        meta_file=None,
        exp_meta_file=None,
        mask_file=None,
        extra_num_frames=8,
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
            mask_file=mask_file,
            extra_num_frames=extra_num_frames,
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
        self.mask_file = kwargs.get("mask_file", None)
        self.extra_num_frames = kwargs.get("extra_num_frames", 8)
        self.imgmap_suffix = kwargs.get("imgmap_suffix", ".jpg")
        self.segmap_suffix = kwargs.get("segmap_suffix", ".png")
        self.use_threads = kwargs.get("use_threads", True)

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)

        metadata = MetadataCatalog.get(f"{self.data_name}")
        metadata.set(
            gt_json=self.data_path,
            data_name=self.data_name,
            ignore_value=self.ignore_value,
            ignore_label=self.ignore_label,
            background_label=self.background_label,
            label_divisor=1000,
        )
        self._metadata = metadata

    def process_batch_anet_video_worker(self, batch):
        ann_dir = osp.join(self.data_root, "anns")
        rets = []
        img_id = 0
        ann_id = 0

        for ann_file in batch:
            with open(osp.join(ann_dir, ann_file), "r") as f:
                ann_data = json.load(f)
            # {"video_id": "v_00Dk03Jr70M", "segment_id": "0", "new_caption": "A <p> man </p> [SEG:0] enters a room and piles plaster onto a base", "refined_caption": "In the video, <p> a man </p> [SEG:0] is seen entering a room and piling plaster onto a base. He starts by putting plaster on a box and then spreading it on the wall. He then moves to another wall and repeats the process.", "ann": {"process_clss": [["man"]], "tokens": ["A", "man", "enters", "a", "room", "and", "piles", "plaster", "onto", "a", "base"], "frame_ind": [6], "process_idx": [[1]], "timestamps": [0.0, 15.23], "process_bnd_box": [[4, 114, 195, 304]], "crowds": [0]}, "seg_token_to_obj": {"[SEG:0]": {"bbox": [4, 114, 195, 304], "frame_id": 6, "process_idx": [1], "crowds": 0}}}
            vid_id = ann_data["video_id"]
            segment_id = ann_data["segment_id"]
            refined_caption = ann_data["refined_caption"]

            image_files = [
                osp.join(vid_id, segment_id, img_file)
                for img_file in sorted(os.listdir(osp.join(self.video_folder, vid_id, segment_id)))
                if img_file.endswith(self.imgmap_suffix)
            ]
            width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

            # NOTE: anet_vid_gcgseg dataset is low quality, so we need to filter out some frames, even we drop it in the vid_gcgseg training set!!!
            seg_tokens = re.findall(
                r"<p>\s*[^<]+\s*</p>\s*(\[SEG:\d+\])", refined_caption
            )  # ["[SEG:0], [SEG:1], [SEG:2]"]
            labels = [label.strip() for label in re.findall(r"<p>(.*?)</p>\s*\[SEG:\d+\]", refined_caption)]
            # "<p> x </p> [SEG:i]" -> "<p>x</p>"; other "<p> x </p>" -> "x"
            formatted_caption = re.sub(
                r"<p>\s*([^<]+?)\s*</p>\s*\[SEG:\d+\]|<p>\s*([^<]+?)\s*</p>",
                lambda m: (
                    f"{DEFAULT_PSTART_TOKEN}{m.group(1).strip()}{DEFAULT_PEND_TOKEN}"
                    if m.group(1) is not None
                    else m.group(2).strip()
                ),
                refined_caption,
            )
            # drop unmatched "<p>" / "</p>" left in malformed captions (keep paired tags)
            _parts = re.split(r"(<\/?p>)", formatted_caption)
            _out, _opens = [], []
            for _tok in _parts:
                if _tok == DEFAULT_PSTART_TOKEN:
                    _opens.append(len(_out))
                    _out.append(_tok)
                elif _tok == DEFAULT_PEND_TOKEN:
                    if _opens:
                        _opens.pop()
                        _out.append(_tok)
                else:
                    _out.append(_tok)
            for _idx in reversed(_opens):
                _out[_idx] = ""
            formatted_caption = "".join(_out)
            # drop any remaining raw "[SEG:i]" tokens (sometimes appear without "<p>...</p>")
            formatted_caption = re.sub(r"\s*\[SEG:\d+\]", "", formatted_caption)
            formatted_caption = re.sub(r" +", " ", formatted_caption)
            cleaned_caption = re.sub(r"<p>\s*([^<]+?)\s*</p>", r"\1", formatted_caption)
            frame_ids = [ann_data["seg_token_to_obj"][seg_token]["frame_id"] for seg_token in seg_tokens]
            anns = [[] for _ in range(len(image_files))]

            for cat_id, (seg_token, frame_id) in enumerate(zip(seg_tokens, frame_ids)):
                seg_token_id = seg_token.split(":")[1].split("]")[0]
                segmap_file = osp.join(
                    f"{vid_id}____{segment_id}",
                    f"{str(seg_token_id).zfill(2)}",
                    "mask.png",
                )
                segmap = Image.open(segmap_file).convert("L")
                binary_mask = np.array(segmap).astype(np.uint8)
                anns[frame_id].append(
                    {
                        "id": ann_id,
                        "category_id": cat_id,
                        "segmentation": encode_mask(binary_mask),
                        "frame_id": frame_id,
                    }
                )
                ann_id += 1

            if cat_id + 1 != formatted_caption.count("<p>"):
                continue

            # assert (
            #     cat_id + 1 == formatted_caption.count("<p>") == formatted_caption.count("</p>")
            # ), f"cat_id: {cat_id}, formatted_caption: {formatted_caption}"

            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns, self.num_frames)
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                # self.woann_cnt += 1 # Cannot update in worker
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_sampled_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                cur_sampled_anns = copy.deepcopy(
                    sampled_anns[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_anns
                )
                # filter frames with no annotations
                cur_sampled_extra_image_files = [
                    image_file if self.data_mode == "train" and len(ann) > 0 else image_file
                    for image_file, ann in zip(cur_sampled_image_files, cur_sampled_anns)
                ]
                cur_sampled_anns = [
                    ann if self.data_mode == "train" and len(ann) > 0 else ann for ann in cur_sampled_anns
                ]
                if len(cur_sampled_extra_image_files) == 0 or len(cur_sampled_anns) == 0:
                    continue
                cur_sampled_extra_image_files, cur_sampled_anns = self._sample_frames(
                    cur_sampled_extra_image_files, cur_sampled_anns, self.extra_num_frames
                )
                for j in range(max(math.ceil(len(cur_sampled_extra_image_files) / self.extra_num_frames), 1)):
                    cur_sub_sampled_extra_image_files = copy.deepcopy(
                        cur_sampled_extra_image_files[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_extra_image_files
                    )
                    cur_sub_sampled_anns = copy.deepcopy(
                        cur_sampled_anns[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_anns
                    )
                    if len(cur_sub_sampled_extra_image_files) == 0 or len(cur_sub_sampled_anns) == 0:
                        continue
                    if len(cur_sub_sampled_extra_image_files) < 2 and self.extra_num_frames > 1:
                        continue

                    images = [
                        {"id": img_id + i, "file_name": image_file, "height": height, "width": width}
                        for i, image_file in enumerate(cur_sub_sampled_extra_image_files)
                    ]
                    for image_anns, image in zip(cur_sub_sampled_anns, images):
                        for ann in image_anns:
                            ann["image_id"] = image["id"]

                    img_id += len(images)

                    vid_info = {
                        "video_id": vid_id,
                        "video_name": vid_id,
                        "images": images,
                        "file_names": cur_sampled_image_files,
                        "extra_file_names": cur_sub_sampled_extra_image_files,
                        "width": width,
                        "height": height,
                        "sample_id": i,
                        "chunk_id": i * int(self.num_frames // self.extra_num_frames) + j,
                        "labels": labels,
                        "caption": cleaned_caption,
                        "num_frames": self.num_frames,
                        "extra_num_frames": self.extra_num_frames,
                    }

                    rets.append(
                        {
                            "video_id": vid_id,
                            "image_files": cur_sampled_image_files,
                            "extra_image_files": cur_sub_sampled_extra_image_files,
                            "image_sizes": [(height, width)] * len(cur_sub_sampled_extra_image_files),
                            "annotations": cur_sub_sampled_anns,
                            "video_info": vid_info,
                            "caption": formatted_caption,
                        }
                    )

        return rets

    def _load_anet_gcgseg_data(self):
        ann_dir = osp.join(self.data_root, "anns")
        ann_files = os.listdir(ann_dir)

        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(f"Using {num_workers} workers for processing videos", logger="current")

        batch_size = max(1, min(16, len(ann_files) // num_workers if len(ann_files) > 0 else 1))
        batches = [ann_files[i : i + batch_size] for i in range(0, len(ann_files), batch_size)]

        video_rets = []
        if self.use_threads:
            print_log(
                f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks",
                logger="current",
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.process_batch_anet_video_worker, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                    batch_results = future.result()
                    if batch_results:
                        video_rets.extend(batch_results)
        else:
            with mp.Pool(num_workers) as pool:
                for batch_results in tqdm(
                    pool.imap(self.process_batch_anet_video_worker, batches),
                    total=len(batches),
                    desc=f"Loading {self.data_name} dataset",
                    ncols=80,
                ):
                    if batch_results:
                        video_rets.extend(batch_results)

        # Post-process to fix IDs
        global_img_id = 0
        global_ann_id = 0

        for ret in video_rets:
            # Update image IDs
            if "video_info" in ret and "images" in ret["video_info"]:
                for img, image_anns in zip(ret["video_info"]["images"], ret["annotations"]):
                    img["id"] = global_img_id
                    for ann in image_anns:
                        ann["image_id"] = global_img_id
                        ann["id"] = global_ann_id
                        global_ann_id += 1
                    global_img_id += 1

        return video_rets

    def process_batch_mevis_video_worker(self, batch):
        caption_dir = osp.join(self.data_root, self.data_split)
        rets = []
        img_id = 0
        ann_id = 0

        for vid_id in batch:
            cur_exp_meta_data = self.exp_meta_data[vid_id]
            image_files = [
                osp.join(vid_id, img_file)
                for img_file in sorted(os.listdir(osp.join(self.video_folder, vid_id)))
                if img_file.endswith(self.imgmap_suffix)
            ]
            width, height = Image.open(osp.join(self.video_folder, image_files[0])).size
            caption_file = osp.join(caption_dir, vid_id + ".json")
            if not osp.exists(caption_file):
                # print_log(f"Caption file not found: {caption_file}", logger="current")
                continue
            with open(caption_file, "r") as f:
                caption = json.load(f)["caption"]
            # {"caption": "In this scene, two foxes are positioned on a roof. One [fox is standing on a tube at the back of the roof facing left](1), while another [fox is sitting at the front of the roof facing right](2)."}
            list_of_obj_ids, labels = get_phrase_and_obj_ids_from_caption(caption)
            formatted_caption = format_caption(caption)
            cleaned_caption = re.sub(r"<p>\s*([^<]+?)\s*</p>", r"\1", formatted_caption)

            anns = [[] for _ in range(len(image_files))]
            obj2ann = {
                str(obj_id): ann_id
                for _, exp in cur_exp_meta_data["expressions"].items()
                for obj_id, ann_id in zip(exp["obj_id"], exp["anno_id"])
            }
            for cat_id, obj_ids in enumerate(list_of_obj_ids):
                for frame_id in range(len(image_files)):
                    for obj_id in obj_ids:
                        ann_id = obj2ann[str(obj_id)]
                        rle_masks = self.mask_data[str(ann_id)]
                        segmentation = rle_masks[frame_id]
                        if segmentation is None:
                            continue
                        entry = next((ann for ann in anns[frame_id] if ann["category_id"] == cat_id), None)
                        if entry is None:
                            anns[frame_id].append(
                                {
                                    "id": ann_id,
                                    "segmentation": [segmentation],
                                    "category_id": cat_id,
                                    "frame_id": frame_id,
                                }
                            )
                        else:
                            entry["segmentation"].append(segmentation)
                        ann_id += 1

            assert (
                cat_id + 1 == formatted_caption.count("<p>") == formatted_caption.count("</p>")
            ), f"cat_id: {cat_id}, formatted_caption: {formatted_caption}"

            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns, self.num_frames)
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                # self.woann_cnt += 1 # Cannot update in worker
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_sampled_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                cur_sampled_anns = copy.deepcopy(
                    sampled_anns[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_anns
                )
                # filter frames with no annotations
                cur_sampled_extra_image_files = [
                    image_file if self.data_mode == "train" and len(ann) > 0 else image_file
                    for image_file, ann in zip(cur_sampled_image_files, cur_sampled_anns)
                ]
                cur_sampled_anns = [
                    ann if self.data_mode == "train" and len(ann) > 0 else ann for ann in cur_sampled_anns
                ]
                if len(cur_sampled_extra_image_files) == 0 or len(cur_sampled_anns) == 0:
                    continue
                cur_sampled_extra_image_files, cur_sampled_anns = self._sample_frames(
                    cur_sampled_extra_image_files, cur_sampled_anns, self.extra_num_frames
                )
                if len(cur_sampled_extra_image_files) == 0 or len(cur_sampled_anns) == 0:
                    continue
                for j in range(max(math.ceil(len(cur_sampled_extra_image_files) / self.extra_num_frames), 1)):
                    cur_sub_sampled_extra_image_files = copy.deepcopy(
                        cur_sampled_extra_image_files[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_extra_image_files
                    )
                    cur_sub_sampled_anns = copy.deepcopy(
                        cur_sampled_anns[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_anns
                    )
                    if len(cur_sub_sampled_extra_image_files) == 0 or len(cur_sub_sampled_anns) == 0:
                        continue
                    if len(cur_sub_sampled_extra_image_files) < 2 and self.extra_num_frames > 1:
                        continue

                    images = [
                        {"id": img_id + i, "file_name": image_file, "height": height, "width": width}
                        for i, image_file in enumerate(cur_sub_sampled_extra_image_files)
                    ]
                    for image_anns, image in zip(cur_sub_sampled_anns, images):
                        for ann in image_anns:
                            ann["image_id"] = image["id"]
                    img_id += len(images)

                    vid_info = {
                        "video_id": vid_id,
                        "video_name": vid_id,
                        "images": images,
                        "file_names": cur_sampled_image_files,
                        "extra_file_names": cur_sub_sampled_extra_image_files,
                        "width": width,
                        "height": height,
                        "sample_id": i,
                        "chunk_id": i * int(self.num_frames // self.extra_num_frames) + j,
                        "labels": labels,
                        "caption": cleaned_caption,
                        "num_frames": self.num_frames,
                        "extra_num_frames": self.extra_num_frames,
                    }

                    rets.append(
                        {
                            "video_id": vid_id,
                            "image_files": cur_sampled_image_files,
                            "extra_image_files": cur_sub_sampled_extra_image_files,
                            "image_sizes": [(height, width)] * len(cur_sub_sampled_extra_image_files),
                            "annotations": cur_sub_sampled_anns,
                            "video_info": vid_info,
                            "caption": formatted_caption,
                        }
                    )

        return rets

    def _load_mevis_gcgseg_data(self):
        with open(self.exp_meta_file, "r") as f:
            self.exp_meta_data = json.load(f)["videos"]
        with open(self.mask_file, "r") as f:
            self.mask_data = json.load(f)

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
                futures = [executor.submit(self.process_batch_mevis_video_worker, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                    batch_results = future.result()
                    if batch_results:
                        video_rets.extend(batch_results)
        else:
            with mp.Pool(num_workers) as pool:
                for batch_results in tqdm(
                    pool.imap(self.process_batch_mevis_video_worker, batches),
                    total=len(batches),
                    desc=f"Loading {self.data_name} dataset",
                    ncols=80,
                ):
                    if batch_results:
                        video_rets.extend(batch_results)

        # Post-process to fix IDs
        global_img_id = 0
        global_ann_id = 0

        for ret in video_rets:
            # Update image IDs
            if "video_info" in ret and "images" in ret["video_info"]:
                for img, image_anns in zip(ret["video_info"]["images"], ret["annotations"]):
                    for ann in image_anns:
                        ann["image_id"] = global_img_id
                        ann["id"] = global_ann_id
                        global_ann_id += 1
                    img["id"] = global_img_id
                    global_img_id += 1

        return video_rets

    def process_batch_video_gcg_worker(self, batch):
        rets = []
        img_id = 0
        ann_id = 0

        for vid_info in batch:
            prefix = (
                osp.join(vid_info["dataset_split"], "train", "JPEGImages")
                if vid_info["dataset_split"] == "yt19"
                else osp.join(vid_info["dataset_split"], "train" if self.data_split == "train" else "val")
            )
            image_files = [osp.join(prefix, img_file) for img_file in sorted(vid_info["file_names"])]
            width, height = Image.open(osp.join(self.video_folder, image_files[0])).size
            assert (
                height == vid_info["height"] and width == vid_info["width"] and len(image_files) == vid_info["length"]
            )
            # {
            #     # ...
            #     "dense_cap": {
            #         "v_id2o_id": {"0": 1, "1": 2, "2": 5, "3": 6},
            #         "token_pos": [1, 44, 63, 80],
            #         "mask_id": [1, 2, 5, 6],
            #         "caption": "A man is riding a small red bike with blue wheels on a wooden track indoors. He is wearing a long green jacket and a hat. There are people watching in the background. He rides the bike in a circle around the track. Another man in a red coat and a hat is  riding a small yellow bike with red wheels. A person wearing a dark shirt is sitting down and waving at the people on the bikes. A person wearing a white shirt and dark pants is standing on the side of the velodrome and watching the others ride bikes.",
            #     }
            # }

            token_pos = vid_info["dense_cap"]["token_pos"]
            mask_id = vid_info["dense_cap"]["mask_id"]
            caption = vid_info["dense_cap"]["caption"]

            if len(token_pos) != len(mask_id):
                continue

            formatted_caption = " ".join(
                [
                    f"{DEFAULT_PSTART_TOKEN}{word}{DEFAULT_PEND_TOKEN}" if i in token_pos else word
                    for i, word in enumerate(caption.split(" "))
                ]
            )
            cleaned_caption = re.sub(r"<p>\s*([^<]+?)\s*</p>", r"\1", formatted_caption)
            labels = [caption.split(" ")[i] for i in token_pos]
            pos2mask = defaultdict(list)
            for _token_pos, _mask_id in zip(token_pos, mask_id):
                pos2mask[_token_pos].append(_mask_id)

            anns = [[] for _ in range(len(image_files))]
            for cat_id, (_token_pos, _mask_ids) in enumerate(pos2mask.items()):
                for frame_id in range(len(image_files)):
                    segmentation = []
                    for _mask_id in _mask_ids:
                        _segmentation = self.mask_id2ann[_mask_id]["segmentations"][frame_id]
                        if _segmentation is None:
                            continue
                        segmentation.append(_segmentation)
                    if len(segmentation) == 0:
                        continue
                    anns[frame_id].append(
                        {
                            "id": ann_id,
                            "category_id": cat_id,
                            "frame_id": frame_id,
                            "segmentation": segmentation,
                        }
                    )
                    ann_id += 1

            assert (
                cat_id + 1 == formatted_caption.count("<p>") == formatted_caption.count("</p>")
            ), f"cat_id: {cat_id}, formatted_caption: {formatted_caption}"

            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns, self.num_frames)
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                # self.woann_cnt += 1 # Cannot update in worker
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_sampled_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                cur_sampled_anns = copy.deepcopy(
                    sampled_anns[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_anns
                )
                # filter frames with no annotations
                cur_sampled_extra_image_files = [
                    image_file if self.data_mode == "train" and len(ann) > 0 else image_file
                    for image_file, ann in zip(cur_sampled_image_files, cur_sampled_anns)
                ]
                cur_sampled_anns = [
                    ann if self.data_mode == "train" and len(ann) > 0 else ann for ann in cur_sampled_anns
                ]
                if len(cur_sampled_extra_image_files) == 0 or len(cur_sampled_anns) == 0:
                    continue
                cur_sampled_extra_image_files, cur_sampled_anns = self._sample_frames(
                    cur_sampled_extra_image_files, cur_sampled_anns, self.extra_num_frames
                )
                for j in range(max(math.ceil(len(cur_sampled_extra_image_files) / self.extra_num_frames), 1)):
                    cur_sub_sampled_extra_image_files = copy.deepcopy(
                        cur_sampled_extra_image_files[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_extra_image_files
                    )
                    cur_sub_sampled_anns = copy.deepcopy(
                        cur_sampled_anns[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_anns
                    )
                    if len(cur_sub_sampled_extra_image_files) == 0 or len(cur_sub_sampled_anns) == 0:
                        continue
                    if len(cur_sub_sampled_extra_image_files) < 2 and self.extra_num_frames > 1:
                        continue

                    images = [
                        {"id": img_id + i, "file_name": image_file, "height": height, "width": width}
                        for i, image_file in enumerate(cur_sub_sampled_extra_image_files)
                    ]
                    for image_anns, image in zip(cur_sub_sampled_anns, images):
                        for ann in image_anns:
                            ann["image_id"] = image["id"]
                    img_id += len(images)

                    _vid_info = {
                        "video_id": vid_info["id"],
                        "video_name": "/".join(image_files[0].split("/")[:-1]),
                        "images": images,
                        "file_names": cur_sampled_image_files,
                        "extra_file_names": cur_sub_sampled_extra_image_files,
                        "width": width,
                        "height": height,
                        "sample_id": i,
                        "chunk_id": i * int(self.num_frames // self.extra_num_frames) + j,
                        "labels": labels,
                        "caption": cleaned_caption,
                        "num_frames": self.num_frames,
                        "extra_num_frames": self.extra_num_frames,
                    }

                    rets.append(
                        {
                            "video_id": vid_info["id"],
                            "image_files": cur_sampled_image_files,
                            "extra_image_files": cur_sub_sampled_extra_image_files,
                            "image_sizes": [(height, width)] * len(cur_sub_sampled_extra_image_files),
                            "annotations": cur_sub_sampled_anns,
                            "video_info": _vid_info,
                            "caption": formatted_caption,
                        }
                    )
        return rets

    def _load_video_gcgseg_data(self):
        with open(osp.join(self.data_root, "instruction_data", f"{self.data_split}.json"), "r") as f:
            instruction_data = json.load(f)

        video_infos = instruction_data["videos"]
        annotations = instruction_data["annotations"]
        self.mask_id2ann = {ann["id"]: ann for ann in annotations}

        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(f"Using {num_workers} workers for processing videos", logger="current")

        batch_size = max(1, min(16, len(video_infos) // num_workers if len(video_infos) > 0 else 1))
        batches = [video_infos[i : i + batch_size] for i in range(0, len(video_infos), batch_size)]

        video_rets = []
        if self.use_threads:
            print_log(
                f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks",
                logger="current",
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.process_batch_video_gcg_worker, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                    batch_results = future.result()
                    if batch_results:
                        video_rets.extend(batch_results)
        else:
            with mp.Pool(num_workers) as pool:
                for batch_results in tqdm(
                    pool.imap(self.process_batch_video_gcg_worker, batches),
                    total=len(batches),
                    desc=f"Loading {self.data_name} dataset",
                    ncols=80,
                ):
                    if batch_results:
                        video_rets.extend(batch_results)

        # Post-process to fix IDs
        global_img_id = 0
        global_ann_id = 0

        for ret in video_rets:
            # Update image IDs
            if "video_info" in ret and "images" in ret["video_info"]:
                for img, image_anns in zip(ret["video_info"]["images"], ret["annotations"]):
                    for ann in image_anns:
                        ann["image_id"] = global_img_id
                        ann["id"] = global_ann_id
                        global_ann_id += 1
                    img["id"] = global_img_id
                    global_img_id += 1

        return video_rets

    def process_batch_vidstg_video_worker(self, batch):
        caption_dir = osp.join(self.data_root, f"{self.data_split}_captions")
        rets = []
        img_id = 0
        ann_id = 0

        for caption_file in batch:
            vid_id = osp.basename(caption_file).split(".")[0]
            with open(osp.join(caption_dir, caption_file), "r") as f:
                caption = json.load(f)["caption"]
            # {"caption": "In this scene, two foxes are positioned on a roof. One [fox is standing on a tube at the back of the roof facing left](1), while another [fox is sitting at the front of the roof facing right](2)."}
            list_of_obj_ids, labels = get_phrase_and_obj_ids_from_caption(caption)
            formatted_caption = format_caption(caption)
            cleaned_caption = re.sub(r"<p>\s*([^<]+?)\s*</p>", r"\1", formatted_caption)

            image_files = [
                osp.join(vid_id, "frames", img_file)
                for img_file in sorted(os.listdir(osp.join(self.video_folder, vid_id, "frames")))
                if img_file.endswith(self.imgmap_suffix)
            ]
            width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

            anns = [[] for _ in range(len(image_files))]
            for cat_id, obj_ids in enumerate(list_of_obj_ids):
                for frame_id, image_file in enumerate(image_files):
                    for obj_id in obj_ids:
                        segmap_file = osp.basename(image_file).replace(self.imgmap_suffix, self.segmap_suffix)
                        segmap_file = osp.join(vid_id, "masks", str(obj_id).zfill(3), segmap_file)
                        if not osp.exists(osp.join(self.gt_video_folder, segmap_file)):
                            print_log(f"Segmap file not found: {segmap_file}", logger="current")
                            continue
                        segmap = Image.open(osp.join(self.gt_video_folder, segmap_file))
                        binary_mask = (np.array(segmap) / 255).astype(np.uint8)
                        anns[frame_id].append(
                            {
                                "id": ann_id,
                                "category_id": cat_id,
                                "frame_id": frame_id,
                                "segmentation": encode_mask(binary_mask),
                            }
                        )
                        ann_id += 1

            assert (
                cat_id + 1 == formatted_caption.count("<p>") == formatted_caption.count("</p>")
            ), f"cat_id: {cat_id}, formatted_caption: {formatted_caption}"

            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns, self.num_frames)
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                # self.woann_cnt += 1 # Cannot update in worker
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_sampled_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                cur_sampled_anns = copy.deepcopy(
                    sampled_anns[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_anns
                )
                # filter frames with no annotations
                cur_sampled_extra_image_files = [
                    image_file if self.data_mode == "train" and len(ann) > 0 else image_file
                    for image_file, ann in zip(cur_sampled_image_files, cur_sampled_anns)
                ]
                cur_sampled_anns = [
                    ann if self.data_mode == "train" and len(ann) > 0 else ann for ann in cur_sampled_anns
                ]
                if len(cur_sampled_extra_image_files) == 0 or len(cur_sampled_anns) == 0:
                    continue
                cur_sampled_extra_image_files, cur_sampled_anns = self._sample_frames(
                    cur_sampled_extra_image_files, cur_sampled_anns, self.extra_num_frames
                )
                for j in range(max(math.ceil(len(cur_sampled_extra_image_files) / self.extra_num_frames), 1)):
                    cur_sub_sampled_extra_image_files = copy.deepcopy(
                        cur_sampled_extra_image_files[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_extra_image_files
                    )
                    cur_sub_sampled_anns = copy.deepcopy(
                        cur_sampled_anns[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_anns
                    )
                    if len(cur_sub_sampled_extra_image_files) == 0 or len(cur_sub_sampled_anns) == 0:
                        continue
                    if len(cur_sub_sampled_extra_image_files) < 2 and self.extra_num_frames > 1:
                        continue

                    images = [
                        {"id": img_id + i, "file_name": image_file, "height": height, "width": width}
                        for i, image_file in enumerate(cur_sub_sampled_extra_image_files)
                    ]
                    for image_anns, image in zip(cur_sub_sampled_anns, images):
                        for ann in image_anns:
                            ann["image_id"] = image["id"]
                    img_id += len(images)

                    vid_info = {
                        "video_id": vid_id,
                        "video_name": vid_id,
                        "images": images,
                        "file_names": cur_sampled_image_files,
                        "extra_file_names": cur_sub_sampled_extra_image_files,
                        "width": width,
                        "height": height,
                        "sample_id": i,
                        "chunk_id": i * int(self.num_frames // self.extra_num_frames) + j,
                        "labels": labels,
                        "caption": cleaned_caption,
                        "num_frames": self.num_frames,
                        "extra_num_frames": self.extra_num_frames,
                    }

                    rets.append(
                        {
                            "video_id": vid_id,
                            "image_files": cur_sampled_image_files,
                            "extra_image_files": cur_sub_sampled_extra_image_files,
                            "image_sizes": [(height, width)] * len(cur_sub_sampled_extra_image_files),
                            "annotations": cur_sub_sampled_anns,
                            "video_info": vid_info,
                            "caption": formatted_caption,
                        }
                    )
        return rets

    def _load_vidstg_gcgseg_data(self):
        caption_dir = osp.join(self.data_root, f"{self.data_split}_captions")
        caption_files = [f for f in os.listdir(caption_dir) if f.endswith(".json")]

        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(f"Using {num_workers} workers for processing videos", logger="current")

        batch_size = max(1, min(16, len(caption_files) // num_workers if len(caption_files) > 0 else 1))
        batches = [caption_files[i : i + batch_size] for i in range(0, len(caption_files), batch_size)]

        video_rets = []
        if self.use_threads:
            print_log(
                f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks",
                logger="current",
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.process_batch_vidstg_video_worker, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                    batch_results = future.result()
                    if batch_results:
                        video_rets.extend(batch_results)
        else:
            with mp.Pool(num_workers) as pool:
                for batch_results in tqdm(
                    pool.imap(self.process_batch_vidstg_video_worker, batches),
                    total=len(batches),
                    desc=f"Loading {self.data_name} dataset",
                    ncols=80,
                ):
                    if batch_results:
                        video_rets.extend(batch_results)

        # Post-process to fix IDs
        global_img_id = 0
        global_ann_id = 0

        for ret in video_rets:
            # Update image IDs
            if "video_info" in ret and "images" in ret["video_info"]:
                for img, image_anns in zip(ret["video_info"]["images"], ret["annotations"]):
                    for ann in image_anns:
                        ann["image_id"] = global_img_id
                        ann["id"] = global_ann_id
                        global_ann_id += 1
                    img["id"] = global_img_id
                    global_img_id += 1

        return video_rets

    def _load_hcstvg_gcgseg_data(self):
        return self._load_vidstg_gcgseg_data()

    def process_batch_ytvos_video_worker(self, batch):
        caption_dir = osp.join(self.data_root, self.data_split)
        rets = []
        img_id = 0
        ann_id = 0

        for vid_id in batch:
            assert vid_id in self.exp_meta_data.keys(), f"Video ID {vid_id} not found in {self.exp_meta_file}"
            cur_meta_data = self.meta_data[vid_id]
            cur_exp_meta_data = self.exp_meta_data[vid_id]
            image_files = [
                osp.join(vid_id, f"{frame_id}{self.imgmap_suffix}") for frame_id in cur_exp_meta_data["frames"]
            ]
            width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

            caption_file = osp.join(caption_dir, vid_id + ".json")
            if not osp.exists(caption_file):
                # print_log(f"Caption file not found: {caption_file}", logger="current")
                continue
            with open(caption_file, "r") as f:
                caption = json.load(f)["caption"]
            # {"caption": "In this scene, two foxes are positioned on a roof. One [fox is standing on a tube at the back of the roof facing left](1), while another [fox is sitting at the front of the roof facing right](2)."}
            list_of_obj_ids, labels = get_phrase_and_obj_ids_from_caption(caption)
            formatted_caption = format_caption(caption)
            cleaned_caption = re.sub(r"<p>\s*([^<]+?)\s*</p>", r"\1", formatted_caption)

            anns = [[] for _ in range(len(image_files))]
            obj2frames = {obj_id: obj_info["frames"] for obj_id, obj_info in cur_meta_data["objects"].items()}
            for cat_id, obj_ids in enumerate(list_of_obj_ids):
                for obj_id in obj_ids:
                    for frame_id, image_file in enumerate(image_files):
                        frame = image_file.split("/")[-1].split(".")[0]
                        if frame not in obj2frames[obj_id]:
                            continue
                        segmap_file = image_file.replace(self.imgmap_suffix, self.segmap_suffix)
                        segmap_file = osp.join(self.gt_video_folder, segmap_file)
                        if not osp.exists(segmap_file):
                            print_log(f"Segmap file not found: {segmap_file}", logger="current")
                            continue
                        segmap = Image.open(segmap_file)
                        segmap = np.array(segmap).astype(np.uint8)
                        binary_mask = segmap == int(obj_id)
                        if binary_mask.sum() == 0:
                            continue
                        anns[frame_id].append(
                            {
                                "id": ann_id,
                                "category_id": cat_id,
                                "frame_id": frame_id,
                                "segmentation": encode_mask(binary_mask),
                            }
                        )
                        ann_id += 1

            assert (
                cat_id + 1 == formatted_caption.count("<p>") == formatted_caption.count("</p>")
            ), f"cat_id: {cat_id}, formatted_caption: {formatted_caption}"

            sampled_image_files, sampled_anns = self._sample_frames(image_files, anns, self.num_frames)
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                # self.woann_cnt += 1 # Cannot update in worker
                continue
            for i in range(max(math.ceil(len(sampled_image_files) / self.num_frames), 1)):
                cur_sampled_image_files = copy.deepcopy(
                    sampled_image_files[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_image_files
                )
                cur_sampled_anns = copy.deepcopy(
                    sampled_anns[i * self.num_frames : (i + 1) * self.num_frames]
                    if self.num_frames > 0
                    else sampled_anns
                )
                # filter frames with no annotations
                cur_sampled_extra_image_files = [
                    image_file if self.data_mode == "train" and len(ann) > 0 else image_file
                    for image_file, ann in zip(cur_sampled_image_files, cur_sampled_anns)
                ]
                cur_sampled_anns = [
                    ann if self.data_mode == "train" and len(ann) > 0 else ann for ann in cur_sampled_anns
                ]
                if len(cur_sampled_extra_image_files) == 0 or len(cur_sampled_anns) == 0:
                    continue
                cur_sampled_extra_image_files, cur_sampled_anns = self._sample_frames(
                    cur_sampled_extra_image_files, cur_sampled_anns, self.extra_num_frames
                )
                for j in range(max(math.ceil(len(cur_sampled_extra_image_files) / self.extra_num_frames), 1)):
                    cur_sub_sampled_extra_image_files = copy.deepcopy(
                        cur_sampled_extra_image_files[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_extra_image_files
                    )
                    cur_sub_sampled_anns = copy.deepcopy(
                        cur_sampled_anns[j * self.extra_num_frames : (j + 1) * self.extra_num_frames]
                        if self.extra_num_frames > 0
                        else cur_sampled_anns
                    )
                    if len(cur_sub_sampled_extra_image_files) == 0 or len(cur_sub_sampled_anns) == 0:
                        continue
                    if len(cur_sub_sampled_extra_image_files) < 2 and self.extra_num_frames > 1:
                        continue

                    images = [
                        {"id": img_id + k, "file_name": image_file, "height": height, "width": width}
                        for k, image_file in enumerate(cur_sub_sampled_extra_image_files)
                    ]
                    for image_anns, image in zip(cur_sub_sampled_anns, images):
                        for ann in image_anns:
                            ann["image_id"] = image["id"]
                    img_id += len(images)

                    vid_info = {
                        "video_id": vid_id,
                        "video_name": vid_id,
                        "images": images,
                        "file_names": cur_sampled_image_files,
                        "extra_file_names": cur_sub_sampled_extra_image_files,
                        "width": width,
                        "height": height,
                        "sample_id": i,
                        "chunk_id": i * int(self.num_frames // self.extra_num_frames) + j,
                        "labels": labels,
                        "caption": cleaned_caption,
                        "num_frames": self.num_frames,
                        "extra_num_frames": self.extra_num_frames,
                    }

                    rets.append(
                        {
                            "video_id": vid_id,
                            "image_files": cur_sampled_image_files,
                            "extra_image_files": cur_sub_sampled_extra_image_files,
                            "image_sizes": [(height, width)] * len(cur_sub_sampled_extra_image_files),
                            "annotations": cur_sub_sampled_anns,
                            "video_info": vid_info,
                            "caption": formatted_caption,
                        }
                    )

        return rets

    def _load_ytvos_gcgseg_data(self):
        with open(self.exp_meta_file, "r") as f:
            self.exp_meta_data = json.load(f)["videos"]

        with open(self.meta_file, "r") as f:
            self.meta_data = json.load(f)["videos"]

        vid_ids = list(self.meta_data.keys())
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
                futures = [executor.submit(self.process_batch_ytvos_video_worker, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Loading {self.data_name} dataset", ncols=80):
                    batch_results = future.result()
                    if batch_results:
                        video_rets.extend(batch_results)
        else:
            with mp.Pool(num_workers) as pool:
                for batch_results in tqdm(
                    pool.imap(self.process_batch_ytvos_video_worker, batches),
                    total=len(batches),
                    desc=f"Loading {self.data_name} dataset",
                    ncols=80,
                ):
                    if batch_results:
                        video_rets.extend(batch_results)

        # Post-process to fix IDs
        global_img_id = 0
        global_ann_id = 0

        for ret in video_rets:
            # Update image IDs
            if "video_info" in ret and "images" in ret["video_info"]:
                for img, image_anns in zip(ret["video_info"]["images"], ret["annotations"]):
                    for ann in image_anns:
                        ann["image_id"] = global_img_id
                        ann["id"] = global_ann_id
                        global_ann_id += 1
                    img["id"] = global_img_id
                    global_img_id += 1

        return video_rets

    def _load_ann_data(self):
        rets = None
        if self.data_path is not None and osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                rets = json.load(f)

        if (
            rets is None
            or max([video_ret["video_info"]["num_frames"] for video_ret in rets]) > self.num_frames
            or max([video_ret["video_info"]["extra_num_frames"] for video_ret in rets]) > self.extra_num_frames
        ):
            if "anet_gcg" in self.data_path:
                # ActivityNet Entities GCG
                rets = self._load_anet_gcgseg_data()
            elif "mevis_gcg" in self.data_path:
                # MeViS GCG
                rets = self._load_mevis_gcgseg_data()
            elif "video_gcg" in self.data_path:
                # Burst-YTVIS GCG
                rets = self._load_video_gcgseg_data()
            elif "vidstg_gcg" in self.data_path:
                # VidSTG GCG
                rets = self._load_vidstg_gcgseg_data()
            elif "hcstvg_gcg" in self.data_path:
                # HC-STVG GCG
                rets = self._load_hcstvg_gcgseg_data()
            elif "ytvos_gcg" in self.data_path:
                # Refer-YTVOS GCG
                rets = self._load_ytvos_gcgseg_data()
            else:
                raise ValueError(f"Invalid dataset: {self.data_path}")

            os.makedirs(osp.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(rets, f)

        self._set_metadata()

        return rets

    def _decode_ann_data(self, data_dict):
        image_sizes = data_dict["image_sizes"]
        annotations = data_dict["annotations"]
        mask_labels = []
        class_labels = []
        height, width = image_sizes[0]

        for img_anns in annotations:
            _mask_labels = []
            _class_labels = []
            for ann in img_anns:
                category_id = ann["category_id"]
                segmentation = ann["segmentation"]
                binary_mask = decode_mask(segmentation, height, width)
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

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("image_files", None) is not None:
            # NOTE: do NOT mutate lists while iterating; it will skip elements and break alignment.
            image_files_in = list(data_dict["image_files"])
            annotations_in = data_dict.get("annotations", None)
            image_sizes_in = data_dict.get("image_sizes", None)
            pil_images, image_files, kept_idx = [], [], []
            for i, image_file in enumerate(image_files_in):
                try:
                    pil_images.append(Image.open(osp.join(self.video_folder, image_file)).convert("RGB"))
                    image_files.append(image_file)
                    kept_idx.append(i)
                except Exception as e:
                    print(f"Error loading image_file {image_file}: {e}")
            data_dict["image_files"] = image_files

            if "extra_image_files" not in data_dict:
                extra_pil_images = pil_images
                data_dict["extra_image_files"] = image_files
                if image_sizes_in is not None:
                    data_dict["image_sizes"] = [image_sizes_in[i] for i in kept_idx if i < len(image_sizes_in)]
                if annotations_in is not None:
                    data_dict["annotations"] = [annotations_in[i] for i in kept_idx if i < len(annotations_in)]
            else:
                extra_image_files_in = list(data_dict["extra_image_files"])
                extra_pil_images, extra_image_files = [], []
                kept_annotations = [] if annotations_in is not None else None
                kept_image_sizes = [] if image_sizes_in is not None else None
                for i, image_file in enumerate(extra_image_files_in):
                    try:
                        extra_pil_images.append(Image.open(osp.join(self.video_folder, image_file)).convert("RGB"))
                        extra_image_files.append(image_file)
                        if kept_image_sizes is not None and i < len(image_sizes_in):
                            kept_image_sizes.append(image_sizes_in[i])
                        if kept_annotations is not None and i < len(annotations_in):
                            kept_annotations.append(annotations_in[i])
                    except Exception as e:
                        print(f"Error loading extra_image_file {image_file}: {e}")

                # Keep metadata aligned with extra frames.
                data_dict["extra_image_files"] = extra_image_files
                if kept_image_sizes is not None:
                    data_dict["image_sizes"] = kept_image_sizes
                if kept_annotations is not None:
                    data_dict["annotations"] = kept_annotations
            if self.video_processor is not None:
                video_images = pil_images
                if self.expand2square:
                    video_images = [
                        expand2square(pil_image, tuple(int(x * 255) for x in self.video_processor.image_mean))
                        for pil_image in pil_images
                    ]
                output = self.video_processor.preprocess(video_images, do_sample_frames=False, return_tensors="pt")
                pixel_values_videos = (
                    output["pixel_values_videos"][0]
                    if output["pixel_values_videos"].ndim == 4
                    else output["pixel_values_videos"]
                )
                video_grid_thw = output.get("video_grid_thw", None)
                data_dict["pixel_values_videos"] = pixel_values_videos
                data_dict["video_grid_thw"] = video_grid_thw
            elif self.image_processor is not None:
                video_images = pil_images
                if self.expand2square:
                    video_images = [
                        expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                        for pil_image in video_images
                    ]
                output = self.image_processor.preprocess(video_images, return_tensors="pt")
                pixel_values_videos = output["pixel_values"]
                data_dict["pixel_values_videos"] = pixel_values_videos

            if self.extra_image_processor is not None:
                data_dict.update(self._decode_ann_data(data_dict))
                extra_output = self.extra_image_processor.preprocess(
                    extra_pil_images, data_dict["mask_labels"], return_tensors="pt"
                )
                data_dict["extra_pixel_values"] = extra_output["pixel_values"]
                data_dict["scaled_size"] = extra_output["scaled_sizes"].tolist()
                data_dict["mask_labels"] = extra_output.get("mask_labels", None)
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, use_vision_token=True))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        else:
            if hasattr(self.video_processor, "crop_size"):
                crop_size = self.video_processor.crop_size
            else:
                crop_size = self.video_processor.size
            # placeholder for crop_size
            lengths = [1600, 1536] if self.pixel_values_ndim == 2 else [384, 384]
            crop_size = (
                {"height": lengths[0], "width": lengths[1]}
                if crop_size is None or "height" not in crop_size or "width" not in crop_size
                else crop_size
            )
            data_dict["pixel_values_videos"] = (
                torch.zeros(2, 3, crop_size["height"], crop_size["width"])
                if self.pixel_values_ndim == 3
                else torch.zeros(2, crop_size["height"], crop_size["width"])
            )
            data_dict["video_grid_thw"] = torch.tensor([[2, 40, 40]]) if self.pixel_values_ndim == 2 else None
            if self.extra_image_processor is not None:
                if hasattr(self.extra_image_processor, "crop_size"):
                    crop_size = self.extra_image_processor.crop_size
                else:
                    crop_size = self.extra_image_processor.size
                data_dict["extra_pixel_values"] = torch.zeros(2, 3, crop_size["height"], crop_size["width"])
                data_dict["video_info"] = {"image_files": None}
                data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
                data_dict["image_sizes"] = [{"height": crop_size["height"], "width": crop_size["width"]}] * 2
                data_dict["mask_labels"] = torch.zeros(2, 0, crop_size["height"], crop_size["width"])
                data_dict["class_labels"] = torch.zeros(2, 0)
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, use_vision_token=False))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        return data_dict
