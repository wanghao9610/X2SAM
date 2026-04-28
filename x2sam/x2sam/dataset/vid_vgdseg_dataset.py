import copy
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain, combinations

import numpy as np
import torch
from panopticapi.utils import rgb2id
from PIL import Image
from pycocotools import mask as mask_utils
from skimage.measure import label, regionprops
from tqdm import tqdm

from x2sam.structures import BoxMode
from x2sam.utils import comm
from x2sam.utils.logging import print_log
from x2sam.utils.palette import get_palette

from .utils.catalog import MetadataCatalog
from .utils.image import expand2square
from .utils.mask import decode_mask, encode_mask
from .utils.vprompt import (
    generate_box_vprompt,
    generate_mask_vprompt,
    generate_point_vprompt,
    generate_scribble_vprompt,
)
from .utils.ytvos import YTVOS
from .vid_base_dataset import VidBaseDataset


class VidVGDSegDataset(VidBaseDataset):
    def __init__(
        self,
        *args,
        min_area=5,
        max_retries=1000,
        point_radius=10,
        scribble_radius=5,
        vprompt_index=-1,
        source_data_path=None,
        visual_prompt_type="point_visual_prompt",
        sampler_input_feat="pixel_values",
        use_negative_sample=False,
        use_threads=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            min_area=min_area,
            max_retries=max_retries,
            point_radius=point_radius,
            scribble_radius=scribble_radius,
            vprompt_index=vprompt_index,
            source_data_path=source_data_path,
            visual_prompt_type=visual_prompt_type,
            sampler_input_feat=sampler_input_feat,
            use_negative_sample=use_negative_sample,
            use_threads=use_threads,
            **kwargs,
        )

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.min_area = kwargs.get("min_area", 5)
        self.max_retries = kwargs.get("max_retries", 1000)
        self.point_radius = kwargs.get("point_radius", 10)
        self.scribble_radius = kwargs.get("scribble_radius", 5)
        self.vprompt_index = kwargs.get("vprompt_index", -1)
        self.source_data_path = kwargs.get("source_data_path", None)
        self.visual_prompt_type = kwargs.get("visual_prompt_type", "point_visual_prompt")
        self.use_negative_sample = kwargs.get("use_negative_sample", False)
        self.sampler_input_feat = kwargs.get("sampler_input_feat", "pixel_values")
        self.use_threads = kwargs.get("use_threads", True)
        # for VIPSeg (video panoptic) conversion
        self.pan_segmap_folder = kwargs.get("pan_segmap_folder", None)
        self.imgmap_suffix = kwargs.get("imgmap_suffix", ".jpg")
        self.segmap_suffix = kwargs.get("segmap_suffix", ".png")

    def _set_metadata(self, **kwargs):
        gt_json = kwargs.get("gt_json", None)
        cats = kwargs.get("cats", None)
        cats = sorted(cats, key=lambda x: x["id"])
        cat_colors = [x["color"] for x in cats] if "color" in cats[0] else get_palette("random", len(cats))
        dataset_id_to_contiguous_id = {x["id"]: i for i, x in enumerate(cats)}
        cat_id_to_name = {x["id"]: x["name"] for x in cats}
        cat_id_to_color = {x["id"]: cat_colors[dataset_id_to_contiguous_id[x["id"]]] for x in cats}

        thing_cats = [x for x in cats if x.get("isthing", 1) == 1]
        thing_cat_ids = [x["id"] for x in thing_cats]
        # only keep the thing categories
        thing_cat_id_to_contiguous_id = {cat_id: cont_id for cont_id, cat_id in enumerate(thing_cat_ids)}
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
            label_divisor=1000,
        )
        self._metadata = metadata

    def _get_visual_prompts(self, mask):
        label_mask = label(mask)
        props = [prop for prop in regionprops(label_mask) if prop.area > self.min_area]
        point_visual_prompt = generate_point_vprompt(mask, props, self.max_retries, self.point_radius)
        scribble_visual_prompt = generate_scribble_vprompt(mask, props, self.max_retries, self.scribble_radius)
        box_visual_prompt = generate_box_vprompt(mask, props)
        mask_visual_prompt = generate_mask_vprompt(mask)

        return (
            encode_mask(point_visual_prompt),
            encode_mask(scribble_visual_prompt),
            encode_mask(box_visual_prompt),
            encode_mask(mask_visual_prompt),
        )

    def _process_ytvis_batch_videos(self, vid_ids_batch, coco_api):
        current_process = mp.current_process()
        pid = current_process.pid

        rets = []
        for vid_id in tqdm(vid_ids_batch, desc=f"Process {pid}"):
            _vid_info = coco_api.loadVids(vid_id)[0]
            ann_ids = coco_api.getAnnIds(vidIds=[vid_id])
            _image_files = sorted(_vid_info["file_names"])
            _anns = coco_api.loadAnns(ann_ids)

            anns = []
            image_files = []
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
                image_files.append(_image_files[i])
                anns.append(img_anns)

            if len(anns) == 0:
                self.woann_cnt += 1
                continue

            vid_info = {
                "video_id": vid_id,
                "image_files": image_files,
                "height": _vid_info["height"],
                "width": _vid_info["width"],
            }

            for ann in anns:
                for _ann in ann:
                    mask = decode_mask(_ann["segmentation"], _vid_info["height"], _vid_info["width"])
                    point_visual_prompt, scribble_visual_prompt, box_visual_prompt, mask_visual_prompt = (
                        self._get_visual_prompts(mask)
                    )
                    visual_prompts = {
                        "point_visual_prompt": point_visual_prompt,
                        "scribble_visual_prompt": scribble_visual_prompt,
                        "box_visual_prompt": box_visual_prompt,
                        "mask_visual_prompt": mask_visual_prompt,
                    }
                    _ann["visual_prompts"] = visual_prompts

            if len(image_files) < 2:
                print_log(f"{vid_id} in {self.data_name} has less than 2 frames.")
                continue

            rets.append(
                {
                    "image_files": image_files,
                    "video_id": _vid_info["id"],
                    "image_sizes": [(_vid_info["height"], _vid_info["width"])] * len(image_files),
                    "annotations": anns,
                    "video_info": vid_info,
                }
            )

        return rets

    def _mp_process_ytvis_ann_data(self, vid_ids, coco_api):
        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(
            f"Creating {self.data_name} gt_json, which will take a while, you can download the gt_json from https://huggingface.co/hao9610/X2SAM/resolve/main/vid_vgdseg_annotations",
            logger="current",
        )
        print_log(f"Processing {len(vid_ids)} videos with {num_workers} workers...", logger="current")

        batch_size = max(8, min(128, len(vid_ids) // num_workers))
        vid_id_batches = [vid_ids[i : i + batch_size] for i in range(0, len(vid_ids), batch_size)]

        rets = []
        if self.use_threads:
            print_log(f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks", logger="current")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                process_func = partial(self._process_ytvis_batch_videos, coco_api=coco_api)
                futures = [executor.submit(process_func, batch) for batch in vid_id_batches]
                for future in tqdm(futures, desc=f"Processing {self.data_name}", ncols=80):
                    batch_results = future.result()
                    if batch_results is not None:
                        rets.extend(batch_results)
        else:
            chunk_size = max(1, len(vid_id_batches) // num_workers // 2)
            with mp.Pool(num_workers) as pool:
                process_func = partial(self._process_ytvis_batch_videos, coco_api=coco_api)
                for i, batch_results in enumerate(
                    tqdm(
                        pool.imap_unordered(process_func, vid_id_batches, chunk_size=chunk_size),
                        total=len(vid_id_batches),
                        desc=f"Processing {self.data_name}",
                        ncols=80,
                    )
                ):
                    if batch_results is not None:
                        rets.extend(batch_results)

        rets = [r for r in rets if r is not None]
        return rets

    def _process_vipseg_batch_videos(self, batch, cat_ids):
        current_process = mp.current_process()
        pid = current_process.pid

        rets = []
        for _vid_info, _ann_info in tqdm(batch, desc=f"Process {pid}"):
            try:
                vid_id = _vid_info.get("video_id", _vid_info.get("id", None))
                if vid_id is None:
                    continue

                images = sorted(_vid_info.get("images", []), key=lambda x: x.get("id", 0))
                anns = sorted(_ann_info.get("annotations", []), key=lambda x: x.get("image_id", x.get("id", 0)))
                if len(images) == 0 or len(anns) == 0:
                    continue

                image_files = []
                annotations = []

                for frame_idx, (img_info, ann_info) in enumerate(zip(images, anns)):
                    file_name = img_info.get("file_name", None) or ann_info.get("file_name", None)
                    if file_name is None:
                        continue

                    # VIPSeg images are stored as {video_id}/{frame_name}
                    image_file = f"{vid_id}/{file_name}" if "/" not in str(file_name) else str(file_name)

                    segmap_file = image_file.replace(self.imgmap_suffix, self.segmap_suffix)
                    segmap_file = osp.join(self.pan_segmap_folder, segmap_file)
                    seg_map = Image.open(segmap_file).convert("RGB")
                    seg_map = rgb2id(np.array(seg_map))

                    segments_info = ann_info.get("segments_info", [])
                    img_anns = []
                    for seg in segments_info:
                        if int(seg.get("iscrowd", 0)) != 0:
                            continue
                        cat_id = seg.get("category_id", None)
                        if cat_id is None or cat_id not in cat_ids:
                            continue
                        seg_id = seg.get("id", None)
                        if seg_id is None:
                            continue

                        mask = seg_map == seg_id
                        if mask.sum() == 0:
                            continue

                        point_visual_prompt, scribble_visual_prompt, box_visual_prompt, mask_visual_prompt = (
                            self._get_visual_prompts(mask.astype(np.uint8))
                        )
                        img_anns.append(
                            {
                                "id": int(seg_id) * 100 + frame_idx,
                                "segmentation": encode_mask(mask.astype(np.uint8)),
                                "bbox": [0.0, 0.0, _vid_info["width"], _vid_info["height"]],  # placeholder
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "iscrowd": 0,
                                "category_id": int(cat_id),
                                "visual_prompts": {
                                    "point_visual_prompt": point_visual_prompt,
                                    "scribble_visual_prompt": scribble_visual_prompt,
                                    "box_visual_prompt": box_visual_prompt,
                                    "mask_visual_prompt": mask_visual_prompt,
                                },
                            }
                        )

                    if len(img_anns) == 0:
                        continue
                    image_files.append(image_file)
                    annotations.append(img_anns)

                if len(image_files) < 2:
                    continue

                height = images[0].get("height", None)
                width = images[0].get("width", None)
                if height is None or width is None:
                    width, height = Image.open(osp.join(self.video_folder, image_files[0])).size

                vid_info = {
                    "video_id": vid_id,
                    "image_files": image_files,
                    "height": height,
                    "width": width,
                }

                rets.append(
                    {
                        "image_files": image_files,
                        "video_id": vid_id,
                        "image_sizes": [(height, width)] * len(image_files),
                        "annotations": annotations,
                        "video_info": vid_info,
                    }
                )
            except Exception as e:
                print_log(f"Error processing VIPSeg video {_vid_info.get('video_id', None)}: {e}", logger="current")
                continue

        return rets

    def _mp_process_vipseg_ann_data(self, coco_data, cat_ids):
        assert (
            self.pan_segmap_folder is not None
        ), "VIPSeg VGDSeg needs pan_segmap_folder to build annotations from panoptic masks."

        videos = coco_data.get("videos", [])
        ann_videos = coco_data.get("annotations", [])
        assert len(videos) > 0 and len(ann_videos) > 0, f"Invalid VIPSeg json: {self.source_data_path}"

        videos = sorted(videos, key=lambda x: x.get("video_id", x.get("id", None)))
        ann_videos = sorted(ann_videos, key=lambda x: x.get("video_id", x.get("id", None)))
        pairs = list(zip(videos, ann_videos))

        num_workers = min(64, max(1, mp.cpu_count() - 10))
        print_log(
            f"Creating {self.data_name} gt_json (VIPSeg panoptic -> vgdseg), which will take a while; "
            "you can also download a prebuilt gt_json from "
            "https://huggingface.co/hao9610/X2SAM/resolve/main/vipseg_vgdseg_annotations",
            logger="current",
        )
        print_log(f"Processing {len(pairs)} videos with {num_workers} workers...", logger="current")

        batch_size = max(1, min(16, len(pairs) // num_workers if len(pairs) > 0 else 1))
        batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

        rets = []
        if self.use_threads:
            print_log(f"Using ThreadPoolExecutor with {num_workers} threads for I/O-intensive tasks", logger="current")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                process_func = partial(self._process_vipseg_batch_videos, cat_ids=cat_ids)
                futures = [executor.submit(process_func, batch) for batch in batches]
                for future in tqdm(futures, desc=f"Processing {self.data_name}", ncols=80):
                    batch_results = future.result()
                    if batch_results is not None:
                        rets.extend(batch_results)
        else:
            chunk_size = max(1, len(batches) // num_workers // 2)
            with mp.Pool(num_workers) as pool:
                process_func = partial(self._process_vipseg_batch_videos, cat_ids=cat_ids)
                for batch_results in tqdm(
                    pool.imap_unordered(process_func, batches, chunk_size=chunk_size),
                    total=len(batches),
                    desc=f"Processing {self.data_name}",
                    ncols=80,
                ):
                    if batch_results is not None:
                        rets.extend(batch_results)

        rets = [r for r in rets if r is not None]
        return rets

    def _sample_cats(self, cat_ids, video_anns, vprompt_anns):
        def _sample(items, num=None):
            num = len(items) if num is None or num < 0 else min(len(items), num)
            items = random.sample(items, num)
            return items

        pos_cat_ids = sorted({ann["category_id"] for ann in vprompt_anns if ann["category_id"] in cat_ids})
        assert len(pos_cat_ids) > 0, f"len(pos_cat_ids) == 0, {vprompt_anns}"

        if self.data_mode == "train":
            if self.use_negative_sample and random.random() < 0.5:
                neg_cat_ids = sorted(set(cat_ids) - set(pos_cat_ids))
                num_neg = random.randint(0, max(self.num_class, self.num_class - len(neg_cat_ids)))
                sampled_neg_cat_ids = _sample(neg_cat_ids, num_neg)
                sampled_cat_ids = _sample(pos_cat_ids + sampled_neg_cat_ids)
            else:
                sampled_cat_ids = _sample(pos_cat_ids)
        else:
            sampled_cat_ids = pos_cat_ids

        for anns in video_anns:
            for i in range(len(anns) - 1, -1, -1):
                if anns[i]["category_id"] not in sampled_cat_ids:
                    anns.pop(i)
                else:
                    anns[i]["category_id"] = sampled_cat_ids.index(anns[i]["category_id"])

        return sampled_cat_ids

    def _get_vprompt_index(self, anns):
        """
        Get the index of the vprompt that contains the most categories.
        """
        vprompt_index = 0
        all_cat_ids = [_ann["category_id"] for ann in anns for _ann in ann]
        frame_cat_ids = [sorted(set([_ann["category_id"] for _ann in ann])) for ann in anns]
        sorted_cat_ids = sorted(set(all_cat_ids), key=all_cat_ids.count, reverse=True)
        cat_id_combinations = list(
            chain.from_iterable(combinations(sorted_cat_ids, r) for r in range(len(sorted_cat_ids), 0, -1))
        )
        for cat_id_combination in cat_id_combinations:
            if sorted(cat_id_combination) in frame_cat_ids:
                vprompt_index = frame_cat_ids.index(sorted(cat_id_combination))
                break

        return vprompt_index

    def _load_ytvis_vgdseg_data(self):
        coco_api = YTVOS(self.source_data_path)
        vid_ids = sorted(coco_api.getVidIds())
        cats = coco_api.loadCats(coco_api.getCatIds())
        cat_ids = sorted([cat["id"] for cat in cats])

        if osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                _rets = json.load(f)

            self.woann_cnt = len(vid_ids) - len(_rets)
        else:
            _rets = self._mp_process_ytvis_ann_data(vid_ids, coco_api)
            basedir = osp.dirname(self.data_path)
            os.makedirs(basedir, exist_ok=True)
            print_log(f"Saving {self.data_name} gt_json to {self.data_path}...", logger="current")
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(_rets, f)

        rets = []
        img_id = 0
        for ret in _rets:
            assert len(ret["image_files"]) == len(
                ret["annotations"]
            ), "len(ret['image_files']) != len(ret['annotations'])"
            sampled_image_files, sampled_anns = self._sample_frames(ret["image_files"], ret["annotations"])
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                self.woann_cnt += 1
                continue
            width, height = Image.open(osp.join(self.video_folder, sampled_image_files[0])).size
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

                images = [
                    {"id": img_id + i, "file_name": image_file, "height": height, "width": width}
                    for i, image_file in enumerate(cur_image_files)
                ]
                img_id += len(images)

                vid_info = {
                    "video_id": ret["video_id"],
                    "video_name": osp.dirname(cur_image_files[0]),
                    "images": images,
                    "file_names": cur_image_files,
                    "height": ret["video_info"]["height"],
                    "width": ret["video_info"]["width"],
                    "chunk_id": i,
                }
                rets.append(
                    {
                        "video_id": ret["video_id"],
                        "image_files": cur_image_files,
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "annotations": cur_anns,
                        "video_info": vid_info,
                    }
                )

        for ret in rets:
            # select the frame that contains the most categories as the vprompt frame
            vprompt_index = (
                self._get_vprompt_index(ret["annotations"])
                if self.data_mode == "train" or self.vprompt_index == -1
                else self.vprompt_index
            )
            sampled_cat_ids = self._sample_cats(
                cat_ids,
                ret["annotations"],
                ret["annotations"][vprompt_index],
            )
            assert len(sampled_cat_ids) > 0, f"len(sampled_cat_ids) == 0, {ret['annotations'][vprompt_index]}"
            ret["sampled_labels"] = sampled_cat_ids
            ret["contiguous_labels"] = cat_ids
            ret["vprompt_index"] = vprompt_index

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
            self._set_metadata(cats=cats, gt_json=tmp_file)
        else:
            self._set_metadata(cats=cats)

        return rets

    def _load_vipseg_vgdseg_data(self):
        with open(self.source_data_path, "r") as f:
            coco_data = json.load(f)

        cats = coco_data.get("categories", None)
        assert cats is not None and len(cats) > 0, f"Invalid VIPSeg json: {self.source_data_path}"

        cat_ids = sorted([cat["id"] for cat in cats if cat.get("isthing", 1) == 1])
        if len(cat_ids) == 0:
            cat_ids = sorted([cat["id"] for cat in cats])

        if osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                _rets = json.load(f)
        else:
            assert (
                self.pan_segmap_folder is not None
            ), "VIPSeg VGDSeg needs pan_segmap_folder to build annotations from panoptic masks."
            _rets = self._mp_process_vipseg_ann_data(coco_data, cat_ids)

            basedir = osp.dirname(self.data_path)
            os.makedirs(basedir, exist_ok=True)
            print_log(f"Saving {self.data_name} gt_json to {self.data_path}...", logger="current")
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(_rets, f)

        # Chunk videos into fixed-length clips, and sample categories like ytvis.
        rets = []
        img_id = 0
        for ret in _rets:
            if ret is None:
                continue
            assert len(ret["image_files"]) == len(
                ret["annotations"]
            ), "len(ret['image_files']) != len(ret['annotations'])"
            sampled_image_files, sampled_anns = self._sample_frames(ret["image_files"], ret["annotations"])
            if len(sampled_image_files) == 0 or len(sampled_anns) == 0:
                self.woann_cnt += 1
                continue
            width, height = Image.open(osp.join(self.video_folder, sampled_image_files[0])).size
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

                images = [
                    {"id": img_id + j, "file_name": image_file, "height": height, "width": width}
                    for j, image_file in enumerate(cur_image_files)
                ]
                img_id += len(images)

                vid_info = {
                    "video_id": ret["video_id"],
                    "video_name": osp.dirname(cur_image_files[0]),
                    "images": images,
                    "file_names": cur_image_files,
                    "height": ret["video_info"].get("height", height),
                    "width": ret["video_info"].get("width", width),
                    "chunk_id": i,
                }
                rets.append(
                    {
                        "video_id": ret["video_id"],
                        "image_files": cur_image_files,
                        "image_sizes": [(vid_info["height"], vid_info["width"])] * len(cur_image_files),
                        "annotations": cur_anns,
                        "video_info": vid_info,
                    }
                )

        for ret in rets:
            vprompt_index = (
                self._get_vprompt_index(ret["annotations"])
                if self.data_mode == "train" or self.vprompt_index == -1
                else self.vprompt_index
            )
            sampled_cat_ids = self._sample_cats(
                cat_ids,
                ret["annotations"],
                ret["annotations"][vprompt_index],
            )
            assert len(sampled_cat_ids) > 0, f"len(sampled_cat_ids) == 0, {ret['annotations'][vprompt_index]}"
            ret["sampled_labels"] = sampled_cat_ids
            ret["contiguous_labels"] = cat_ids
            ret["vprompt_index"] = vprompt_index

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
            self._set_metadata(cats=cats, gt_json=tmp_file)
        else:
            self._set_metadata(cats=cats)

        return rets

    def _decode_ytvis_ann_data(self, data_dict):
        image_sizes = data_dict["image_sizes"]
        anns = data_dict["annotations"]
        vprompt_index = (
            data_dict["vprompt_index"] if self.data_mode == "train" or self.vprompt_index == -1 else self.vprompt_index
        )
        mask_labels = []
        class_labels = []
        vprompt_masks = []

        vprompt_catids = sorted(set([ann["category_id"] for ann in anns[vprompt_index]]))
        for ann, image_size in zip(anns, image_sizes):
            _mask_labels = []
            _class_labels = []
            _vprompt_masks = []
            height, width = image_size
            for _ann in ann:
                segmentation = _ann["segmentation"]
                visual_prompts = _ann["visual_prompts"]
                category_id = _ann["category_id"]
                if category_id not in vprompt_catids:
                    continue
                if self.data_mode == "train":
                    keys = list(visual_prompts.keys())
                    while keys:
                        key = random.choice(keys)
                        visual_masks = decode_mask(visual_prompts[key], height, width)
                        if visual_masks.sum() > 0:
                            break
                        keys.remove(key)
                    if not keys:
                        print_log(
                            f"{data_dict['image_files']} has no visual prompts, ann_id: {ann['id']}",
                            logger="current",
                        )
                        continue
                else:
                    visual_masks = (
                        decode_mask(visual_prompts[self.visual_prompt_type], height, width)
                        if visual_prompts.get(self.visual_prompt_type, None) is not None
                        else None
                    )
                    if visual_masks is None or visual_masks.sum() == 0:
                        print_log(
                            f"{data_dict['image_file']} has no {self.visual_prompt_type}, ann_id: {ann['id']}",
                            logger="current",
                        )
                        continue

                assert visual_masks.sum() > 0, f"visual_masks.sum() == 0, {data_dict['image_file']}, {ann['id']}"

                binary_mask = decode_mask(segmentation, height, width)
                _vprompt_masks.append(visual_masks)
                _mask_labels.append(binary_mask)
                _class_labels.append(category_id)

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
            vprompt_masks.append(_vprompt_masks)
            mask_labels.append(_mask_labels)
            class_labels.append(_class_labels)

        return mask_labels, class_labels, vprompt_masks, vprompt_index

    def _decode_vipseg_ann_data(self, data_dict):
        image_sizes = data_dict["image_sizes"]
        anns = data_dict["annotations"]
        vprompt_index = (
            data_dict["vprompt_index"] if self.data_mode == "train" or self.vprompt_index == -1 else self.vprompt_index
        )

        mask_labels = []
        class_labels = []
        vprompt_masks = []

        vprompt_catids = sorted(set([ann["category_id"] for ann in anns[vprompt_index]]))
        for ann, image_size in zip(anns, image_sizes):
            _mask_labels = []
            _class_labels = []
            _vprompt_masks = []
            height, width = image_size
            for _ann in ann:
                segmentation = _ann.get("segmentation", None)
                category_id = _ann.get("category_id", None)
                if segmentation is None or category_id is None:
                    continue
                if category_id not in vprompt_catids:
                    continue

                binary_mask = decode_mask(segmentation, height, width)

                visual_prompts = _ann.get("visual_prompts", None)
                if visual_prompts is None:
                    visual_masks = binary_mask
                else:
                    if self.data_mode == "train":
                        keys = list(visual_prompts.keys())
                        while keys:
                            key = random.choice(keys)
                            visual_masks = decode_mask(visual_prompts[key], height, width)
                            if visual_masks is not None and visual_masks.sum() > 0:
                                break
                            keys.remove(key)
                        if not keys:
                            visual_masks = binary_mask
                    else:
                        vp = visual_prompts.get(self.visual_prompt_type, None)
                        visual_masks = decode_mask(vp, height, width) if vp is not None else None
                        if visual_masks is None or visual_masks.sum() == 0:
                            visual_masks = binary_mask

                _vprompt_masks.append(visual_masks)
                _mask_labels.append(binary_mask)
                _class_labels.append(category_id)

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

            vprompt_masks.append(_vprompt_masks)
            mask_labels.append(_mask_labels)
            class_labels.append(_class_labels)

        return mask_labels, class_labels, vprompt_masks, vprompt_index

    def _load_ann_data(self):
        if "yt" in self.data_name:
            rets = self._load_ytvis_vgdseg_data()
        elif "vipseg" in self.data_name:
            rets = self._load_vipseg_vgdseg_data()
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

        return rets

    def _decode_ann_data(self, data_dict):
        if "yt" in self.data_name:
            mask_labels, class_labels, vprompt_masks, vprompt_index = self._decode_ytvis_ann_data(data_dict)
        elif "vipseg" in self.data_name:
            mask_labels, class_labels, vprompt_masks, vprompt_index = self._decode_vipseg_ann_data(data_dict)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

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
        vprompt_masks = vprompt_masks[vprompt_index]

        data_dict.update(
            {
                "mask_labels": mask_labels,
                "class_labels": class_labels,
                "vprompt_masks": vprompt_masks,
                "vprompt_index": vprompt_index,
            }
        )

        return data_dict

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("image_files", None) is not None:
            image_files = data_dict["image_files"]
            extra_image_files = data_dict.get("extra_image_files", image_files)
            # qwenvl series will patch along the time dimension
            image_files = (
                sorted(image_files * getattr(self.video_processor, "temporal_patch_size", 1))
                if self.video_processor is not None and self.sampler_input_feat == "pixel_values"
                else sorted(image_files)
            )
            pil_images = [
                Image.open(osp.join(self.video_folder, image_file)).convert("RGB") for image_file in image_files
            ]
            extra_pil_images = [
                Image.open(osp.join(self.video_folder, image_file)).convert("RGB") for image_file in extra_image_files
            ]
            if self.video_processor is not None:
                video_images = pil_images
                if self.expand2square:
                    video_images = [
                        expand2square(pil_image, tuple(int(x * 255) for x in self.video_processor.image_mean))
                        for pil_image in pil_images
                    ]
                output = self.video_processor.preprocess(video_images, return_tensors="pt")
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
                    extra_pil_images,
                    data_dict["mask_labels"],
                    data_dict["vprompt_masks"],
                    return_tensors="pt",
                )
                data_dict["extra_pixel_values"] = extra_output["pixel_values"]
                data_dict["scaled_size"] = extra_output["scaled_sizes"].tolist()
                data_dict["mask_labels"] = extra_output.get("mask_labels", None)
                data_dict["vprompt_masks"] = extra_output.get("vprompt_masks", None)
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
                torch.zeros(3, crop_size["height"], crop_size["width"])
                if self.pixel_values_ndim == 3
                else torch.zeros(crop_size["height"], crop_size["width"])
            )
            data_dict["video_grid_thw"] = torch.tensor([[1, 40, 40]]) if self.pixel_values_ndim == 2 else None
            if self.extra_image_processor is not None:
                if hasattr(self.extra_image_processor, "crop_size"):
                    crop_size = self.extra_image_processor.crop_size
                else:
                    crop_size = self.extra_image_processor.size
                data_dict["extra_pixel_values"] = torch.zeros(3, crop_size["height"], crop_size["width"])
                data_dict["image_info"] = {"image_file": None}
                data_dict["scaled_size"] = (crop_size["height"], crop_size["width"])
                data_dict["video_size"] = (0, crop_size["height"], crop_size["width"])
                data_dict["mask_labels"] = torch.zeros(0, crop_size["height"], crop_size["width"])
                data_dict["vprompt_masks"] = torch.zeros(0, crop_size["height"], crop_size["width"])
                data_dict["class_labels"] = torch.zeros(0)
                data_dict["task_name"] = self.task_name
            data_dict.update(self._get_input_ids(data_dict, use_vision_token=False))
            data_dict.update(self._get_cond_ids(data_dict))
            data_dict.update(self._get_seg_ids(data_dict))
        return data_dict
