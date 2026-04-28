import copy
import math
import os.path as osp
from itertools import chain

import torch
from PIL import Image

from .base_dataset import BaseDataset
from .utils.image import expand2square


class VidBaseDataset(BaseDataset):
    def __init__(self, frame_mode="span", num_frames=-1, max_frame=-1, frame_stride=-1, *args, **kwargs):
        super().__init__(
            *args,
            frame_mode=frame_mode,
            num_frames=num_frames,
            max_frame=max_frame,
            frame_stride=frame_stride,
            **kwargs,
        )

    @property
    def video_length(self):
        return (
            [len(data_dict.get("image_files", [self.num_frames])) for data_dict in self.data] * math.ceil(self.repeats)
        )[: int(len(self.data) * self.repeats)]

    def custom_init(self, **kwargs):
        self.frame_mode = kwargs.get("frame_mode", "span")
        self.num_frames = kwargs.get("num_frames", -1)
        self.max_frame = kwargs.get("max_frame", -1)
        self.frame_stride = kwargs.get("frame_stride", -1)

    def _sample_frames(self, imgs, anns=None, num_frames=None):
        frame_length = len(imgs)
        num_frames = num_frames if num_frames is not None else self.num_frames
        if self.frame_mode == "span" and num_frames > 0:
            indices = list(
                chain(
                    *sorted(
                        [
                            list(range(start, frame_length, max(1, frame_length // num_frames)))[:num_frames]
                            for start in range(math.ceil(frame_length / num_frames))
                        ],
                        key=lambda x: x[0],
                    )
                )
            )
        elif self.frame_mode == "stride" and self.frame_stride > 0:
            indices = list(
                chain(
                    *sorted(
                        [
                            group[i : i + num_frames]
                            for start in range(self.frame_stride)
                            for group in [list(range(start, frame_length, self.frame_stride))]
                            for i in range(0, len(group), num_frames)
                            if len(group[i : i + num_frames]) > 1
                        ],
                        key=lambda x: x[0],
                    )
                )
            )
        else:
            raise ValueError(
                f"Invalid frame_mode: {self.frame_mode}, num_frames: {num_frames}, frame_stride: {self.frame_stride}"
            )

        imgs = [imgs[i] for i in indices]
        if anns is not None:
            anns = [anns[i] for i in indices]
        return imgs, anns

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("image_files", None) is not None:
            image_files = data_dict["image_files"]
            extra_image_files = data_dict.get("extra_image_files", image_files)
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
