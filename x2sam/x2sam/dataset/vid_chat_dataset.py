import copy
import ctypes
import json
import os
import os.path as osp
from collections import defaultdict
from contextlib import contextmanager

import torch

from x2sam.utils.constants import DEFAULT_VIDEO_TOKEN
from x2sam.utils.logging import print_log

from .utils.image import expand2square
from .utils.load import load_video
from .vid_base_dataset import VidBaseDataset


def _set_ffmpeg_log_level(level):
    """Set FFmpeg log level using ctypes.

    FFmpeg log levels:
        AV_LOG_QUIET   = -8  (suppress all)
        AV_LOG_PANIC   =  0
        AV_LOG_FATAL   =  8
        AV_LOG_ERROR   = 16
        AV_LOG_WARNING = 24
        AV_LOG_INFO    = 32
        AV_LOG_VERBOSE = 40
        AV_LOG_DEBUG   = 48
    """
    lib_names = [
        "libavutil.so",  # Linux
        "libavutil.dylib",  # macOS
        "avutil",  # Windows / fallback
        "libavutil.so.56",
        "libavutil.so.57",
        "libavutil.so.58",
        "libavutil.so.59",
    ]

    for lib_name in lib_names:
        try:
            libavutil = ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
            libavutil.av_log_set_level(level)
            return True
        except OSError:
            continue
    return False


@contextmanager
def suppress_ffmpeg_output():
    """Context manager to suppress FFmpeg/libav warning messages (supports PyAV, torchcodec, etc.)."""
    # FFmpeg log levels
    AV_LOG_QUIET = -8
    AV_LOG_WARNING = 24

    # Save original file descriptors
    original_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        # 1. Try to set av logging level if PyAV is available
        try:
            import av

            av.logging.set_level(av.logging.PANIC)
        except ImportError:
            pass

        # 2. Set FFmpeg log level via ctypes (for torchcodec and direct FFmpeg usage)
        _set_ffmpeg_log_level(AV_LOG_QUIET)

        # 3. Redirect stderr at file descriptor level (catches C-level output)
        os.dup2(devnull_fd, 2)

        yield

    finally:
        # Restore stderr file descriptor
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
        os.close(devnull_fd)

        # Restore av logging level
        try:
            import av

            av.logging.set_level(av.logging.WARNING)
        except ImportError:
            pass

        # Restore FFmpeg log level
        _set_ffmpeg_log_level(AV_LOG_WARNING)


class VidChatDataset(VidBaseDataset):
    def __init__(self, *args, source_data_path=None, **kwargs):
        super().__init__(*args, source_data_path=source_data_path, **kwargs)

    def custom_init(self, **kwargs):
        super().custom_init(**kwargs)
        self.source_data_path = kwargs.get("source_data_path", None)

    def _set_metadata(self, **kwargs):
        super()._set_metadata(**kwargs)

    def _load_ann_data(self):
        if self.data_path is not None and osp.exists(self.data_path):
            with open(self.data_path, "r") as f:
                data = json.load(f)
            return data

        if osp.exists(self.source_data_path):
            with open(self.source_data_path, "r") as f:
                data = json.load(f)

        video_id_ann_map = defaultdict(list)
        for ann in data:
            video_id_ann_map[ann["video_id"]].append(ann)

        rets = []
        for video_id, anns in video_id_ann_map.items():
            conversations = []
            for i, ann in enumerate(anns):
                conversations.extend(
                    [
                        {"from": "human", "value": f"{DEFAULT_VIDEO_TOKEN}\n{ann['q']}" if i == 0 else ann["q"]},
                        {"from": "gpt", "value": ann["a"]},
                    ]
                )
            video_file = next(
                (
                    f"{video_id}{ext}"
                    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov")
                    if osp.exists(osp.join(self.video_folder, f"{video_id}{ext}"))
                ),
                None,
            )
            if video_file is None:
                continue
            rets.append({"id": video_id, "video_file": video_file, "conversations": conversations})

        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(rets, f)

        return rets

    def __getitem__(self, index):
        index = index % self.data_length
        data_dict = copy.deepcopy(self.data[index])
        if data_dict.get("video_file", None) is not None:
            video_file = osp.join(self.video_folder, data_dict["video_file"])
            if self.video_processor is not None:
                try:
                    # TODO: pre-sample frames
                    with suppress_ffmpeg_output():
                        output = self.video_processor.preprocess(
                            video_file,
                            num_frames=self.num_frames,
                            fps=None,
                            do_sample_frames=True,
                            return_metadata=True,
                            return_tensors="pt",
                        )
                except Exception as e:
                    print_log(f"Error processing video {video_file}: {e}", logger="current")
                    # Use deterministic fallback index to prevent rank divergence in DDP
                    new_index = (index + 1) % self.data_length
                    return self.__getitem__(new_index)

                pixel_values_videos = (
                    output["pixel_values_videos"][0]
                    if output["pixel_values_videos"].ndim == 4
                    else output["pixel_values_videos"]
                )
                video_grid_thw = output.get("video_grid_thw", None)
                video_metadata = output.get("video_metadata", None)
                data_dict["pixel_values_videos"] = pixel_values_videos
                data_dict["video_grid_thw"] = video_grid_thw
                data_dict["video_metadata"] = video_metadata
            elif self.image_processor is not None:
                try:
                    video_images = load_video(video_file, num_frames=self.num_frames, fps=None, do_sample_frames=True)
                except Exception as e:
                    print_log(f"Error processing video {video_file}: {e}", logger="current")
                    # Use deterministic fallback index to prevent rank divergence in DDP
                    new_index = (index + 1) % self.data_length
                    return self.__getitem__(new_index)
                    
                if self.expand2square:
                    video_images = [
                        expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                        for pil_image in video_images
                    ]
                output = self.image_processor.preprocess(video_images, return_tensors="pt")
                pixel_values_videos = output["pixel_values"]
                data_dict["pixel_values_videos"] = pixel_values_videos
            data_dict.update(self._get_input_ids(data_dict, use_vision_token=True))
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
            data_dict.update(self._get_input_ids(data_dict, use_vision_token=False))
        return data_dict
