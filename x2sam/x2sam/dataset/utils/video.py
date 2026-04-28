import math
from itertools import chain
from typing import Union

import numpy as np


def sample_frames(imgs, num_frames=None, frame_mode="span", frame_stride=1):
    frame_length = len(imgs)
    num_frames = num_frames if num_frames is not None else frame_length
    if frame_mode == "span" and num_frames > 0:
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
    elif frame_mode == "stride" and frame_stride > 0:
        indices = list(
            chain(
                *sorted(
                    [
                        group[i : i + num_frames]
                        for start in range(frame_stride)
                        for group in [list(range(start, frame_length, frame_stride))]
                        for i in range(0, len(group), num_frames)
                        if len(group[i : i + num_frames]) > 1
                    ],
                    key=lambda x: x[0],
                )
            )
        )
    else:
        raise ValueError(f"Invalid frame_mode: {frame_mode}, num_frames: {num_frames}, frame_stride: {frame_stride}")

    return [imgs[i] for i in indices]


def calculate_timestamps(indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
    if not isinstance(indices, list):
        indices = indices.tolist()
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
    timestamps = [idx / video_fps for idx in indices]
    # @JJJYmmm frames are merged by self.merge_size, \
    # so we need to average the timestamps between the first/last frame within the temporal patch
    timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]
    return timestamps
