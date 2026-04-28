import base64
import json
import os
import os.path as osp
from functools import partial
from io import BytesIO
from typing import Optional, Union

import numpy as np
import requests
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata
from transformers.video_utils import load_video as hf_load_video

from x2sam.utils.logging import print_log

from ..utils.video import sample_frames


def sample_frames_from_metadata(
    metadata: VideoMetadata,
    num_frames: Optional[int] = None,
    fps: Optional[Union[int, float]] = None,
    **kwargs,
):
    """
    Uniformly sample frame indices from video metadata.
    """
    if fps is not None and num_frames is not None:
        raise ValueError("`num_frames` and `fps` are mutually exclusive, please use only one!")

    total_num_frames = metadata.total_num_frames
    if total_num_frames is None:
        raise ValueError("`metadata.total_num_frames` is required for frame sampling.")

    if num_frames is None and fps is not None:
        if metadata.fps is None:
            raise ValueError("Asked to sample `fps` frames per second but `metadata.fps` is missing.")
        num_frames = int(total_num_frames / metadata.fps * fps)

    if num_frames is not None:
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be > 0, got {num_frames}.")
        if num_frames > total_num_frames:
            print_log(
                f"Video frames length is less than `num_frames={num_frames}`, using `total_num_frames={total_num_frames}`.",
                logger="current",
            )
            num_frames = total_num_frames
        indices = torch.linspace(0, total_num_frames - 1, steps=num_frames).long()
    else:
        indices = torch.arange(0, total_num_frames).long()

    return indices


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def load_image(image_file, threshold=128, mode="RGB", to_numpy=False, to_tensor=False):
    assert mode in ["RGB", "L"]
    if to_numpy and to_tensor:
        raise ValueError("`to_numpy` and `to_tensor` cannot both be True.")

    if isinstance(image_file, Image.Image):
        image = image_file.convert(mode)
    elif isinstance(image_file, np.ndarray):
        image = Image.fromarray(image_file.astype(np.uint8)).convert(mode)
    elif isinstance(image_file, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.permute(1, 2, 0).numpy()
        elif image.ndim == 2:
            image = image.numpy()
        else:
            image = image.numpy()
        image = Image.fromarray(image.astype(np.uint8)).convert(mode)
    elif isinstance(image_file, str):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert(mode)
        else:
            image = Image.open(image_file).convert(mode)
    else:
        raise ValueError(f"Unsupported image input: {type(image_file)}")

    if mode == "L" and threshold is not None:
        image = image.point(lambda x: 1 if x > threshold else 0, mode="L")

    if to_numpy:
        image = np.array(image)
    if to_tensor:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = torch.from_numpy(np.ascontiguousarray(image.copy()))
    return image


def _stack_frames(frames, to_numpy=False, to_tensor=False):
    if to_tensor and len(frames) > 0 and isinstance(frames[0], torch.Tensor):
        return torch.stack(frames, dim=0)
    if to_numpy and len(frames) > 0 and isinstance(frames[0], np.ndarray):
        return np.stack(frames, axis=0)
    return frames


def _convert_decoded_frame(frame, threshold=128, mode="RGB", to_numpy=False, to_tensor=False):
    return load_image(
        frame,
        threshold=threshold,
        mode=mode,
        to_numpy=to_numpy,
        to_tensor=to_tensor,
    )


def _is_image_path(path: str):
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def _load_frames_from_directory(
    video_dir,
    num_frames=8,
    frame_mode="span",
    threshold=128,
    mode="RGB",
    to_numpy=False,
    to_tensor=False,
    do_sample_frames=True,
):
    image_files = sorted(os.listdir(video_dir))
    if do_sample_frames:
        image_files = sample_frames(image_files, num_frames=num_frames, frame_mode=frame_mode)[:num_frames]

    frames = []
    for image_file in image_files:
        image = load_image(
            osp.join(video_dir, image_file),
            threshold=threshold,
            mode=mode,
            to_numpy=to_numpy,
            to_tensor=to_tensor,
        )
        frames.append(image)
    return _stack_frames(frames, to_numpy=to_numpy, to_tensor=to_tensor)


def _load_frames_from_list(
    frame_list,
    threshold=128,
    mode="RGB",
    to_numpy=False,
    to_tensor=False,
    do_sample_frames=True,
):
    if do_sample_frames:
        raise ValueError("Sampling frames from a list of images is not supported! Set `do_sample_frames=False`.")

    frames = [
        load_image(
            frame,
            threshold=threshold,
            mode=mode,
            to_numpy=to_numpy,
            to_tensor=to_tensor,
        )
        for frame in frame_list
    ]
    return _stack_frames(frames, to_numpy=to_numpy, to_tensor=to_tensor)


def _load_video_via_hf(
    video_source,
    num_frames=8,
    fps=None,
    threshold=128,
    mode="RGB",
    to_numpy=False,
    to_tensor=False,
    do_sample_frames=True,
    **kwargs,
):
    sample_indices_fn = None
    if do_sample_frames:
        sample_indices_fn = partial(
            sample_frames_from_metadata,
            num_frames=num_frames,
            fps=fps,
            **kwargs,
        )

    video, _ = hf_load_video(
        video_source,
        sample_indices_fn=sample_indices_fn,
    )
    video = _stack_frames(
        [
            _convert_decoded_frame(
                frame,
                threshold=threshold,
                mode=mode,
                to_numpy=to_numpy,
                to_tensor=to_tensor,
            )
            for frame in video
        ]
    )

    return video


def load_video(
    video_file,
    num_frames=None,
    fps=None,
    frame_mode="span",
    threshold=128,
    mode="RGB",
    to_numpy=False,
    to_tensor=False,
    do_sample_frames=False,
    **kwargs,
):
    assert mode in ["RGB", "L"]
    if to_numpy and to_tensor:
        raise ValueError("`to_numpy` and `to_tensor` cannot both be True.")

    # 1. list/tuple: treat as image sequence
    if isinstance(video_file, (list, tuple)):
        return _load_frames_from_list(
            video_file,
            threshold=threshold,
            mode=mode,
            to_numpy=to_numpy,
            to_tensor=to_tensor,
            do_sample_frames=do_sample_frames,
        )

    # 2. PIL / ndarray / tensor: single image
    if isinstance(video_file, (Image.Image, np.ndarray, torch.Tensor)):
        image = load_image(
            video_file,
            threshold=threshold,
            mode=mode,
            to_numpy=to_numpy,
            to_tensor=to_tensor,
        )
        return _stack_frames([image], to_numpy=to_numpy, to_tensor=to_tensor)

    # 3. str input
    if isinstance(video_file, str):
        # URL
        if video_file.startswith("http://") or video_file.startswith("https://"):
            if _is_image_path(video_file):
                image = load_image(
                    video_file,
                    threshold=threshold,
                    mode=mode,
                    to_numpy=to_numpy,
                    to_tensor=to_tensor,
                )
                return _stack_frames([image], to_numpy=to_numpy, to_tensor=to_tensor)

            return _load_video_via_hf(
                video_file,
                num_frames=num_frames,
                fps=fps,
                threshold=threshold,
                mode=mode,
                to_numpy=to_numpy,
                to_tensor=to_tensor,
                do_sample_frames=do_sample_frames,
                **kwargs,
            )

        # directory of frames
        if osp.isdir(video_file):
            return _load_frames_from_directory(
                video_file,
                num_frames=num_frames,
                frame_mode=frame_mode,
                threshold=threshold,
                mode=mode,
                to_numpy=to_numpy,
                to_tensor=to_tensor,
                do_sample_frames=do_sample_frames,
            )

        # file path
        if osp.isfile(video_file):
            if _is_image_path(video_file):
                image = load_image(
                    video_file,
                    threshold=threshold,
                    mode=mode,
                    to_numpy=to_numpy,
                    to_tensor=to_tensor,
                )
                return _stack_frames([image], to_numpy=to_numpy, to_tensor=to_tensor)

            return _load_video_via_hf(
                video_file,
                num_frames=num_frames,
                fps=fps,
                threshold=threshold,
                mode=mode,
                to_numpy=to_numpy,
                to_tensor=to_tensor,
                do_sample_frames=do_sample_frames,
                **kwargs,
            )

    raise ValueError(f"Unsupported video input: {video_file}")
