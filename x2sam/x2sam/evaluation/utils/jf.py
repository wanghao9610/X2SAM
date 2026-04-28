import math
import multiprocessing

import cv2
import numpy as np
from skimage.morphology import disk
from tabulate import tabulate

from x2sam.dataset.utils.mask import decode_mask


class JFStat:
    """Statistics for J and F scores."""

    def __init__(self):
        self.j_score = 0.0
        self.f_score = 0.0
        self.jf_score = 0.0
        self.count = 0

    def update(self, j, f, n=1):
        self.j_score += j
        self.f_score += f
        self.count += n

    def __iadd__(self, jf_stat):
        """In-place addition operator for combining JFStat objects."""
        self.j_score += jf_stat.j_score
        self.f_score += jf_stat.f_score
        self.count += jf_stat.count
        return self

    def __add__(self, jf_stat):
        """Addition operator for combining JFStat objects."""
        result = JFStat()
        result.j_score = self.j_score + jf_stat.j_score
        result.f_score = self.f_score + jf_stat.f_score
        result.count = self.count + jf_stat.count
        return result

    def average(self):
        self.j_score = self.j_score / self.count
        self.f_score = self.f_score / self.count
        self.jf_score = (self.j_score + self.f_score) / 2

    def __repr__(self) -> str:
        headers = ["Metric", "J", "F", "J&F"]
        data = [["Value (%)", f"{self.j_score * 100:.2f}", f"{self.f_score * 100:.2f}", f"{self.jf_score * 100:.2f}"]]

        table = tabulate(data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center")
        return str(table)


def calculate_J(gt_mask, pred_mask):
    """Calculate the J Score(actually IoU) between the ground truth mask and the predicted mask."""
    gt_mask = gt_mask.astype(np.bool)
    pred_mask = pred_mask.astype(np.bool)
    intersection = np.sum(gt_mask & pred_mask, axis=(-2, -1))
    union = np.sum(gt_mask | pred_mask, axis=(-2, -1))
    J = intersection / union
    if J.ndim == 0:
        J = 1.0 if np.isclose(union, 0) else J
    else:
        J[np.isclose(union, 0)] = 1
    return J


def calculate_F(gt_mask, pred_mask, boundary_threshold=0.008):
    """Calculate the F Score(boundary aware F1 Score) between the ground truth mask and the predicted mask."""
    bound_pixel_size = (
        boundary_threshold if boundary_threshold >= 1 else np.ceil(boundary_threshold * np.linalg.norm(gt_mask.shape))
    )
    # binary mask to boundary mask
    gt_boundary_mask = binary_mask_to_boundary_mask(gt_mask)
    pred_boundary_mask = binary_mask_to_boundary_mask(pred_mask)
    # dilate boundary mask
    gt_dilated_boundary_mask = cv2.dilate(gt_boundary_mask.astype(np.uint8), disk(bound_pixel_size).astype(np.uint8))
    pred_dilated_boundary_mask = cv2.dilate(
        pred_boundary_mask.astype(np.uint8), disk(bound_pixel_size).astype(np.uint8)
    )
    # iou
    gt_matched_boundary_mask = gt_boundary_mask * pred_dilated_boundary_mask
    pred_matched_boundary_mask = pred_boundary_mask * gt_dilated_boundary_mask
    # area
    num_gt_boundary_pixels = np.sum(gt_boundary_mask)
    num_pred_boundary_pixels = np.sum(pred_boundary_mask)
    # precision and recall
    if num_pred_boundary_pixels == 0 and num_gt_boundary_pixels > 0:
        precision = 1
        recall = 0
    elif num_pred_boundary_pixels > 0 and num_gt_boundary_pixels == 0:
        precision = 0
        recall = 1
    elif num_pred_boundary_pixels == 0 and num_gt_boundary_pixels == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(pred_matched_boundary_mask) / num_pred_boundary_pixels
        recall = np.sum(gt_matched_boundary_mask) / num_gt_boundary_pixels

    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    return F


def binary_mask_to_boundary_mask(mask, width=None, height=None):
    """Convert a binary mask to a boundary mask."""
    mask = mask.astype(bool)
    mask[mask > 0] = 1

    assert np.atleast_3d(mask).shape[2] == 1

    width = mask.shape[1] if width is None else width
    height = mask.shape[0] if height is None else height

    h, w = mask.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), "Can't convert %dx%d seg to %dx%d bmap." % (
        w,
        h,
        width,
        height,
    )

    e = np.zeros_like(mask)
    s = np.zeros_like(mask)
    se = np.zeros_like(mask)

    e[:, :-1] = mask[:, 1:]
    s[:-1, :] = mask[1:, :]
    se[:-1, :-1] = mask[1:, 1:]

    b = mask ^ e | mask ^ s | mask ^ se
    b[-1, :] = mask[-1, :] ^ e[-1, :]
    b[:, -1] = mask[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        boundary_mask = b
    else:
        boundary_mask = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if boundary_mask[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    boundary_mask[j, i] = 1

    return boundary_mask


def jf_compute_single_core(annotation_set):
    jf_stat = JFStat()
    for gt_mask, pred_mask in annotation_set:
        gt_mask = decode_mask(gt_mask, *gt_mask["size"])
        pred_mask = decode_mask(pred_mask, *pred_mask["size"])
        j = calculate_J(gt_mask, pred_mask)
        f = calculate_F(gt_mask, pred_mask)
        jf_stat.update(j, f)
    return jf_stat


def jf_compute_multi_core(matched_annotations_list):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for annotation_set in annotations_split:
        p = workers.apply_async(jf_compute_single_core, args=(annotation_set,))
        processes.append(p)

    workers.close()
    workers.join()
    jf_stat = JFStat()
    for p in processes:
        jf_stat += p.get()

    return jf_stat


def jf_compute(annotations, predictions):
    gt_masks = []
    pred_masks = []
    for video_id, vid_preds in predictions.items():
        vid_anns = annotations.get(video_id, None)
        if vid_anns is None:
            print(f"No ground truth found for video {video_id}")
            continue
        for img_preds, img_anns in zip(vid_preds, vid_anns):
            for img_pred, img_ann in zip(img_preds, img_anns):
                gt_masks.append(img_ann["segmentation"])
                pred_masks.append(img_pred["segmentation"])
    matched_annotations_list = list(zip(gt_masks, pred_masks))
    jf_stat = jf_compute_multi_core(matched_annotations_list)
    jf_stat.average()
    return jf_stat


def print_jf_results(jf_res):
    headers = ["Metric", "J", "F", "J&F"]
    data = [["Value (%)", f"{jf_res.j_score * 100:.2f}", f"{jf_res.f_score * 100:.2f}", f"{jf_res.jf_score * 100:.2f}"]]
    table = tabulate(data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center")

    return str(table)
