import argparse
import collections
import json
import multiprocessing
import os.path as osp
import time
from typing import Any, Mapping, MutableMapping, Sequence, Text

import numpy as np
from panopticapi.utils import get_traceback, rgb2id
from PIL import Image
from tabulate import tabulate

_EPSILON = 1e-15


class STQStat:
    """Metric class for the Segmentation and Tracking Quality (STQ).

    Please see the following paper for more details about the metric:

    "STEP: Segmenting and Tracking Every Pixel", Weber et al., arXiv:2102.11859,
    2021.


    The metric computes the geometric mean of two terms.
    - Association Quality: This term measures the quality of the track ID
        assignment for `thing` classes. It is formulated as a weighted IoU
        measure.
    - Segmentation Quality: This term measures the semantic segmentation quality.
        The standard class IoU measure is used for this.

    Example usage:

    stq_obj = segmentation_tracking_quality.STQ(num_classes, things_list,
      ignore_value, label_bit_shift, offset)
    stq_obj.update_state(y_true_1, y_pred_1)
    stq_obj.update_state(y_true_2, y_pred_2)
    ...
    result = stq_obj.result()
    """

    def __init__(
        self, num_classes: int, things_list: Sequence[int], ignore_value: int, label_bit_shift: int, offset: int
    ):
        """Initialization of the STQ metric.

        Args:
          num_classes: Number of classes in the dataset as an integer.
          things_list: A sequence of class ids that belong to `things`.
          ignore_value: The class id to be ignored in evaluation as an integer or
            integer tensor.
          label_bit_shift: The number of bits the class label is shifted as an
            integer -> (class_label << bits) + trackingID
          offset: The maximum number of unique labels as an integer or integer
            tensor.
        """
        self._num_classes = num_classes
        self._ignore_value = ignore_value
        self._things_list = things_list
        self._label_bit_shift = label_bit_shift
        self._bit_mask = (2**label_bit_shift) - 1

        if ignore_value >= num_classes:
            self._confusion_matrix_size = num_classes + 1
            self._include_indices = np.arange(self._num_classes)
        else:
            self._confusion_matrix_size = num_classes
            self._include_indices = np.array([i for i in range(num_classes) if i != self._ignore_value])

        self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
        self._predictions = collections.OrderedDict()
        self._ground_truth = collections.OrderedDict()
        self._intersections = collections.OrderedDict()
        self._sequence_length = collections.OrderedDict()
        self._offset = offset
        lower_bound = num_classes << self._label_bit_shift
        if offset < lower_bound:
            raise ValueError(
                "The provided offset %d is too small. No guarantess "
                "about the correctness of the results can be made. "
                "Please choose an offset that is higher than num_classes"
                " * max_instances_per_category = %d" % lower_bound
            )

    def get_semantic(self, y: np.ndarray) -> np.ndarray:
        """Returns the semantic class from a panoptic label map."""
        return y >> self._label_bit_shift

    @staticmethod
    def _update_dict_stats(stat_dict: MutableMapping[int, np.ndarray], id_array: np.ndarray):
        """Updates a given dict with corresponding counts."""
        ids, counts = np.unique(id_array, return_counts=True)
        for idx, count in zip(ids, counts):
            if idx in stat_dict:
                stat_dict[idx] += count
            else:
                stat_dict[idx] = count

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray, sequence_id=0):
        """Accumulates the segmentation and tracking quality statistics.

        IMPORTANT: When encoding the parameters y_true and y_pred, please be aware
        that the `+` operator binds higher than the label shift `<<` operator.

        Args:
          y_true: The ground-truth panoptic label map for a particular video frame
            (defined as (semantic_map << label_bit_shift) + instance_map).
          y_pred: The predicted panoptic label map for a particular video frame
            (defined as (semantic_map << label_bit_shift) + instance_map).
          sequence_id: The optional ID of the sequence the frames belong to. When no
            sequence is given, all frames are considered to belong to the same
            sequence (default: 0).
        """
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)

        semantic_label = self.get_semantic(y_true)
        semantic_prediction = self.get_semantic(y_pred)
        # Check if the ignore value is outside the range [0, num_classes]. If yes,
        # map `_ignore_value` to `_num_classes`, so it can be used to create the
        # confusion matrix.
        if self._ignore_value > self._num_classes:
            semantic_label = np.where(semantic_label != self._ignore_value, semantic_label, self._num_classes)
            semantic_prediction = np.where(
                semantic_prediction != self._ignore_value, semantic_prediction, self._num_classes
            )
        if sequence_id in self._iou_confusion_matrix_per_sequence:
            idxs = (np.reshape(semantic_label, [-1]) << self._label_bit_shift) + np.reshape(semantic_prediction, [-1])
            unique_idxs, counts = np.unique(idxs, return_counts=True)
            self._iou_confusion_matrix_per_sequence[sequence_id][
                unique_idxs >> self._label_bit_shift, unique_idxs & self._bit_mask
            ] += counts
            self._sequence_length[sequence_id] += 1
        else:
            self._iou_confusion_matrix_per_sequence[sequence_id] = np.zeros(
                (self._confusion_matrix_size, self._confusion_matrix_size), dtype=np.int64
            )
            idxs = np.stack([np.reshape(semantic_label, [-1]), np.reshape(semantic_prediction, [-1])], axis=0)
            np.add.at(self._iou_confusion_matrix_per_sequence[sequence_id], tuple(idxs), 1)

            self._predictions[sequence_id] = {}
            self._ground_truth[sequence_id] = {}
            self._intersections[sequence_id] = {}
            self._sequence_length[sequence_id] = 1

        instance_label = y_true & self._bit_mask  # 0xFFFF == 2 ^ 16 - 1

        label_mask = np.zeros_like(semantic_label, dtype=np.bool)
        prediction_mask = np.zeros_like(semantic_prediction, dtype=np.bool)
        for things_class_id in self._things_list:
            label_mask = np.logical_or(label_mask, semantic_label == things_class_id)
            prediction_mask = np.logical_or(prediction_mask, semantic_prediction == things_class_id)

        # Select the `crowd` region of the current class. This region is encoded
        # instance id `0`.
        iscrowd = np.logical_and(instance_label == 0, label_mask)
        # Select the non-crowd region of the corresponding class as the `crowd`
        # region is ignored for the tracking term.
        label_mask = np.logical_and(label_mask, np.logical_not(iscrowd))
        # Do not punish id assignment for regions that are annotated as `crowd` in
        # the ground-truth.
        prediction_mask = np.logical_and(prediction_mask, np.logical_not(iscrowd))

        seq_preds = self._predictions[sequence_id]
        seq_gts = self._ground_truth[sequence_id]
        seq_intersects = self._intersections[sequence_id]

        # Compute and update areas of ground-truth, predictions and intersections.
        self._update_dict_stats(seq_preds, y_pred[prediction_mask])
        self._update_dict_stats(seq_gts, y_true[label_mask])

        non_crowd_intersection = np.logical_and(label_mask, prediction_mask)
        intersection_ids = y_true[non_crowd_intersection] * self._offset + y_pred[non_crowd_intersection]
        self._update_dict_stats(seq_intersects, intersection_ids)

    def result(self) -> Mapping[Text, Any]:
        """Computes the segmentation and tracking quality.

        Returns:
          A dictionary containing:
            - 'STQ': The total STQ score.
            - 'AQ': The total association quality (AQ) score.
            - 'IoU': The total mean IoU.
            - 'STQ_per_seq': A list of the STQ score per sequence.
            - 'AQ_per_seq': A list of the AQ score per sequence.
            - 'IoU_per_seq': A list of mean IoU per sequence.
            - 'Id_per_seq': A list of string-type sequence Ids to map list index to
                sequence.
            - 'Length_per_seq': A list of the length of each sequence.
        """
        # Compute association quality (AQ)
        num_tubes_per_seq = [0] * len(self._ground_truth)
        aq_per_seq = [0] * len(self._ground_truth)
        iou_per_seq = [0] * len(self._ground_truth)
        id_per_seq = [""] * len(self._ground_truth)

        for index, sequence_id in enumerate(self._ground_truth):
            outer_sum = 0.0
            predictions = self._predictions[sequence_id]
            ground_truth = self._ground_truth[sequence_id]
            intersections = self._intersections[sequence_id]
            num_tubes_per_seq[index] = len(ground_truth)
            id_per_seq[index] = sequence_id

            for gt_id, gt_size in ground_truth.items():
                inner_sum = 0.0
                for pr_id, pr_size in predictions.items():
                    tpa_key = self._offset * gt_id + pr_id
                    if tpa_key in intersections:
                        tpa = intersections[tpa_key]
                        fpa = pr_size - tpa
                        fna = gt_size - tpa
                        inner_sum += tpa * (tpa / (tpa + fpa + fna))

                outer_sum += 1.0 / gt_size * inner_sum
            aq_per_seq[index] = outer_sum

        aq_mean = np.sum(aq_per_seq) / np.maximum(np.sum(num_tubes_per_seq), _EPSILON)
        aq_per_seq = aq_per_seq / np.maximum(num_tubes_per_seq, _EPSILON)

        # Compute IoU scores.
        # The rows correspond to ground-truth and the columns to predictions.
        # Remove fp from confusion matrix for the void/ignore class.
        total_confusion = np.zeros((self._confusion_matrix_size, self._confusion_matrix_size), dtype=np.int64)
        for index, confusion in enumerate(self._iou_confusion_matrix_per_sequence.values()):
            removal_matrix = np.zeros_like(confusion)
            removal_matrix[self._include_indices, :] = 1.0
            confusion *= removal_matrix
            total_confusion += confusion

            # `intersections` corresponds to true positives.
            intersections = confusion.diagonal()
            fps = confusion.sum(axis=0) - intersections
            fns = confusion.sum(axis=1) - intersections
            unions = intersections + fps + fns

            num_classes = np.count_nonzero(unions)
            ious = intersections.astype(np.double) / np.maximum(unions, 1e-15).astype(np.double)
            iou_per_seq[index] = np.sum(ious) / num_classes

        # `intersections` corresponds to true positives.
        intersections = total_confusion.diagonal()
        fps = total_confusion.sum(axis=0) - intersections
        fns = total_confusion.sum(axis=1) - intersections
        unions = intersections + fps + fns

        num_classes = np.count_nonzero(unions)
        ious = intersections.astype(np.double) / np.maximum(unions, _EPSILON).astype(np.double)
        iou_mean = np.sum(ious) / num_classes

        st_quality = np.sqrt(aq_mean * iou_mean)
        st_quality_per_seq = np.sqrt(aq_per_seq * iou_per_seq)

        return {
            "STQ": st_quality,
            "AQ": aq_mean,
            "IoU": float(iou_mean),
            "STQ_per_seq": st_quality_per_seq,
            "AQ_per_seq": aq_per_seq,
            "IoU_per_seq": iou_per_seq,
            "ID_per_seq": id_per_seq,
            "Length_per_seq": list(self._sequence_length.values()),
        }

    def reset_states(self):
        """Resets all states that accumulated data."""
        self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
        self._predictions = collections.OrderedDict()
        self._ground_truth = collections.OrderedDict()
        self._intersections = collections.OrderedDict()
        self._sequence_length = collections.OrderedDict()

    def __iadd__(self, other: "STQStat") -> "STQStat":
        """Merge another STQStat instance into this one."""
        for seq_id in other._iou_confusion_matrix_per_sequence:
            if seq_id not in self._iou_confusion_matrix_per_sequence:
                # New sequence, just copy
                self._iou_confusion_matrix_per_sequence[seq_id] = other._iou_confusion_matrix_per_sequence[
                    seq_id
                ].copy()
                self._predictions[seq_id] = dict(other._predictions[seq_id])
                self._ground_truth[seq_id] = dict(other._ground_truth[seq_id])
                self._intersections[seq_id] = dict(other._intersections[seq_id])
                self._sequence_length[seq_id] = other._sequence_length[seq_id]
            else:
                # Same sequence exists, merge statistics
                self._iou_confusion_matrix_per_sequence[seq_id] += other._iou_confusion_matrix_per_sequence[seq_id]
                self._sequence_length[seq_id] += other._sequence_length[seq_id]
                # Merge predictions
                for k, v in other._predictions[seq_id].items():
                    if k in self._predictions[seq_id]:
                        self._predictions[seq_id][k] += v
                    else:
                        self._predictions[seq_id][k] = v
                # Merge ground_truth
                for k, v in other._ground_truth[seq_id].items():
                    if k in self._ground_truth[seq_id]:
                        self._ground_truth[seq_id][k] += v
                    else:
                        self._ground_truth[seq_id][k] = v
                # Merge intersections
                for k, v in other._intersections[seq_id].items():
                    if k in self._intersections[seq_id]:
                        self._intersections[seq_id][k] += v
                    else:
                        self._intersections[seq_id][k] = v
        return self


@get_traceback
def stq_compute_single_core(
    proc_id, annotation_set, gt_folder, pred_folder, categories, n_classes, ignore_value, bit_shift
):
    """Single core computation of STQ statistics."""
    thing_list = [cat["id"] for cat in categories if cat["isthing"] == 1]
    stq_metric = STQStat(n_classes, thing_list, ignore_value, bit_shift, 2**24)

    for idx, cur_annotation in enumerate(annotation_set):
        video_id, (gt_anns, pred_anns) = next(iter(cur_annotation.items()))
        if idx % 10 == 0:
            print(f"Core: {proc_id}, {idx} from {len(annotation_set)} videos processed")

        # Build GT ID to instance number mapping
        gt_id_to_ins_num_dic = {}
        gt_id_list = []
        for segm in gt_anns:
            for img_info in segm["segments_info"]:
                id_tmp = img_info["id"]
                if id_tmp not in gt_id_list:
                    gt_id_list.append(id_tmp)
        for ii, id_tmp in enumerate(gt_id_list):
            gt_id_to_ins_num_dic[id_tmp] = ii

        # Build prediction ID to instance number mapping
        pred_id_to_ins_num_dic = {}
        pred_id_list = []
        for segm in pred_anns:
            for img_info in segm["segments_info"]:
                id_tmp = img_info["id"]
                if id_tmp not in pred_id_list:
                    pred_id_list.append(id_tmp)
        for ii, id_tmp in enumerate(pred_id_list):
            pred_id_to_ins_num_dic[id_tmp] = ii

        # Process each frame
        for gt_ann, pred_ann in zip(gt_anns, pred_anns):
            # Read GT and prediction panoptic images
            pan_gt = np.array(
                Image.open(osp.join(gt_folder, video_id, gt_ann["file_name"])).convert("RGB"), dtype=np.uint32
            )
            pan_pred = np.array(
                Image.open(osp.join(pred_folder, pred_ann["file_name"])).convert("RGB"), dtype=np.uint32
            )

            pan_gt = rgb2id(pan_gt)
            pan_pred = rgb2id(pan_pred)

            # Build ground truth
            ground_truth_instance = np.ones_like(pan_gt) * ignore_value
            ground_truth_semantic = np.ones_like(pan_gt) * ignore_value
            for el in gt_ann["segments_info"]:
                id_ = el["id"]
                cat_id = el["category_id"]
                ground_truth_semantic[pan_gt == id_] = cat_id
                ground_truth_instance[pan_gt == id_] = gt_id_to_ins_num_dic[id_]

            # Build prediction
            prediction_instance = np.ones_like(pan_pred) * ignore_value
            prediction_semantic = np.ones_like(pan_pred) * ignore_value
            for el in pred_ann["segments_info"]:
                id_ = el["id"]
                cat_id = el["category_id"]
                prediction_semantic[pan_pred == id_] = cat_id
                prediction_instance[pan_pred == id_] = pred_id_to_ins_num_dic[id_]

            valid_mask = ground_truth_semantic != ignore_value
            ground_truth = (ground_truth_semantic[valid_mask] << bit_shift) + ground_truth_instance[valid_mask]
            prediction = (prediction_semantic[valid_mask] << bit_shift) + prediction_instance[valid_mask]

            stq_metric.update_state(ground_truth.astype(np.int32), prediction.astype(np.int32), video_id)

    print(f"Core: {proc_id}, all {len(annotation_set)} videos processed")
    return stq_metric


def stq_compute_multi_core(
    matched_annotations_list, gt_folder, pred_folder, categories, n_classes, ignore_value, bit_shift
):
    """Multi-core computation of STQ statistics."""
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print(f"Number of cores: {cpu_num}, videos per core: {len(annotations_split[0])}")

    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(
            stq_compute_single_core,
            (proc_id, annotation_set, gt_folder, pred_folder, categories, n_classes, ignore_value, bit_shift),
        )
        processes.append(p)

    workers.close()
    workers.join()

    # Merge results from all processes
    thing_list = [cat["id"] for cat in categories if cat["isthing"] == 1]
    stq_stat = STQStat(n_classes, thing_list, ignore_value, bit_shift, 2**24)
    for p in processes:
        stq_stat += p.get()
    return stq_stat


def stq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None, ignore_value=255, bit_shift=16):
    """Compute STQ metrics.

    Args:
        gt_json_file: Path to ground truth JSON file.
        pred_json_file: Path to prediction JSON file.
        gt_folder: Path to ground truth panoptic masks folder. If None, uses gt_json_file path without .json.
        pred_folder: Path to prediction panoptic masks folder. If None, uses pred_json_file path without .json.
        ignore_value: The class id to be ignored in evaluation.
        bit_shift: The number of bits the class label is shifted.

    Returns:
        Dictionary containing STQ, AQ, IoU and per-sequence results.
    """
    start_time = time.time()

    with open(gt_json_file, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_file, "r") as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace(".json", "")
    if pred_folder is None:
        pred_folder = pred_json_file.replace(".json", "")

    categories = gt_json["categories"]
    n_classes = len(categories)

    print("Evaluation STQ metrics:")
    print("Ground truth:")
    print(f"\tSegmentation folder: {gt_folder}")
    print(f"\tJSON file: {gt_json_file}")
    print("Prediction:")
    print(f"\tSegmentation folder: {pred_folder}")
    print(f"\tJSON file: {pred_json_file}")

    if not osp.isdir(gt_folder):
        raise Exception(f"Folder {gt_folder} with ground truth segmentations doesn't exist")
    if not osp.isdir(pred_folder):
        raise Exception(f"Folder {pred_folder} with predicted segmentations doesn't exist")

    # Build prediction annotations dict
    pred_annotations = {el["video_id"]: el["annotations"] for el in pred_json["annotations"]}

    # Build GT annotations dict
    gt_annotations = {el["video_id"]: el["annotations"] for el in gt_json["annotations"]}

    matched_annotations_list = []
    for video_info in gt_json["videos"]:
        video_id = video_info["video_id"]
        if video_id not in pred_annotations:
            print(f"No prediction for the video with id: {video_id}")
            continue

        gt_ann = gt_annotations[video_id]
        pred_ann = pred_annotations[video_id]

        # Ensure explicit alignment by image_id to avoid frame mismatch
        pred_img_ids = [x["image_id"] for x in pred_ann]
        gt_ann_by_id = {x["image_id"]: x for x in gt_ann}

        # Filter and align gt_ann to match pred_ann order
        matched_img_ids = [img_id for img_id in pred_img_ids if img_id in gt_ann_by_id]
        aligned_gt_ann = [gt_ann_by_id[img_id] for img_id in matched_img_ids]

        # Filter pred_ann to only matched frames
        matched_img_ids_set = set(matched_img_ids)
        aligned_pred_ann = [x for x in pred_ann if x["image_id"] in matched_img_ids_set]

        matched_annotations_list.append({video_id: (aligned_gt_ann, aligned_pred_ann)})

    print("Evaluation STQ metrics:")
    print(f"Number of videos: {len(matched_annotations_list)}")

    stq_stat = stq_compute_multi_core(
        matched_annotations_list, gt_folder, pred_folder, categories, n_classes, ignore_value, bit_shift
    )

    result = stq_stat.result()

    print("=" * 80)
    print(f"STQ: {result['STQ']:.4f}")
    print(f"AQ: {result['AQ']:.4f}")
    print(f"IoU: {result['IoU']:.4f}")
    print("=" * 80)

    t_delta = time.time() - start_time
    print(f"Time elapsed: {t_delta:.2f} seconds")

    return result


def print_stq_results(stq_res):
    """Print STQ results in a formatted table."""
    headers = ["Metric", "STQ", "AQ", "IoU"]
    data = [["Value (%)", f"{stq_res['STQ'] * 100:.2f}", f"{stq_res['AQ'] * 100:.2f}", f"{stq_res['IoU'] * 100:.2f}"]]
    table = tabulate(
        data,
        headers=headers,
        tablefmt="outline",
        floatfmt=".2f",
        stralign="center",
        numalign="center",
    )
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="VPSNet eval")
    parser.add_argument("--submit_dir", "-i", type=str, help="test output directory")

    parser.add_argument(
        "--truth_dir",
        type=str,
        help="ground truth directory. Point this to <BASE_DIR>/VIPSeg/vipseg_720p/panomasksRGB "
        "after running the conversion script",
    )

    parser.add_argument(
        "--pan_gt_json_file",
        type=str,
        help="ground truth JSON file. Point this to <BASE_DIR>/VIPSeg/vipseg_720p/panoptic_gt_"
        "VIPSeg_val.json after running the conversion script",
    )

    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json_file", type=str, help="JSON file with ground truth data")
    parser.add_argument("--pred_json_file", type=str, help="JSON file with predictions data")
    parser.add_argument(
        "--gt_folder",
        type=str,
        default=None,
        help="Folder with ground truth COCO format segmentations. "
        "Default: X if the corresponding json file is X.json",
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        default=None,
        help="Folder with prediction COCO format segmentations. "
        "Default: X if the corresponding json file is X.json",
    )
    parser.add_argument("--n_classes", type=int, default=124, help="Number of classes in the dataset")
    parser.add_argument("--ignore_value", type=int, default=255, help="The class id to be ignored in evaluation")
    parser.add_argument("--bit_shift", type=int, default=16, help="The number of bits the class label is shifted")
    args = parser.parse_args()

    result = stq_compute(
        args.gt_json_file,
        args.pred_json_file,
        args.gt_folder,
        args.pred_folder,
        args.n_classes,
        args.ignore_value,
        args.bit_shift,
    )
    print(print_stq_results(result))


if __name__ == "__main__":
    main()
