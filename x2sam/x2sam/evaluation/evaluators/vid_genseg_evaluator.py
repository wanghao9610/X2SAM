import contextlib
import io
import itertools
import json
import logging
import os
import os.path as osp
import tempfile
from collections import defaultdict
from typing import List

import numpy as np
from panopticapi.utils import id2rgb
from PIL import Image
from pycocotools import mask as mask_util
from tabulate import tabulate

from x2sam.dataset.utils.coco import COCO
from x2sam.utils import comm
from x2sam.utils.logging import print_log

from ..utils.map import convert_to_coco_json, derive_coco_results, evaluate_predictions_on_coco, instances_to_coco_json
from ..utils.stq import print_stq_results, stq_compute
from ..utils.vc import mvc_compute
from ..utils.vpq import print_panoptic_results, vpq_compute
from .vid_base_evaluator import VidBaseEvaluator


class VidGenSegEvaluator(VidBaseEvaluator):
    def __init__(
        self,
        *args,
        data_name: str = "vid_genseg",
        encode_masks: bool = False,
        nframes: List[int] = [1, 2, 4, 6],
        evaluation_metrics: List[str] = ["mIoU", "mVC"],
        show_categories: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            evaluation_metrics,
            **kwargs,
        )
        self._encode_masks = encode_masks
        self._nframes = nframes
        self._show_categories = show_categories

    @VidBaseEvaluator.metadata.setter
    def metadata(self, value):
        self._metadata = value
        self._dataset_name = self.data_name
        self._num_classes = len(self._metadata.dataset_id_to_contiguous_id)
        self._contiguous_id_to_dataset_id = {v: k for k, v in self._metadata.dataset_id_to_contiguous_id.items()}
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._thing_contiguous_id_to_dataset_id = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
        if hasattr(self._metadata, "stuff_dataset_id_to_contiguous_id"):
            self._stuff_contiguous_id_to_dataset_id = {
                v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
            }

    def reset(self):
        self._predictions = []
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[segment_info["category_id"]]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[segment_info["category_id"]]
        return segment_info

    def _encode_json_sem_seg(self, sem_seg, input_file_name, ignore_value=255):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if label == ignore_value:
                continue
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle})
        return json_list

    def semantic_process(self, inputs, outputs):
        for video_input, video_output in zip(inputs, outputs):
            video_id = video_input["video_id"]
            chunk_id = video_input["chunk_id"]
            video_preds = []
            for file_name, image_output in zip(video_input["file_names"], video_output):
                segmentation = image_output["segmentation"].to(self._cpu_device)
                sampled_labels = image_output.get("sampled_labels", None)
                pred = np.array(segmentation, dtype=np.uint8)

                if sampled_labels is not None:
                    unique_labels = np.unique(pred).tolist()
                    for unique_label in unique_labels:
                        # with label shift because we shift the gt_segmap by label_shift
                        pred[pred == unique_label] = sampled_labels[unique_label]

                file_name_png = osp.splitext(file_name)[0] + ".png"
                with io.BytesIO() as out:
                    Image.fromarray(pred).save(out, format="PNG")
                    video_preds.append(
                        {
                            "file_name": file_name_png,
                            "png_string": out.getvalue(),
                        }
                    )

            self._predictions.append({"video_id": video_id, "chunk_id": chunk_id, "annotations": video_preds})

    def instance_process(self, inputs, outputs):
        for video_input, video_output in zip(inputs, outputs):
            video_id = video_input["video_id"]
            chunk_id = video_input["chunk_id"]
            video_preds = []
            for image_input, image_output in zip(video_input["images"], video_output):
                prediction = {"image_id": image_input["id"]}

                if "instances" in image_output:
                    instances = image_output["instances"].to(self._cpu_device)
                    prediction["instances"] = instances_to_coco_json(instances, image_input["id"])
                if "proposals" in image_output:
                    prediction["proposals"] = image_output["proposals"].to(self._cpu_device)
                if len(prediction) > 1:
                    video_preds.append(prediction)
            self._predictions.append({"video_id": video_id, "chunk_id": chunk_id, "annotations": video_preds})

    def panoptic_process(self, inputs, outputs):
        for video_input, video_output in zip(inputs, outputs):
            video_id = video_input["video_id"]
            chunk_id = video_input["chunk_id"]
            video_preds = []
            for image_input, image_output in zip(video_input["images"], video_output):
                segmentation, segments_info = (
                    image_output["segmentation"],
                    image_output["segments_info"],
                )
                segmentation = segmentation.to(self._cpu_device)
                segmentation = np.array(segmentation, dtype=int)
                if segments_info is None:
                    # If "segments_info" is None, we assume "segmentation" is a
                    # H*W int32 image storing the panoptic_id in the format of
                    # category_id * label_divisor + instance_id.
                    label_divisor = self._metadata.label_divisor
                    segments_info = []
                    for panoptic_label in np.unique(segmentation):
                        if panoptic_label == 0:
                            # VOID region.
                            continue
                        pred_class = panoptic_label // label_divisor
                        isthing = pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                        segments_info.append(
                            {
                                "id": int(panoptic_label) + 1,
                                "category_id": int(pred_class),
                                "isthing": bool(isthing),
                            }
                        )
                    # Official evaluation script uses 0 for VOID label.
                    segmentation += 1

                file_name = osp.basename(image_input["file_name"])
                file_name_png = f"{video_id}_{chunk_id}_{osp.splitext(file_name)[0]}.png"
                with io.BytesIO() as out:
                    Image.fromarray(id2rgb(segmentation)).save(out, format="PNG")
                    segments_info = [self._convert_category_id(x) for x in segments_info]
                    video_preds.append(
                        {
                            "image_id": image_input["id"],
                            "file_name": file_name_png,
                            "png_string": out.getvalue(),
                            "segments_info": segments_info,
                        }
                    )
            self._predictions.append({"video_id": video_id, "chunk_id": chunk_id, "annotations": video_preds})

    def semantic_evaluate(self, predictions):
        # merge the predictions of the same video
        merged_predictions = defaultdict(list)
        for prediction in predictions:
            video_id = prediction["video_id"]
            anns = prediction["annotations"]
            chunk_id = prediction["chunk_id"]
            merged_predictions[video_id].extend((chunk_id, anns))
        # deduplicate the predictions by file_name when distributed evaluation
        predictions = {
            video_id: list(
                {
                    ann["file_name"]: ann
                    for _, ann_list in sorted(zip(anns[::2], anns[1::2]), key=lambda x: x[0])
                    for ann in ann_list
                }.values()
            )
            for video_id, anns in merged_predictions.items()
        }

        gt_json = osp.realpath(self._metadata.gt_json) if self._metadata.gt_json is not None else None
        gt_segmap_folder = osp.realpath(self._metadata.sem_segmap_folder)
        segmap_suffix = self._metadata.segmap_suffix if hasattr(self._metadata, "segmap_suffix") else ".png"
        ignore_value = self._metadata.ignore_value if hasattr(self._metadata, "ignore_value") else 255
        label_shift = self._metadata.label_shift if hasattr(self._metadata, "label_shift") else 0
        invalid_catids = self._metadata.invalid_catids if hasattr(self._metadata, "invalid_catids") else []

        with tempfile.TemporaryDirectory(prefix="semantic_eval") as pred_dir:
            for _, video_pred in predictions.items():
                for pred in video_pred:
                    file_name = osp.join(pred_dir, pred["file_name"])
                    os.makedirs(osp.dirname(file_name), exist_ok=True)
                    assert not osp.exists(file_name), f"File {file_name} already exists"
                    with open(file_name, "wb") as f:
                        f.write(pred.pop("png_string"))

            if gt_json is not None:
                with open(gt_json, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                json_data["annotations"] = list(itertools.chain(*predictions.values()))

            table = ""
            with contextlib.redirect_stdout(io.StringIO()):
                video_gt_masks = []
                video_pred_masks = []
                for _, video_pred in predictions.items():
                    gt_masks = []
                    pred_masks = []
                    for pred in video_pred:
                        gt_file_name = osp.splitext(pred["file_name"])[0] + segmap_suffix
                        gt_segmap = np.array(Image.open(osp.join(gt_segmap_folder, gt_file_name)), dtype=np.int64)
                        pred_segmap = np.array(Image.open(osp.join(pred_dir, pred["file_name"])), dtype=np.int64)
                        if label_shift != 0:
                            gt_segmap[gt_segmap == 0] = ignore_value
                            gt_segmap = gt_segmap + label_shift
                            gt_segmap[gt_segmap == (ignore_value + label_shift)] = ignore_value

                        for invalid_catid in invalid_catids:
                            gt_segmap[gt_segmap == invalid_catid] = ignore_value
                            pred_segmap[pred_segmap == invalid_catid] = ignore_value

                        gt_masks.append(gt_segmap)
                        pred_masks.append(pred_segmap)

                    video_gt_masks.append(gt_masks)
                    video_pred_masks.append(pred_masks)

            if "mIoU" in self._evaluation_metrics:
                print_log("Evaluating mIoU...", logger="current")

                for gt_masks, pred_masks in zip(video_gt_masks, video_pred_masks):
                    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
                        cur_gt_mask = gt_mask.copy()
                        cur_gt_mask[cur_gt_mask == ignore_value] = self._num_classes
                        self._conf_matrix += np.bincount(
                            (self._num_classes + 1) * pred_mask.reshape(-1) + cur_gt_mask.reshape(-1),
                            minlength=self._conf_matrix.size,
                        ).reshape(self._conf_matrix.shape)

                acc = np.full(self._num_classes, np.nan, dtype=np.float32)
                iou = np.full(self._num_classes, np.nan, dtype=np.float32)
                tp = self._conf_matrix.diagonal()[:-1].astype(np.float32)
                pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float32)
                class_weights = pos_gt / np.sum(pos_gt)
                pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float32)
                acc_valid = pos_gt > 0
                acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
                union = pos_gt + pos_pred - tp
                iou_valid = np.logical_and(acc_valid, union > 0)
                iou[iou_valid] = tp[iou_valid] / union[iou_valid]
                macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
                miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
                fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
                pacc = np.sum(tp) / np.sum(pos_gt)

                headers = ["Metric", "mIoU", "fwIoU", "mACC", "pACC"]
                data = [
                    ["Value (%)", f"{100 * miou:.2f}", f"{100 * fiou:.2f}", f"{100 * macc:.2f}", f"{100 * pacc:.2f}"]
                ]

                if self._show_categories:
                    class_names = [self._metadata.dataset_classes[k] for k in sorted(self._metadata.dataset_classes)]
                    for i, name in enumerate(class_names):
                        data.extend([[f"IoU-{name}", f"{100 * iou[i]:.2f}"], [f"ACC-{name}", f"{100 * acc[i]:.2f}"]])

                table += "mIoU:\n" + tabulate(
                    data,
                    headers=headers,
                    tablefmt="outline",
                    floatfmt=".2f",
                    stralign="center",
                    numalign="center",
                )

            if "mVC" in self._evaluation_metrics:
                print_log(f"Evaluating mVC@{self._nframes}...", logger="current")
                mvc_accs = mvc_compute(video_gt_masks, video_pred_masks, self._nframes)
                headers = ["Metric", *[f"mVC@{n_frame}" for n_frame in self._nframes]]
                data = [
                    [
                        "Value (%)",
                        *[
                            f"{100 * (np.mean(mvc_accs[n_frame]) if len(mvc_accs[n_frame]) > 0 else 0.0):.2f}"
                            for n_frame in self._nframes
                        ],
                    ]
                ]

                table += "\nmVC:\n" + tabulate(
                    data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center"
                )

            return table

    def instance_evaluate(self, predictions):
        # merge the predictions of the same video
        merged_predictions = defaultdict(list)
        for prediction in predictions:
            video_id = prediction["video_id"]
            anns = prediction["annotations"]
            chunk_id = prediction["chunk_id"]
            merged_predictions[video_id].extend((chunk_id, anns))
        # deduplicate and sort the predictions by file_name when distributed evaluation
        predictions = {
            video_id: list(
                {
                    ann["image_id"]: ann
                    for _, ann_list in sorted(zip(anns[::2], anns[1::2]), key=lambda x: x[0])
                    for ann in ann_list
                }.values()
            )
            for video_id, anns in merged_predictions.items()
        }

        with open(osp.realpath(self._metadata.gt_json), "r") as f:
            gt_video_anns = json.load(f)
        # map video_anns to image_anns
        gt_image_anns = []
        for gt_ann in gt_video_anns:
            for image_info, annotations in zip(gt_ann["video_info"]["images"], gt_ann["annotations"]):
                gt_image_anns.append(
                    {
                        "image_id": image_info["id"],
                        "image_info": image_info,
                        "image_file": image_info["file_name"],
                        "image_size": (image_info["height"], image_info["width"]),
                        "annotations": annotations,
                    }
                )

        print_log(f"Trying to convert '{self.data_name}' to COCO format...", logger="current")
        cache_path = osp.join(self._output_dir, f"{self.data_name}_coco_format.json")
        convert_to_coco_json(self.data_name, cache_path, gt_image_anns, allow_cached=False)
        coco_api = COCO(cache_path)

        coco_results = list(
            itertools.chain(*[pred["instances"] for _, video_pred in predictions.items() for pred in video_pred])
        )
        # unmap the category ids for COCO
        dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        num_classes = len(all_contiguous_ids)
        assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

        reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        new_coco_results = []
        for result in coco_results:
            category_id = result["category_id"]
            if category_id not in reverse_id_mapping:
                # print_log(
                #     f"A prediction has class={category_id}, "
                #     f"but the dataset only has {num_classes} classes and "
                #     f"predicted class id should be in [0, {num_classes - 1}].",
                #     logger="current",
                # )
                continue
            result["category_id"] = reverse_id_mapping[category_id]
            new_coco_results.append(result)

        coco_eval = (
            evaluate_predictions_on_coco(
                coco_api,
                new_coco_results,
                "segm",
                
            )
            if len(new_coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )
        table = derive_coco_results(
            coco_eval, "segm", class_names=self._metadata.get("thing_classes"), show_categories=self._show_categories
        )
        return table

    def panoptic_evaluate(self, predictions):
        # merge the predictions of the same video
        merged_predictions = defaultdict(list)
        for prediction in predictions:
            video_id = prediction["video_id"]
            anns = prediction["annotations"]
            chunk_id = prediction["chunk_id"]
            merged_predictions[video_id].extend((chunk_id, anns))
        # deduplicate and sort the predictions by file_name when distributed evaluation
        predictions = {
            video_id: list(
                {
                    ann["file_name"]: ann
                    for _, ann_list in sorted(zip(anns[::2], anns[1::2]), key=lambda x: x[0])
                    for ann in ann_list
                }.values()
            )
            for video_id, anns in merged_predictions.items()
        }

        # PanopticApi requires local files
        gt_json = osp.realpath(self._metadata.gt_json)
        gt_panseg_folder = osp.realpath(self._metadata.pan_segmap_folder)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            for _, video_pred in predictions.items():
                for pred in video_pred:
                    assert not osp.exists(
                        osp.join(pred_dir, pred["file_name"])
                    ), f"File {pred['file_name']} already exists"
                    with open(osp.join(pred_dir, pred["file_name"]), "wb") as f:
                        f.write(pred.pop("png_string"))

            with open(gt_json, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            json_data["annotations"] = [
                {"video_id": video_id, "annotations": annotations} for video_id, annotations in predictions.items()
            ]

            pred_json = osp.join(pred_dir, "predictions.json")
            print_log(f"Saving {self.data_name} predictions to {pred_json}...", logger="current")
            with open(pred_json, "w") as f:
                json.dump(json_data, f)

            table = ""
            for evaluation_metric in self._evaluation_metrics:
                if evaluation_metric == "VPQ":
                    with contextlib.redirect_stdout(io.StringIO()):
                        vpq_res = []
                        avg_res = {}
                        for nframe in self._nframes:
                            print_log(f"Evaluating VPQ@{nframe}...", logger="current")
                            res = vpq_compute(
                                gt_json,
                                osp.realpath(pred_json),
                                gt_folder=gt_panseg_folder,
                                pred_folder=pred_dir,
                                nframe=nframe,
                            )
                            vpq_res.append(res)
                            table += f"VPQ@{nframe}:\n{print_panoptic_results(res)}\n"

                        for group in ["All", "Things", "Stuff"]:
                            avg_res[group] = {
                                "pq": float(np.mean([r[group]["pq"] for r in vpq_res])),
                                "sq": float(np.mean([r[group]["sq"] for r in vpq_res])),
                                "rq": float(np.mean([r[group]["rq"] for r in vpq_res])),
                                "n": int(vpq_res[0][group].get("n", 0)),
                            }
                        table += (
                            f"VPQ@avg({','.join(map(str, self._nframes))}):\n" f"{print_panoptic_results(avg_res)}\n"
                        )

                elif evaluation_metric == "STQ":
                    with contextlib.redirect_stdout(io.StringIO()):
                        print_log("Evaluating STQ...", logger="current")
                        stq_res = stq_compute(
                            gt_json,
                            osp.realpath(pred_json),
                            gt_folder=gt_panseg_folder,
                            pred_folder=pred_dir,
                            ignore_value=self._metadata.ignore_value,
                        )
                        table += f"STQ:\n{print_stq_results(stq_res)}\n"
                else:
                    raise ValueError(f"Unknown evaluation metric: {evaluation_metric}")

        return table

    def process(self, inputs, outputs):
        if "panoptic" in self.data_name:
            self.panoptic_process(inputs, outputs)
        elif "semantic" in self.data_name:
            self.semantic_process(inputs, outputs)
        elif "instance" in self.data_name:
            self.instance_process(inputs, outputs)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

    def evaluate(self):
        if self._distributed:
            comm.synchronize()

            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return
        else:
            predictions = self._predictions

        if not predictions and osp.exists(self._output_dir) and self.support_loading:
            file_path = osp.join(self._output_dir, "predictions.json")
            print_log(f"Loading predictions from {file_path}...", logger="current")
            with open(file_path, "r") as f:
                predictions = json.load(f)
        elif self._output_dir:
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                file_path = osp.join(self._output_dir, "predictions.json")
                print_log(f"Saving {self.data_name} predictions to {self._output_dir}...", logger="current")
                with open(file_path, "w") as f:
                    json.dump(predictions, f)
            except Exception as e:
                print_log(f"Error saving {self.data_name} predictions to {self._output_dir}: {e}", logger="current")

        predictions = [
            pred for pred in predictions if isinstance(pred, dict) and "video_id" in pred and "chunk_id" in pred
        ]

        if not predictions:
            print_log(
                f"{self.__class__.__name__} did not receive valid predictions.",
                logger="current",
                level=logging.WARNING,
            )
            return {}

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")

        if "panoptic" in self.data_name:
            table = self.panoptic_evaluate(predictions)
        elif "semantic" in self.data_name:
            table = self.semantic_evaluate(predictions)
        elif "instance" in self.data_name:
            table = self.instance_evaluate(predictions)
        else:
            raise ValueError(f"Unknown dataset name: {self.data_name}")

        print_log(f"{self.data_name} evaluation results:\n{table}", logger="current")
