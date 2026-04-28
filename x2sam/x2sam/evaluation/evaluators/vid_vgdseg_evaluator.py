import itertools
import json
import logging
import os
import os.path as osp
from collections import defaultdict
from typing import List

from x2sam.dataset.utils.coco import COCO
from x2sam.utils import comm
from x2sam.utils.logging import print_log

from ..utils.map import convert_to_coco_json, derive_coco_results, evaluate_predictions_on_coco, instances_to_coco_json
from .vid_base_evaluator import VidBaseEvaluator


class VidVGDSegEvaluator(VidBaseEvaluator):
    def __init__(
        self,
        *args,
        data_name: str = "vid_vgdseg",
        reverse_cat_ids: bool = True,
        show_categories: bool = False,
        evaluation_metrics: List[str] = ["mAP"],
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            evaluation_metrics,
            **kwargs,
        )
        self._reverse_cat_ids = reverse_cat_ids
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

    def process(self, inputs, outputs):
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

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")

        self._evaluate_predictions(predictions)

    def _evaluate_predictions(self, predictions):
        # merge the predictions of the same video
        merged_predictions = defaultdict(list)
        for prediction in predictions:
            video_id = prediction["video_id"]
            anns = prediction["annotations"]
            chunk_id = prediction["chunk_id"]
            merged_predictions[video_id].extend((chunk_id, anns))
        predictions = {
            video_id: list(itertools.chain(*(p for _, p in sorted(zip(preds[::2], preds[1::2]), key=lambda x: x[0]))))
            for video_id, preds in merged_predictions.items()
        }

        with open(osp.realpath(self._metadata.gt_json), "r") as f:
            gt_video_anns = json.load(f)
        # map video_anns to image_anns
        gt_image_anns = []
        for gt_ann in gt_video_anns:
            sampled_labels = gt_ann["sampled_labels"]
            contiguous_labels = gt_ann["contiguous_labels"]
            for image_info, annotations in zip(gt_ann["video_info"]["images"], gt_ann["annotations"]):
                for annotation in annotations:
                    annotation["category_id"] = contiguous_labels.index(sampled_labels[annotation["category_id"]])
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

        if self._reverse_cat_ids:
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
        else:
            new_coco_results = coco_results

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

        print_log(f"{self.data_name} evaluation results:\n{table}", logger="current")
