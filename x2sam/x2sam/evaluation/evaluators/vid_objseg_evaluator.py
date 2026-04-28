import itertools
import json
import logging
import os
import os.path as osp
from collections import defaultdict
from typing import List

import numpy as np

from x2sam.dataset.utils.mask import encode_mask
from x2sam.utils import comm
from x2sam.utils.logging import print_log

from ..utils.jf import JFStat, jf_compute, print_jf_results
from .vid_base_evaluator import VidBaseEvaluator


class VidObjSegEvaluator(VidBaseEvaluator):
    def __init__(
        self,
        *args,
        data_name: str = "vid_objseg",
        evaluation_metrics: List[str] = ["J", "F", "J&F"],
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            evaluation_metrics,
            **kwargs,
        )
        self.j_f_stat = JFStat()

    def process(self, inputs, outputs):
        for video_input, video_output in zip(inputs, outputs):
            video_id = video_input["video_id"]
            sample_id = video_input["sample_id"]
            chunk_id = video_input["chunk_id"]
            video_preds = []
            for file_name, image_output in zip(video_input["file_names"], video_output):
                pred_mask, segments_info = (
                    image_output["segmentation"],
                    image_output["segments_info"],
                )
                pred_mask = pred_mask.cpu().numpy()
                pred_mask[pred_mask == self._metadata.ignore_value] = 0
                pred_mask = pred_mask.astype(np.uint8)
                video_preds.append(
                    [
                        {
                            "file_name": osp.basename(file_name),
                            "segmentation": encode_mask(pred_mask),
                            "segments_info": segments_info,
                        }
                    ]
                )
            self._predictions.append(
                {"video_id": video_id, "sample_id": sample_id, "chunk_id": chunk_id, "annotations": video_preds}
            )

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
            pred
            for pred in predictions
            if isinstance(pred, dict) and "video_id" in pred and "sample_id" in pred and "chunk_id" in pred
        ]
        if not predictions:
            print_log(
                f"{self.__class__.__name__} did not receive valid predictions.",
                logger="current",
                level=logging.WARNING,
            )
            return {}

        print_log(f"Evaluating {self.data_name} with {len(predictions)} predictions...", logger="current")

        gt_json = osp.realpath(self._metadata.gt_json)
        self._eval_predictions(predictions, gt_json)

    def _eval_predictions(self, predictions, gt_json):
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_anns = json.load(f)

        # merge the annotations of the same video with the same sample_id
        merged_annotations = defaultdict(list)
        for gt_ann in gt_anns:
            video_id = gt_ann["video_id"]
            sample_id = gt_ann["video_info"]["sample_id"]
            chunk_id = gt_ann["video_info"]["chunk_id"]
            anns = gt_ann["annotations"]
            merged_annotations[f"{video_id}{sample_id}"].extend((chunk_id, anns))
        # chunks may be out of order (e.g. after distributed gather), so sort
        # the (chunk_id, anns) pairs by chunk_id before flattening per-frame anns
        annotations = {
            video_id: list(
                itertools.chain(*(a for _, a in sorted(zip(anns[::2], anns[1::2]), key=lambda x: x[0])))
            )
            for video_id, anns in merged_annotations.items()
        }

        # merge the predictions of the same video with the same sample_id
        merged_predictions = defaultdict(list)
        for prediction in predictions:
            video_id = prediction["video_id"]
            sample_id = prediction["sample_id"]
            chunk_id = prediction["chunk_id"]
            preds = prediction["annotations"]
            merged_predictions[f"{video_id}{sample_id}"].extend((chunk_id, preds))
        # video_id is actually the video_id and sample_id concatenated
        predictions = {
            video_id: list(itertools.chain(*(p for _, p in sorted(zip(preds[::2], preds[1::2]), key=lambda x: x[0]))))
            for video_id, preds in merged_predictions.items()
        }
        jf_res = jf_compute(annotations, predictions)
        table = print_jf_results(jf_res)

        print_log(f"{self.data_name} evaluation results:\n{table}", logger="current")
