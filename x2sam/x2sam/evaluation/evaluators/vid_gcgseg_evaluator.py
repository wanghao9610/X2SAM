import copy
import itertools
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import traceback
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from x2sam.dataset.utils.coco import COCO
from x2sam.dataset.utils.mask import decode_mask, encode_mask
from x2sam.utils import comm
from x2sam.utils.logging import print_log

from ..utils.cococapeval import COCOEvalCap
from ..utils.map import create_small_table, derive_coco_results
from ..utils.miou import compute_iou_matrix, compute_miou
from .vid_base_evaluator import VidBaseEvaluator


class VidGCGSegEvaluator(VidBaseEvaluator):

    def __init__(
        self,
        *args,
        data_name: str = "vid_gcgseg",
        text_model: str = "bert-base-uncased",
        evaluation_metrics: list[str] = ["miou", "map", "caption", "recall"],
        **kwargs,
    ):
        super().__init__(
            *args,
            data_name,
            evaluation_metrics,
            **kwargs,
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model)
        self.text_model.eval()

    def _encode_text_embeddings(self, text):
        input_dict = self.text_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        text_embeddings = self.text_model(**input_dict)
        text_embeddings = torch.mean(text_embeddings.last_hidden_state[0], dim=0).detach().numpy()
        return text_embeddings

    def _get_text_similarity(self, text1, text2):
        text1_embeddings = self._encode_text_embeddings(text1)
        text2_embeddings = self._encode_text_embeddings(text2)
        return cosine_similarity([text1_embeddings], [text2_embeddings])[0, 0]

    def _get_rle_masks(self, segmentation, segments_info):
        rle_masks = []
        for seg_info in segments_info:
            binary_mask = (segmentation == seg_info["id"]).astype(np.uint8)
            rle_masks.append(encode_mask(binary_mask))

        return rle_masks

    # follow segmentation evaluation
    def process(self, inputs, outputs):
        for video_input, video_output in zip(inputs, outputs):
            video_id = video_input["video_id"]
            chunk_id = video_input["chunk_id"]
            sample_id = video_input["sample_id"]
            video_preds = []
            for image_input, image_output in zip(video_input["images"], video_output):
                segmentation, segments_info, gcg_phrases, gcg_caption = (
                    image_output["segmentation"],
                    image_output["segments_info"],
                    image_output["gcg_phrases"],
                    image_output["gcg_caption"],
                )
                segmentation = segmentation.to(self._cpu_device)
                segmentation = np.array(segmentation, dtype=np.uint8)
                segmentation = self._get_rle_masks(segmentation, segments_info)
                file_name = os.path.basename(image_input["file_name"])
                annotations = [
                    {
                        "image_id": image_input["id"],
                        "segmentation": seg,
                        "category_id": seg_info["category_id"],
                        "score": seg_info["score"],
                    }
                    for seg, seg_info in zip(segmentation, segments_info)
                ]
                video_preds.append(
                    {
                        "image_id": image_input["id"],
                        "file_name": file_name,
                        "annotations": annotations,
                        "gcg_phrases": gcg_phrases,
                        "gcg_caption": gcg_caption,
                    }
                )
            self._predictions.append(
                {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "sample_id": sample_id,
                    "annotations": video_preds,
                }
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
            if isinstance(pred, dict) and "video_id" in pred and "chunk_id" in pred and "sample_id" in pred
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
        self._evaluate_predictions(predictions, gt_json)

    def _evaluate_predictions(self, predictions, gt_json):
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_video_anns = json.load(f)

        merged_annotations = defaultdict(list)
        merged_gt_captions = defaultdict(list)
        merged_gt_labels = defaultdict(list)
        for gt_ann in gt_video_anns:
            video_id = gt_ann["video_id"]
            chunk_id = gt_ann["video_info"]["chunk_id"]
            sample_id = gt_ann["video_info"]["sample_id"]
            caption = gt_ann["video_info"]["caption"]
            labels = gt_ann["video_info"]["labels"]
            anns = gt_ann["annotations"]
            merged_annotations[f"{video_id}{sample_id}"].extend((chunk_id, anns))
            merged_gt_captions[f"{video_id}{sample_id}"].extend([caption])
            merged_gt_labels[f"{video_id}{sample_id}"].extend([labels])
        annotations = {
            video_id: list(
                itertools.chain(*(a for _, a in sorted(zip(anns[::2], anns[1::2]), key=lambda x: x[0])))
            )
            for video_id, anns in merged_annotations.items()
        }
        gt_captions = {video_id: list(set(captions)) for video_id, captions in sorted(merged_gt_captions.items())}
        gt_labels = {
            video_id: [list(x) for x in {tuple(sorted(label)) for label in labels}]
            for video_id, labels in sorted(merged_gt_labels.items())
        }

        # merge the predictions of the same video
        merged_predictions = defaultdict(list)
        merged_pred_captions = defaultdict(list)
        merged_pred_labels = defaultdict(list)
        for prediction in predictions:
            video_id = prediction["video_id"]
            chunk_id = prediction["chunk_id"]
            sample_id = prediction["sample_id"]
            preds = prediction["annotations"]
            captions = [pred["gcg_caption"] for pred in preds]
            labels = [pred["gcg_phrases"] for pred in preds]
            merged_predictions[f"{video_id}{sample_id}"].extend((chunk_id, preds))
            merged_pred_captions[f"{video_id}{sample_id}"].extend(captions)
            merged_pred_labels[f"{video_id}{sample_id}"].extend(labels)
        predictions = {
            video_id: list(itertools.chain(*(p for _, p in sorted(zip(preds[::2], preds[1::2]), key=lambda x: x[0]))))
            for video_id, preds in merged_predictions.items()
        }
        pred_captions = {video_id: list(set(captions)) for video_id, captions in sorted(merged_pred_captions.items())}
        pred_labels = {
            video_id: list(dict((tuple(sorted(label)), label) for label in labels).values())
            for video_id, labels in sorted(merged_pred_labels.items())
        }

        for metric in self._evaluation_metrics:
            if metric == "miou":
                self._eval_miou(copy.deepcopy(annotations), copy.deepcopy(predictions))
            elif metric == "map":
                self._eval_map(copy.deepcopy(annotations), copy.deepcopy(predictions))
            elif metric == "caption":
                self._eval_caption(copy.deepcopy(gt_captions), copy.deepcopy(pred_captions))
            elif metric == "recall":
                self._eval_recall(
                    copy.deepcopy(annotations),
                    copy.deepcopy(predictions),
                    copy.deepcopy(gt_labels),
                    copy.deepcopy(pred_labels),
                )
            else:
                raise ValueError(f"Metric {metric} not supported")

    @staticmethod
    def _merge_segmentation(segmentation):
        segmentation = mask_utils.merge(
            [
                mask_utils.frPyObjects(seg, *seg["size"]) if isinstance(seg["counts"], list) else seg
                for seg in segmentation
            ]
        )
        return segmentation

    def _eval_miou(self, annotations, predictions):
        mious = []
        # image level evaluation
        for video_id, video_pred in predictions.items():
            video_ann = annotations[video_id]
            for image_preds, image_anns in zip(video_pred, video_ann):
                gt_masks = [
                    decode_mask(
                        (
                            self._merge_segmentation(ann["segmentation"])
                            if isinstance(ann["segmentation"], list)
                            else ann["segmentation"]
                        ),
                        *(
                            ann["segmentation"][0]["size"]
                            if isinstance(ann["segmentation"], list)
                            else ann["segmentation"]["size"]
                        ),
                    )
                    for ann in image_anns
                ]
                pred_masks = [
                    decode_mask(
                        (
                            self._merge_segmentation(ann["segmentation"])
                            if isinstance(ann["segmentation"], list)
                            else ann["segmentation"]
                        ),
                        *(
                            ann["segmentation"][0]["size"]
                            if isinstance(ann["segmentation"], list)
                            else ann["segmentation"]["size"]
                        ),
                    )
                    for ann in image_preds["annotations"]
                ]
                iou = compute_miou(gt_masks, pred_masks)
                mious.append(iou)

        miou_res = float(np.mean(mious) * 100) if mious else 0.0
        data = [["Value (%)", f"{miou_res:.2f}"]]
        headers = ["Metric", "mIoU"]
        table = tabulate(
            data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center"
        )
        print_log(f"{self.data_name} mIoU results:\n{table}", logger="current")

    def _eval_map(self, annotations, predictions):
        gt_data = {
            "info": {"description": "GCGSeg dataset"},
            "categories": [{"id": 1, "name": "object"}],
            "images": [],
            "annotations": [],
        }
        pred_data = []

        # image level evaluation
        for video_id, video_pred in predictions.items():
            video_ann = annotations[video_id]
            for image_preds, image_anns in zip(video_pred, video_ann):
                for image_pred, image_ann in zip(image_preds["annotations"], image_anns):
                    gt_data["images"].append(
                        {
                            "id": image_pred["image_id"],
                            "height": (
                                image_pred["segmentation"][0]["size"][0]
                                if isinstance(image_pred["segmentation"], list)
                                else image_pred["segmentation"]["size"][0]
                            ),
                            "width": (
                                image_pred["segmentation"][0]["size"][1]
                                if isinstance(image_pred["segmentation"], list)
                                else image_pred["segmentation"]["size"][1]
                            ),
                        }
                    )
                    gt_data["annotations"].append(
                        {
                            "id": image_ann["id"],
                            "image_id": image_ann["image_id"],
                            "iscrowd": 0,
                            "area": mask_utils.area(
                                self._merge_segmentation(image_ann["segmentation"])
                                if isinstance(image_ann["segmentation"], list)
                                else image_ann["segmentation"]
                            ),
                            "category_id": 1,  # fake category id, represents the foreground object
                            "segmentation": (
                                self._merge_segmentation(image_ann["segmentation"])
                                if isinstance(image_ann["segmentation"], list)
                                else image_ann["segmentation"]
                            ),
                        }
                    )
                    pred_data.append(
                        {
                            "image_id": image_pred["image_id"],
                            "segmentation": (
                                self._merge_segmentation(image_pred["segmentation"])
                                if isinstance(image_pred["segmentation"], list)
                                else image_pred["segmentation"]
                            ),
                            "category_id": image_pred["category_id"],
                            "score": image_pred["score"],
                        }
                    )

        coco_gt = COCO(dataset=gt_data)
        coco_dt = coco_gt.loadRes(pred_data)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        imgids = sorted(list(set([pred["image_id"] for pred in pred_data])))
        coco_eval.params.imgIds = imgids
        coco_eval.params.catIds = [1]
        print_log(f"{self.data_name} mAP results:", logger="current")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = derive_coco_results(coco_eval, "segm")
        table = create_small_table(results)
        print_log(f"{self.data_name} mAP results:\n{table}", logger="current")

    def _eval_caption(self, gt_captions, pred_captions):
        gt_data = {"info": {"description": "GCGSeg dataset"}, "images": [], "annotations": []}
        pred_data = []
        for i, (video_id, pred_caption) in enumerate(pred_captions.items()):
            gt_caption = gt_captions[video_id]
            assert (
                len(gt_caption) == 1
            ), f"Each video should have exactly one ground truth caption, gt_caption: {gt_caption}"
            cap_sims = [self._get_text_similarity(gt_caption[0], _pred_caption) for _pred_caption in pred_caption]
            best_match_idx = np.argmax(cap_sims)

            gt_data["images"].append({"id": video_id})
            gt_data["annotations"].append({"image_id": video_id, "id": i, "caption": gt_caption[0]})
            pred_data.append({"image_id": video_id, "id": i, "caption": pred_caption[best_match_idx]})

        coco_gt = COCO(dataset=gt_data)
        coco_dt = coco_gt.loadRes(pred_data)
        coco_eval = COCOEvalCap(coco_gt, coco_dt)
        imgids = sorted(list(set([pred["image_id"] for pred in pred_data])))
        coco_eval.params["image_id"] = imgids
        coco_eval.evaluate()

        metrics = list(coco_eval.eval.keys())
        data = [["Value (%)", *[f"{coco_eval.eval[metric] * 100:.2f}" for metric in metrics]]]
        headers = ["Metric", *metrics]
        table = tabulate(
            data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center"
        )

        print_log(f"{self.data_name} caption results:\n{table}", logger="current")

    def _eval_recall(self, annotations, predictions, gt_labels, pred_labels):
        cap_gt_data = {"info": {"description": "GCGSeg dataset"}, "images": [], "annotations": []}
        cap_pred_data = []
        for i, (video_id, pred_label) in enumerate(pred_labels.items()):
            gt_label = gt_labels[video_id]
            assert (
                len(gt_label) == 1
            ), f"Each video should have exactly one set of ground truth labels, gt_label: {gt_label}"
            label_sims = [
                self._get_text_similarity(" ".join(sorted(gt_label[0])), " ".join(sorted(_pred_label)))
                for _pred_label in pred_label
            ]
            best_match_idx = np.argmax(label_sims)
            gt_labels[video_id] = gt_label[0]
            pred_labels[video_id] = pred_label[best_match_idx]
            cap_gt_data["images"].append({"id": video_id})
            cap_gt_data["annotations"].append({"image_id": video_id, "id": i, "labels": gt_label[0]})
            cap_pred_data.append({"image_id": video_id, "id": i, "labels": pred_label[best_match_idx]})

        gt_data = {
            "info": {"description": "GCGSeg dataset"},
            "categories": [{"id": 1, "name": "object"}],
            "images": [],
            "annotations": [],
        }
        pred_data = []
        prd_id = 0
        ann_id = 0
        # video level evaluation
        for video_id, video_pred in predictions.items():
            video_ann = annotations[video_id]
            # TODO: fix this
            if len(video_ann) != len(video_pred):
                print_log(
                    f"len(video_ann) != len(video_pred) in video {video_id}, len(video_ann): {len(video_ann)}, len(video_pred): {len(video_pred)}",
                    logger="current",
                )
                continue
            gt_label = gt_labels[video_id]
            pred_label = pred_labels[video_id]
            first_valid_ann = next(ann[0] for ann in video_ann if ann)
            video_height = (
                first_valid_ann["segmentation"][0]["size"][0]
                if isinstance(first_valid_ann["segmentation"], list)
                else first_valid_ann["segmentation"]["size"][0]
            )
            video_width = (
                first_valid_ann["segmentation"][0]["size"][1]
                if isinstance(first_valid_ann["segmentation"], list)
                else first_valid_ann["segmentation"]["size"][1]
            )
            gt_data["images"].append(
                {
                    "id": video_id,
                    "height": video_height,
                    "width": video_width,
                }
            )
            # each label has only one segmentation
            for gt_cat_id in range(len(gt_label)):
                segmentation = [
                    (
                        self._merge_segmentation(
                            [
                                (
                                    (
                                        self._merge_segmentation(image_ann["segmentation"])
                                        if isinstance(image_ann["segmentation"], list)
                                        else image_ann["segmentation"]
                                    )
                                    if image_ann["category_id"] == gt_cat_id
                                    else encode_mask(np.zeros((video_height, video_width), dtype=np.uint8))
                                )
                                for image_ann in image_anns
                            ]
                        )
                        if image_anns
                        else encode_mask(np.zeros((video_height, video_width), dtype=np.uint8))
                    )
                    for image_anns in video_ann
                ]
                gt_data["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": video_id,  # video_id as the image_id
                        "category_id": gt_cat_id,
                        "segmentation": segmentation,
                    }
                )
                ann_id += 1

            for pred_cat_id in range(len(pred_label)):
                segmentation = [
                    (
                        self._merge_segmentation(
                            [
                                (
                                    (
                                        self._merge_segmentation(image_pred["segmentation"])
                                        if isinstance(image_pred["segmentation"], list)
                                        else image_pred["segmentation"]
                                    )
                                    if image_pred["category_id"] == pred_cat_id
                                    else encode_mask(np.zeros((video_height, video_width), dtype=np.uint8))
                                )
                                for image_pred in image_preds["annotations"]
                            ]
                        )
                        if image_preds["annotations"]
                        else encode_mask(np.zeros((video_height, video_width), dtype=np.uint8))
                    )
                    for image_preds in video_pred
                ]
                pred_data.append(
                    {
                        "id": prd_id,
                        "image_id": video_id,  # video_id as the image_id
                        "category_id": pred_cat_id,
                        "segmentation": segmentation,
                    }
                )
                prd_id += 1

        coco_gt = COCO(dataset=gt_data)
        cap_coco_gt = COCO(dataset=cap_gt_data)
        coco_dt = coco_gt.loadRes(pred_data)
        cap_coco_dt = cap_coco_gt.loadRes(cap_pred_data)
        imgids = sorted(list(set([pred["image_id"] for pred in pred_data])))

        # Pre-compute text similarities for all images to avoid PyTorch model in workers
        text_sims = {}
        for imgid in imgids:
            cap_gt_ann_ids = cap_coco_gt.getAnnIds(imgIds=[imgid])
            cap_gt_anns = cap_coco_gt.loadAnns(cap_gt_ann_ids)[0]
            cap_dt_ann_ids = cap_coco_dt.getAnnIds(imgIds=[imgid])
            cap_dt_anns = cap_coco_dt.loadAnns(cap_dt_ann_ids)[0]

            cap_gt_labels = cap_gt_anns["labels"]
            cap_dt_labels = cap_dt_anns["labels"]

            text_sim = np.zeros((len(cap_gt_labels), len(cap_dt_labels)))
            for i, gt_label in enumerate(cap_gt_labels):
                for j, dt_label in enumerate(cap_dt_labels):
                    text_sim[i, j] = self._get_text_similarity(gt_label, dt_label)
            text_sims[imgid] = text_sim

        # Initialize multiprocessing pool
        num_workers = min(16, len(imgids))
        with mp.Pool(num_workers) as pool:
            process_func = partial(
                self._process_single_image,
                coco_gt=coco_gt,
                cap_coco_gt=cap_coco_gt,
                coco_dt=coco_dt,
                cap_coco_dt=cap_coco_dt,
                text_sims=text_sims,
            )
            results = list(
                tqdm(pool.imap_unordered(process_func, imgids), total=len(imgids), desc="Calculating recall")
            )

        # Sum up results
        p_cnt = sum(p for p, _ in results)
        tp_cnt = sum(tp for _, tp in results)

        recall = tp_cnt / p_cnt if p_cnt > 0 else 0
        headers = ["Metric", "Recall"]
        data = [["Value (%)", f"{recall * 100:.2f}"]]
        table = tabulate(
            data, headers=headers, tablefmt="outline", floatfmt=".2f", stralign="center", numalign="center"
        )

        print_log(f"{self.data_name} recall results:\n{table}", logger="current")

    def _process_single_image(self, imgid, coco_gt, cap_coco_gt, coco_dt, cap_coco_dt, text_sims=None):
        try:
            imginfo = coco_gt.loadImgs([imgid])[0]
            height, width = imginfo["height"], imginfo["width"]

            gt_ann_ids = coco_gt.getAnnIds(imgIds=[imgid])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)
            dt_ann_ids = coco_dt.getAnnIds(imgIds=[imgid])
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            cap_gt_ann_ids = cap_coco_gt.getAnnIds(imgIds=[imgid])
            cap_gt_anns = cap_coco_gt.loadAnns(cap_gt_ann_ids)[0]
            cap_dt_ann_ids = cap_coco_dt.getAnnIds(imgIds=[imgid])
            cap_dt_anns = cap_coco_dt.loadAnns(cap_dt_ann_ids)[0]

            cap_gt_labels = cap_gt_anns["labels"]
            cap_dt_labels = cap_dt_anns["labels"]

            text_sim = text_sims[imgid]
            best_matches = self._find_best_matches(
                gt_anns, cap_gt_labels, dt_anns, cap_dt_labels, height, width, text_sims=text_sim
            )
            return len(cap_gt_labels), len(best_matches)
        except Exception as e:
            print_log(f"Error processing image {imgid}: {e}\n{traceback.format_exc()}", logger="current")
            return 0, 0

    def _find_best_matches(
        self,
        gt_anns,
        gt_labels,
        dt_anns,
        dt_labels,
        height,
        width,
        text_sims=None,
        iou_threshold=0.5,
        text_sim_threshold=0.5,
    ):
        best_matches = []

        # Compute pair-wise IoU
        # Decode masks for each annotation, preserving frame dimension for frame-wise IoU computation
        gt_masks = [
            [decode_mask(segmentation, height, width) for segmentation in image_anns["segmentation"]]
            for image_anns in gt_anns
        ]
        pred_masks = [
            [decode_mask(segmentation, height, width) for segmentation in image_anns["segmentation"]]
            for image_anns in dt_anns
        ]

        # there are some bad cases that the len(gt_masks[0]) != len(pred_masks[0])
        # TODO: fix this
        try:
            ious = compute_iou_matrix(gt_masks, pred_masks)
        except Exception as e:
            print_log(
                f"Error computing IoU: gt_masks: {len(gt_masks), [len(mask) for mask in gt_masks]}, pred_masks: {len(pred_masks), [len(mask) for mask in pred_masks]}, {e}\n{traceback.format_exc()}",
                logger="current",
            )
            return []

        # Use pre-computed text similarities if provided, otherwise compute them
        if text_sims is None:
            text_sims = np.zeros((len(gt_labels), len(dt_labels)))
            for i, gt_label in enumerate(gt_labels):
                for j, dt_label in enumerate(dt_labels):
                    text_sims[i, j] = self._get_text_similarity(gt_label, dt_label)

        # Find one-to-one matches satisfying both IoU and text similarity thresholds
        while ious.size > 0:
            max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
            if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
                break  # No admissible pair found

            best_matches.append(max_iou_idx)

            # Remove selected annotations from consideration
            ious[max_iou_idx[0], :] = 0
            ious[:, max_iou_idx[1]] = 0
            text_sims[max_iou_idx[0], :] = 0
            text_sims[:, max_iou_idx[1]] = 0

        return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]
