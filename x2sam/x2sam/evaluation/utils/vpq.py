import argparse
import copy
import json
import multiprocessing
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import PIL.Image as Image
from panopticapi.utils import get_traceback, rgb2id
from tabulate import tabulate

OFFSET = 256 * 256 * 256
VOID = 0


class VPQStatCat:
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class VPQStat:
    def __init__(self):
        self.pq_per_cat = defaultdict(VPQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def vpq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info["isthing"] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0, "iou": 0.0, "tp": 0, "fp": 0, "fn": 0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {
                "pq": pq_class,
                "sq": sq_class,
                "rq": rq_class,
                "iou": iou,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            pq += pq_class
            sq += sq_class
            rq += rq_class
        return {"pq": pq / n, "sq": sq / n, "rq": rq / n, "n": n}, per_class_results


@get_traceback
def vpq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories, nframe):
    vpq_stat = VPQStat()

    idx = 0

    for cur_annotation in annotation_set:
        video_id, (gt_anns, pred_anns) = next(iter(cur_annotation.items()))
        if idx % 10 == 0:
            print("Core: {}, {} from {} images processed".format(proc_id, idx, len(annotation_set)))
        idx += 1

        for i in range(0, len(gt_anns) - nframe + 1):
            vid_pan_gt, vid_pan_pred = [], []
            gt_segms_list, pred_segms_list = [], []

            for gt_ann, pred_ann in zip(gt_anns[i : i + nframe], pred_anns[i : i + nframe]):
                pan_gt = np.array(
                    Image.open(osp.join(gt_folder, video_id, gt_ann["file_name"])).convert("RGB"),
                    dtype=np.uint32,
                )
                pan_gt = rgb2id(pan_gt)
                pan_pred = np.array(
                    Image.open(osp.join(pred_folder, pred_ann["file_name"])).convert("RGB"), dtype=np.uint32
                )
                pan_pred = rgb2id(pan_pred)
                gt_segms = {}
                pred_segms = {}

                for el in gt_ann["segments_info"]:
                    if el["id"] in gt_segms:
                        gt_segms[el["id"]]["area"] += el["area"]
                    else:
                        gt_segms[el["id"]] = copy.deepcopy(el)
                for el in pred_ann["segments_info"]:
                    if el["id"] in pred_segms:
                        pred_segms[el["id"]]["area"] += el["area"]
                    else:
                        pred_segms[el["id"]] = copy.deepcopy(el)

                # predicted segments area calculation + prediction sanity checks
                pred_labels_set = set(el["id"] for el in pred_ann["segments_info"])
                labels, labels_cnt = np.unique(pan_pred, return_counts=True)
                for label, label_cnt in zip(labels, labels_cnt):
                    if label not in pred_segms:
                        if label == VOID:
                            continue
                        raise KeyError(
                            "Segment with ID {} is presented in PNG and not presented in JSON.".format(label)
                        )
                    pred_segms[label]["area"] = label_cnt
                    pred_labels_set.remove(label)
                    if pred_segms[label]["category_id"] not in categories:
                        raise KeyError(
                            "Segment with ID {} has unknown category_id {}.".format(
                                label, pred_segms[label]["category_id"]
                            )
                        )
                if len(pred_labels_set) != 0:
                    raise KeyError(
                        "The following segment IDs {} are presented in JSON and not presented in PNG.".format(
                            list(pred_labels_set)
                        )
                    )

                vid_pan_gt.append(pan_gt)
                vid_pan_pred.append(pan_pred)
                gt_segms_list.append(gt_segms)
                pred_segms_list.append(pred_segms)

            #### Step 2. Concatenate the collected items -> tube-level.
            vid_pan_gt = np.stack(vid_pan_gt)  # [nf,H,W]
            vid_pan_pred = np.stack(vid_pan_pred)  # [nf,H,W]
            vid_gt_segms, vid_pred_segms = {}, {}
            for gt_segms, pred_segms in zip(gt_segms_list, pred_segms_list):
                # aggregate into tube 'area'
                for k in gt_segms.keys():
                    if k not in vid_gt_segms:
                        vid_gt_segms[k] = gt_segms[k]
                    else:
                        vid_gt_segms[k]["area"] += gt_segms[k]["area"]
                for k in pred_segms.keys():
                    if k not in vid_pred_segms:
                        vid_pred_segms[k] = pred_segms[k]
                    else:
                        vid_pred_segms[k]["area"] += pred_segms[k]["area"]

            #### Step3. Confusion matrix calculation
            vid_pan_gt_pred = vid_pan_gt.astype(np.uint64) * OFFSET + vid_pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(vid_pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            #### Step4. Tube matching
            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple

                if gt_label not in vid_gt_segms:
                    continue
                if pred_label not in vid_pred_segms:
                    continue
                if vid_gt_segms[gt_label]["iscrowd"] == 1:
                    continue
                if vid_gt_segms[gt_label]["category_id"] != vid_pred_segms[pred_label]["category_id"]:
                    continue

                union = (
                    vid_pred_segms[pred_label]["area"]
                    + vid_gt_segms[gt_label]["area"]
                    - intersection
                    - gt_pred_map.get((VOID, pred_label), 0)
                )
                iou = intersection / union
                assert iou <= 1.0, "INVALID IOU VALUE : %d" % (gt_label)
                # count true positives
                if iou > 0.5:
                    vpq_stat[vid_gt_segms[gt_label]["category_id"]].tp += 1
                    vpq_stat[vid_gt_segms[gt_label]["category_id"]].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false negatives
            crowd_labels_dict = {}
            for gt_label, gt_info in vid_gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info["iscrowd"] == 1:
                    crowd_labels_dict[gt_info["category_id"]] = gt_label
                    continue
                vpq_stat[gt_info["category_id"]].fn += 1
                fn += 1

            # count false positives
            for pred_label, pred_info in vid_pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info["category_id"] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info["category_id"]], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info["area"] > 0.5:
                    continue
                vpq_stat[pred_info["category_id"]].fp += 1
                fp += 1

    print("Core: {}, all {} videos processed".format(proc_id, len(annotation_set)))
    return vpq_stat


def vpq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, nframe):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, videos per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(
            vpq_compute_single_core, (proc_id, annotation_set, gt_folder, pred_folder, categories, nframe)
        )
        processes.append(p)

    workers.close()
    workers.join()
    vpq_stat = VPQStat()
    for p in processes:
        vpq_stat += p.get()
    return vpq_stat


def vpq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None, nframe=2):
    start_time = time.time()
    with open(gt_json_file, "r") as f:
        gt_json = json.load(f)
    with open(pred_json_file, "r") as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace(".json", "")
    if pred_folder is None:
        pred_folder = pred_json_file.replace(".json", "")
    categories = {el["id"]: el for el in gt_json["categories"]}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not osp.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not osp.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el["video_id"]: el["annotations"] for el in pred_json["annotations"]}
    matched_annotations_list = []
    for _gt_ann in gt_json["annotations"]:
        video_id = _gt_ann["video_id"]
        if video_id not in pred_annotations:
            print(f"no prediction for the video with id: {video_id}")
            continue

        # update the gt_ann with the video_id according to the image_id in the gt_json
        # ensure explicit alignment by image_id to avoid frame mismatch
        pred_ann = pred_annotations[video_id]
        pred_ann_by_id = {x["image_id"]: x for x in pred_ann}
        gt_ann_by_id = {x["image_id"]: x for x in _gt_ann["annotations"]}
        # find matched image_ids and sort them to ensure temporal order
        matched_img_ids = sorted(set(pred_ann_by_id.keys()) & set(gt_ann_by_id.keys()))
        # align gt and pred annotations in sorted temporal order
        aligned_gt_ann = [gt_ann_by_id[img_id] for img_id in matched_img_ids]
        aligned_pred_ann = [pred_ann_by_id[img_id] for img_id in matched_img_ids]
        matched_annotations_list.append({video_id: (aligned_gt_ann, aligned_pred_ann)})

    print("Evaluation video panoptic segmentation metrics:")
    print("Number of videos: {}".format(len(matched_annotations_list)))
    print("Number of frames per tube: {}".format(nframe))

    vpq_stat = vpq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, nframe)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = vpq_stat.vpq_average(categories, isthing=isthing)
        if name == "All":
            results["per_class"] = per_class_results

    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print(
            "{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
                name,
                100 * results[name]["pq"],
                100 * results[name]["sq"],
                100 * results[name]["rq"],
                results[name]["n"],
            )
        )

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


def print_panoptic_results(pq_res):
    headers = ["Metric", "PQ", "SQ", "RQ", "# Categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
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
    parser = argparse.ArgumentParser(description="Video Panoptic Quality evaluation")
    parser.add_argument("--submit_dir", "-i", type=str, help="Test output directory", required=True)
    parser.add_argument(
        "--truth_dir",
        type=str,
        help="Ground truth directory. Point this to <BASE_DIR>/VIPSeg/vipseg_720p/panomasksRGB "
        "after running the conversion script",
        required=True,
    )
    parser.add_argument(
        "--pan_gt_json_file",
        type=str,
        help="Ground truth JSON file. Point this to <BASE_DIR>/VIPSeg/vipseg_720p/panoptic_gt_"
        "VIPSeg_val.json after running the conversion script",
        required=True,
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
        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json",
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        default=None,
        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json",
    )
    parser.add_argument("-nframes", type=list, default=[1], help="Numbers of evaluation frame per tube.")
    args = parser.parse_args()
    for nframe in args.nframes:
        table = vpq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder, nframe)
        print(print_panoptic_results(table))


if __name__ == "__main__":
    main()
