import multiprocessing

import numpy as np


def mvc_compute_single_core(video_gt_masks, video_pred_masks, n_frames):
    mvc_accs = {n_frame: [] for n_frame in n_frames}
    for gt_masks, pred_masks in zip(video_gt_masks, video_pred_masks):
        for n_frame in n_frames:
            for i in range(len(gt_masks) - n_frame):
                height, width = gt_masks[i].shape
                gt_global_common = np.ones((height, width), dtype=np.uint8)
                pred_global_common = np.ones((height, width), dtype=np.uint8)
                gt_mask_i = gt_masks[i]
                pred_mask_i = pred_masks[i]
                for j in range(1, n_frame):
                    gt_mask_j = gt_masks[i + j]
                    pred_mask_j = pred_masks[i + j]
                    gt_common = gt_mask_i == gt_mask_j
                    pred_common = pred_mask_i == pred_mask_j
                    gt_global_common = np.logical_and(gt_global_common, gt_common)
                    pred_global_common = np.logical_and(pred_global_common, pred_common)
                global_common = gt_global_common * pred_global_common
                mvc_accs[n_frame].append(global_common.sum() / (gt_global_common.sum() + 1e-6))

    return mvc_accs


def mvc_compute_multi_core(mvc_accs, video_gt_masks, video_pred_masks, nframes=[8, 16]):
    cpu_num = multiprocessing.cpu_count()
    num_videos = len(video_gt_masks)
    chunk_size = (num_videos + cpu_num - 1) // cpu_num  # Ceiling division

    video_gt_masks_splits = []
    video_pred_masks_splits = []
    for i in range(0, num_videos, chunk_size):
        end_idx = min(i + chunk_size, num_videos)
        video_gt_masks_splits.append(video_gt_masks[i:end_idx])
        video_pred_masks_splits.append(video_pred_masks[i:end_idx])
    workers = multiprocessing.Pool(cpu_num)
    processes = []
    for video_gt_masks_split, video_pred_masks_split in zip(video_gt_masks_splits, video_pred_masks_splits):
        p = workers.apply_async(mvc_compute_single_core, (video_gt_masks_split, video_pred_masks_split, nframes))
        processes.append(p)

    workers.close()
    workers.join()
    for p in processes:
        p_mvc_accs = p.get()
        for n_frame in nframes:
            mvc_accs[n_frame].extend(p_mvc_accs[n_frame])


def mvc_compute(video_gt_masks, video_pred_masks, nframes=[8, 16]):
    mvc_accs = {n_frame: [] for n_frame in nframes}
    mvc_compute_multi_core(mvc_accs, video_gt_masks, video_pred_masks, nframes)
    return mvc_accs
