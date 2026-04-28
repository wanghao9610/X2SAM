from collections import deque
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from x2sam.structures.masks import pairwise_mask_iou

from ...utils.process import cond_id_postprocess, sem_seg_postprocess


def vid_objseg_postprocess_fn(
    outputs,
    image_sizes,
    vprompt_masks: Optional[List[TensorType]] = None,
    cond_ids: Optional[List[TensorType]] = None,
    scaled_sizes: Optional[List[TensorType]] = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    use_bg_embeds: Optional[bool] = False,
    ignore_value: int = 255,
    memory_size: int = 4,
    alpha: float = 0.2,
    **kwargs,
) -> List[Dict]:

    # [batch_size, num_queries, num_classes]
    class_queries_logits = outputs.class_queries_logits
    # [batch_size, num_queries, height, width]
    masks_queries_logits = outputs.masks_queries_logits
    scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes
    if isinstance(image_sizes[0][0], tuple) or isinstance(image_sizes[0][0], list):
        image_sizes = [image_size for image_size in image_sizes[0]]
    if isinstance(scaled_sizes[0][0], tuple) or isinstance(scaled_sizes[0][0], list):
        scaled_sizes = [scaled_size for scaled_size in scaled_sizes[0]]
    if vprompt_masks is not None and isinstance(vprompt_masks[0], torch.Tensor):
        vprompt_masks = [vprompt_masks[0] for _ in range(len(image_sizes))]

    batch_size = class_queries_logits.shape[0]
    cond_ids = cond_ids.expand(batch_size, -1) if cond_ids is not None else None

    results = []
    memory = deque(maxlen=memory_size)

    for i in range(batch_size):
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]
        image_size = image_sizes[i]
        scaled_size = scaled_sizes[i]
        cond_id = cond_ids[i] if cond_ids is not None else None
        vprompt_mask = vprompt_masks[i] if vprompt_masks is not None else None

        mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])
        vprompt_mask = (
            sem_seg_postprocess(vprompt_mask, scaled_size, image_size[0], image_size[1], mode="nearest")
            if vprompt_masks is not None
            else None
        )
        pred_mask = (mask_pred > 0).float()
        anchor_mask = (vprompt_mask == 255).float().squeeze(0)

        if cond_id is not None and not use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            pred_scores = mask_cls.sigmoid() @ cond_id_map.T
            # NOTE: objseg postprocess does not use sqrt
        elif cond_id is not None and use_bg_embeds:
            cond_id_map = cond_id_postprocess(cond_id, training=False).to(mask_cls.dtype)
            pred_scores = mask_cls @ cond_id_map.T
            pred_scores = F.softmax(pred_scores, dim=-1)[:, :-1]
        else:
            # the last class is __background__
            pred_scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        num_classes = pred_scores.shape[-1]
        assert num_classes == 1

        keep = (pred_scores >= threshold) | (pred_scores == pred_scores.max())
        keep = keep.squeeze(-1)
        mask_pred = mask_pred[keep]
        pred_mask = pred_mask[keep]
        pred_scores = pred_scores[keep]

        # Build multi-reference memory: pinned anchor + recent clean frames.
        # Weights use the per-frame confidence so that the anchor (w=1.0)
        # always dominates when recent frames are uncertain.
        ref_masks: List[torch.Tensor] = [anchor_mask]
        ref_weights: List[float] = [1.0]
        for m, s in memory:
            ref_masks.append(m)
            ref_weights.append(s)
        ref = torch.stack(ref_masks, dim=0).to(pred_mask.device, dtype=pred_mask.dtype)
        w = torch.tensor(ref_weights, device=pred_mask.device, dtype=pred_mask.dtype)

        # Weighted mean of multi-reference IoU -> [Q]
        iou = pairwise_mask_iou(pred_mask, ref)
        geom_score = (iou * w.unsqueeze(0)).sum(dim=-1) / w.sum().clamp(min=1e-6)
        combined_score = alpha * geom_score + (1 - alpha) * pred_scores.squeeze(-1)
        top_index = combined_score.argmax(dim=0)
        top_score = combined_score[top_index].squeeze(-1).clamp(min=0)

        mask_pred = mask_pred[top_index]
        mask_prob = mask_pred.sigmoid()

        # 255 is the ignore index
        segmentation = torch.full(
            (image_size[0], image_size[1]), ignore_value, dtype=torch.long, device=mask_pred.device
        )
        segmentation[mask_prob > mask_threshold] = 1
        if float(top_score.item()) > threshold:
            binary_mask = (segmentation == 1).to(pred_mask.dtype)
            memory.append((binary_mask, float(top_score.item())))

        segments_info = {
            "id": 0,
            "label_id": 0,
            "was_fused": False,
            "score": round(top_score.item(), 6),
        }

        results.append({"segmentation": segmentation, "segments_info": segments_info, "vprompt_masks": vprompt_mask})

    return [results]