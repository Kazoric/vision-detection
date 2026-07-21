import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from utils.ssd_utils import jaccard_iou

class SSDMultiBoxLoss(nn.Module):
    def __init__(self, num_classes: int, overlap_thresh: float = 0.5, neg_pos_ratio: float = 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.variances = [0.1, 0.2]

    def forward(self, loc_preds: torch.Tensor, cls_preds: torch.Tensor, targets: List[Dict], priors: torch.Tensor, img_size=(300, 300)) -> Dict[str, torch.Tensor]:
        B, num_priors, _ = loc_preds.shape
        device = loc_preds.device
        img_w, img_h = img_size

        priors_corner = torch.zeros_like(priors)
        priors_corner[:, :2] = priors[:, :2] - priors[:, 2:] / 2.0
        priors_corner[:, 2:] = priors[:, :2] + priors[:, 2:] / 2.0

        loc_targets = torch.zeros_like(loc_preds)
        cls_targets = torch.zeros((B, num_priors), dtype=torch.long, device=device)

        for b in range(B):
            gt_boxes = targets[b]["boxes"]
            labels = targets[b]["labels"]

            if len(gt_boxes) == 0:
                continue

            gt_boxes_norm = gt_boxes.clone()
            gt_boxes_norm[:, [0, 2]] /= float(img_w)
            gt_boxes_norm[:, [1, 3]] /= float(img_h)

            overlaps = jaccard_iou(gt_boxes_norm, priors_corner)
            best_prior_overlap, best_prior_idx = overlaps.max(1)
            best_truth_overlap, best_truth_idx = overlaps.max(0)

            for idx, prior_idx in enumerate(best_prior_idx):
                best_truth_idx[prior_idx] = idx
                best_truth_overlap[prior_idx] = 2.0

            assigned_labels = labels[best_truth_idx]
            assigned_labels[best_truth_overlap < self.overlap_thresh] = 0
            cls_targets[b] = assigned_labels

            matched_gt = gt_boxes_norm[best_truth_idx]
            gt_cxcy = (matched_gt[:, :2] + matched_gt[:, 2:]) / 2.0
            gt_wh = matched_gt[:, 2:] - matched_gt[:, :2]

            g_hat_cxcy = (gt_cxcy - priors[:, :2]) / (priors[:, 2:] * self.variances[0])
            g_hat_wh = torch.log(gt_wh / priors[:, 2:] + 1e-5) / self.variances[1]
            loc_targets[b] = torch.cat([g_hat_cxcy, g_hat_wh], dim=1)

        pos_mask = cls_targets > 0
        num_pos = pos_mask.sum()

        if num_pos > 0:
            loss_box = F.smooth_l1_loss(loc_preds[pos_mask], loc_targets[pos_mask], reduction='sum') / B
        else:
            loss_box = torch.tensor(0.0, device=device, requires_grad=True)

        cls_loss_all = F.cross_entropy(cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduction='none')
        cls_loss_all = cls_loss_all.view(B, num_priors)

        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0.0

        _, loss_idx = cls_loss_neg.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos_per_img = pos_mask.sum(1, keepdim=True)
        num_neg_per_img = torch.clamp(num_pos_per_img * self.neg_pos_ratio, max=num_priors - 1).long()
        neg_mask = idx_rank < num_neg_per_img

        final_mask = pos_mask | neg_mask
        loss_class = cls_loss_all[final_mask].sum() / B

        return {
            "loss_box": loss_box,
            "loss_class": loss_class,
            "loss_obj": torch.tensor(0.0, device=device),
            "loss_noobj": torch.tensor(0.0, device=device)
        }