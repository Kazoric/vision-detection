import math
from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calcul de la loss CIoU vectorisé sur des boîtes au format (x1, y1, x2, y2)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)
    union_area = (w1 * h1) + (w2 * h2) - inter_area + eps
    iou = inter_area / union_area

    # Terme de distance entre centres et diagonale englobante
    center_dist = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4.0
    enclose_diag = ((torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)) ** 2 +
                    (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)) ** 2) + eps

    # Aspect ratio
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return iou - (center_dist / enclose_diag) - (alpha * v)

class YoloLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        strides: List[int] = [8, 16, 32],
        lambda_box: float = 7.5,
        lambda_cls: float = 0.5,
        scale_ranges: List[Tuple[float, float]] = [(0, 64), (64, 128), (128, 1e5)],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.scale_ranges = scale_ranges

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal Loss pour équilibrer fond et objets."""
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, gamma)
        return focal_weight * bce

    def forward(
        self,
        predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        targets: Union[List[Dict], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = next(iter(predictions.values()))[0].device
        targets = self._format_targets(targets, device)

        tot_loss_box = torch.tensor(0.0, device=device)
        tot_loss_cls = torch.tensor(0.0, device=device)
        total_num_pos = 0

        for scale_idx, (_, (cls_score, bbox_pred)) in enumerate(predictions.items()):
            stride = self.strides[scale_idx]
            min_size, max_size = self.scale_ranges[scale_idx]

            cls_score = cls_score.permute(0, 2, 3, 1)  # [B, H, W, C]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # [B, H, W, 4]
            B, H, W, _ = cls_score.shape

            target_cls, target_box, pos_mask = self._build_targets(
                targets, B, H, W, stride, min_size, max_size, device
            )

            num_pos = pos_mask.sum().item()
            total_num_pos += num_pos

            # Loss Classification avec Focal Loss
            focal_loss = self._focal_loss(cls_score, target_cls)
            tot_loss_cls += focal_loss.sum()

            # Loss Bounding Box (uniquement sur les cellules positives)
            if num_pos > 0:
                b_idx, y_idx, x_idx = torch.where(pos_mask)

                dist_pred = F.softplus(bbox_pred[b_idx, y_idx, x_idx]) * stride
                pred_x1 = (x_idx.float() + 0.5) * stride - dist_pred[:, 0]
                pred_y1 = (y_idx.float() + 0.5) * stride - dist_pred[:, 1]
                pred_x2 = (x_idx.float() + 0.5) * stride + dist_pred[:, 2]
                pred_y2 = (y_idx.float() + 0.5) * stride + dist_pred[:, 3]

                pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
                target_boxes = target_box[b_idx, y_idx, x_idx]

                ciou = bbox_ciou(pred_boxes, target_boxes)
                tot_loss_box += (1.0 - ciou).sum()

        num_pos_norm = max(total_num_pos, 1)
        loss_box = (tot_loss_box / num_pos_norm) * self.lambda_box
        loss_class = (tot_loss_cls / num_pos_norm) * self.lambda_cls

        return {
            "loss_box": loss_box,
            "loss_class": loss_class,
        }

    def _format_targets(self, targets: Union[List, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Convertit une liste de cibles (format SSD List[Dict]) en Tensor 2D [N, 6]."""
        if isinstance(targets, torch.Tensor):
            return targets.to(device)

        if not targets:
            return torch.empty((0, 6), device=device)

        formatted_list = []
        for b_idx, t in enumerate(targets):
            if isinstance(t, dict):
                boxes = t["boxes"].to(device)
                labels = t["labels"].to(device)
                if len(boxes) > 0:
                    b_vec = torch.full((len(boxes), 1), b_idx, device=device, dtype=boxes.dtype)
                    labels_vec = labels.unsqueeze(1).to(dtype=boxes.dtype)
                    formatted_list.append(torch.cat([b_vec, labels_vec, boxes], dim=1))
            elif isinstance(t, torch.Tensor) and len(t) > 0:
                b_vec = torch.full((len(t), 1), b_idx, device=device, dtype=t.dtype)
                formatted_list.append(torch.cat([b_vec, t.to(device)], dim=1))

        if formatted_list:
            return torch.cat(formatted_list, dim=0)
        return torch.empty((0, 6), device=device)

    def _build_targets(
        self,
        targets: torch.Tensor,
        batch_size: int,
        grid_h: int,
        grid_w: int,
        stride: int,
        min_size: float,
        max_size: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_cls = torch.zeros((batch_size, grid_h, grid_w, self.num_classes), device=device)
        target_box = torch.zeros((batch_size, grid_h, grid_w, 4), device=device)
        pos_mask = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.bool, device=device)

        if targets is None or targets.numel() == 0:
            return target_cls, target_box, pos_mask

        batch_idx = targets[:, 0].long().clamp(0, batch_size - 1)
        cls_id = targets[:, 1].long().clamp(0, self.num_classes - 1)
        xmin, ymin, xmax, ymax = targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5]

        # Filtrage par échelle
        max_dim = torch.max(xmax - xmin, ymax - ymin)
        scale_mask = (max_dim >= min_size) & (max_dim < max_size)

        if not scale_mask.any():
            return target_cls, target_box, pos_mask

        batch_idx, cls_id = batch_idx[scale_mask], cls_id[scale_mask]
        xmin, ymin = xmin[scale_mask], ymin[scale_mask]
        xmax, ymax = xmax[scale_mask], ymax[scale_mask]

        gx = ((xmin + xmax) / (2.0 * stride)).long().clamp(0, grid_w - 1)
        gy = ((ymin + ymax) / (2.0 * stride)).long().clamp(0, grid_h - 1)

        pos_mask[batch_idx, gy, gx] = True
        target_box[batch_idx, gy, gx] = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        target_cls[batch_idx, gy, gx, cls_id] = 1.0

        return target_cls, target_box, pos_mask