import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


# class YOLOLoss(nn.Module):
#     """
#     Fonction de perte YOLOv3 calculée sur les 3 échelles de sortie.
#     Match chaque vérité terrain (Ground Truth) avec l'ancre la plus proche parmi les 9 disponibles.
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         lambda_box: float = 0.05,
#         lambda_obj: float = 1.0,
#         lambda_cls: float = 0.5,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.lambda_box = lambda_box
#         self.lambda_obj = lambda_obj
#         self.lambda_cls = lambda_cls

#         self.bce_obj = nn.BCEWithLogitsLoss()
#         self.bce_cls = nn.BCEWithLogitsLoss()
#         self.smooth_l1 = nn.SmoothL1Loss()

#     def forward(
#         self,
#         raw_outputs: List[torch.Tensor],
#         targets: List[Dict[str, torch.Tensor]],
#         anchors: torch.Tensor,
#         img_size: Tuple[int, int] = (300, 300)
#     ) -> Dict[str, torch.Tensor]:

#         device = raw_outputs[0].device
#         B = raw_outputs[0].size(0)

#         loss_box = torch.tensor(0.0, device=device)
#         loss_obj = torch.tensor(0.0, device=device)
#         loss_cls = torch.tensor(0.0, device=device)

#         target_objs, target_boxes, target_clss = self._build_targets(
#             raw_outputs, targets, anchors, img_size
#         )

#         for scale_idx, raw_out in enumerate(raw_outputs):
#             H, W = raw_out.shape[2:]

#             preds = raw_out.view(
#                 B, 3, 5 + self.num_classes, H, W
#             ).permute(0, 1, 3, 4, 2).contiguous()

#             pred_tx = preds[..., 0]
#             pred_ty = preds[..., 1]
#             pred_tw = preds[..., 2]
#             pred_th = preds[..., 3]
#             pred_obj = preds[..., 4]
#             pred_cls = preds[..., 5:]

#             t_obj = target_objs[scale_idx]
#             t_box = target_boxes[scale_idx]
#             t_cls = target_clss[scale_idx]

#             obj_mask = (t_obj == 1.0)
#             noobj_mask = (t_obj == 0.0)

#             # --- 1. Perte d'Objectité rééquilibrée ---
#             loss_obj_pos = self.bce_obj(pred_obj[obj_mask], t_obj[obj_mask])
#             loss_obj_neg = self.bce_obj(pred_obj[noobj_mask], t_obj[noobj_mask])
            
#             # Le fond est pondéré à 0.5 pour ne pas étouffer les signaux positifs
#             loss_scale_obj = (loss_obj_pos if obj_mask.sum() > 0 else 0.0) + 0.5 * loss_obj_neg
#             loss_obj += loss_scale_obj

#             # --- 2. Pertes de Boîtes et Classes (Uniquement si présence d'objet) ---
#             if obj_mask.sum() > 0:
#                 loss_box += F.smooth_l1_loss(torch.sigmoid(pred_tx[obj_mask]), t_box[..., 0][obj_mask])
#                 loss_box += F.smooth_l1_loss(torch.sigmoid(pred_ty[obj_mask]), t_box[..., 1][obj_mask])
#                 loss_box += F.smooth_l1_loss(pred_tw[obj_mask], t_box[..., 2][obj_mask])
#                 loss_box += F.smooth_l1_loss(pred_th[obj_mask], t_box[..., 3][obj_mask])

#                 loss_cls += self.bce_cls(pred_cls[obj_mask], t_cls[obj_mask])

#         total_loss = (
#             self.lambda_box * loss_box +
#             self.lambda_obj * loss_obj +
#             self.lambda_cls * loss_cls
#         )

#         return {
#             "loss": total_loss,
#             "loss_box": loss_box,
#             "loss_obj": loss_obj,
#             "loss_cls": loss_cls
#         }

#     def _build_targets(
#         self,
#         raw_outputs: List[torch.Tensor],
#         targets: List[Dict[str, torch.Tensor]],
#         anchors: torch.Tensor,
#         img_size: Tuple[int, int]
#     ):
#         device = raw_outputs[0].device
#         B = raw_outputs[0].size(0)
#         img_h, img_w = img_size

#         target_objs = []
#         target_boxes = []
#         target_clss = []

#         for raw_out in raw_outputs:
#             H, W = raw_out.shape[2:]
#             target_objs.append(torch.zeros((B, 3, H, W), device=device, dtype=torch.float32))
#             target_boxes.append(torch.zeros((B, 3, H, W, 4), device=device, dtype=torch.float32))
#             target_clss.append(torch.zeros((B, 3, H, W, self.num_classes), device=device, dtype=torch.float32))

#         # Les 9 ancres au format plat (9, 2)
#         all_anchors = anchors.view(-1, 2)

#         for b in range(B):
#             gt_boxes = targets[b]["boxes"]    # (N, 4) en coordonnées pixels [x1, y1, x2, y2]
#             gt_labels = targets[b]["labels"]  # (N,)

#             if gt_boxes.size(0) == 0:
#                 continue

#             # Conversion des vérités terrain en coordonnées normalisées [cx, cy, w, h]
#             w_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) / img_w
#             h_gt = (gt_boxes[:, 3] - gt_boxes[:, 1]) / img_h
#             cx_gt = (gt_boxes[:, 0] + gt_boxes[:, 2]) / (2.0 * img_w)
#             cy_gt = (gt_boxes[:, 1] + gt_boxes[:, 3]) / (2.0 * img_h)

#             wa_norm = all_anchors[:, 0] / img_w
#             ha_norm = all_anchors[:, 1] / img_h

#             for i in range(gt_boxes.size(0)):
#                 w, h = w_gt[i], h_gt[i]
#                 cx, cy = cx_gt[i], cy_gt[i]
#                 label = gt_labels[i]

#                 # Calcul de l'IoU entre la boite GT et les 9 ancres (alignées en (0, 0))
#                 inter = torch.min(w, wa_norm) * torch.min(h, ha_norm)
#                 union = (w * h) + (wa_norm * ha_norm) - inter
#                 anchor_ious = inter / (union + 1e-16)

#                 # Recherche de la meilleure ancre parmi les 9
#                 best_anchor_idx = torch.argmax(anchor_ious).item()
#                 scale_idx = best_anchor_idx // 3  # Échelle (0, 1 ou 2)
#                 anchor_idx = best_anchor_idx % 3  # Ancre au sein de cette échelle (0, 1 ou 2)

#                 H, W = raw_outputs[scale_idx].shape[2:]

#                 # Indexation de la cellule dans la grille
#                 gx, gy = cx * W, cy * H
#                 gi, gj = int(gx), int(gy)

#                 gi = max(0, min(gi, W - 1))
#                 gj = max(0, min(gj, H - 1))

#                 # Calcul des offsets cibles
#                 tx = gx - gi
#                 ty = gy - gj
#                 p_w = anchors[scale_idx, anchor_idx, 0] / img_w
#                 p_h = anchors[scale_idx, anchor_idx, 1] / img_h

#                 tw = torch.log(w / (p_w + 1e-16) + 1e-16)
#                 th = torch.log(h / (p_h + 1e-16) + 1e-16)

#                 # Assignation des valeurs cibles
#                 target_objs[scale_idx][b, anchor_idx, gj, gi] = 1.0
#                 target_boxes[scale_idx][b, anchor_idx, gj, gi] = torch.tensor(
#                     [tx, ty, tw, th], device=device
#                 )

#                 # Encodage de la classe
#                 cls_idx = label.item()
#                 if cls_idx < self.num_classes:
#                     target_clss[scale_idx][b, anchor_idx, gj, gi, cls_idx] = 1.0

#         return target_objs, target_boxes, target_clss


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """ Calcule la Complete IoU (CIoU) entre deux ensembles de boîtes (x1, y1, x2, y2) normalisées. """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1) + (w2 * h2) - inter_area + eps

    iou = inter_area / union_area

    center_x1, center_y1 = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
    center_x2, center_y2 = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0
    center_distance = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    return iou - (center_distance / enclose_diagonal) - (alpha * v)


class YOLOLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lambda_box: float = 2.5,     # Réajusté pour la CIoU Loss
        lambda_obj: float = 1.0,
        lambda_cls: float = 0.5,
        labels_are_one_indexed: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.labels_are_one_indexed = labels_are_one_indexed

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="sum")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        raw_outputs: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        anchors: torch.Tensor,
        img_size: Tuple[int, int] = (300, 300)
    ) -> Dict[str, torch.Tensor]:

        device = raw_outputs[0].device
        B = raw_outputs[0].size(0)

        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        target_objs, target_boxes, target_clss = self._build_targets(
            raw_outputs, targets, anchors, img_size
        )

        # Compter le nombre total de cibles positives dans tout le batch
        num_positives = sum((t == 1.0).sum().item() for t in target_objs)
        num_targets = max(1, num_positives)

        for scale_idx, raw_out in enumerate(raw_outputs):
            H, W = raw_out.shape[2:]

            preds = raw_out.view(
                B, 3, 5 + self.num_classes, H, W
            ).permute(0, 1, 3, 4, 2).contiguous()

            pred_tx = preds[..., 0]
            pred_ty = preds[..., 1]
            pred_tw = preds[..., 2]
            pred_th = preds[..., 3]
            pred_obj = preds[..., 4]
            pred_cls = preds[..., 5:]

            t_obj = target_objs[scale_idx]
            t_box = target_boxes[scale_idx]
            t_cls = target_clss[scale_idx]

            obj_mask = (t_obj == 1.0)
            noobj_mask = (t_obj == 0.0)

            # 1. Perte d'Objectité (Somme des cibles positives + 0.5 * négatives)
            obj_loss_pos = self.bce_obj(pred_obj[obj_mask], t_obj[obj_mask])
            obj_loss_neg = self.bce_obj(pred_obj[noobj_mask], t_obj[noobj_mask])
            loss_obj += (obj_loss_pos + 0.5 * obj_loss_neg) / num_targets

            if obj_mask.sum() > 0:
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"
                )

                scale_anchors = anchors[scale_idx].to(device)
                anchor_w = scale_anchors[:, 0].view(1, 3, 1, 1)
                anchor_h = scale_anchors[:, 1].view(1, 3, 1, 1)

                # Boîtes prédites (x1, y1, x2, y2)
                pred_cx = (torch.sigmoid(pred_tx) + grid_x) / W
                pred_cy = (torch.sigmoid(pred_ty) + grid_y) / H
                pred_w = (torch.exp(pred_tw) * anchor_w) / img_size[1]
                pred_h = (torch.exp(pred_th) * anchor_h) / img_size[0]

                pred_x1 = pred_cx - pred_w / 2.0
                pred_y1 = pred_cy - pred_h / 2.0
                pred_x2 = pred_cx + pred_w / 2.0
                pred_y2 = pred_cy + pred_h / 2.0
                pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

                # Boîtes cibles (x1, y1, x2, y2)
                target_cx = (t_box[..., 0] + grid_x) / W
                target_cy = (t_box[..., 1] + grid_y) / H
                target_w = (torch.exp(t_box[..., 2]) * anchor_w) / img_size[1]
                target_h = (torch.exp(t_box[..., 3]) * anchor_h) / img_size[0]

                target_x1 = target_cx - target_w / 2.0
                target_y1 = target_cy - target_h / 2.0
                target_x2 = target_cx + target_w / 2.0
                target_y2 = target_cy + target_h / 2.0
                target_boxes_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=-1)

                # 2. Perte CIoU normalisée par num_targets
                ciou = bbox_ciou(pred_boxes_xyxy[obj_mask], target_boxes_xyxy[obj_mask])
                loss_box += (1.0 - ciou).sum() / num_targets

                # 3. Perte de Classification normalisée par num_targets
                loss_cls += self.bce_cls(pred_cls[obj_mask], t_cls[obj_mask]) / num_targets

        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )

        return {
            "loss": total_loss,
            "loss_box": loss_box,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls
        }

    def _build_targets(
        self,
        raw_outputs: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        anchors: torch.Tensor,
        img_size: Tuple[int, int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        device = raw_outputs[0].device
        B = raw_outputs[0].size(0)
        img_h, img_w = img_size

        target_objs = []
        target_boxes = []
        target_clss = []

        for raw_out in raw_outputs:
            H, W = raw_out.shape[2:]
            target_objs.append(torch.zeros((B, 3, H, W), device=device, dtype=torch.float32))
            target_boxes.append(torch.zeros((B, 3, H, W, 4), device=device, dtype=torch.float32))
            target_clss.append(torch.zeros((B, 3, H, W, self.num_classes), device=device, dtype=torch.float32))

        all_anchors = anchors.view(-1, 2)

        for b in range(B):
            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]

            if gt_boxes.size(0) == 0:
                continue

            w_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) / img_w
            h_gt = (gt_boxes[:, 3] - gt_boxes[:, 1]) / img_h
            cx_gt = (gt_boxes[:, 0] + gt_boxes[:, 2]) / (2.0 * img_w)
            cy_gt = (gt_boxes[:, 1] + gt_boxes[:, 3]) / (2.0 * img_h)

            wa_norm = all_anchors[:, 0] / img_w
            ha_norm = all_anchors[:, 1] / img_h

            for i in range(gt_boxes.size(0)):
                w, h = w_gt[i], h_gt[i]
                cx, cy = cx_gt[i], cy_gt[i]
                label = gt_labels[i].item()

                cls_idx = label - 1 if self.labels_are_one_indexed else label

                inter = torch.min(w, wa_norm) * torch.min(h, ha_norm)
                union = (w * h) + (wa_norm * ha_norm) - inter
                anchor_ious = inter / (union + 1e-16)

                best_anchor_idx = torch.argmax(anchor_ious).item()
                scale_idx = best_anchor_idx // 3
                anchor_idx = best_anchor_idx % 3

                H, W = raw_outputs[scale_idx].shape[2:]

                gx, gy = cx * W, cy * H
                gi, gj = int(gx), int(gy)

                gi = max(0, min(gi, W - 1))
                gj = max(0, min(gj, H - 1))

                tx = gx - gi
                ty = gy - gj

                p_w = anchors[scale_idx, anchor_idx, 0] / img_w
                p_h = anchors[scale_idx, anchor_idx, 1] / img_h

                tw = torch.log(w / (p_w + 1e-16) + 1e-16)
                th = torch.log(h / (p_h + 1e-16) + 1e-16)

                target_objs[scale_idx][b, anchor_idx, gj, gi] = 1.0
                target_boxes[scale_idx][b, anchor_idx, gj, gi] = torch.tensor(
                    [tx, ty, tw, th], device=device, dtype=torch.float32
                )

                if 0 <= cls_idx < self.num_classes:
                    target_clss[scale_idx][b, anchor_idx, gj, gi, cls_idx] = 1.0

        return target_objs, target_boxes, target_clss