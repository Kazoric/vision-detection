# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Dict, Tuple


# # class YOLOLoss(nn.Module):
# #     """
# #     Fonction de perte YOLOv3 calculée sur les 3 échelles de sortie.
# #     Match chaque vérité terrain (Ground Truth) avec l'ancre la plus proche parmi les 9 disponibles.
# #     """

# #     def __init__(
# #         self,
# #         num_classes: int,
# #         lambda_box: float = 0.05,
# #         lambda_obj: float = 1.0,
# #         lambda_cls: float = 0.5,
# #     ):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         self.lambda_box = lambda_box
# #         self.lambda_obj = lambda_obj
# #         self.lambda_cls = lambda_cls

# #         self.bce_obj = nn.BCEWithLogitsLoss()
# #         self.bce_cls = nn.BCEWithLogitsLoss()
# #         self.smooth_l1 = nn.SmoothL1Loss()

# #     def forward(
# #         self,
# #         raw_outputs: List[torch.Tensor],
# #         targets: List[Dict[str, torch.Tensor]],
# #         anchors: torch.Tensor,
# #         img_size: Tuple[int, int] = (300, 300)
# #     ) -> Dict[str, torch.Tensor]:

# #         device = raw_outputs[0].device
# #         B = raw_outputs[0].size(0)

# #         loss_box = torch.tensor(0.0, device=device)
# #         loss_obj = torch.tensor(0.0, device=device)
# #         loss_cls = torch.tensor(0.0, device=device)

# #         target_objs, target_boxes, target_clss = self._build_targets(
# #             raw_outputs, targets, anchors, img_size
# #         )

# #         for scale_idx, raw_out in enumerate(raw_outputs):
# #             H, W = raw_out.shape[2:]

# #             preds = raw_out.view(
# #                 B, 3, 5 + self.num_classes, H, W
# #             ).permute(0, 1, 3, 4, 2).contiguous()

# #             pred_tx = preds[..., 0]
# #             pred_ty = preds[..., 1]
# #             pred_tw = preds[..., 2]
# #             pred_th = preds[..., 3]
# #             pred_obj = preds[..., 4]
# #             pred_cls = preds[..., 5:]

# #             t_obj = target_objs[scale_idx]
# #             t_box = target_boxes[scale_idx]
# #             t_cls = target_clss[scale_idx]

# #             obj_mask = (t_obj == 1.0)
# #             noobj_mask = (t_obj == 0.0)

# #             # --- 1. Perte d'Objectité rééquilibrée ---
# #             loss_obj_pos = self.bce_obj(pred_obj[obj_mask], t_obj[obj_mask])
# #             loss_obj_neg = self.bce_obj(pred_obj[noobj_mask], t_obj[noobj_mask])
            
# #             # Le fond est pondéré à 0.5 pour ne pas étouffer les signaux positifs
# #             loss_scale_obj = (loss_obj_pos if obj_mask.sum() > 0 else 0.0) + 0.5 * loss_obj_neg
# #             loss_obj += loss_scale_obj

# #             # --- 2. Pertes de Boîtes et Classes (Uniquement si présence d'objet) ---
# #             if obj_mask.sum() > 0:
# #                 loss_box += F.smooth_l1_loss(torch.sigmoid(pred_tx[obj_mask]), t_box[..., 0][obj_mask])
# #                 loss_box += F.smooth_l1_loss(torch.sigmoid(pred_ty[obj_mask]), t_box[..., 1][obj_mask])
# #                 loss_box += F.smooth_l1_loss(pred_tw[obj_mask], t_box[..., 2][obj_mask])
# #                 loss_box += F.smooth_l1_loss(pred_th[obj_mask], t_box[..., 3][obj_mask])

# #                 loss_cls += self.bce_cls(pred_cls[obj_mask], t_cls[obj_mask])

# #         total_loss = (
# #             self.lambda_box * loss_box +
# #             self.lambda_obj * loss_obj +
# #             self.lambda_cls * loss_cls
# #         )

# #         return {
# #             "loss": total_loss,
# #             "loss_box": loss_box,
# #             "loss_obj": loss_obj,
# #             "loss_cls": loss_cls
# #         }

# #     def _build_targets(
# #         self,
# #         raw_outputs: List[torch.Tensor],
# #         targets: List[Dict[str, torch.Tensor]],
# #         anchors: torch.Tensor,
# #         img_size: Tuple[int, int]
# #     ):
# #         device = raw_outputs[0].device
# #         B = raw_outputs[0].size(0)
# #         img_h, img_w = img_size

# #         target_objs = []
# #         target_boxes = []
# #         target_clss = []

# #         for raw_out in raw_outputs:
# #             H, W = raw_out.shape[2:]
# #             target_objs.append(torch.zeros((B, 3, H, W), device=device, dtype=torch.float32))
# #             target_boxes.append(torch.zeros((B, 3, H, W, 4), device=device, dtype=torch.float32))
# #             target_clss.append(torch.zeros((B, 3, H, W, self.num_classes), device=device, dtype=torch.float32))

# #         # Les 9 ancres au format plat (9, 2)
# #         all_anchors = anchors.view(-1, 2)

# #         for b in range(B):
# #             gt_boxes = targets[b]["boxes"]    # (N, 4) en coordonnées pixels [x1, y1, x2, y2]
# #             gt_labels = targets[b]["labels"]  # (N,)

# #             if gt_boxes.size(0) == 0:
# #                 continue

# #             # Conversion des vérités terrain en coordonnées normalisées [cx, cy, w, h]
# #             w_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) / img_w
# #             h_gt = (gt_boxes[:, 3] - gt_boxes[:, 1]) / img_h
# #             cx_gt = (gt_boxes[:, 0] + gt_boxes[:, 2]) / (2.0 * img_w)
# #             cy_gt = (gt_boxes[:, 1] + gt_boxes[:, 3]) / (2.0 * img_h)

# #             wa_norm = all_anchors[:, 0] / img_w
# #             ha_norm = all_anchors[:, 1] / img_h

# #             for i in range(gt_boxes.size(0)):
# #                 w, h = w_gt[i], h_gt[i]
# #                 cx, cy = cx_gt[i], cy_gt[i]
# #                 label = gt_labels[i]

# #                 # Calcul de l'IoU entre la boite GT et les 9 ancres (alignées en (0, 0))
# #                 inter = torch.min(w, wa_norm) * torch.min(h, ha_norm)
# #                 union = (w * h) + (wa_norm * ha_norm) - inter
# #                 anchor_ious = inter / (union + 1e-16)

# #                 # Recherche de la meilleure ancre parmi les 9
# #                 best_anchor_idx = torch.argmax(anchor_ious).item()
# #                 scale_idx = best_anchor_idx // 3  # Échelle (0, 1 ou 2)
# #                 anchor_idx = best_anchor_idx % 3  # Ancre au sein de cette échelle (0, 1 ou 2)

# #                 H, W = raw_outputs[scale_idx].shape[2:]

# #                 # Indexation de la cellule dans la grille
# #                 gx, gy = cx * W, cy * H
# #                 gi, gj = int(gx), int(gy)

# #                 gi = max(0, min(gi, W - 1))
# #                 gj = max(0, min(gj, H - 1))

# #                 # Calcul des offsets cibles
# #                 tx = gx - gi
# #                 ty = gy - gj
# #                 p_w = anchors[scale_idx, anchor_idx, 0] / img_w
# #                 p_h = anchors[scale_idx, anchor_idx, 1] / img_h

# #                 tw = torch.log(w / (p_w + 1e-16) + 1e-16)
# #                 th = torch.log(h / (p_h + 1e-16) + 1e-16)

# #                 # Assignation des valeurs cibles
# #                 target_objs[scale_idx][b, anchor_idx, gj, gi] = 1.0
# #                 target_boxes[scale_idx][b, anchor_idx, gj, gi] = torch.tensor(
# #                     [tx, ty, tw, th], device=device
# #                 )

# #                 # Encodage de la classe
# #                 cls_idx = label.item()
# #                 if cls_idx < self.num_classes:
# #                     target_clss[scale_idx][b, anchor_idx, gj, gi, cls_idx] = 1.0

# #         return target_objs, target_boxes, target_clss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import List, Dict, Tuple


# def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
#     """ Calcule la Complete IoU (CIoU) entre deux ensembles de boîtes (x1, y1, x2, y2) normalisées. """
#     b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
#     b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

#     inter_x1 = torch.max(b1_x1, b2_x1)
#     inter_y1 = torch.max(b1_y1, b2_y1)
#     inter_x2 = torch.min(b1_x2, b2_x2)
#     inter_y2 = torch.min(b1_y2, b2_y2)

#     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
#     union_area = (w1 * h1) + (w2 * h2) - inter_area + eps

#     iou = inter_area / union_area

#     center_x1, center_y1 = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
#     center_x2, center_y2 = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0
#     center_distance = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

#     enclose_x1 = torch.min(b1_x1, b2_x1)
#     enclose_y1 = torch.min(b1_y1, b2_y1)
#     enclose_x2 = torch.max(b1_x2, b2_x2)
#     enclose_y2 = torch.max(b1_y2, b2_y2)
#     enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

#     v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
#     with torch.no_grad():
#         alpha = v / ((1.0 - iou) + v + eps)

#     return iou - (center_distance / enclose_diagonal) - (alpha * v)


# class YOLOLoss(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         lambda_box: float = 2.5,     # Réajusté pour la CIoU Loss
#         lambda_obj: float = 1.0,
#         lambda_cls: float = 0.5,
#         labels_are_one_indexed: bool = False
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.lambda_box = lambda_box
#         self.lambda_obj = lambda_obj
#         self.lambda_cls = lambda_cls
#         self.labels_are_one_indexed = labels_are_one_indexed

#         self.bce_obj = nn.BCEWithLogitsLoss(reduction="sum")
#         self.bce_cls = nn.BCEWithLogitsLoss(reduction="sum")

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

#         # Compter le nombre total de cibles positives dans tout le batch
#         num_positives = sum((t == 1.0).sum().item() for t in target_objs)
#         num_targets = max(1, num_positives)

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

#             # 1. Perte d'Objectité (Somme des cibles positives + 0.5 * négatives)
#             obj_loss_pos = self.bce_obj(pred_obj[obj_mask], t_obj[obj_mask])
#             obj_loss_neg = self.bce_obj(pred_obj[noobj_mask], t_obj[noobj_mask])
#             loss_obj += (obj_loss_pos + 0.5 * obj_loss_neg) / num_targets

#             if obj_mask.sum() > 0:
#                 grid_y, grid_x = torch.meshgrid(
#                     torch.arange(H, device=device),
#                     torch.arange(W, device=device),
#                     indexing="ij"
#                 )

#                 scale_anchors = anchors[scale_idx].to(device)
#                 anchor_w = scale_anchors[:, 0].view(1, 3, 1, 1)
#                 anchor_h = scale_anchors[:, 1].view(1, 3, 1, 1)

#                 # Boîtes prédites (x1, y1, x2, y2)
#                 pred_cx = (torch.sigmoid(pred_tx) + grid_x) / W
#                 pred_cy = (torch.sigmoid(pred_ty) + grid_y) / H
#                 pred_w = (torch.exp(pred_tw) * anchor_w) / img_size[1]
#                 pred_h = (torch.exp(pred_th) * anchor_h) / img_size[0]

#                 pred_x1 = pred_cx - pred_w / 2.0
#                 pred_y1 = pred_cy - pred_h / 2.0
#                 pred_x2 = pred_cx + pred_w / 2.0
#                 pred_y2 = pred_cy + pred_h / 2.0
#                 pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

#                 # Boîtes cibles (x1, y1, x2, y2)
#                 target_cx = (t_box[..., 0] + grid_x) / W
#                 target_cy = (t_box[..., 1] + grid_y) / H
#                 target_w = (torch.exp(t_box[..., 2]) * anchor_w) / img_size[1]
#                 target_h = (torch.exp(t_box[..., 3]) * anchor_h) / img_size[0]

#                 target_x1 = target_cx - target_w / 2.0
#                 target_y1 = target_cy - target_h / 2.0
#                 target_x2 = target_cx + target_w / 2.0
#                 target_y2 = target_cy + target_h / 2.0
#                 target_boxes_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=-1)

#                 # 2. Perte CIoU normalisée par num_targets
#                 ciou = bbox_ciou(pred_boxes_xyxy[obj_mask], target_boxes_xyxy[obj_mask])
#                 loss_box += (1.0 - ciou).sum() / num_targets

#                 # 3. Perte de Classification normalisée par num_targets
#                 loss_cls += self.bce_cls(pred_cls[obj_mask], t_cls[obj_mask]) / num_targets

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
#     ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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

#         all_anchors = anchors.view(-1, 2)

#         for b in range(B):
#             gt_boxes = targets[b]["boxes"]
#             gt_labels = targets[b]["labels"]

#             if gt_boxes.size(0) == 0:
#                 continue

#             w_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) / img_w
#             h_gt = (gt_boxes[:, 3] - gt_boxes[:, 1]) / img_h
#             cx_gt = (gt_boxes[:, 0] + gt_boxes[:, 2]) / (2.0 * img_w)
#             cy_gt = (gt_boxes[:, 1] + gt_boxes[:, 3]) / (2.0 * img_h)

#             wa_norm = all_anchors[:, 0] / img_w
#             ha_norm = all_anchors[:, 1] / img_h

#             for i in range(gt_boxes.size(0)):
#                 w, h = w_gt[i], h_gt[i]
#                 cx, cy = cx_gt[i], cy_gt[i]
#                 label = gt_labels[i].item()

#                 cls_idx = label - 1 if self.labels_are_one_indexed else label

#                 inter = torch.min(w, wa_norm) * torch.min(h, ha_norm)
#                 union = (w * h) + (wa_norm * ha_norm) - inter
#                 anchor_ious = inter / (union + 1e-16)

#                 best_anchor_idx = torch.argmax(anchor_ious).item()
#                 scale_idx = best_anchor_idx // 3
#                 anchor_idx = best_anchor_idx % 3

#                 H, W = raw_outputs[scale_idx].shape[2:]

#                 gx, gy = cx * W, cy * H
#                 gi, gj = int(gx), int(gy)

#                 gi = max(0, min(gi, W - 1))
#                 gj = max(0, min(gj, H - 1))

#                 tx = gx - gi
#                 ty = gy - gj

#                 p_w = anchors[scale_idx, anchor_idx, 0] / img_w
#                 p_h = anchors[scale_idx, anchor_idx, 1] / img_h

#                 tw = torch.log(w / (p_w + 1e-16) + 1e-16)
#                 th = torch.log(h / (p_h + 1e-16) + 1e-16)

#                 target_objs[scale_idx][b, anchor_idx, gj, gi] = 1.0
#                 target_boxes[scale_idx][b, anchor_idx, gj, gi] = torch.tensor(
#                     [tx, ty, tw, th], device=device, dtype=torch.float32
#                 )

#                 if 0 <= cls_idx < self.num_classes:
#                     target_clss[scale_idx][b, anchor_idx, gj, gi, cls_idx] = 1.0

#         return target_objs, target_boxes, target_clss


# models/losses/yolo_loss.py
# loss/yolo_loss.py

# import math
# from typing import Dict, List, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def bbox_ciou(
#     box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7
# ) -> torch.Tensor:
#     """Calcul CIoU sur des boîtes (x1, y1, x2, y2) en pixels."""
#     b1_x1, b1_y1, b1_x2, b1_y2 = (
#         box1[..., 0],
#         box1[..., 1],
#         box1[..., 2],
#         box1[..., 3],
#     )
#     b2_x1, b2_y1, b2_x2, b2_y2 = (
#         box2[..., 0],
#         box2[..., 1],
#         box2[..., 2],
#         box2[..., 3],
#     )

#     inter_x1 = torch.max(b1_x1, b2_x1)
#     inter_y1 = torch.max(b1_y1, b2_y1)
#     inter_x2 = torch.min(b1_x2, b2_x2)
#     inter_y2 = torch.min(b1_y2, b2_y2)

#     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
#         inter_y2 - inter_y1, min=0
#     )

#     w1, h1 = torch.clamp(b1_x2 - b1_x1, min=eps), torch.clamp(
#         b1_y2 - b1_y1, min=eps
#     )
#     w2, h2 = torch.clamp(b2_x2 - b2_x1, min=eps), torch.clamp(
#         b2_y2 - b2_y1, min=eps
#     )
#     union_area = (w1 * h1) + (w2 * h2) - inter_area + eps

#     iou = inter_area / union_area

#     center_x1, center_y1 = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
#     center_x2, center_y2 = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0
#     center_distance = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

#     enclose_x1 = torch.min(b1_x1, b2_x1)
#     enclose_y1 = torch.min(b1_y1, b2_y1)
#     enclose_x2 = torch.max(b1_x2, b2_x2)
#     enclose_y2 = torch.max(b1_y2, b2_y2)
#     enclose_diagonal = (
#         (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
#     )

#     v = (4.0 / (math.pi**2)) * torch.pow(
#         torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
#     )
#     with torch.no_grad():
#         alpha = v / ((1.0 - iou) + v + eps)

#     return iou - (center_distance / enclose_diagonal) - (alpha * v)


# class YoloLoss(nn.Module):

#     def __init__(
#         self,
#         num_classes: int,
#         strides: List[int] = [8, 16, 32],
#         lambda_box: float = 7.5,
#         lambda_cls: float = 0.5,
#         # Plages de tailles d'objets recommandées par échelle [min_size, max_size]
#         scale_ranges: List[Tuple[float, float]] = [
#             (0, 64),
#             (64, 128),
#             (128, 1e5),
#         ],
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.strides = strides
#         self.lambda_box = lambda_box
#         self.lambda_cls = lambda_cls
#         self.scale_ranges = scale_ranges

#         self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

#     def forward(
#         self,
#         predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
#         targets: torch.Tensor,
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         first_key = next(iter(predictions))
#         device = predictions[first_key][0].device

#         tot_loss_box = torch.tensor(0.0, device=device)
#         tot_loss_cls = torch.tensor(0.0, device=device)
#         total_num_pos = 0

#         for scale_idx, (level_name, (cls_score, bbox_pred)) in enumerate(
#             predictions.items()
#         ):
#             stride = (
#                 self.strides[scale_idx]
#                 if scale_idx < len(self.strides)
#                 else 8 * (2**scale_idx)
#             )
#             min_size, max_size = self.scale_ranges[scale_idx]

#             cls_score = cls_score.permute(0, 2, 3, 1)  # [B, H, W, num_classes]
#             bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # [B, H, W, 4]
#             batch_size, grid_h, grid_w, _ = cls_score.shape

#             # 1. Construction des cibles filtrées par échelle
#             target_cls, target_box, pos_mask = self._build_targets(
#                 targets,
#                 batch_size,
#                 grid_h,
#                 grid_w,
#                 stride,
#                 min_size,
#                 max_size,
#                 device,
#             )

#             num_pos = pos_mask.sum().item()
#             total_num_pos += num_pos

#             # Loss Classification (calculée sur toutes les cellules)
#             tot_loss_cls += self.bce_loss(cls_score, target_cls).sum()

#             # Loss Bounding Box (calculée UNIQUEMENT sur les cellules positives)
#             if num_pos > 0:
#                 grid_y, grid_x = torch.meshgrid(
#                     torch.arange(grid_h, device=device),
#                     torch.arange(grid_w, device=device),
#                     indexing="ij",
#                 )
#                 grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)
#                 grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)

#                 # Décodage Anchor-Free (LTRB distances)
#                 dist_pred = F.softplus(bbox_pred) * stride

#                 pred_x1 = (grid_x + 0.5) * stride - dist_pred[..., 0]
#                 pred_y1 = (grid_y + 0.5) * stride - dist_pred[..., 1]
#                 pred_x2 = (grid_x + 0.5) * stride + dist_pred[..., 2]
#                 pred_y2 = (grid_y + 0.5) * stride + dist_pred[..., 3]
#                 pred_boxes = torch.stack(
#                     [pred_x1, pred_y1, pred_x2, pred_y2], dim=-1
#                 )

#                 # Extraction des cibles en pixels (x1, y1, x2, y2)
#                 target_boxes = target_box

#                 # CIoU Loss
#                 ciou = bbox_ciou(
#                     pred_boxes[pos_mask], target_boxes[pos_mask]
#                 )
#                 tot_loss_box += (1.0 - ciou).sum()

#         num_pos_clamp = max(total_num_pos, 1)

#         loss_box = (tot_loss_box / num_pos_clamp) * self.lambda_box
#         loss_cls = (tot_loss_cls / num_pos_clamp) * self.lambda_cls

#         total_loss = loss_cls + loss_box

#         return total_loss, {
#             "loss_total": total_loss.detach(),
#             "loss_cls": loss_cls.detach(),
#             "loss_box": loss_box.detach(),
#         }

#     def _build_targets(
#         self,
#         targets: torch.Tensor,
#         batch_size: int,
#         grid_h: int,
#         grid_w: int,
#         stride: int,
#         min_size: float,
#         max_size: float,
#         device: torch.device,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

#         target_cls = torch.zeros(
#             (batch_size, grid_h, grid_w, self.num_classes), device=device
#         )
#         target_box = torch.zeros(
#             (batch_size, grid_h, grid_w, 4), device=device
#         )
#         pos_mask = torch.zeros(
#             (batch_size, grid_h, grid_w), dtype=torch.bool, device=device
#         )

#         if targets is None or targets.numel() == 0:
#             return target_cls, target_box, pos_mask

#         # 1. Extraction des coordonnées
#         batch_idx = targets[:, 0].long().clamp(0, batch_size - 1)
#         cls_id = targets[:, 1].long().clamp(0, self.num_classes - 1)
#         xmin_pix, ymin_pix = targets[:, 2], targets[:, 3]
#         xmax_pix, ymax_pix = targets[:, 4], targets[:, 5]

#         w_pix = xmax_pix - xmin_pix
#         h_pix = ymax_pix - ymin_pix

#         # 2. Filtrage par échelle (seuls les objets de taille adaptée sont retenus pour ce niveau)
#         max_dim = torch.max(w_pix, h_pix)
#         scale_mask = (max_dim >= min_size) & (max_dim < max_size)

#         if not scale_mask.any():
#             return target_cls, target_box, pos_mask

#         # Filtrage des tenseurs
#         batch_idx = batch_idx[scale_mask]
#         cls_id = cls_id[scale_mask]
#         xmin_pix, ymin_pix = xmin_pix[scale_mask], ymin_pix[scale_mask]
#         xmax_pix, ymax_pix = xmax_pix[scale_mask], ymax_pix[scale_mask]

#         cx_pix = (xmin_pix + xmax_pix) / 2.0
#         cy_pix = (ymin_pix + ymax_pix) / 2.0

#         gx = (cx_pix / stride).long().clamp(0, grid_w - 1)
#         gy = (cy_pix / stride).long().clamp(0, grid_h - 1)

#         # 3. Remplissage
#         pos_mask[batch_idx, gy, gx] = True

#         target_box[batch_idx, gy, gx, 0] = xmin_pix
#         target_box[batch_idx, gy, gx, 1] = ymin_pix
#         target_box[batch_idx, gy, gx, 2] = xmax_pix
#         target_box[batch_idx, gy, gx, 3] = ymax_pix

#         target_cls[batch_idx, gy, gx, cls_id] = 1.0

#         return target_cls, target_box, pos_mask



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


# class YoloLoss(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         strides: List[int] = [8, 16, 32],
#         lambda_box: float = 7.5,
#         lambda_cls: float = 0.5,
#         scale_ranges: List[Tuple[float, float]] = [(0, 64), (64, 128), (128, 1e5)],
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.strides = strides
#         self.lambda_box = lambda_box
#         self.lambda_cls = lambda_cls
#         self.scale_ranges = scale_ranges
#         self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

#     def _format_targets(self, targets: Union[List, torch.Tensor], device: torch.device) -> torch.Tensor:
#         """Convertit une liste de cibles (format SSD List[Dict]) en Tensor 2D [N, 6]."""
#         if isinstance(targets, torch.Tensor):
#             return targets.to(device)

#         if not targets:
#             return torch.empty((0, 6), device=device)

#         formatted_list = []
#         for b_idx, t in enumerate(targets):
#             if isinstance(t, dict):
#                 boxes = t["boxes"].to(device)
#                 labels = t["labels"].to(device)
#                 if len(boxes) > 0:
#                     b_vec = torch.full((len(boxes), 1), b_idx, device=device, dtype=boxes.dtype)
#                     labels_vec = labels.unsqueeze(1).to(dtype=boxes.dtype)
#                     formatted_list.append(torch.cat([b_vec, labels_vec, boxes], dim=1))
#             elif isinstance(t, torch.Tensor) and len(t) > 0:
#                 b_vec = torch.full((len(t), 1), b_idx, device=device, dtype=t.dtype)
#                 formatted_list.append(torch.cat([b_vec, t.to(device)], dim=1))

#         if formatted_list:
#             return torch.cat(formatted_list, dim=0)
#         return torch.empty((0, 6), device=device)

#     def forward(
#         self,
#         predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
#         targets: Union[List[Dict], torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         device = next(iter(predictions.values()))[0].device
#         targets = self._format_targets(targets, device)

#         tot_loss_box = torch.tensor(0.0, device=device)
#         tot_loss_cls = torch.tensor(0.0, device=device)
#         total_num_pos = 0

#         for scale_idx, (_, (cls_score, bbox_pred)) in enumerate(predictions.items()):
#             stride = self.strides[scale_idx] if scale_idx < len(self.strides) else 8 * (2**scale_idx)
#             min_size, max_size = self.scale_ranges[scale_idx]

#             cls_score = cls_score.permute(0, 2, 3, 1)  # [B, H, W, C]
#             bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # [B, H, W, 4]
#             B, H, W, _ = cls_score.shape

#             target_cls, target_box, pos_mask = self._build_targets(
#                 targets, B, H, W, stride, min_size, max_size, device
#             )

#             num_pos = pos_mask.sum().item()
#             total_num_pos += num_pos

#             # Loss Classification
#             tot_loss_cls += self.bce_loss(cls_score, target_cls)

#             # Loss Bounding Box (décodage uniquement sur les cellules positives)
#             if num_pos > 0:
#                 b_idx, y_idx, x_idx = torch.where(pos_mask)

#                 dist_pred = F.softplus(bbox_pred[b_idx, y_idx, x_idx]) * stride
#                 pred_x1 = (x_idx + 0.5) * stride - dist_pred[:, 0]
#                 pred_y1 = (y_idx + 0.5) * stride - dist_pred[:, 1]
#                 pred_x2 = (x_idx + 0.5) * stride + dist_pred[:, 2]
#                 pred_y2 = (y_idx + 0.5) * stride + dist_pred[:, 3]

#                 pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
#                 target_boxes = target_box[b_idx, y_idx, x_idx]

#                 ciou = bbox_ciou(pred_boxes, target_boxes)
#                 tot_loss_box += (1.0 - ciou).sum()

#         num_pos_norm = max(total_num_pos, 1)
#         loss_box = (tot_loss_box / num_pos_norm) * self.lambda_box
#         loss_class = (tot_loss_cls / num_pos_norm) * self.lambda_cls

#         return {
#             "loss_box": loss_box,
#             "loss_class": loss_class,
#             "loss_obj": torch.tensor(0.0, device=device),
#             "loss_noobj": torch.tensor(0.0, device=device),
#         }

#     def _build_targets(
#         self,
#         targets: torch.Tensor,
#         batch_size: int,
#         grid_h: int,
#         grid_w: int,
#         stride: int,
#         min_size: float,
#         max_size: float,
#         device: torch.device,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

#         target_cls = torch.zeros((batch_size, grid_h, grid_w, self.num_classes), device=device)
#         target_box = torch.zeros((batch_size, grid_h, grid_w, 4), device=device)
#         pos_mask = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.bool, device=device)

#         if targets is None or targets.numel() == 0:
#             return target_cls, target_box, pos_mask

#         batch_idx = targets[:, 0].long().clamp(0, batch_size - 1)
#         cls_id = targets[:, 1].long().clamp(0, self.num_classes - 1)
#         xmin, ymin, xmax, ymax = targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5]

#         # Filtrage par échelle
#         max_dim = torch.max(xmax - xmin, ymax - ymin)
#         scale_mask = (max_dim >= min_size) & (max_dim < max_size)

#         if not scale_mask.any():
#             return target_cls, target_box, pos_mask

#         batch_idx, cls_id = batch_idx[scale_mask], cls_id[scale_mask]
#         xmin, ymin = xmin[scale_mask], ymin[scale_mask]
#         xmax, ymax = xmax[scale_mask], ymax[scale_mask]

#         gx = ((xmin + xmax) / (2.0 * stride)).long().clamp(0, grid_w - 1)
#         gy = ((ymin + ymax) / (2.0 * stride)).long().clamp(0, grid_h - 1)

#         pos_mask[batch_idx, gy, gx] = True
#         target_box[batch_idx, gy, gx] = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
#         target_cls[batch_idx, gy, gx, cls_id] = 1.0

#         return target_cls, target_box, pos_mask


import torch
import torch.nn as nn
import torch.nn.functional as F

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