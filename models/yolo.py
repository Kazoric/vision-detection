import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from typing import List, Dict, Optional, Tuple

from core.config import Config
from core.model_base import Model
# Assurez-vous d'avoir une perte YOLO adaptée (ex: YOLOLoss avec CIoU + BCE)
from loss.yolo_loss import YOLOLoss


class ConvBlock(nn.Module):
    """ Bloc Conv + BatchNorm + LeakyReLU classique pour le neck YOLO """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """ Bloc ResNet Bottleneck standard """
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """ Backbone ResNet qui extrait les cartes C3, C4, C5 """
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, planes=64, blocks=2, stride=1)   # C2
        self.layer2 = self._make_layer(Bottleneck, planes=128, blocks=2, stride=2)  # C3 (512 ch)
        self.layer3 = self._make_layer(Bottleneck, planes=256, blocks=2, stride=2)  # C4 (1024 ch)
        self.layer4 = self._make_layer(Bottleneck, planes=512, blocks=2, stride=2)  # C5 (2048 ch)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        c3 = self.layer2(x)  # Échelle 1/8
        c4 = self.layer3(c3) # Échelle 1/16
        c5 = self.layer4(c4) # Échelle 1/32
        return c3, c4, c5


class YOLOResNetModel(Model):
    """
    Détecteur YOLOv3-style construit sur un backbone ResNet50/Bottleneck.
    Exécute une prédiction multi-échelle sur 3 niveaux avec un FPN (Feature Pyramid Network).
    """

    def __init__(
        self,
        config: Config,
        score_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
        **kwargs
    ):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.num_anchors_per_scale = 3

        # Ancres normalisées (largeur, hauteur) par échelle [Petite, Moyenne, Grande]
        # Adaptables selon votre jeu de données
        self.default_anchors = [
            [(10, 13), (16, 30), (33, 23)],      # Ancres pour la plus grande résolution (C3)
            [(30, 61), (62, 45), (59, 119)],     # Ancres pour la résolution moyenne (C4)
            [(116, 90), (156, 198), (373, 326)]  # Ancres pour la plus petite résolution (C5)
        ]

        super().__init__(config=config, device=device)

    @property
    def name(self) -> str:
        return "YOLO_ResNet"

    def _build_architecture(self) -> None:
        self.backbone = ResNetBackbone()

        # Nombre de valeurs prédites par ancre : [x, y, w, h, obj_score] + C classes
        self.out_channels_per_anchor = 5 + self.num_classes - 1

        # --- FPN Neck & Têtes de détection ---
        # Branche C5 (1/32)
        self.conv_c5 = ConvBlock(2048, 512, kernel_size=1)
        self.head_c5 = nn.Conv2d(512, self.num_anchors_per_scale * self.out_channels_per_anchor, kernel_size=1)

        # Fusion C5 -> C4
        self.up_c5 = ConvBlock(512, 256, kernel_size=1)
        self.conv_c4 = ConvBlock(1024 + 256, 256, kernel_size=3, padding=1)
        self.head_c4 = nn.Conv2d(256, self.num_anchors_per_scale * self.out_channels_per_anchor, kernel_size=1)

        # Fusion C4 -> C3
        self.up_c4 = ConvBlock(256, 128, kernel_size=1)
        self.conv_c3 = ConvBlock(512 + 128, 128, kernel_size=3, padding=1)
        self.head_c3 = nn.Conv2d(128, self.num_anchors_per_scale * self.out_channels_per_anchor, kernel_size=1)

        # Enregistrement des ancres sous forme de buffers PyTorch
        self.register_buffer("anchors", torch.tensor(self.default_anchors, dtype=torch.float32))

        self.criterion = YOLOLoss(num_classes=self.num_classes - 1)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        # Extraction des caractéristiques multi-échelles
        c3, c4, c5 = self.backbone(images)

        # Top-down FPN
        p5 = self.conv_c5(c5)
        out_p5 = self.head_c5(p5)

        p5_up = F.interpolate(self.up_c5(p5), size=c4.shape[2:], mode="nearest")
        p4 = self.conv_c4(torch.cat([c4, p5_up], dim=1))
        out_p4 = self.head_c4(p4)

        p4_up = F.interpolate(self.up_c4(p4), size=c3.shape[2:], mode="nearest")
        p3 = self.conv_c3(torch.cat([c3, p4_up], dim=1))
        out_p3 = self.head_c3(p3)

        raw_outputs = [out_p3, out_p4, out_p5]

        if self.training and targets is not None:
            # Transmettre directement les prédictions brutes au Loss YOLO
            return self.criterion(raw_outputs, targets, self.anchors)
            pass

        return self.predict_decoded(
            raw_outputs,
            image_shape=images.shape[2:],
            confidence_threshold=self.score_thresh,
            iou_threshold=self.iou_thresh
        )

    def predict_decoded(
        self,
        raw_outputs: List[torch.Tensor],
        image_shape: Tuple[int, int],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, torch.Tensor]]:
        """ Decoding des sorties YOLO + NMS par image """
        batch_size = raw_outputs[0].size(0)
        img_h, img_w = image_shape
        device = raw_outputs[0].device

        all_decoded_boxes = []
        all_scores = []
        all_labels = []

        for scale_idx, raw_out in enumerate(raw_outputs):
            B, _, H, W = raw_out.shape
            
            # Reshape: (B, num_anchors, H, W, 5 + num_classes)
            predictions = raw_out.view(
                B, self.num_anchors_per_scale, self.out_channels_per_anchor, H, W
            ).permute(0, 1, 3, 4, 2).contiguous()

            # Activation des prédictions
            tx = torch.sigmoid(predictions[..., 0])
            ty = torch.sigmoid(predictions[..., 1])
            tw = predictions[..., 2]
            th = predictions[..., 3]
            objectness = torch.sigmoid(predictions[..., 4])
            class_probs = torch.softmax(predictions[..., 5:], dim=-1)

            # Grille de coordonnées (grid_x, grid_y)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )

            # Ancres de l'échelle courante
            scale_anchors = self.anchors[scale_idx].to(device) # (3, 2)
            anchor_w = scale_anchors[:, 0].view(1, 3, 1, 1)
            anchor_h = scale_anchors[:, 1].view(1, 3, 1, 1)

            # Décodage des coordonnées réelles (normalisées puis aux dimensions de l'image)
            cx = (tx + grid_x) / W
            cy = (ty + grid_y) / H
            pw = (torch.exp(tw) * anchor_w) / img_w
            ph = (torch.exp(th) * anchor_h) / img_h

            x1 = (cx - pw / 2.0) * img_w
            y1 = (cy - ph / 2.0) * img_h
            x2 = (cx + pw / 2.0) * img_w
            y2 = (cy + ph / 2.0) * img_h

            boxes = torch.stack([x1, y1, x2, y2], dim=-1) # (B, A, H, W, 4)

            # Calcul du score final : P(Object) * P(Class)
            scores, labels = torch.max(class_probs, dim=-1)
            final_scores = objectness * scores

            # Aplatir les dimensions spatiales et d'ancres pour l'évaluation
            all_decoded_boxes.append(boxes.view(B, -1, 4))
            all_scores.append(final_scores.view(B, -1))
            all_labels.append(labels.view(B, -1))

        # Concaténation des prédictions de toutes les échelles
        all_decoded_boxes = torch.cat(all_decoded_boxes, dim=1) # (B, Total_Anchors, 4)
        all_scores = torch.cat(all_scores, dim=1)               # (B, Total_Anchors)
        all_labels = torch.cat(all_labels, dim=1)               # (B, Total_Anchors)

        predictions_list = []
        for b in range(batch_size):
            b_boxes = all_decoded_boxes[b]
            b_scores = all_scores[b]
            b_labels = all_labels[b]

            # Filtrage selon le seuil de confiance
            mask = b_scores > confidence_threshold
            b_boxes = b_boxes[mask]
            b_scores = b_scores[mask]
            b_labels = b_labels[mask]

            if b_boxes.size(0) == 0:
                predictions_list.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.zeros(0, dtype=torch.float32, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device)
                })
                continue

            # NMS supprimant les chevauchements
            keep = batched_nms(b_boxes, b_scores, b_labels, iou_threshold)

            predictions_list.append({
                "boxes": b_boxes[keep],
                "scores": b_scores[keep],
                "labels": b_labels[keep]
            })

        return predictions_list

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "architecture": "YOLO_ResNet"
        }

# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.ops import nms
# from typing import List, Dict, Optional

# from core.config import Config
# from core.model_base import Model


# class YOLOHead(nn.Module):
#     """Tête de prédiction YOLO projetant sur une grille S x S."""
#     def __init__(self, in_channels: int, num_classes: int, grid_size: int = 10):
#         super().__init__()
#         self.grid_size = grid_size
#         self.num_classes = num_classes
#         self.out_channels = 1 + 4 + num_classes  # [obj_score, xc, yc, w, h, class_0, ...]

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(512, self.out_channels, kernel_size=1)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.conv(x)


# class SimplifiedYoloLoss(nn.Module):
#     """Module de perte YOLO retournant un dictionnaire de pertes."""
#     def __init__(self, num_classes: int, coord_weight: float = 10.0, noobj_weight: float = 0.1):
#         super().__init__()
#         self.num_classes = num_classes
#         self.coord_weight = coord_weight
#         self.noobj_weight = noobj_weight

#         self.mse = nn.MSELoss(reduction="sum")
#         self.bce = nn.BCEWithLogitsLoss(reduction="sum")
#         self.ce = nn.CrossEntropyLoss(reduction="sum")

#     def forward(self, predictions: torch.Tensor, targets: list, img_size=(300, 300)) -> Dict[str, torch.Tensor]:
#         B, _, S_h, S_w = predictions.shape
#         img_w, img_h = img_size

#         obj_mask = torch.zeros((B, S_h, S_w), device=predictions.device)
#         target_boxes = torch.zeros((B, 4, S_h, S_w), device=predictions.device)
#         target_classes = torch.zeros((B, S_h, S_w), dtype=torch.long, device=predictions.device)

#         for b in range(len(targets)):
#             boxes = targets[b]["boxes"]
#             labels = targets[b]["labels"]

#             for idx in range(len(boxes)):
#                 xmin, ymin, xmax, ymax = boxes[idx]
#                 label = labels[idx]

#                 xc = (xmin + xmax) / 2.0 / img_w
#                 yc = (ymin + ymax) / 2.0 / img_h
#                 w = (xmax - xmin) / img_w
#                 h = (ymax - ymin) / img_h

#                 i = min(max(int(xc * S_w), 0), S_w - 1)
#                 j = min(max(int(yc * S_h), 0), S_h - 1)

#                 obj_mask[b, j, i] = 1.0
#                 target_boxes[b, :, j, i] = torch.tensor([xc, yc, w, h], device=predictions.device)
#                 target_classes[b, j, i] = label

#         pred_obj = predictions[:, 0, :, :]
#         # Sigmoïde obligatoire pour contraindre les prédictions de boîtes entre 0 et 1
#         pred_boxes = torch.sigmoid(predictions[:, 1:5, :, :])
#         pred_classes = predictions[:, 5:, :, :]

#         has_obj = (obj_mask == 1.0)
#         no_obj = (obj_mask == 0.0)

#         loss_obj = self.bce(pred_obj[has_obj], obj_mask[has_obj]) / B
#         loss_noobj = (self.bce(pred_obj[no_obj], obj_mask[no_obj]) * self.noobj_weight) / B

#         if has_obj.sum() > 0:
#             p_boxes = pred_boxes.permute(0, 2, 3, 1)[has_obj]
#             t_boxes = target_boxes.permute(0, 2, 3, 1)[has_obj]
#             loss_box = (self.mse(p_boxes, t_boxes) * self.coord_weight) / B

#             p_classes = pred_classes.permute(0, 2, 3, 1)[has_obj]
#             t_classes = target_classes[has_obj]
#             loss_class = self.ce(p_classes, t_classes) / B
#         else:
#             loss_box = torch.tensor(0.0, device=predictions.device, requires_grad=True)
#             loss_class = torch.tensor(0.0, device=predictions.device, requires_grad=True)

#         return {
#             "loss_obj": loss_obj,
#             "loss_noobj": loss_noobj,
#             "loss_box": loss_box,
#             "loss_class": loss_class
#         }


# class YOLOModel(Model):
#     """Détecteur YOLO basé sur un backbone ResNet18."""

#     def __init__(
#         self,
#         config: Config,
#         grid_size: int = 10,
#         dropout: float = 0.1,
#         score_thresh: float = 0.15,
#         iou_thresh: float = 0.45,
#         device: Optional[str] = None,
#         **kwargs
#     ):
#         self.grid_size = grid_size
#         self.dropout = dropout
#         self.score_thresh = score_thresh
#         self.iou_thresh = iou_thresh
#         self.num_classes = getattr(config, "num_classes", kwargs.get("num_classes", 20))
        
#         super().__init__(config=config, device=device)

#     @property
#     def name(self) -> str:
#         return f"YOLO_ResNet18_Grid{self.grid_size}"

#     def _build_architecture(self) -> None:
#         """Instancie le backbone ResNet18 et la tête de détection."""
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         # On retire Average Pooling et FC Layer (les 2 derniers éléments)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         self.dropout_layer = nn.Dropout2d(p=self.dropout) if self.dropout > 0 else nn.Identity()
        
#         # ResNet18 produit 512 canaux en sortie de layer4
#         self.head = YOLOHead(in_channels=512, num_classes=self.num_classes, grid_size=self.grid_size)
#         self.criterion = SimplifiedYoloLoss(num_classes=self.num_classes)

#         self._initialize_custom_weights()

#     def _initialize_custom_weights(self) -> None:
#         for m in self.head.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, images, targets=None):
#         if isinstance(images, list):
#             images = torch.stack(images)

#         features = self.backbone(images)
#         features = self.dropout_layer(features)
#         predictions = self.head(features)

#         # Mode entraînement : renvoie le dictionnaire de pertes
#         if self.training and targets is not None:
#             return self.criterion(predictions, targets)

#         # Mode évaluation : renvoie les détections décodées
#         return self.predict_decoded(
#             predictions,
#             confidence_threshold=self.score_thresh,
#             iou_threshold=self.iou_thresh
#         )

#     def predict_decoded(
#         self,
#         preds: torch.Tensor,
#         confidence_threshold: float = 0.15,
#         iou_threshold: float = 0.45,
#         img_size=(300, 300)
#     ) -> List[Dict[str, torch.Tensor]]:
#         """Décode les prédictions brutes en coordonnées d'image avec NMS (vectorisé sur GPU)."""
#         B, _, S_h, S_w = preds.shape
#         img_w, img_h = img_size
#         device = preds.device

#         # Application des activations
#         obj_scores = torch.sigmoid(preds[:, 0, :, :])                # (B, S_h, S_w)
#         boxes = torch.sigmoid(preds[:, 1:5, :, :])                   # (B, 4, S_h, S_w)
#         class_probs = torch.softmax(preds[:, 5:, :, :], dim=1)        # (B, num_classes, S_h, S_w)

#         predictions_list = []

#         for b in range(B):
#             keep_mask = obj_scores[b] > confidence_threshold

#             if not keep_mask.any():
#                 predictions_list.append({
#                     "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
#                     "scores": torch.zeros(0, dtype=torch.float32, device=device),
#                     "labels": torch.zeros(0, dtype=torch.long, device=device)
#                 })
#                 continue

#             scores = obj_scores[b][keep_mask]
#             b_boxes = boxes[b].permute(1, 2, 0)[keep_mask]           # (N, 4) -> [xc, yc, w, h]
#             b_classes = class_probs[b].permute(1, 2, 0)[keep_mask]     # (N, num_classes)

#             class_scores, labels = torch.max(b_classes, dim=-1)

#             # Conversion (xc, yc, w, h) -> (xmin, ymin, xmax, ymax) en pixels
#             xc = b_boxes[:, 0] * img_w
#             yc = b_boxes[:, 1] * img_h
#             w  = b_boxes[:, 2] * img_w
#             h  = b_boxes[:, 3] * img_h

#             xmin = (xc - w / 2.0).clamp(min=0.0, max=float(img_w))
#             ymin = (yc - h / 2.0).clamp(min=0.0, max=float(img_h))
#             xmax = (xc + w / 2.0).clamp(min=0.0, max=float(img_w))
#             ymax = (yc + h / 2.0).clamp(min=0.0, max=float(img_h))

#             # Filtrage des boîtes invalides (aire nulle ou négative)
#             valid_area = (xmax > xmin) & (ymax > ymin)
#             if not valid_area.any():
#                 predictions_list.append({
#                     "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
#                     "scores": torch.zeros(0, dtype=torch.float32, device=device),
#                     "labels": torch.zeros(0, dtype=torch.long, device=device)
#                 })
#                 continue

#             decoded_boxes = torch.stack([xmin[valid_area], ymin[valid_area], xmax[valid_area], ymax[valid_area]], dim=1)
#             valid_scores = scores[valid_area]
#             valid_labels = labels[valid_area]

#             # Suppressions des doublons (NMS)
#             keep_nms = nms(decoded_boxes, valid_scores, iou_threshold)

#             predictions_list.append({
#                 "boxes": decoded_boxes[keep_nms],
#                 "scores": valid_scores[keep_nms],
#                 "labels": valid_labels[keep_nms]
#             })

#         return predictions_list

#     def get_model_specific_params(self) -> dict:
#         return {
#             "num_classes": self.num_classes,
#             "grid_size": self.grid_size,
#             "dropout": self.dropout,
#             "score_thresh": self.score_thresh,
#             "iou_thresh": self.iou_thresh,
#             "input_size": (300, 300),
#             "architecture": "YOLO_ResNet18"
#         }