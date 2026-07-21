import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from typing import List, Dict, Tuple

from core.config import Config
from core.model_base import Model


class Bottleneck(nn.Module):
    """
    Bloc Bottleneck standard de ResNet-50 (1x1 conv -> 3x3 conv -> 1x1 conv)
    L'expansion est de 4 (ex: 64 canaux d'entrée -> 256 canaux de sortie).
    """
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        # 1x1 Conv : Réduction du nombre de canaux
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 Conv : Extraction spatiale
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 Conv : Restauration/Expansion des canaux (planes * 4)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Si le stride > 1 ou si le nombre de canaux change, ajuster l'identité
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """ Backbone ResNet-50 construit manuellement """
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        # --- Stem Inicial ---
        # Image (3, 300, 300) -> Conv7x7 -> (64, 150, 150) -> MaxPool -> (64, 75, 75)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- ResNet Layers ---
        # Layer 1 : 3 blocs (Sortie: 256 canaux, 75x75)
        self.layer1 = self._make_layer(Bottleneck, planes=64, blocks=2, stride=1)
        
        # Layer 2 : 4 blocs (Sortie: 512 canaux, 38x38) -> SOURCE SSD 1
        self.layer2 = self._make_layer(Bottleneck, planes=128, blocks=2, stride=2)
        
        # Layer 3 : 6 blocs (Sortie: 1024 canaux, 19x19) -> SOURCE SSD 2
        self.layer3 = self._make_layer(Bottleneck, planes=256, blocks=2, stride=2)
        
        # Layer 4 : 3 blocs (Sortie: 2048 canaux, 10x10) -> SOURCE SSD 3
        self.layer4 = self._make_layer(Bottleneck, planes=512, blocks=2, stride=2)

        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        # Création du chemin de projection si résolution réduite ou canaux modifiés
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        # Premier bloc du groupe (peut réduire la taille spatiale via stride)
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        # Blocs suivants du même groupe (stride 1)
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """ Initialisation Kaiming Normal pour les Conv2d """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# =====================================================================
# 2. SSD PRIORS & MULTIBOX LOSS
# =====================================================================

def generate_ssd_priors() -> torch.Tensor:
    """ Génère les 8732 boîtes ancres par défaut pour SSD300 """
    feature_maps = [38, 19, 10, 5, 3, 1]
    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    priors = []
    for k, f in enumerate(feature_maps):
        for i in range(f):
            for j in range(f):
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                s_k = min_sizes[k] / 300.0
                priors.append([cx, cy, s_k, s_k])

                s_k_prime = (s_k * (max_sizes[k] / 300.0)) ** 0.5
                priors.append([cx, cy, s_k_prime, s_k_prime])

                for ar in aspect_ratios[k]:
                    priors.append([cx, cy, s_k * (ar ** 0.5), s_k / (ar ** 0.5)])
                    priors.append([cx, cy, s_k / (ar ** 0.5), s_k * (ar ** 0.5)])

    priors = torch.tensor(priors, dtype=torch.float32)
    return torch.clamp(priors, 0.0, 1.0)


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    A, B = box_a.size(0), box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


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


# =====================================================================
# 3. SSD DETECTOR INTEGRATING CUSTOM RESNET
# =====================================================================

class SSDResNetBackbone(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("priors", generate_ssd_priors())

        # Instanciation de notre ResNet Fait Maison
        self.resnet = ResNetBackbone()

        # Couches Supplémentaires SSD (Sources 4, 5 et 6)
        # Source 4 : 10x10 -> 5x5
        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Source 5 : 5x5 -> 3x3
        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        # Source 6 : 3x3 -> 1x1
        self.extra_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Têtes MultiBox (Localisation & Classification)
        num_boxes = [4, 6, 6, 6, 4, 4]
        in_channels = [512, 1024, 2048, 512, 256, 256]

        self.loc_layers = nn.ModuleList([
            nn.Conv2d(in_ch, n_box * 4, kernel_size=3, padding=1)
            for in_ch, n_box in zip(in_channels, num_boxes)
        ])

        self.cls_layers = nn.ModuleList([
            nn.Conv2d(in_ch, n_box * num_classes, kernel_size=3, padding=1)
            for in_ch, n_box in zip(in_channels, num_boxes)
        ])

        self.criterion = SSDMultiBoxLoss(num_classes=num_classes)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        loc_preds, cls_preds = self.compute_raw_predictions(images)

        if self.training and targets is not None:
            return self.criterion(loc_preds, cls_preds, targets, self.priors)

        return self.predict_decoded(loc_preds, cls_preds)

    def compute_raw_predictions(self, x: torch.Tensor):
        sources = []

        # Stream principal ResNet-50
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        
        x = self.resnet.layer2(x)
        sources.append(x)  # Source 1 : 38x38 (512 ch)

        x = self.resnet.layer3(x)
        sources.append(x)  # Source 2 : 19x19 (1024 ch)

        x = self.resnet.layer4(x)
        sources.append(x)  # Source 3 : 10x10 (2048 ch)

        # Layers supplémentaires SSD
        x = self.extra_conv1(x)
        sources.append(x)  # Source 4 : 5x5 (512 ch)

        x = self.extra_conv2(x)
        sources.append(x)  # Source 5 : 3x3 (256 ch)

        x = self.extra_conv3(x)
        sources.append(x)  # Source 6 : 1x1 (256 ch)

        # Pass passage dans les têtes MultiBox
        loc_outputs, cls_outputs = [], []
        for (feat, l_conv, c_conv) in zip(sources, self.loc_layers, self.cls_layers):
            loc_outputs.append(l_conv(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1))
            cls_outputs.append(c_conv(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1))

        loc_flat = torch.cat(loc_outputs, dim=1)
        cls_flat = torch.cat(cls_outputs, dim=1)

        loc_final = loc_flat.view(loc_flat.size(0), -1, 4)
        cls_final = cls_flat.view(cls_flat.size(0), -1, self.num_classes)

        return loc_final, cls_final

    def predict_decoded(self, loc_preds: torch.Tensor, cls_preds: torch.Tensor, confidence_threshold: float = 0.15, iou_threshold: float = 0.45, img_size=(300, 300)) -> List[Dict[str, torch.Tensor]]:
        B = loc_preds.size(0)
        img_w, img_h = img_size
        device = loc_preds.device

        cx = loc_preds[..., 0] * 0.1 * self.priors[:, 2] + self.priors[:, 0]
        cy = loc_preds[..., 1] * 0.1 * self.priors[:, 3] + self.priors[:, 1]
        w  = torch.exp(loc_preds[..., 2] * 0.2) * self.priors[:, 2]
        h  = torch.exp(loc_preds[..., 3] * 0.2) * self.priors[:, 3]

        decoded_boxes = torch.zeros_like(loc_preds)
        decoded_boxes[..., 0] = (cx - w / 2.0) * img_w
        decoded_boxes[..., 1] = (cy - h / 2.0) * img_h
        decoded_boxes[..., 2] = (cx + w / 2.0) * img_w
        decoded_boxes[..., 3] = (cy + h / 2.0) * img_h
        decoded_boxes = torch.clamp(decoded_boxes, min=0.0, max=float(max(img_w, img_h)))

        cls_probs = F.softmax(cls_preds, dim=-1)

        predictions_list = []
        for b in range(B):
            b_boxes, b_scores, b_labels = [], [], []

            for c in range(1, self.num_classes):
                scores = cls_probs[b, :, c]
                mask = scores > confidence_threshold

                if mask.sum() == 0:
                    continue

                b_boxes.append(decoded_boxes[b, mask])
                b_scores.append(scores[mask])
                b_labels.append(torch.full((mask.sum(),), fill_value=c, dtype=torch.long, device=device))

            if len(b_boxes) > 0:
                t_boxes = torch.cat(b_boxes, dim=0)
                t_scores = torch.cat(b_scores, dim=0)
                t_labels = torch.cat(b_labels, dim=0)

                keep_nms = batched_nms(t_boxes, t_scores, t_labels, iou_threshold)

                predictions_list.append({
                    "boxes": t_boxes[keep_nms],
                    "scores": t_scores[keep_nms],
                    "labels": t_labels[keep_nms]
                })
            else:
                predictions_list.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.zeros(0, dtype=torch.float32, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device)
                })

        return predictions_list


class SSDResNetModel(Model):
    """ Modèle SSD-ResNet50 """
    def __init__(self, config: Config, score_thresh: float = 0.15, iou_thresh: float = 0.45, **kwargs):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        super().__init__(config=config, **kwargs)

    @property
    def name(self) -> str:
        return "SSD300_ResNet"

    def build_model(self) -> nn.Module:
        print("[INFO] Building SSD300 with ResNet")
        return SSDResNetBackbone(num_classes=self.num_classes)

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "input_size": (300, 300),
            "architecture": "SSD_ResNet"
        }