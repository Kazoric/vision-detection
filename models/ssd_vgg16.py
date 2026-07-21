import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List, Dict, Tuple, Optional

from core.config import Config
from core.model_base import Model


class L2Norm(nn.Module):
    """
    L2 Normalization essential for the conv4_3 layer.
    """
    def __init__(self, n_channels, scale=20):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant_(self.gamma, scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        x = torch.div(x, norm)
        return self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x

def generate_ssd_priors() -> torch.Tensor:
    """ Generates default 8732 anchor boxes for SSD300 in [cx, cy, w, h] normalized coordinates """
    feature_maps = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    aspect_ratios = [
        [2],           # 38x38 (4 boîtes)
        [2, 3],        # 19x19 (6 boîtes)
        [2, 3],        # 10x10 (6 boîtes)
        [2, 3],        # 5x5   (6 boîtes)
        [2],           # 3x3   (4 boîtes)
        [2]            # 1x1   (4 boîtes)
    ]

    priors = []
    for k, f in enumerate(feature_maps):
        for i in range(f):
            for j in range(f):
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                # Boîte de taille min
                s_k = min_sizes[k] / 300.0
                priors.append([cx, cy, s_k, s_k])

                # Boîte supplémentaire s'_k = sqrt(s_k * s_{k+1})
                s_k_prime = (s_k * (max_sizes[k] / 300.0)) ** 0.5
                priors.append([cx, cy, s_k_prime, s_k_prime])

                # Ratio d'aspects
                for ar in aspect_ratios[k]:
                    priors.append([cx, cy, s_k * (ar ** 0.5), s_k / (ar ** 0.5)])
                    priors.append([cx, cy, s_k / (ar ** 0.5), s_k * (ar ** 0.5)])

    priors = torch.tensor(priors, dtype=torch.float32)
    return torch.clamp(priors, 0.0, 1.0)


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """ Intersection entre 2 ensembles de boîtes [xmin, ymin, xmax, ymax] """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """ Calcul de l'IoU entre box_a et box_b """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union

class SSDMultiBoxLoss(nn.Module):
    """ Perte SSD MultiBox compatible avec la structure du Trainer YOLO """
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

        # Conversion des priors [cx, cy, w, h] vers [xmin, ymin, xmax, ymax]
        priors_corner = torch.zeros_like(priors)
        priors_corner[:, :2] = priors[:, :2] - priors[:, 2:] / 2.0
        priors_corner[:, 2:] = priors[:, :2] + priors[:, 2:] / 2.0

        loc_targets = torch.zeros_like(loc_preds)
        cls_targets = torch.zeros((B, num_priors), dtype=torch.long, device=device)

        # Matching Ground Truth / Priors pour chaque image du batch
        for b in range(B):
            gt_boxes = targets[b]["boxes"]
            labels = targets[b]["labels"]

            if len(gt_boxes) == 0:
                continue

            # Normalisation des coordonnées [xmin, ymin, xmax, ymax] dans [0, 1]
            gt_boxes_norm = gt_boxes.clone()
            gt_boxes_norm[:, [0, 2]] /= float(img_w)
            gt_boxes_norm[:, [1, 3]] /= float(img_h)

            overlaps = jaccard_iou(gt_boxes_norm, priors_corner) # [num_gt, num_priors]
            
            best_prior_overlap, best_prior_idx = overlaps.max(1)
            best_truth_overlap, best_truth_idx = overlaps.max(0)

            # Assurer que chaque GT a au moins 1 prior associé
            for idx, prior_idx in enumerate(best_prior_idx):
                best_truth_idx[prior_idx] = idx
                best_truth_overlap[prior_idx] = 2.0

            # Label 0 est réservé au Background dans SSD
            assigned_labels = labels[best_truth_idx]
            assigned_labels[best_truth_overlap < self.overlap_thresh] = 0
            cls_targets[b] = assigned_labels

            # Encodage des offsets de localisation
            matched_gt = gt_boxes_norm[best_truth_idx]
            gt_cxcy = (matched_gt[:, :2] + matched_gt[:, 2:]) / 2.0
            gt_wh = matched_gt[:, 2:] - matched_gt[:, :2]
            
            g_hat_cxcy = (gt_cxcy - priors[:, :2]) / (priors[:, 2:] * self.variances[0])
            g_hat_wh = torch.log(gt_wh / priors[:, 2:] + 1e-5) / self.variances[1]
            loc_targets[b] = torch.cat([g_hat_cxcy, g_hat_wh], dim=1)

        pos_mask = cls_targets > 0 # [B, num_priors]
        num_pos = pos_mask.sum()

        # 1. Smooth L1 Localization Loss
        if num_pos > 0:
            loss_box = F.smooth_l1_loss(loc_preds[pos_mask], loc_targets[pos_mask], reduction='sum') / B
        else:
            loss_box = torch.tensor(0.0, device=device, requires_grad=True)

        # 2. Hard Negative Mining pour la Classification Loss (Ratio Neg/Pos = 3:1)
        cls_loss_all = F.cross_entropy(cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduction='none')
        cls_loss_all = cls_loss_all.view(B, num_priors)

        # Filtre des positifs
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[pos_mask] = 0.0 

        _, loss_idx = cls_loss_neg.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos_per_img = pos_mask.sum(1, keepdim=True)
        num_neg_per_img = torch.clamp(num_pos_per_img * self.neg_pos_ratio, max=num_priors - 1).long()
        neg_mask = idx_rank < num_neg_per_img

        # Perte Totale de Classification (Positifs + Negatifs sélectionnés)
        final_mask = pos_mask | neg_mask
        loss_class = cls_loss_all[final_mask].sum() / B

        return {
            "loss_box": loss_box,
            "loss_class": loss_class,
            "loss_obj": torch.tensor(0.0, device=device),    # Maintient la compatibilité
            "loss_noobj": torch.tensor(0.0, device=device)  # Maintient la compatibilité
        }

class SSDVGG16Backbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)
        self.register_buffer("priors", generate_ssd_priors())

        # ------------------------------------------------------------
        # VGG16 architecture
        # ------------------------------------------------------------
        # Block 1 : Input 300x300 -> Output 150x150
        self.vgg_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc 2 : Input 150x150 -> Output 75x75
        self.vgg_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc 3 : Input 75x75 -> Output 38x38
        self.vgg_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
        )

        # Bloc 4 : Input 38x38
        self.vgg_block4_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1) # Output 1
        )
        self.L2Norm = L2Norm(512, scale=20)
        # Output 19x19
        self.vgg_block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloc 5 : Input 19x19 -> Output 19x19
        self.vgg_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # FC layers to convolutions layers
        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1) # Output 2

        # ------------------------------------------------------------
        # SSD layers
        # ------------------------------------------------------------
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # Output 3 (10x10)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Output 4 (5x5)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # Output 5 (3x3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # Output 6 (1x1)

        # ------------------------------------------------------------
        # PREDICTION HEADS (MultiBox)
        # ------------------------------------------------------------
        num_boxes = [4, 6, 6, 6, 4, 4]

        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512,  num_boxes[0] * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, num_boxes[1] * 4, kernel_size=3, padding=1),
            nn.Conv2d(512,  num_boxes[2] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256,  num_boxes[3] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256,  num_boxes[4] * 4, kernel_size=3, padding=1),
            nn.Conv2d(256,  num_boxes[5] * 4, kernel_size=3, padding=1)
        ])

        self.cls_layers = nn.ModuleList([
            nn.Conv2d(512,  num_boxes[0] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, num_boxes[1] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512,  num_boxes[2] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256,  num_boxes[3] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256,  num_boxes[4] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256,  num_boxes[5] * num_classes, kernel_size=3, padding=1)
        ])

        self.criterion = SSDMultiBoxLoss(num_classes=num_classes)
        self._init_weights()

    def _init_weights(self):
        """Kaiming/He initialization for new layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        loc_preds, cls_preds = self.compute_raw_predictions(images)

        # Mode entraînement : renvoie le dictionnaire de pertes
        if self.training and targets is not None:
            return self.criterion(loc_preds, cls_preds, targets, self.priors)

        # Mode évaluation : renvoie les prédictions décodées identiques à YOLO
        return self.predict_decoded(loc_preds, cls_preds)
    
    def compute_raw_predictions(self, x):
        sources = list()
        loc_outputs = list()
        cls_outputs = list()

        # --- VGG Forward ---
        x = self.vgg_block1(x)
        x = self.vgg_block2(x)
        x = self.vgg_block3(x)
        
        # Intermediate extraction of conv4_3
        x = self.vgg_block4_conv(x)
        conv4_3_activated = self.relu(x)
        sources.append(self.L2Norm(conv4_3_activated)) # Source 1 (38x38)
        
        x = self.vgg_block4_pool(conv4_3_activated)
        x = self.vgg_block5(x)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        sources.append(x) # Source 2 (19x19)

        # --- SSD Layers Forward ---
        x = self.relu(self.conv8_1(x))
        x = self.relu(self.conv8_2(x))
        sources.append(x) # Source 3 (10x10)

        x = self.relu(self.conv9_1(x))
        x = self.relu(self.conv9_2(x))
        sources.append(x) # Source 4 (5x5)

        x = self.relu(self.conv10_1(x))
        x = self.relu(self.conv10_2(x))
        sources.append(x) # Source 5 (3x3)

        x = self.relu(self.conv11_1(x))
        x = self.relu(self.conv11_2(x))
        sources.append(x) # Source 6 (1x1)

        # --- MultiBox Heads ---
        for (feat, l_conv, c_conv) in zip(sources, self.loc_layers, self.cls_layers):
            loc_outputs.append(l_conv(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1))
            cls_outputs.append(c_conv(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1))

        loc_flat = torch.cat(loc_outputs, dim=1)
        cls_flat = torch.cat(cls_outputs, dim=1)

        loc_final = loc_flat.view(loc_flat.size(0), -1, 4)
        cls_final = cls_flat.view(cls_flat.size(0), -1, self.num_classes)

        return loc_final, cls_final
    
    def predict_decoded(self, loc_preds: torch.Tensor, cls_preds: torch.Tensor, confidence_threshold: float = 0.15, iou_threshold: float = 0.45, img_size=(300, 300)) -> List[Dict[str, torch.Tensor]]:
        """ Décode les tenseurs bruts SSD en coordonnées d'images clippées avec NMS par classe """
        B = loc_preds.size(0)
        img_w, img_h = img_size
        device = loc_preds.device

        # Décodage des boîtes [cx, cy, w, h] vers [xmin, ymin, xmax, ymax]
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

        # Probabilités des classes via Softmax
        cls_probs = F.softmax(cls_preds, dim=-1)

        predictions_list = []
        for b in range(B):
            b_boxes, b_scores, b_labels = [], [], []

            # Filtrage classe par classe (en ignorant la classe 0 : background)
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

                # NMS Batched par classe
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

class SSDVGG16Model(Model):
    """ Wrapper SSDVGG16Model inheriting from your base Model class """
    def __init__(self, config: Config, score_thresh: float = 0.15, iou_thresh: float = 0.45, **kwargs):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        # SSD300 also uses multi-scale feature maps, so no grid_size needed here either
        super().__init__(config=config, **kwargs)

    @property
    def name(self) -> str:
        return "SSD300_VGG16"

    def build_model(self) -> nn.Module:
        print(f"[INFO] Building SSD300 VGG16 Network (Native resolution: 300x300)...")

        model = SSDVGG16Backbone(num_classes=self.num_classes)

        # self.post_processor = SSDPostProcessor(
        #     score_thresh=self.score_thresh, 
        #     iou_thresh=self.iou_thresh
        # )

        # model.priors = generate_ssd_priors().to(config.device)
        
        return model
    
    # def setup_post_processing(self):
    #     # On instancie le processeur dédié
    #     self.post_processor = SSDPostProcessor(
    #         score_thresh=self.score_thresh, 
    #         iou_thresh=self.iou_thresh
    #     )

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "input_size": (300, 300),  # The original VGG16 SSD is calibrated for 300x300
            "architecture": "SSD_VGG16"
        }
    

from torchvision.ops import batched_nms

class SSDPostProcessor:
    def __init__(self, score_thresh: float = 0.15, iou_thresh: float = 0.45, background_label: int = 0):
        """
        Args:
            score_thresh (float): Minimum confidence score to consider a detection valid.
            iou_thresh (float): Overlap threshold for Non-Maximum Suppression (NMS).
            background_label (int): Index of the background class (usually 0 in SSD).
        """
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.background_label = background_label
        
        # Standard SSD variances used during encoding/decoding stabilization
        self.variances = [0.1, 0.1, 0.2, 0.2]

    def decode_boxes(self, loc_preds: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """
        Converts predicted localization offsets into absolute bounding box coordinates [xmin, ymin, xmax, ymax].
        
        Args:
            loc_preds (Tensor): Predicted offsets with shape [Batch, 8732, 4]
            priors (Tensor): Default anchor boxes with shape [8732, 4] in [cx, cy, w, h] format
            
        Returns:
            Tensor: Decoded boxes with shape [Batch, 8732, 4] in [xmin, ymin, xmax, ymax] format
        """
        # Extract prior center and dimensions
        priors_cx = priors[:, 0]
        priors_cy = priors[:, 1]
        priors_w  = priors[:, 2]
        priors_h  = priors[:, 3]

        # Apply inverse transformation formulas using broadcasting over the batch dimension
        cx = loc_preds[..., 0] * self.variances[0] * priors_w + priors_cx
        cy = loc_preds[..., 1] * self.variances[1] * priors_h + priors_cy
        w  = torch.exp(loc_preds[..., 2] * self.variances[2]) * priors_w
        h  = torch.exp(loc_preds[..., 3] * self.variances[3]) * priors_h

        # Convert [cx, cy, w, h] format into standard coordinate format [xmin, ymin, xmax, ymax]
        decoded_boxes = torch.zeros_like(loc_preds)
        decoded_boxes[..., 0] = cx - w / 2.0  # xmin
        decoded_boxes[..., 1] = cy - h / 2.0  # ymin
        decoded_boxes[..., 2] = cx + w / 2.0  # xmax
        decoded_boxes[..., 3] = cy + h / 2.0  # ymax

        # Clip coordinates to keep them inside a normalized [0, 1] frame boundary
        return torch.clamp(decoded_boxes, min=0.0, max=1.0)

    def __call__(self, loc_preds: torch.Tensor, cls_preds: torch.Tensor, priors: torch.Tensor) -> list:
        """
        Processes raw model outputs to yield final filtered detections per batch item.
        
        Args:
            loc_preds (Tensor): Raw localization tensor from model forward [Batch, 8732, 4]
            cls_preds (Tensor): Raw classification logits from model forward [Batch, 8732, num_classes]
            priors (Tensor): Default anchor coordinates [8732, 4]
            
        Returns:
            list of dict: A list containing detection dictionaries for each image in the batch.
                          Each dict contains keys: 'boxes', 'scores', and 'labels'.
        """
        batch_size = loc_preds.size(0)
        num_classes = cls_preds.size(2)
        
        # 1. Decode all boxes for the entire batch
        all_decoded_boxes = self.decode_boxes(loc_preds, priors)
        
        # 2. Compute probabilities using Softmax over the class logits
        all_cls_scores = torch.softmax(cls_preds, dim=-1)
        
        batch_results = []

        # Process each image in the batch individually since final detection counts vary
        for i in range(batch_size):
            img_boxes = all_decoded_boxes[i]       # Shape: [8732, 4]
            img_scores = all_cls_scores[i]        # Shape: [8732, num_classes]

            # Lists to stack valid predictions before applying global NMS
            valid_boxes = []
            valid_scores = []
            valid_labels = []

            # Iterate through all object classes, completely skipping background (label 0)
            for c in range(num_classes):
                if c == self.background_label:
                    continue
                
                class_scores = img_scores[:, c]
                score_mask = class_scores > self.score_thresh
                
                if score_mask.sum() == 0:
                    continue
                
                # Extract coordinates and scores matching our filter criteria
                valid_boxes.append(img_boxes[score_mask])
                valid_scores.append(class_scores[score_mask])
                # Generate matching categorical labels for tracking inside the NMS step
                valid_labels.append(torch.full_like(class_scores[score_mask], fill_value=c, dtype=torch.long))

            # If no boxes passed the confidence threshold across any class, return empty tensors
            if len(valid_boxes) == 0:
                batch_results.append({
                    "boxes": torch.empty((0, 4), device=loc_preds.device),
                    "scores": torch.empty((0,), device=loc_preds.device),
                    "labels": torch.empty((0,), dtype=torch.long, device=loc_preds.device)
                })
                continue

            # Concatenate collected class arrays into consolidated single tensors
            final_img_boxes = torch.cat(valid_boxes, dim=0)
            final_img_scores = torch.cat(valid_scores, dim=0)
            final_img_labels = torch.cat(valid_labels, dim=0)

            # 3. Apply Multi-Class Batched NMS
            # batched_nms separates objects natively by class index to avoid cross-class masking
            keep_indices = batched_nms(
                boxes=final_img_boxes,
                scores=final_img_scores,
                idxs=final_img_labels,
                iou_threshold=self.iou_thresh
            )

            # Extract the actual post-processed safe outputs
            batch_results.append({
                "boxes": final_img_boxes[keep_indices],
                "scores": final_img_scores[keep_indices],
                "labels": final_img_labels[keep_indices]
            })

        return batch_results