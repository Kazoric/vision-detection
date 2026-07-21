import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from typing import List, Dict

from core.config import Config
from core.model_base import Model
from loss.ssd_loss import SSDMultiBoxLoss
from utils.ssd_utils import generate_ssd_priors


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
        super().__init__(config=config, **kwargs)

    @property
    def name(self) -> str:
        return "SSD300_VGG16"

    def build_model(self) -> nn.Module:
        print(f"[INFO] Building SSD300 VGG16 Network (Native resolution: 300x300)...")
        return SSDVGG16Backbone(num_classes=self.num_classes)

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "input_size": (300, 300),
            "architecture": "SSD_VGG16"
        }