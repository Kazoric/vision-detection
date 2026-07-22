import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from typing import List, Dict, Optional

from core.config import Config
from core.model_base import Model
from loss.ssd_loss import SSDMultiBoxLoss
from utils.ssd_utils import generate_ssd_priors


class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck block (1x1 conv -> 3x3 conv -> 1x1 conv).
    Expansion factor is 4 (e.g. 64 input channels -> 256 output channels).
    """
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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # If stride > 1 or channel count changes, adjust identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """ Custom-built ResNet backbone """
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        # Initial Stem: Image (3, 300, 300) -> Conv7x7 -> (64, 150, 150) -> MaxPool -> (64, 75, 75)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(Bottleneck, planes=64, blocks=2, stride=1)
        self.layer2 = self._make_layer(Bottleneck, planes=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(Bottleneck, planes=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(Bottleneck, planes=512, blocks=2, stride=2)

        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        # Create projection shortcut if spatial size is reduced or channels change
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        # First block in group (may reduce spatial size via stride)
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion

        # Subsequent blocks in same group (stride 1)
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """ Kaiming Normal initialization for Conv2d """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SSDResNetModel(Model):
    """
    SSD300 detector built on a ResNet backbone.
    Inherits directly from Model.
    """

    def __init__(
        self,
        config: Config,
        score_thresh: float = 0.15,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
        **kwargs
    ):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        super().__init__(config=config, device=device)

    @property
    def name(self) -> str:
        return "SSD300_ResNet"

    def _build_architecture(self) -> None:
        """ Instantiates PyTorch submodules for SSD-ResNet """
        self.register_buffer("priors", generate_ssd_priors())

        # ResNet Backbone
        self.resnet = ResNetBackbone()

        # Extra SSD Layers (Sources 4, 5 and 6)
        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.extra_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # MultiBox Heads (Localization & Classification)
        num_boxes = [4, 6, 6, 6, 4, 4]
        in_channels = [512, 1024, 2048, 512, 256, 256]

        self.loc_layers = nn.ModuleList([
            nn.Conv2d(in_ch, n_box * 4, kernel_size=3, padding=1)
            for in_ch, n_box in zip(in_channels, num_boxes)
        ])

        self.cls_layers = nn.ModuleList([
            nn.Conv2d(in_ch, n_box * self.num_classes, kernel_size=3, padding=1)
            for in_ch, n_box in zip(in_channels, num_boxes)
        ])

        self.criterion = SSDMultiBoxLoss(num_classes=self.num_classes)

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        loc_preds, cls_preds = self.compute_raw_predictions(images)

        if self.training and targets is not None:
            return self.criterion(loc_preds, cls_preds, targets, self.priors)

        return self.predict_decoded(
            loc_preds, cls_preds,
            confidence_threshold=self.score_thresh,
            iou_threshold=self.iou_thresh
        )

    def compute_raw_predictions(self, x: torch.Tensor):
        sources = []

        # Main ResNet stream
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        x = self.resnet.layer2(x)
        sources.append(x)  # Source 1: 38x38 (512 ch)

        x = self.resnet.layer3(x)
        sources.append(x)  # Source 2: 19x19 (1024 ch)

        x = self.resnet.layer4(x)
        sources.append(x)  # Source 3: 10x10 (2048 ch)

        # Extra SSD layers
        x = self.extra_conv1(x)
        sources.append(x)  # Source 4: 5x5 (512 ch)

        x = self.extra_conv2(x)
        sources.append(x)  # Source 5: 3x3 (256 ch)

        x = self.extra_conv3(x)
        sources.append(x)  # Source 6: 1x1 (256 ch)

        # Pass through MultiBox heads
        loc_outputs, cls_outputs = [], []
        for (feat, l_conv, c_conv) in zip(sources, self.loc_layers, self.cls_layers):
            loc_outputs.append(l_conv(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1))
            cls_outputs.append(c_conv(feat).permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1))

        loc_flat = torch.cat(loc_outputs, dim=1)
        cls_flat = torch.cat(cls_outputs, dim=1)

        loc_final = loc_flat.view(loc_flat.size(0), -1, 4)
        cls_final = cls_flat.view(cls_flat.size(0), -1, self.num_classes)

        return loc_final, cls_final

    def predict_decoded(
        self,
        loc_preds: torch.Tensor,
        cls_preds: torch.Tensor,
        confidence_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        img_size=(300, 300)
    ) -> List[Dict[str, torch.Tensor]]:
        B = loc_preds.size(0)
        img_w, img_h = img_size
        device = loc_preds.device

        cx = loc_preds[..., 0] * 0.1 * self.priors[:, 2] + self.priors[:, 0]
        cy = loc_preds[..., 1] * 0.1 * self.priors[:, 3] + self.priors[:, 1]
        w = torch.exp(loc_preds[..., 2] * 0.2) * self.priors[:, 2]
        h = torch.exp(loc_preds[..., 3] * 0.2) * self.priors[:, 3]

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

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "input_size": (300, 300),
            "architecture": "SSD_ResNet"
        }