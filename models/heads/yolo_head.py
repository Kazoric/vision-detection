from typing import Dict, Tuple
import math
import torch
import torch.nn as nn

from .base import BaseHead


class YoloHead(BaseHead):
    """
    Decoupled YOLO Detection Head module.
    
    Separates bounding box regression and classification branches for each feature scale.
    """

    def __init__(self, in_channels: Dict[str, int], num_classes: int = 80) -> None:
        """
        Args:
            in_channels (Dict[str, int]): Dictionary of input channel sizes per level.
            num_classes (int): Number of target object categories.
        """
        super().__init__(in_channels=in_channels)
        self.num_classes = num_classes

        self.cls_convs = nn.ModuleDict()
        self.reg_convs = nn.ModuleDict()
        self.cls_preds = nn.ModuleDict()
        self.reg_preds = nn.ModuleDict()

        # Build classification and regression sub-branches per feature level (e.g., "p3", "p4", "p5")
        for key, ch in self.in_channels.items():
            # Classification branch
            self.cls_convs[key] = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            )
            self.cls_preds[key] = nn.Conv2d(ch, self.num_classes, kernel_size=1)

            # Bounding box regression branch (4 coords: x, y, w, h)
            self.reg_convs[key] = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            )
            self.reg_preds[key] = nn.Conv2d(ch, 4, kernel_size=1)

        # Initialisation du biais prioritaire (-4.59)
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cls_preds.values():
            nn.init.constant_(conv.bias, bias_init)

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            features (Dict[str, torch.Tensor]): Map of stage names to feature maps.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Map of stage names 
            to tuples of (cls_scores, bbox_preds).
        """
        outputs = {}
        for key, x in features.items():
            cls_feat = self.cls_convs[key](x)
            cls_score = self.cls_preds[key](cls_feat)

            reg_feat = self.reg_convs[key](x)
            bbox_pred = self.reg_preds[key](reg_feat)

            outputs[key] = (cls_score, bbox_pred)

        return outputs