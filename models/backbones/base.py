from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for all feature extraction backbones (e.g., ResNet, EfficientNet).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts multi-scale feature maps from an input image tensor.

        Args:
            x (torch.Tensor): Input image tensor with shape (B, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping feature stage names 
                                     (e.g., "c2", "c3") to their feature map tensors.
        """
        pass

    @property
    @abstractmethod
    def output_channels(self) -> Dict[str, int]:
        """
        Dictionary mapping feature stage names to their corresponding channel counts.

        Example:
            {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}
        """
        pass