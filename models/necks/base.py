from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict


class BaseNeck(nn.Module, ABC):
    """
    Abstract base class for all Neck architectures (e.g., FPN, PANet, BiFPN).
    
    The Neck receives multi-scale feature maps from the Backbone 
    and produces fused/enriched feature maps for the Head.
    """
    def __init__(self, in_channels: Dict[str, int]):
        """
        Args:
            in_channels: Dictionary mapping feature level names to channel counts
                         (sourced directly from backbone.output_channels).
                         Example: {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}
        """
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extracts multi-scale feature maps from the input image tensor.
        """
        pass

    @property
    @abstractmethod
    def output_channels(self) -> Dict[str, int]:
        """
        Dictionary mapping feature stage names to their output channel counts.
        Example: {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}
        """
        pass