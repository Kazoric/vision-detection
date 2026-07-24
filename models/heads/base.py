from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple

import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for all task-specific heads (e.g., Detection Head, Segmentation Head).
    
    Receives feature maps from the Neck (or Backbone) and processes them to produce 
    final network outputs (e.g., class logits, bounding box regressions).
    """

    def __init__(self, in_channels: Dict[str, int]) -> None:
        """
        Args:
            in_channels (Dict[str, int]): Map of level names to channel counts.
                                          Sourced from neck.output_channels or backbone.output_channels.
                                          Example: {"p3": 256, "p4": 256, "p5": 256}
        """
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Union[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, ...]]]:
        """
        Processes multi-scale feature maps into task-specific predictions.

        Args:
            features (Dict[str, torch.Tensor]): Fused feature maps from neck or backbone.

        Returns:
            Raw network predictions structured per scale level or task.
        """
        pass