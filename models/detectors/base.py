import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple

from models.backbones.base import BaseBackbone
from models.necks.base import BaseNeck
from models.heads.base import BaseHead


class BaseDetector(nn.Module, ABC):
    """
    Abstract base class for all object detectors (e.g., YOLO, RetinaNet, Faster R-CNN).

    Combines a Backbone (feature extraction), an optional Neck (feature fusion),
    and a Head (task prediction) into an end-to-end model.
    """

    def __init__(
        self,
        backbone: Optional[BaseBackbone] = None,
        neck: Optional[BaseNeck] = None,
        head: Optional[BaseHead] = None
    ):
        """
        Args:
            backbone (Optional[BaseBackbone]): Feature extractor backbone.
            neck (Optional[BaseNeck]): Optional multi-scale feature fusion neck (e.g., FPN).
            head (Optional[BaseHead]): Task-specific prediction head.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        # Minimal model metadata
        self.num_classes: Optional[int] = None
        self.name: str = self.__class__.__name__

    @abstractmethod
    def forward(
        self, x: torch.Tensor, targets: Optional[Any] = None
    ) -> Union[Dict[str, Any], torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through backbone -> neck -> head.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).
            targets (Optional[Any]): Ground truth annotations for training loss calculation.

        Returns:
            Raw predictions during inference, or computed loss values during training.
        """
        pass

    @classmethod
    @abstractmethod
    def _build_architecture(cls, config: Any) -> "BaseDetector":
        """
        Instantiates network sub-modules (backbone, neck, head) from configuration.
        Must be implemented by concrete detector subclasses.
        """
        raise NotImplementedError("Each detector must implement `_build_architecture`.")

    @classmethod
    def from_config(cls, config: Any) -> "BaseDetector":
        """
        Factory method to build and initialize the model architecture from config.
        """
        model = cls._build_architecture(config)
        model.num_classes = getattr(config.model, "num_classes", None)
        model.name = getattr(config.model, "name", cls.__name__)
        return model