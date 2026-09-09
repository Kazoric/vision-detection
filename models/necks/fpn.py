from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNeck
from .registry import register_neck

@register_neck("fpn")
class FPN(BaseNeck):
    """
    Feature Pyramid Network (FPN) neck implementation.
    
    Ref: "Feature Pyramid Networks for Object Detection" (Lin et al., 2017)
    
    It builds a top-down pathway with lateral connections to construct
    a feature pyramid with high-level semantics at all scales.
    """

    def __init__(self, in_channels: Dict[str, int], out_channels: int = 256) -> None:
        """
        Args:
            in_channels (Dict[str, int]): Dictionary of input channel sizes per stage from backbone.
                                          Example: {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}
            out_channels (int): Uniform channel size for all output feature maps (default: 256).
        """
        super().__init__(in_channels=in_channels)
        self._out_channels = out_channels

        self.lateral_convs = nn.ModuleDict()
        self.fpn_convs = nn.ModuleDict()

        # Build 1x1 lateral convolutions and 3x3 smoothing convolutions for each stage
        for key, in_ch in in_channels.items():
            # 1x1 conv to reduce/project input channels to out_channels
            self.lateral_convs[key] = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_channels,
                kernel_size=1
            )
            # 3x3 conv to smooth out anti-aliasing caused by upsampling
            self.fpn_convs[key] = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the FPN.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping stage names (e.g. "c2", "c3")
                                              to feature map tensors.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping output stage names (e.g. "p2", "p3")
                                     to processed feature map tensors.
        """
        # Sort keys from highest to lowest feature level (e.g. ["c5", "c4", "c3", "c2"])
        sorted_keys = sorted(inputs.keys(), reverse=True)

        lateral_maps = {}
        prev_feature = None

        # Top-down pathway with lateral connections
        for key in sorted_keys:
            lateral = self.lateral_convs[key](inputs[key])

            if prev_feature is not None:
                # Upsample higher-level feature map to match current lateral spatial size
                top_down = F.interpolate(
                    prev_feature,
                    size=lateral.shape[-2:],
                    mode="nearest"
                )
                lateral = lateral + top_down

            prev_feature = lateral
            lateral_maps[key] = lateral

        # Smooth features using 3x3 convs and map output keys (e.g. "c2" -> "p2")
        outputs = {}
        for key in inputs.keys():
            out_key = key.replace("c", "p")
            outputs[out_key] = self.fpn_convs[key](lateral_maps[key])

        return outputs

    @property
    def output_channels(self) -> Dict[str, int]:
        """
        Returns the output channel count for each produced feature level.
        Maps input keys (e.g., "c2") to output keys (e.g., "p2") with `out_channels`.

        Example:
            {"p2": 256, "p3": 256, "p4": 256, "p5": 256}
        """
        return {
            key.replace("c", "p"): self._out_channels
            for key in self.in_channels.keys()
        }