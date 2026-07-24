import torch
import torch.nn as nn
from typing import Dict, List
from models.necks.base import BaseNeck


class SSD300Neck(BaseNeck):
    """
    Neck spécifique à SSD300 :
    Reçoit [c3 (38x38), c4 (19x19), c5 (10x10)] du backbone,
    puis ajoute 3 couches extra pour obtenir 6 feature maps au total.
    """

    def __init__(
        self,
        in_channels: Dict[str, int],  # ex: {"c3": 512, "c4": 1024, "c5": 2048}
        source_layer: str = "c5",
        extra_channels: List[int] = [512, 256, 256],  # 3 couches extra
    ):
        super().__init__(in_channels=in_channels)
        self.source_layer = source_layer
        self.extra_layers = nn.ModuleDict()
        self._output_channels = dict(in_channels)

        curr_channels = in_channels[source_layer]

        # 1. extra_1 : 10x10 -> 5x5 (stride=2, padding=1)
        self.extra_layers["extra_1"] = nn.Sequential(
            nn.Conv2d(curr_channels, extra_channels[0] // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(extra_channels[0] // 2, extra_channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self._output_channels["extra_1"] = extra_channels[0]

        # 2. extra_2 : 5x5 -> 3x3 (stride=1, padding=0)
        self.extra_layers["extra_2"] = nn.Sequential(
            nn.Conv2d(extra_channels[0], extra_channels[1] // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(extra_channels[1] // 2, extra_channels[1], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self._output_channels["extra_2"] = extra_channels[1]

        # 3. extra_3 : 3x3 -> 1x1 (stride=1, padding=0)
        self.extra_layers["extra_3"] = nn.Sequential(
            nn.Conv2d(extra_channels[1], extra_channels[2] // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(extra_channels[2] // 2, extra_channels[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self._output_channels["extra_3"] = extra_channels[2]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # On conserve c3, c4 et c5 du backbone
        outputs = dict(inputs)
        x = inputs[self.source_layer]

        # On applique les 3 extensions
        for name, layer in self.extra_layers.items():
            x = layer(x)
            outputs[name] = x

        # Contient les 6 clés: c3, c4, c5, extra_1, extra_2, extra_3
        return outputs

    @property
    def output_channels(self) -> Dict[str, int]:
        return self._output_channels