import torch
from torch import nn
from typing import List, Dict, Optional, Tuple

from .base import BaseBackbone

class Bottleneck(nn.Module):
    """ Bloc ResNet Bottleneck standard """
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

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ResNetBackbone(BaseBackbone):

    def __init__(self, out_indices: Tuple[str, ...] = ("c2", "c3", "c4", "c5")):
        super().__init__()
        self.in_planes = 64
        self.out_indices = set(out_indices)
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(Bottleneck, planes=64, blocks=2, stride=1)   # C2 (stride 4)
        self.layer2 = self._make_layer(Bottleneck, planes=128, blocks=2, stride=2)  # C3 (stride 8)
        self.layer3 = self._make_layer(Bottleneck, planes=256, blocks=2, stride=2)  # C4 (stride 16)
        self.layer4 = self._make_layer(Bottleneck, planes=512, blocks=2, stride=2)  # C5 (stride 32)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
            downsample = None
            if stride != 1 or self.in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
    
            layers = [block(self.in_planes, planes, stride, downsample)]
            self.in_planes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.in_planes, planes))
    
            return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        outputs = {}
        c2 = self.layer1(x)
        if "c2" in self.out_indices: outputs["c2"] = c2
        
        c3 = self.layer2(c2)
        if "c3" in self.out_indices: outputs["c3"] = c3
        
        c4 = self.layer3(c3)
        if "c4" in self.out_indices: outputs["c4"] = c4
        
        c5 = self.layer4(c4)
        if "c5" in self.out_indices: outputs["c5"] = c5

        return outputs

    @property
    def output_channels(self) -> Dict[str, int]:
        channels = {
            "c2": 64 * Bottleneck.expansion,   # 256
            "c3": 128 * Bottleneck.expansion,  # 512
            "c4": 256 * Bottleneck.expansion,  # 1024
            "c5": 512 * Bottleneck.expansion,  # 2048
        }

        return {k: v for k, v in channels.items() if k in self.out_indices}