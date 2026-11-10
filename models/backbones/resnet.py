import torch
from torch import nn
from typing import List, Dict, Optional, Tuple

from .base import BaseBackbone
from .registry import register_backbone

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)

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

@register_backbone("resnet")
class ResNetBackbone(BaseBackbone):

    def __init__(self, block: str = "basic", layers: Tuple[int, ...] = (2, 2, 2, 2), out_indices: Tuple[str, ...] = ("c2", "c3", "c4", "c5"), **kwargs):
        super().__init__()
        self.in_planes = 64
        self.out_indices = set(out_indices)
        block_map = {
            "basic": BasicBlock,
            "bottleneck": Bottleneck,
        }

        if block not in block_map:
            raise ValueError(f"Unknown block : {block}")

        self.block_cls = block_map[block]
        
        # Stem
        # self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        # self.maxpool = nn.Identity()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        widths = (64, 128, 256, 512)
        strides = (1, 2, 2, 2)

        self.stages = nn.Sequential(*[
            self._make_layer(self.block_cls, w, n, s)
            for w, n, s in zip(widths, layers, strides)
        ])


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
        stage_names = ("c2", "c3", "c4", "c5")

        for name, stage in zip(stage_names, self.stages):
            x = stage(x)
            if name in self.out_indices:
                outputs[name] = x

        return outputs

    @property
    def output_channels(self) -> Dict[str, int]:
        expansion = self.block_cls.expansion
        channels = {
            "c2": 64 * expansion,
            "c3": 128 * expansion,
            "c4": 256 * expansion,
            "c5": 512 * expansion,
        }

        return {k: v for k, v in channels.items() if k in self.out_indices}