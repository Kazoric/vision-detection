import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from .base import BaseBackbone

# --- ConvNeXt Utilities ---

class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: 
    channels_last (default) or channels_first. 
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ 
    GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Stochastic Depth (DropPath) function."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """DropPath module wrapper."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block. 
    Equivalent to TransformerEncoderBlock but with convolutions.
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 1. Depthwise conv (7x7) -> simulates the large attention window of ViT
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        
        # 2. Norm (LayerNorm)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 3. Pointwise convs (Inverted MLP: dim -> 4*dim -> dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 4. Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # Permute to switch from (N, C, H, W) to (N, H, W, C) for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
            
        # Return to (N, C, H, W) format
        x = x.permute(0, 3, 1, 2) 

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2Backbone(BaseBackbone):
    """ConvNeXt V2 Architecture."""

    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        drop_path_rate: float = 0.0,
        out_indices: Tuple[str, ...] = ("c2", "c3", "c4", "c5"),
        **kwargs
    ):
        super().__init__()
        self.out_indices = set(out_indices)
        self.dims = dims
        
        # 1. Stem (Patchify similar to ViT but with Conv)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # 2. Downsampling layers between stages
        self.downsample_layer2 = nn.Sequential(
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[0], dims[0+1], kernel_size=2, stride=2),
        )
        self.downsample_layer3 = nn.Sequential(
                LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[1], dims[1+1], kernel_size=2, stride=2),
        )
        self.downsample_layer4 = nn.Sequential(
                LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[2], dims[2+1], kernel_size=2, stride=2),
        )

        # 3. ConvNeXt stages (sequence of blocks)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0

        self.stage1 = nn.Sequential(
            *[ConvNeXtBlock(dim=dims[0], drop_path=dp_rates[cur + j]) 
                for j in range(depths[0])]
        )
        cur += depths[0]
        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock(dim=dims[1], drop_path=dp_rates[cur + j]) 
                for j in range(depths[1])]
        )
        cur += depths[1]
        self.stage3 = nn.Sequential(
            *[ConvNeXtBlock(dim=dims[2], drop_path=dp_rates[cur + j]) 
                for j in range(depths[2])]
        )
        cur += depths[2]
        self.stage4 = nn.Sequential(
            *[ConvNeXtBlock(dim=dims[3], drop_path=dp_rates[cur + j]) 
                for j in range(depths[3])]
        )
        cur += depths[3]

        # Init weights
        self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)

    #     elif isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        # Iterate through the 4 stages (pyramidal architecture)
        x = self.stem(x)
        c2 = self.stage1(x)
        if "c2" in self.out_indices: outputs["c2"] = c2

        x = self.downsample_layer2(c2)
        c3 = self.stage2(x)
        if "c3" in self.out_indices: outputs["c3"] = c3

        x = self.downsample_layer3(c3)
        c4 = self.stage3(x)
        if "c4" in self.out_indices: outputs["c4"] = c4

        x = self.downsample_layer4(c4)
        c5 = self.stage4(x)
        if "c5" in self.out_indices: outputs["c5"] = c5

        # for i, f in enumerate(outputs):
        #     print(f, outputs[f].shape)
        return outputs

    @property
    def output_channels(self) -> Dict[str, int]:
        channels = {
            "c2": self.dims[0],
            "c3": self.dims[1],
            "c4": self.dims[2],
            "c5": self.dims[3],
        }

        return {k: v for k, v in channels.items() if k in self.out_indices}