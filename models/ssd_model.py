import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Dict

# On réutilise tes blocs existants
from core.model_base import Model
# (Assure-toi d'importer BasicBlock et BottleneckBlock depuis ton fichier)

class BasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, dropout=0.1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.dropout = nn.Dropout(dropout)

        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x

        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # add identity
        x = self.dropout(x)
        x += self.a * identity
        x = torch.relu(x)

        return x
    
class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, dropout=0.1):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.dropout = nn.Dropout(dropout)

        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x

        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # add identity
        x = self.dropout(x)
        x += self.a * identity
        x = torch.relu(x)
        
        return x

class ResNetSSDBackbone(nn.Module):
    """
    Backbone multi-échelle pour SSD utilisant tes blocs ResNet personnalisés.
    Retourne un dictionnaire contenant 6 cartes de caractéristiques (Feature Maps).
    """
    def __init__(self, layer_list: List[int], block=BottleneckBlock, dropout: float = 0.1):
        super().__init__()
        self.block = block
        
        # --- STAGES STANDARDS RESNET ---
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Construction des 4 couches ResNet classiques
        self.layer1 = self._make_layer(block, layer_list[0], 64, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(block, layer_list[1], 128, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, layer_list[2], 256, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(block, layer_list[3], 512, stride=2, dropout=dropout)

        # Échelles de sortie de base (déduction dynamique des canaux selon le bloc)
        c3_out = 256 * block.expansion  # ex: 1024 si Bottleneck
        c4_out = 512 * block.expansion  # ex: 2048 si Bottleneck

        # --- COUCHES SUPPLÉMENTAIRES SSD (Extra Layers) ---
        # Permettent de descendre jusqu'à une résolution de 1x1 pour les très grands objets
        self.extra1 = nn.Sequential(
            nn.Conv2d(c4_out, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # Divise la résolution par 2
        )
        self.extra2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.extra3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.extra4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0) # Tombe à 1x1 si l'entrée est petite
        )

        # Attribut CRUCIAL requis par le module SSD de PyTorch :
        # Indique le nombre de canaux (channels) de chaque feature map retournée
        self.out_channels = [c3_out, c4_out, 512, 256, 256, 256]

    def _make_layer(self, block, blocks, planes, stride=1, dropout=0.1):
        ii_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes * block.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.in_channels, planes, i_downsample=ii_downsample, stride=stride, dropout=dropout))
        self.in_channels = planes * block.expansion
        for _ in range(blocks - 1):
            layers.append(block(self.in_channels, planes, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Passage dans les premières couches
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # Feature Map 1 : Sortie du Layer 3 (Haute résolution pour petits objets)
        f1 = self.layer3(x)
        
        # Feature Map 2 : Sortie du Layer 4
        f2 = self.layer4(f1)
        
        # Feature Maps 3 à 6 : Sorties des couches supplémentaires SSD
        f3 = self.extra1(f2)
        f4 = self.extra2(f3)
        f5 = self.extra3(f4)
        f6 = self.extra4(f5)

        # On renvoie un dictionnaire (clé ordonnée textuelle demandée par torchvision)
        out = OrderedDict()
        out["0"] = f1
        out["1"] = f2
        out["2"] = f3
        out["3"] = f4
        out["4"] = f5
        out["5"] = f6
        return out
    


from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from core.config import Config  # Ta dataclass de config globale

class SSDModel(Model):
    """
    Modèle de Détection d'Objets SSD s'intégrant dans le framework de base.
    """
    def __init__(self, config: Config, layer_list=[3, 4, 6, 3], block='Bottleneck', dropout=0.1, **kwargs):
        self.layer_list = layer_list
        self.block_str = block
        self.dropout = dropout

        if block == 'Bottleneck':
            self.block = BottleneckBlock
        elif block == 'Basic':
            self.block = BasicBlock
        else:
            raise ValueError(f"Type de bloc inconnu : {block}")
        
        # Appel de l'initialisation parente qui va déclencher build_model()
        super().__init__(config=config, **kwargs)

    # 1. FIX : Implémentation correcte de la propriété abstraite 'name'
    @property
    def name(self) -> str:
        return f"SSD_ResNet_{self.block_str}"

    # 2. Implémentation de la méthode abstraite 'build_model'
    def build_model(self) -> nn.Module:
        print(f"[INFO] Construction du Backbone ResNet Multi-Échelle ({self.block_str})...")
        
        backbone = ResNetSSDBackbone(
            layer_list=self.layer_list,
            block=self.block,
            dropout=self.dropout
        )

        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=aspect_ratios,
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300]
        )

        print(f"[INFO] Assemblage du modèle SSD complet (Classes: {self.num_classes})...")
        model = SSD(
            backbone=backbone,
            anchor_generator=anchor_generator,
            size=(300, 300),
            num_classes=self.num_classes
        )

        self._initialize_custom_weights(model)
        return model
    
    def _initialize_custom_weights(self, model: nn.Module):
        """ Applique ta logique d'initialisation des poids """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "layer_list": self.layer_list,
            "block": self.block_str,
            "dropout": self.dropout,
            "input_size": (300, 300)
        }