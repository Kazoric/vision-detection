import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from models.heads.base import BaseHead


class SSDHead(BaseHead):
    """
    Task Head pour SSD (MultiBox Head).
    Applique des convolutions parallèles de localisation et de classification
    sur chaque niveau de feature map du Neck/Backbone.
    """

    # Config par défaut des ancres pour SSD300 (4 ou 6 par position selon le niveau)
    DEFAULT_ANCHORS_PER_LEVEL = {
        "c3": 4,
        "c4": 6,
        "c5": 6,
        "extra_1": 6,
        "extra_2": 4,
        "extra_3": 4,
    }

    def __init__(
        self,
        in_channels: Dict[str, int],
        num_classes: int,
        num_anchors_per_level: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Args:
            in_channels (Dict[str, int]): Canaux issus du neck (ex: neck.output_channels).
            num_classes (int): Nombre total de classes (incluant le fond / background).
            num_anchors_per_level (Optional[Dict[str, int]]): Nombre d'ancres associées à chaque niveau.
        """
        super().__init__(in_channels=in_channels)
        self.num_classes = num_classes

        # Utilise la configuration par défaut si aucune n'est spécifiée
        if num_anchors_per_level is None:
            self.num_anchors = {
                k: self.DEFAULT_ANCHORS_PER_LEVEL.get(k, 4) for k in in_channels.keys()
            }
        else:
            self.num_anchors = num_anchors_per_level

        self.cls_convs = nn.ModuleDict()
        self.loc_convs = nn.ModuleDict()

        # Instanciation des couches de prédiction pour chaque niveau
        for level, channels in in_channels.items():
            n_anchors = self.num_anchors[level]

            # Preds de classification : (N_ancres * Num_classes) canaux
            self.cls_convs[level] = nn.Conv2d(
                channels, n_anchors * num_classes, kernel_size=3, padding=1
            )
            # Preds de localisation : (N_ancres * 4) canaux
            self.loc_convs[level] = nn.Conv2d(
                channels, n_anchors * 4, kernel_size=3, padding=1
            )

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features (Dict[str, torch.Tensor]): Cartes de caractéristiques issues du Neck.

        Returns:
            Dict[str, torch.Tensor]: Contient 'loc_preds' (B, Total_Anchors, 4) 
                                     et 'cls_preds' (B, Total_Anchors, num_classes).
        """
        all_loc_preds = []
        all_cls_preds = []

        # Parcours ordonné de tous les niveaux de résolution
        for level in self.in_channels.keys():
            x = features[level]
            b = x.shape[0]

            # 1. Localisation : (B, N_anchors * 4, H, W) -> (B, H * W * N_anchors, 4)
            loc = self.loc_convs[level](x)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(b, -1, 4)
            all_loc_preds.append(loc)

            # 2. Classification : (B, N_anchors * C, H, W) -> (B, H * W * N_anchors, C)
            cls = self.cls_convs[level](x)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(b, -1, self.num_classes)
            all_cls_preds.append(cls)

        # Concaténation de toutes les ancres de toutes les feature maps
        # Formats finaux: (B, Total_Anchors, 4) et (B, Total_Anchors, num_classes)
        loc_preds = torch.cat(all_loc_preds, dim=1)
        cls_preds = torch.cat(all_cls_preds, dim=1)

        return {
            "loc_preds": loc_preds,
            "cls_preds": cls_preds,
        }