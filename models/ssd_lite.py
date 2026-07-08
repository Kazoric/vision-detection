import torch
import torch.nn as nn
# from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from typing import List, Dict, Optional

from core.config import Config
from core.model_base import Model  # Ton modèle de base abstrait


class SSDMobileNetNetwork(nn.Module):
    """
    Réseau SSD Lite basé sur MobileNetV3 Large.
    Torchvision encapsule nativement la gestion des ancres, le calcul des pertes (entraînement)
    et le décodage complet avec NMS (évaluation).
    """
    def __init__(self, num_classes: int, score_thresh: float = 0.15, iou_thresh: float = 0.45):
        super().__init__()
        # Chargement de SSD Lite avec le backbone MobileNetV3 pré-entraîné.
        # Note : Dans l'écosystème torchvision detection, la classe 0 est TOUJOURS le Background.
        self.model = ssdlite320_mobilenet_v3_large(
            weights=None,  # On ne prend pas les poids SSD complets car le num_classes varie
            num_classes=num_classes,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT
        )
        
        # Injection directe des seuils de post-processing dans l'architecture torchvision
        self.model.score_thresh = score_thresh
        self.model.nms_thresh = iou_thresh

    def forward(self, images, targets=None):
        # Torchvision Object Detection attend une liste de Tensors [C, H, W]
        if isinstance(images, torch.Tensor):
            # Si les images arrivent sous forme de batch (B, C, H, W), on les sépare en liste
            images = [img for img in images]
        elif isinstance(images, list):
            images = [img for img in images]

        # --- MODE ENTRAÎNEMENT ---
        if self.training and targets is not None:
            # torchvision renvoie nativement : {"bbox_regression": tensor, "classification": tensor}
            raw_losses = self.model(images, targets)
            
            # Remappage défensif des clés pour que ton Trainer (conçu pour YOLO) ne plante pas
            return {
                "loss_box": raw_losses["bbox_regression"],
                "loss_class": raw_losses["classification"],
                "loss_obj": torch.tensor(0.0, device=raw_losses["bbox_regression"].device),
                "loss_noobj": torch.tensor(0.0, device=raw_losses["bbox_regression"].device)
            }
            
        # --- MODE ÉVALUATION ---
        # torchvision renvoie directement le format décodé attendu :
        # [{'boxes': tensor, 'scores': tensor, 'labels': tensor}, ...]
        return self.model(images)


class SSDLiteModel(Model):
    """ Wrapper SSDModel héritant de ta classe de base Model """
    def __init__(self, config: Config, score_thresh: float = 0.15, iou_thresh: float = 0.45, **kwargs):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        # Suppression de grid_size car SSD utilise des cartes de caractéristiques multi-échelles
        super().__init__(config=config, **kwargs)

    @property
    def name(self) -> str:
        return "SSDLite_MobileNetV3_Large_320"

    def build_model(self) -> nn.Module:
        print(f"[INFO] Construction du Réseau SSDLite MobileNetV3 Large (Résolution native : 320x320)...")
        model = SSDMobileNetNetwork(
            num_classes=self.num_classes,
            score_thresh=self.score_thresh,
            iou_thresh=self.iou_thresh
        )
        # Pas besoin d'initialisation manuelle forcée (_initialize_custom_weights), 
        # torchvision gère déjà l'initialisation de sa nouvelle tête de classification/régression.
        return model

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "input_size": (320, 320),  # La version standard de torchvision est calibrée sur du 320x320
            "architecture": "SSDLite_MobileNetV3"
        }