from typing import Dict, Any, Optional, Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from models.detectors.base import BaseDetector
from models.backbones.base import BaseBackbone
from models.backbones import build_backbone
from models.necks.base import BaseNeck
from models.necks import build_neck
from models.heads.base import BaseHead
from models.heads import build_head
from models.backbones.resnet import ResNetBackbone
from models.necks.ssd_neck import SSD300Neck
from models.heads.ssd_head import SSDHead

from loss.ssd_loss import SSDMultiBoxLoss
from utils.ssd_utils import generate_ssd_priors


class SSDDetector(BaseDetector):
    """
    Détecteur SSD300 complet (Single Shot MultiBox Detector).
    
    Combine :
      1. Backbone (ex: ResNet) -> extrait c3, c4, c5
      2. SSD300Neck           -> ajoute extra_1, extra_2, extra_3
      3. SSDHead             -> prédit les boîtes (loc) et classes (cls) sur les 6 échelles
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        num_classes: int = 10,
        image_size: Tuple[int, int] = [300, 300],
        score_thresh: float = 0.15,
        iou_thresh: float = 0.45,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__(backbone=backbone, neck=neck, head=head)
        self.num_classes = num_classes
        self.image_size = image_size
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        # self.criterion = criterion  # Module de perte (ex: MultiBoxLoss)
        self.criterion = SSDMultiBoxLoss(num_classes=self.num_classes)

        feature_map_sizes = self._infer_feature_map_sizes(self.image_size)

        # 2. Génération automatique des priors adaptés aux feature maps
        priors = generate_ssd_priors(
            feature_map_sizes=feature_map_sizes,
            image_size=self.image_size,
        )

        self.register_buffer("priors", priors)

    def _infer_feature_map_sizes(self, image_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """ Effectue un 'dummy forward' pour déterminer dynamiquement (H, W) de chaque sortie. """
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size[0], image_size[1])
            feats = self.backbone(dummy)
            if self.neck is not None:
                feats = self.neck(feats)

            if isinstance(feats, dict):
                feature_maps = [f.shape[2:] for f in feats.values()]
            elif isinstance(feats, (list, tuple)):
                feature_maps = [f.shape[2:] for f in feats]
            else:
                raise ValueError(f"Type de features non pris en charge : {type(feats)}")

        return [(int(h), int(w)) for h, w in feature_maps]

    def forward(
        self, x: torch.Tensor, targets: Optional[Any] = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Passage avant complet : Image -> Backbone -> Neck -> Head.

        Args:
            x (torch.Tensor): Batch d'images au format (B, 3, 300, 300).
            targets (Optional[Any]): Ground truths (boîtes + classes) pour l'entraînement.

        Returns:
            - En entraînement (avec targets & criterion) : Dict contenant la perte ('loss', 'loss_loc', 'loss_cls').
            - En inférence (ou sans targets) : Dict contenant les prédictions brutes ('loc_preds', 'cls_preds').
        """
        # 1. Extraction de caractéristiques multi-échelles (ex: {"c3": ..., "c4": ..., "c5": ...})
        features = self.backbone(x)

        # 2. Extension des cartes de caractéristiques (-> {"c3", "c4", "c5", "extra_1", "extra_2", "extra_3"})
        if self.neck is not None:
            features = self.neck(features)

        # 3. Prédictions parallèles de localisation et de classification
        # predictions = {"loc_preds": (B, 8732, 4), "cls_preds": (B, 8732, num_classes)}
        preds = self.head(features)

        loc_preds = preds["loc_preds"]
        cls_preds = preds["cls_preds"]

        # 4. Calcul de la perte si nous sommes en phase d'entraînement
        if self.training and targets is not None:
            if self.criterion is None:
                raise RuntimeError(
                    "Un 'criterion' (MultiBoxLoss) doit être défini pour entraîner le modèle."
                )
            loss_dict = self.criterion(loc_preds, cls_preds, targets, self.priors)
            return loss_dict

        return loc_preds, cls_preds

    @classmethod
    def _build_architecture(cls, config: Any) -> "SSDDetector":
        """
        Instancie Backbone, Neck et Head dynamiquement à partir de la configuration.
        """
        
        # 1. Backbone
        backbone_cfg = config.model.backbone
        backbone = build_backbone(
            name=backbone_cfg["type"],
            out_indices=backbone_cfg.get("out_indices", ["c3", "c4", "c5"]),
            **backbone_cfg.get("backbone_kwargs", {})
        )

        # 2. Neck
        neck = None
        if hasattr(config.model, "neck") and config.model.neck is not None:
            neck = build_neck(
                name=config.model.neck["type"],
                in_channels=backbone.output_channels,
                source_layer=config.model.neck.get("source_layer", "c5"),
                extra_channels=config.model.neck.get("extra_channels", [512, 256, 256]),
            )

        if hasattr(config.model, "neck") and config.model.neck is not None:
            head = build_head(
                name=config.model.head["type"],
                in_channels=neck.output_channels,
                num_classes=config.model.num_classes,
                num_anchors_per_level=config.model.head.get("num_anchors_per_level", None),
            )

        # 3. Modèle SSD
        return cls(
            backbone=backbone,
            neck=neck,
            head=head,
            num_classes=config.model.num_classes,
            image_size = config.model.image_size,
            score_thresh=getattr(config.model, "score_thresh", 0.25),
            iou_thresh=getattr(config.model, "iou_thresh", 0.45),
        )

    def predict_decoded(
            self,
            loc_preds: torch.Tensor,
            cls_preds: torch.Tensor,
            confidence_threshold: float = 0.15,
            iou_threshold: float = 0.45,
            img_size=(300, 300)
        ) -> List[Dict[str, torch.Tensor]]:
            B = loc_preds.size(0)
            img_w, img_h = img_size
            device = loc_preds.device
    
            cx = loc_preds[..., 0] * 0.1 * self.priors[:, 2] + self.priors[:, 0]
            cy = loc_preds[..., 1] * 0.1 * self.priors[:, 3] + self.priors[:, 1]
            w = torch.exp(loc_preds[..., 2] * 0.2) * self.priors[:, 2]
            h = torch.exp(loc_preds[..., 3] * 0.2) * self.priors[:, 3]
    
            decoded_boxes = torch.zeros_like(loc_preds)
            decoded_boxes[..., 0] = (cx - w / 2.0) * img_w
            decoded_boxes[..., 1] = (cy - h / 2.0) * img_h
            decoded_boxes[..., 2] = (cx + w / 2.0) * img_w
            decoded_boxes[..., 3] = (cy + h / 2.0) * img_h
            decoded_boxes = torch.clamp(decoded_boxes, min=0.0, max=float(max(img_w, img_h)))
    
            cls_probs = F.softmax(cls_preds, dim=-1)
    
            predictions_list = []
            for b in range(B):
                b_boxes, b_scores, b_labels = [], [], []
    
                for c in range(1, self.num_classes):
                    scores = cls_probs[b, :, c]
                    mask = scores > confidence_threshold
    
                    if mask.sum() == 0:
                        continue
    
                    b_boxes.append(decoded_boxes[b, mask])
                    b_scores.append(scores[mask])
                    b_labels.append(torch.full((mask.sum(),), fill_value=c, dtype=torch.long, device=device))
    
                if len(b_boxes) > 0:
                    t_boxes = torch.cat(b_boxes, dim=0)
                    t_scores = torch.cat(b_scores, dim=0)
                    t_labels = torch.cat(b_labels, dim=0)
    
                    keep_nms = batched_nms(t_boxes, t_scores, t_labels, iou_threshold)
    
                    predictions_list.append({
                        "boxes": t_boxes[keep_nms],
                        "scores": t_scores[keep_nms],
                        "labels": t_labels[keep_nms]
                    })
                else:
                    predictions_list.append({
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                        "scores": torch.zeros(0, dtype=torch.float32, device=device),
                        "labels": torch.zeros(0, dtype=torch.long, device=device)
                    })
    
            return predictions_list

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, confidence_threshold: float = 0.25, iou_threshold: float = 0.45
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Méthode dédiée à l'inférence : prend les images et renvoie les boîtes filtrées par NMS.
        """
        self.eval()
        loc_preds, cls_preds = self.forward(x)
        return self.predict_decoded(loc_preds, cls_preds, confidence_threshold, iou_threshold)