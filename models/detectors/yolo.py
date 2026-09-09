from typing import Dict, Any, Optional, List, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from models.backbones.base import BaseBackbone
from models.backbones import build_backbone
from models.necks.base import BaseNeck
from models.necks import build_neck
from models.heads.base import BaseHead
from models.heads import build_head
from models.heads.yolo_head import YoloHead
from loss.yolo_loss import YoloLoss
from .base import BaseDetector
from .registry import register_detector


@register_detector("yolo")
class Yolo(BaseDetector):
    """
    YOLO Object Detector integrating Backbone, Neck, Decoupled Head, 
    Anchor management, Post-processing (NMS), and Loss computation.
    """

    def __init__(
        self,
        backbone: BaseBackbone,
        neck: Optional[BaseNeck] = None,
        head: Optional[BaseHead] = None,
        num_classes: int = 80,
        score_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        strides: List[int] = [8, 16, 32],
    ) -> None:
        """
        Args:
            backbone (BaseBackbone): Feature extractor backbone.
            neck (Optional[BaseNeck]): Feature pyramid fusion neck (e.g., FPN).
            head (Optional[BaseHead]): Custom head module. If None, YoloHead is created.
            num_classes (int): Number of object categories.
            score_thresh (float): Minimum confidence threshold for inference.
            iou_thresh (float): IoU threshold for NMS.
            strides (List[int]): Strides for P3, P4, P5.
        """
        head_in_channels = (
            neck.output_channels if neck is not None else backbone.output_channels
        )

        if head is None:
            head = YoloHead(in_channels=head_in_channels, num_classes=num_classes)

        super().__init__(backbone=backbone, neck=neck, head=head)

        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.strides = strides

        # Loss function initialization
        self.criterion = YoloLoss(num_classes=num_classes, strides=strides)

    def forward(
        self, x: torch.Tensor, targets: Optional[Any] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: Backbone -> Neck (optional) -> Head.

        Args:
            x (torch.Tensor): Input batch of images (B, 3, H, W).
            targets (Optional[Any]): Ground truth targets for loss computation.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Map of scale keys to (cls_scores, bbox_preds).
        """
        # 1. Feature extraction
        feats = self.backbone(x)

        # 2. Multi-scale feature fusion
        if self.neck is not None:
            feats = self.neck(feats)

        # 3. Prediction head
        raw_outputs = self.head(feats)

        if targets is not None:
            return self.criterion(raw_outputs, targets)

        return raw_outputs

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, confidence_threshold: float = 0.25, iou_threshold: float = 0.45
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Méthode dédiée à l'inférence : prend les images et renvoie les boîtes filtrées par NMS.
        """
        self.eval()
        raw_outputs = self.forward(x)
        return self.post_process(raw_outputs, x.shape[2:], confidence_threshold, iou_threshold)

    def post_process(
        self,
        predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        img_size: Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> List[Dict[str, torch.Tensor]]:
        """Décode les prédictions multi-échelles et applique le NMS.

        Returns:
            List[Dict[str, torch.Tensor]]: Liste (une entrée par image du
            batch) de dictionnaires:
                - "boxes": Tensor [N, 4] (x1, y1, x2, y2)
                - "scores": Tensor [N] (confiance)
                - "labels": Tensor [N] (id de classe, dtype=torch.long)
        """
        first_key = next(iter(predictions))
        batch_size = predictions[first_key][0].shape[0]
        device = predictions[first_key][0].device
        img_h, img_w = img_size

        all_decoded_boxes = []
        all_decoded_scores = []

        # 1. Décodage de toutes les échelles (p3, p4, p5)
        for scale_idx, (level, (cls_score, bbox_pred)) in enumerate(
            predictions.items()
        ):
            stride = (
                self.strides[scale_idx]
                if scale_idx < len(self.strides)
                else 8 * (2**scale_idx)
            )

            cls_prob = torch.sigmoid(cls_score).permute(
                0, 2, 3, 1
            )  # [B, H, W, num_classes]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # [B, H, W, 4]

            _, grid_h, grid_w, _ = cls_prob.shape

            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_h, device=device),
                torch.arange(grid_w, device=device),
                indexing="ij",
            )

            grid_x_px = (grid_x.float() + 0.5) * stride
            grid_y_px = (grid_y.float() + 0.5) * stride

            # Anchor-Free LTRB distances
            dist = F.softplus(bbox_pred) * stride

            x1 = (grid_x_px - dist[..., 0]).clamp(0, img_w)
            y1 = (grid_y_px - dist[..., 1]).clamp(0, img_h)
            x2 = (grid_x_px + dist[..., 2]).clamp(0, img_w)
            y2 = (grid_y_px + dist[..., 3]).clamp(0, img_h)

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            all_decoded_boxes.append(boxes.view(batch_size, -1, 4))
            all_decoded_scores.append(
                cls_prob.view(batch_size, -1, self.num_classes)
            )

        all_boxes = torch.cat(all_decoded_boxes, dim=1)
        all_scores = torch.cat(all_decoded_scores, dim=1)

        output = []

        # 2. Filtrage NMS image par image
        for img_idx in range(batch_size):
            img_boxes = all_boxes[img_idx]
            img_scores = all_scores[img_idx]

            max_scores, class_ids = torch.max(img_scores, dim=1)

            mask = max_scores > conf_thres
            if not mask.any():
                output.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty(
                            (0,), dtype=torch.long, device=device
                        ),
                    }
                )
                continue

            valid_boxes = img_boxes[mask]
            valid_scores = max_scores[mask]
            valid_classes = class_ids[mask]

            keep = batched_nms(
                boxes=valid_boxes,
                scores=valid_scores,
                idxs=valid_classes,
                iou_threshold=iou_thres,
            )

            # Format dict attendu par le predictor
            output.append(
                {
                    "boxes": valid_boxes[keep],
                    "scores": valid_scores[keep],
                    "labels": valid_classes[keep].long(),
                }
            )

        return output

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Returns key configuration metadata for logging/saving."""
        return {
            "num_classes": self.num_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "strides": self.strides,
            "architecture": "Modular_YOLO"
        }

    @classmethod
    def _build_architecture(cls, config: Any) -> "Yolo":
        """Instanciation automatique à partir de la configuration."""
        model_cfg = config.model

        # 1. Backbone
        backbone_cfg = config.model.backbone
        backbone = build_backbone(
            name=backbone_cfg["type"],
            out_indices=backbone_cfg.get("out_indices", ["c3", "c4", "c5"]),
            **backbone_cfg.get("backbone_kwargs", {})
        )

        # 2. Neck
        neck = None
        if hasattr(model_cfg, "neck") and model_cfg.neck is not None:
            neck = build_neck(
                name=model_cfg.neck["type"],
                in_channels=backbone.output_channels,
                out_channels=model_cfg.neck.get("out_channels", 256)
            )

        if hasattr(config.model, "head") and config.model.head is not None:
            head = build_head(
                name=config.model.head["type"],
                in_channels=neck.output_channels,
                num_classes=config.model.num_classes,
            )

        # 3. Modèle YOLO Anchor-Free
        return cls(
            backbone=backbone,
            neck=neck,
            head=head,
            num_classes=model_cfg.num_classes,
            score_thresh=getattr(model_cfg, "score_thresh", 0.25),
            iou_thresh=getattr(model_cfg, "iou_thresh", 0.45),
            strides=getattr(model_cfg, "strides", [8, 16, 32]),
        )