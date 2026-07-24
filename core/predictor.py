import torch
import torch.nn as nn
from typing import List, Dict


class Predictor:
    """
    Performs inference on a trained object detection model.

    The Predictor never modifies the model's state (no .train() call).

    Args:
        model (nn.Module): Trained PyTorch network.
        device (str): 'cuda' or 'cpu'.
    """

    def __init__(self, model: nn.Module, device: str) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)

    def predict(
        self,
        inputs: List[torch.Tensor],
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts objects on a list of images.

        Args:
            inputs (List[Tensor]): Images in [C, H, W] format.
            confidence_threshold: Minimum score to keep a detection.

        Returns:
            List of dicts (one per image) with keys 'boxes', 'labels', 'scores'.
            Boxes are in absolute coordinates [xmin, ymin, xmax, ymax], on CPU.
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(inputs, list):
                inputs_device = torch.stack([img.to(self.device) for img in inputs])
            else:
                inputs_device = inputs.to(self.device)
            # raw_outputs = self.model(inputs_device)
            raw_outputs = self.model.predict(inputs_device, confidence_threshold=confidence_threshold)

        results = []
        for i, output in enumerate(raw_outputs):
            _, img_h, img_w = inputs[i].shape

            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]

            # Clip coordinates to image physical dimensions
            if boxes.shape[0] > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_h)

            # Filter by confidence threshold
            keep = scores >= confidence_threshold
            results.append({
                "boxes": boxes[keep].cpu(),
                "labels": labels[keep].cpu(),
                "scores": scores[keep].cpu(),
            })

        return results

    def predict_on_loader(
        self,
        dataloader: torch.utils.data.DataLoader,
        confidence_threshold: float = 0.5,
    ):
        """
        Runs inference over a complete DataLoader.

        Returns:
            (all_targets, all_preds): two lists of dicts.
        """
        all_targets = []
        all_preds = []

        for images, targets in dataloader:
            preds = self.predict(list(images), confidence_threshold=confidence_threshold)
            all_preds.extend(preds)
            all_targets.extend(list(targets))

        return all_targets, all_preds