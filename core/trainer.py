import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional, List, Dict

import core.metrics as core_metrics
from core.metrics import compute_dataset_tp_fp


class Trainer:
    """
    Trains an object detection model and evaluates its metrics.

    Args:
        model (nn.Module): PyTorch network to train.
        optimizer (Optimizer): Associated optimizer.
        device (str): 'cuda' or 'cpu'.
        scheduler (LRScheduler, optional): Learning rate scheduler.
        num_classes (int): Total number of classes (including background).
        metrics_config (MetricsConfig, optional): Metrics configuration (from Config.metrics).
        on_best_model (callable, optional): Callback triggered when a new best model is found.
                                            Signature: fn(epoch: int, metric_value: float)
                                            Typically: checkpoint.save
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        num_classes: Optional[int] = None,
        metrics_config = None,  # MetricsConfig | None
        on_best_model: Optional[Callable[[int, float], None]] = None,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.metrics_config = metrics_config
        self.on_best_model = on_best_model

        # Best metric tracking parameters
        if metrics_config is not None:
            self.monitor_metric = metrics_config.monitor_metric
            self.monitor_mode = metrics_config.monitor_mode.lower()
        else:
            self.monitor_metric = "val_loss"
            self.monitor_mode = "min"

        assert self.monitor_mode in ("max", "min"), \
            "monitor_mode must be 'max' or 'min'"

        # History logs
        self.train_loss: List[float] = []
        self.valid_loss: List[float] = []
        self.lr_history: List[float] = []
        self.valid_metrics: Dict[str, List[float]] = {}

        # Internal state
        self.start_epoch = 0
        self.best_metric_value = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch_metrics: dict = {}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> None:
        """Main training loop."""

        for epoch in range(self.start_epoch, epochs):

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)

            self.model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for images, targets in pbar:
                images_device = [img.to(self.device, non_blocking=True) for img in images]
                targets_device = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                loss_dict = self.model(images_device, targets_device)
                losses = sum(loss_dict.values())
                losses.backward()
                self.optimizer.step()

                running_loss += losses.item()
                pbar.set_postfix({"batch_loss": f"{losses.item():.4f}"})

            epoch_train_loss = running_loss / len(train_loader)
            self.train_loss.append(epoch_train_loss)
            print(f"{'Train':<12} | Loss: {epoch_train_loss:.4f} | LR: {current_lr:.2e}")

            # --- Validation ---
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                self._maybe_save_best(epoch, val_loss, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            print()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data_loader: DataLoader):
        """
        Computes validation loss and object detection metrics.

        The model remains in `.train()` mode to compute losses
        (torchvision behavior), but gradients are disabled.
        A second pass in `.eval()` provides decoded predictions for evaluation metrics.

        Returns:
            (val_loss: float, metric_outputs: dict)
        """
        self.model.train()
        running_loss = 0.0
        all_preds: List[dict] = []
        all_targets: List[dict] = []

        with torch.no_grad():
            for images, targets in data_loader:
                images_device = [img.to(self.device, non_blocking=True) for img in images]
                targets_device = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images_device, targets_device)
                running_loss += sum(loss_dict.values()).item()

                if self.metrics_config is not None:
                    self.model.eval()
                    if hasattr(self.model, "predict"):
                        preds = self.model.predict(images_device, confidence_threshold=0.15)
                    else:
                        preds = self.model(images_device)
                    self.model.train()

                    all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
                    all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

        val_loss = running_loss / len(data_loader)
        self.valid_loss.append(val_loss)

        # --- Metrics ---
        metric_outputs: dict = {}
        metrics_str = ""
        if self.metrics_config is not None and all_preds:
            metric_outputs = self._compute_metrics(all_targets, all_preds)
            for name, value in metric_outputs.items():
                self.valid_metrics.setdefault(name, []).append(value)
            metrics_str = " | " + " | ".join(f"{n}: {v:.4f}" for n, v in metric_outputs.items())

        print(f"{'Validation':<12} | Loss: {val_loss:.4f}{metrics_str}")
        return val_loss, metric_outputs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        targets: List[Dict[str, torch.Tensor]],
        predictions: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Computes all configured metrics reusing a single TP/FP pass.
        """
        iou_threshold = 0.5  # Default value
        # Try retrieving the threshold from the first found config
        for _, params in self.metrics_config.configs.values():
            if "iou_threshold" in params:
                iou_threshold = params["iou_threshold"]
                break

        raw_data = compute_dataset_tp_fp(
            predictions=predictions,
            targets=targets,
            num_classes=self.num_classes,
            iou_threshold=iou_threshold,
        )

        metric_outputs: dict = {}
        for name, (func, params) in self.metrics_config.configs.items():
            # Dynamic resolution if func is a string
            if isinstance(func, str):
                if not hasattr(core_metrics, func):
                    raise AttributeError(f"Function '{func}' not found in core.metrics")
                func = getattr(core_metrics, func)

            score = func(**raw_data, num_classes=self.num_classes, **params)

            if isinstance(score, dict):
                metric_outputs.update(score)
            else:
                metric_outputs[name] = score

        return metric_outputs

    def _is_better(self, value: float) -> bool:
        if self.monitor_mode == "max":
            return value > self.best_metric_value
        return value < self.best_metric_value

    def _maybe_save_best(self, epoch: int, val_loss: float, val_metrics: dict) -> None:
        """Decides whether to save and updates best_epoch_metrics."""
        if self.monitor_metric in ("loss", "val_loss"):
            current_value = val_loss
        else:
            default = float("-inf") if self.monitor_mode == "max" else float("inf")
            current_value = val_metrics.get(self.monitor_metric, default)

        if self._is_better(current_value):
            self.best_metric_value = current_value
            self.best_epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": self.train_loss[-1],
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "monitor_metric": self.monitor_metric,
                "monitor_value": current_value,
            }
            if self.on_best_model is not None:
                self.on_best_model(epoch + 1, current_value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_final_metrics(self) -> dict:
        """Returns metrics from the best epoch."""
        return self.best_epoch_metrics

    def resume_from(self, epoch: int, best_metric_value: float) -> None:
        """
        Restores Trainer internal state after loading a checkpoint.

        Args:
            epoch: Epoch to resume from (= epoch_saved + 1).
            best_metric_value: Best known metric value prior to resuming.
        """
        self.start_epoch = epoch
        self.best_metric_value = best_metric_value