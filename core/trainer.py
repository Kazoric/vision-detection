import os
import json
from datetime import datetime
from dataclasses import is_dataclass, asdict
from typing import Callable, Optional, List, Dict, Any

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import core.metrics as core_metrics
from core.metrics import compute_dataset_tp_fp


class Trainer:
    """
    Manages the training, validation loop, optimization, and experiment tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        scheduler: Optional[LRScheduler] = None,
        num_classes: Optional[int] = None,
        metrics_config: Optional[Any] = None,
        on_best_model: Optional[Callable[[int, float], None]] = None,
        config: Optional[Any] = None,
        run_id: Optional[str] = None,
    ) -> None:

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.metrics_config = metrics_config
        self.on_best_model = on_best_model
        self.config = config

        # --- Setup Run ID ---
        if run_id is None and config is not None:
            model_name = getattr(model, "name", model.__class__.__name__)
            dataset_name = getattr(config.experiment, "dataset_name", "dataset")
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"{model_name}_{dataset_name}_{date}"
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Best metric tracking parameters
        if metrics_config is not None:
            self.monitor_metric = metrics_config.monitor_metric
            self.monitor_mode = metrics_config.monitor_mode.lower()
        else:
            self.monitor_metric = "val_loss"
            self.monitor_mode = "min"

        assert self.monitor_mode in ("max", "min"), "monitor_mode must be 'max' or 'min'"

        # History logs
        self.train_loss: List[float] = []
        self.valid_loss: List[float] = []
        self.lr_history: List[float] = []
        self.valid_metrics: Dict[str, List[float]] = {}

        # Internal state
        self.start_epoch = 0
        self.best_metric_value = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch_metrics: dict = {}

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: Any,
        device: str = "cpu",
        metrics_config: Optional[Any] = None,
        on_best_model: Optional[Callable[[int, float], None]] = None,
    ) -> "Trainer":
        """
        Factory method building Optimizer, Scheduler, and Trainer instance from a configuration object.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Build Optimizer
        optimizer_cls = getattr(optim, config.optimizer.type)
        optimizer_params = getattr(config.optimizer, "params", {})
        optimizer = optimizer_cls(
            model.parameters(),
            lr=config.training.lr,
            **optimizer_params
        )

        # 2. Build Scheduler
        scheduler = cls._build_scheduler(optimizer, config)

        # 3. Build Run ID
        run_id = getattr(config.experiment, "run_id", None)

        num_classes = getattr(config.model, "num_classes", None)

        return cls(
            model=model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            num_classes=num_classes,
            metrics_config=metrics_config,
            on_best_model=on_best_model,
            config=config,
            run_id=run_id,
        )

    @staticmethod
    def _build_scheduler(optimizer: torch.optim.Optimizer, config: Any) -> Optional[LRScheduler]:
        """Builds warmup + main learning rate scheduler from configuration."""
        has_warmup = getattr(config.training, "warm_up_epochs", 0) > 0
        has_main = getattr(config.scheduler, "type", None) is not None

        if not has_warmup and not has_main:
            return None

        warmup_sched = None
        if has_warmup:
            warmup_sched = LinearLR(
                optimizer,
                start_factor=0.05,
                end_factor=1.0,
                total_iters=config.training.warm_up_epochs,
            )

        main_sched = None
        if has_main:
            scheduler_cls = getattr(optim.lr_scheduler, config.scheduler.type)
            scheduler_params = getattr(config.scheduler, "params", {})
            main_sched = scheduler_cls(optimizer, **scheduler_params)

        if warmup_sched and main_sched:
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[config.training.warm_up_epochs],
            )

        return warmup_sched or main_sched

    # ------------------------------------------------------------------
    # Training Loop & Evaluation
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
                images_device = torch.stack(images).to(self.device, non_blocking=True)
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

    def evaluate(self, data_loader: DataLoader):
        """Computes validation loss and metrics."""
        self.model.train()
        running_loss = 0.0
        all_preds: List[dict] = []
        all_targets: List[dict] = []

        with torch.no_grad():
            for images, targets in data_loader:
                images_device = torch.stack(images).to(self.device, non_blocking=True)
                targets_device = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images_device, targets_device)
                running_loss += sum(loss_dict.values()).item()

                if self.metrics_config is not None:
                    self.model.eval()
                    if hasattr(self.model, "predict"):
                        preds = self.model.predict(images_device, confidence_threshold=0.15)
                    else:
                        loc_preds, cls_preds = self.model(images_device)
                        preds = self.model.predict_decoded(loc_preds, cls_preds)
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
    # Metrics & Metadata Helpers
    # ------------------------------------------------------------------

    def save_hyperparams(self, extra_results: Optional[Dict] = None) -> None:
        """Saves configuration and results to experiments/<run_id>/meta.json."""
        if self.config is None:
            print("[WARN] No config bound to Trainer. Skipping saving hyperparams.")
            return

        meta = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        
        # In case experiment field is a dict or dataclass
        if isinstance(meta.get("experiment"), dict):
            meta["experiment"]["run_id"] = self.run_id

        if extra_results is not None:
            meta["results"] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "best_validation_results": extra_results,
            }

        path = os.path.join("experiments", self.run_id, "meta.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

        print(f"[INFO] Configuration saved: {path}")

    def _compute_metrics(
        self,
        targets: List[Dict[str, torch.Tensor]],
        predictions: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        iou_threshold = 0.5
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

    def get_final_metrics(self) -> dict:
        return self.best_epoch_metrics

    def resume_from(self, epoch: int, best_metric_value: float) -> None:
        self.start_epoch = epoch
        self.best_metric_value = best_metric_value