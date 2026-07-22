import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR

from core.config import Config


class Model(nn.Module, ABC):
    """
    Abstract base class for all detection models, inheriting from nn.Module.

    Responsibilities:
      - Define the common abstract interface (name, _build_architecture, forward)
      - Call _build_architecture() so that the subclass instantiates its PyTorch layers
      - Move the module to the device (cuda / cpu)
      - Initialize the optimizer (self.optimizer) and the scheduler (self.scheduler)
      - Handle the run_id and save the meta-configuration (save_hyperparams)
    """

    def __init__(self, config: Config, device: Optional[str] = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Attributes extracted from the config ---
        self.num_classes = config.model.num_classes
        self.dataset_name = config.experiment.dataset_name
        self.lr = config.training.lr

        # --- Run ID ---
        run_id = config.experiment.run_id
        if run_id is None:
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"{self.name}_{self.dataset_name}_{date}"
        self.run_id = run_id

        # --- Instantiation of PyTorch layers from the subclass ---
        self._build_architecture()

        # Move module to the device (CUDA / CPU)
        self.to(self.device)

        # --- Optimizer ---
        optimizer_cls = getattr(optim, config.optimizer.type)
        self.optimizer = optimizer_cls(
            self.parameters(),
            lr=self.lr,
            **config.optimizer.params
        )

        # --- Scheduler (warm-up + main scheduler) ---
        self.scheduler = self._build_scheduler(config)

    # ------------------------------------------------------------------
    # Abstract Interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Textual identifier of the model (e.g., 'SSD300_ResNet')."""
        pass

    @abstractmethod
    def _build_architecture(self) -> None:
        """
        Abstract method called during initialization.
        The subclass must instantiate its PyTorch sub-modules here
        (e.g., self.resnet = ..., self.loc_layers = ..., self.criterion = ...).
        """
        pass

    @abstractmethod
    def forward(self, images, targets=None):
        """
        Forward pass of the PyTorch model.
        In training mode (self.training=True): returns a dictionary of losses.
        In evaluation mode (self.training=False): returns decoded predictions.
        """
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_scheduler(self, config: Config) -> Optional[object]:
        """Builds the scheduler from the configuration."""
        has_warmup = config.training.warm_up_epochs > 0
        has_main = config.scheduler.type is not None

        if not has_warmup and not has_main:
            return None

        warmup_sched = None
        if has_warmup:
            warmup_sched = LinearLR(
                self.optimizer,
                start_factor=0.05,
                end_factor=1.0,
                total_iters=config.training.warm_up_epochs,
            )

        main_sched = None
        if has_main:
            scheduler_cls = getattr(optim.lr_scheduler, config.scheduler.type)
            main_sched = scheduler_cls(self.optimizer, **config.scheduler.params)

        if warmup_sched and main_sched:
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[config.training.warm_up_epochs],
            )

        return warmup_sched or main_sched

    def save_hyperparams(self, extra_results: Optional[Dict] = None) -> None:
        """
        Saves the complete configuration and final results in
        experiments/<run_id>/meta.json.
        """
        meta = asdict(self.config)
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