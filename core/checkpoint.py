import os
from typing import Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages model checkpoint saving and loading.

    Saves into: experiments/<run_id>/best_model.pth

    The checkpoint stores:
      - model_state_dict
      - optimizer_state_dict
      - epoch (number of the epoch just completed)
      - monitor_metric (name of monitored metric)
      - best_metric_value (best metric value seen so far)

    Typical usage in execution script:
        ckpt = CheckpointManager(
            model=net, optimizer=optim, run_id="my_exp", monitor_metric="mAP"
        )
        # Save when a new best model is found:
        ckpt.save(epoch=5, metric_value=0.82)
        # To resume training:
        state = ckpt.load(load_optimizer=True)
        start_epoch = state["epoch"] + 1
        best_metric_val = state["best_metric_value"]
    """

    FILENAME = "best_model.pt"

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        run_id: str,
        monitor_metric: str = "mAP",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.run_id = run_id
        self.monitor_metric = monitor_metric
        self.save_dir = os.path.join("experiments", run_id)

    # ------------------------------------------------------------------
    # Utility property
    # ------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.save_dir, self.FILENAME)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, epoch: int, metric_value: float) -> None:
        """
        Saves current checkpoint.

        Args:
            epoch: Number of the epoch just completed.
            metric_value: Current value of the monitored metric.
        """
        os.makedirs(self.save_dir, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "monitor_metric": self.monitor_metric,
            "best_metric_value": metric_value,
        }

        torch.save(payload, self.checkpoint_path)
        print(
            f"[Checkpoint] Saved (epoch {epoch}, "
            f"{self.monitor_metric}={metric_value:.4f}) → {self.checkpoint_path}"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, load_optimizer: bool = True) -> dict:
        """
        Loads latest saved checkpoint and restores states.

        Args:
            load_optimizer: If True, also restores optimizer state.

        Returns:
            The complete checkpoint dict. Retrieves state["epoch"]
            and state["best_metric_value"] in calling script.

        Raises:
            FileNotFoundError: If no checkpoint exists for this run_id.
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"[Checkpoint] No checkpoint found: {self.checkpoint_path}"
            )

        state = torch.load(self.checkpoint_path, map_location="cpu")

        self.model.load_state_dict(state["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        epoch = state.get("epoch", 0)
        metric_value = state.get("best_metric_value", None)
        print(
            f"[Checkpoint] Loaded from {self.checkpoint_path} "
            f"— resuming at epoch {epoch + 1}"
            + (f", best {self.monitor_metric}={metric_value:.4f}" if metric_value is not None else "")
        )

        return state

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Returns True if a checkpoint exists for this run."""
        return os.path.exists(self.checkpoint_path)