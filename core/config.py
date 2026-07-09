from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

@dataclass
class ExperimentConfig:
    dataset_name: str
    run_id: Optional[str] = None
    save_checkpoints: bool = True

@dataclass
class ModelConfig:
    name: str
    num_classes: int

@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    warm_up: bool = False
    warm_up_epochs: int = 5

@dataclass
class OptimizerConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SchedulerConfig:
    type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricsConfig:
    monitor_metric: str = "mAP"
    monitor_mode: str = "max"
    configs: Dict[str, Any] = field(
        default_factory=lambda: {"mAP": ("raw_compute_map", {"iou_threshold": 0.5})}
    )

@dataclass
class Config:
    """ Dataclass maîtresse regroupant toutes les sous-configurations """
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    metrics: MetricsConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """ Crée récursivement l'objet Config depuis un dictionnaire standard (ex: YAML) """
        metrics_raw = d.get("metrics", {}).copy()
        
        monitor_metric = metrics_raw.pop("monitor_metric", "mAP")
        monitor_mode = metrics_raw.pop("monitor_mode", "max")
        
        return cls(
            experiment=ExperimentConfig(**d["experiment"]),
            model=ModelConfig(**d["model"]),
            training=TrainingConfig(**d["training"]),
            optimizer=OptimizerConfig(**d["optimizer"]),
            scheduler=SchedulerConfig(**d.get("scheduler", {"type": None, "params": {}})),
            metrics=MetricsConfig(
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                configs=metrics_raw
            )
        )