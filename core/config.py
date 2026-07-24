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
    score_thresh: float = 0.25
    iou_thresh: float = 0.45
    # Dictionnaires de configuration pour la fabrique (build) des sous-modules
    backbone: Dict[str, Any] = field(default_factory=dict)
    neck: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None
    strides: List[int] = field(default_factory=lambda: [8, 16, 32])  # P3=8, P4=16, P5=32

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=d["name"],
            num_classes=d["num_classes"],
            score_thresh=d.get("score_thresh", 0.25),
            iou_thresh=d.get("iou_thresh", 0.45),
            backbone=d.get("backbone", {}),
            neck=d.get("neck"),
            head=d.get("head"),
            strides=d.get("strides", [8, 16, 32]),
        )

@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    warm_up_epochs: int = 0

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
            model=ModelConfig.from_dict(d["model"]),
            training=TrainingConfig(**d["training"]),
            optimizer=OptimizerConfig(**d["optimizer"]),
            scheduler=SchedulerConfig(**d.get("scheduler", {"type": None, "params": {}})),
            metrics=MetricsConfig(
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                configs=metrics_raw
            )
        )