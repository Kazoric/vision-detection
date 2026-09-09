# vision-detection

A modular, config-driven object detection framework built from scratch in PyTorch. It implements two detector families — **SSD** (anchor-based) and an **anchor-free YOLO** — on top of interchangeable backbones, necks, and heads, with a shared training loop, checkpointing, and evaluation pipeline.

## Features

- **Two detector architectures**
  - `SSDDetector` — SSD300-style detector with auto-generated priors/anchors and MultiBox loss.
  - `Yolo` — anchor-free, decoupled-head detector with LTRB distance regression, in the style of FCOS/YOLOX.
- **Modular building blocks**, each selected by name from a registry and wired together by the detector's `from_config` factory:
  - Backbones: `ResNet` (bottleneck ResNet-50 style), `ConvNeXt`, `ConvNeXtV2`.
  - Necks: `SSDNeck` (extra feature levels for SSD), `FPN`.
  - Heads: `SSDHead`, `YoloHead`.
- **Config system** (`core/config.py`): a set of dataclasses (`ExperimentConfig`, `ModelConfig`, `TrainingConfig`, `OptimizerConfig`, `SchedulerConfig`, `MetricsConfig`) that can be built from a plain dict (e.g. loaded from YAML) via `Config.from_dict`.
- **Trainer** (`core/trainer.py`): training/validation loop, LR scheduling (with optional linear warm-up composed with any `torch.optim.lr_scheduler`), best-model tracking on a configurable monitored metric, and metric computation.
- **Metrics** (`core/metrics.py`): IoU computation, prediction-to-ground-truth matching, and dataset-level TP/FP accumulation for mAP.
- **Checkpointing** (`core/checkpoint.py`): saves/restores model + optimizer state, epoch, and best metric value to `experiments/<run_id>/best_model.pt`.
- **Prediction** (`core/predictor.py`): inference wrapper that runs a trained model over single batches or a full `DataLoader` and returns boxes/labels/scores in absolute coordinates.
- **Visualization** (`core/visualizer.py`): plots training/validation loss and metric curves (matplotlib/seaborn), saved under `experiments/<run_id>/plots`.
- **Data loading** (`data_loader.py`): a generic dataset for images + YOLO-format `.txt` annotations, dataset mean/std computation, and a `get_detection_loaders` helper that builds train/val `DataLoader`s from a standard `train/valid` folder layout.

## Project structure

```
vision-detection/
├── core/
│   ├── config.py         # Dataclass-based experiment/model/training configuration
│   ├── model_base.py      # Abstract nn.Module base with optimizer/scheduler wiring
│   ├── trainer.py         # Training & evaluation loop
│   ├── predictor.py        # Inference helper
│   ├── checkpoint.py       # Checkpoint save/load
│   ├── metrics.py          # IoU, matching, mAP-related metrics
│   └── visualizer.py       # Loss/metric plotting
├── models/
│   ├── backbones/          # ResNet, ConvNeXt, ConvNeXtV2 (+ registry)
│   ├── necks/               # FPN, SSDNeck (+ registry)
│   ├── heads/                # SSDHead, YoloHead (+ registry)
│   └── detectors/            # SSDDetector, Yolo, BaseDetector
├── loss/
│   ├── ssd_loss.py          # SSD MultiBox loss
│   └── yolo_loss.py         # YOLO loss
├── utils/
│   └── ssd_utils.py         # SSD prior/anchor generation
└── data_loader.py            # Dataset, transforms, DataLoader factory
```

## Requirements

The code relies on:

- `torch` and `torchvision`
- `numpy`
- `Pillow` (`PIL`)
- `tqdm`
- `matplotlib` and `seaborn` (for `core/visualizer.py`)

There is no `requirements.txt` in the repository yet; install the packages above (ideally with a CUDA-enabled PyTorch build if you plan to train on GPU).

```bash
pip install torch torchvision numpy pillow tqdm matplotlib seaborn
```

## Dataset format

Datasets are expected in YOLO annotation format, laid out as:

```
data/<DatasetName>/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

Each label file is a `.txt` with one line per object: `class_id x_center y_center width height`, all normalized to `[0, 1]`. `get_detection_loaders` in `data_loader.py` will:

- fall back to the training split for validation if no `valid/` folder exists,
- compute (and cache to `stats.json`) per-dataset normalization mean/std unless disabled,
- return ready-to-use `train_loader` / `val_loader` objects.

## Usage

The entry point is `run.py`. The whole experiment — dataset, backbone, neck, head, detector, training and optimizer settings — is described in a single config dict (`yaml_config` in `run.py`), turned into a `Config` object via `Config.from_dict`.

### Running an experiment

```bash
python run.py
```

This will load the dataset, build the model, train it, save the best checkpoint, plot metrics, run a sample inference on the first validation image, and compute final mAP/F1.

### Configuring a model

Every building block (backbone, neck, head, detector) is selected by a `type` string, resolved through a registry, so switching architectures is a pure config change (editing the `yaml_config` dict in `run.py`) :

```python
"model": {
    "image_size": [640, 640],
    "num_classes": 13,
    "score_thresh": 0.25,
    "iou_thresh": 0.45,
    "backbone": {
        "type": "ConvNeXtV2",              # "ResNet" | "ConvNeXt" | "ConvNeXtV2"
        "out_indices": ["c3", "c4", "c5"],
        "backbone_kwargs": {
            "dims": [40, 80, 160, 320],
            "depths": [2, 2, 6, 2],
        },
    },
    "neck": {
        "type": "FPN",                      # "FPN" | "SSDNeck"
        "out_channels": 256,
    },
    "head": {
        "type": "YoloHead",                 # "YoloHead" | "SSDHead"
    },
    "detector": {
        "type": "Yolo",                     # "Yolo" | "SSD"
        "detector_kwargs": {
            "strides": [8, 16, 32],
        },
    },
}
```

The model itself is instantiated in one line, regardless of which detector/backbone/neck/head combination the config describes:

```python
from models.detectors import build_detector

model = build_detector(config)
```

Swapping `Yolo` for `SSDDetector`, or `ConvNeXtV2` for `ResNet`, is purely a config change — e.g. set `model.detector.type` to `"SSD"` and `model.backbone.type` to `"ResNet"`.

### Adding a new backbone / neck / head / detector

Each component type has its own registry under `models/<component>/registry.py`. To add a new one:

1. Implement the class (e.g. a new backbone in `models/backbones/`).
2. Register it with the matching decorator:
```python
   from .registry import register_backbone

   @register_backbone("EfficientNet")
   class EfficientNetBackbone(nn.Module):
       ...
```
3. Import the new module from the component's `__init__.py` so the decorator actually runs at import time:
```python
   # models/backbones/__init__.py
   from .efficientnet import EfficientNetBackbone
```
4. Reference it by name in the config (`"backbone": {"type": "EfficientNet", ...}`) — no other code changes required.

The same pattern applies to `models/necks/`, `models/heads/`, and `models/detectors/`.

### Configuring the rest of the experiment

Besides `model`, the config dict covers everything else needed to run and track an experiment, each mapped to its own dataclass in `core/config.py`.

**`experiment`** (`ExperimentConfig`) — identifies the run and controls checkpointing:
```python
"experiment": {
    "dataset_name": "Chess",           # expects data/<dataset_name>/train and /valid
    "run_id": "yolo_convnextv2_test",  # used for experiments/<run_id>/... paths
    "save_checkpoints": True,          # whether the Trainer saves the best model
}
```

**`training`** (`TrainingConfig`) — core training loop hyperparameters:
```python
"training": {
    "lr": 0.0001,
    "batch_size": 16,
    "epochs": 10,
    "warm_up_epochs": 0,               # linear LR warm-up, composed with the scheduler below
}
```

**`optimizer`** (`OptimizerConfig`) — passed straight through to a `torch.optim` optimizer:
```python
"optimizer": {
    "type": "AdamW",
    "params": {
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
    },
}
```

**`scheduler`** (`SchedulerConfig`, optional) — any `torch.optim.lr_scheduler`, wrapped by the `Trainer` and combined with `warm_up_epochs` if set. Comment it out to train at a constant LR:
```python
"scheduler": {
    "type": "CosineAnnealingLR",
    "params": {
        "T_max": 50,
        "eta_min": 1e-5,
    },
},
```

**`metrics`** (`MetricsConfig`) — which metric the `Trainer` tracks for best-checkpoint selection, plus any additional metrics computed at evaluation time. Each entry maps a metric name to a `(function_name, kwargs)` pair resolved against `core/metrics.py`:
```python
"metrics": {
    "monitor_metric": "mAP",           # this is what CheckpointManager watches
    "monitor_mode": "max",             # "max" or "min"
    "mAP": ("raw_compute_map", {"iou_threshold": 0.5}),
}
```
Extra metrics can be added at runtime the same way `run.py` does before final evaluation:
```python
config.metrics.configs["detection_f1"] = ("raw_compute_precision_recall_f1", {"iou_threshold": 0.5})
```

Together, `experiment` + `training` + `optimizer` + `scheduler` + `metrics` + `model` form the single `yaml_config` dict passed to `Config.from_dict(...)` — this is the only place you need to edit to change an experiment.

## Notes

- Experiment artifacts (checkpoints, metadata, plots) are written to `experiments/<run_id>/` and are git-ignored, along with `data/`, `checkpoints/`, and local scratch files.
