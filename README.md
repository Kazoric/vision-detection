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

<!-- ## Usage

The repository currently exposes the framework's building blocks (config, models, trainer, data loading) as a library rather than a ready-made CLI script — `run.py` is listed in `.gitignore`, so each user is expected to write their own entry-point script. A typical training script looks like this:

```python
from core.config import Config
from core.trainer import Trainer
from core.checkpoint import CheckpointManager
from data_loader import get_detection_loaders
from models.detectors.ssd import SSDDetector   # or models.detectors.yolo.Yolo

# 1. Load configuration (e.g. from a YAML file parsed into a dict)
config = Config.from_dict(config_dict)

# 2. Build data loaders
train_loader, val_loader = get_detection_loaders(
    dataset_root="data/MyDataset",
    batch_size=config.training.batch_size,
    image_size=config.model.image_size,
)

# 3. Build the model from config
model = SSDDetector.from_config(config)

# 4. Build the trainer and run training
trainer = Trainer.from_config(model, config, device=model.device)
trainer.train(train_loader, val_loader, epochs=config.training.epochs)

# 5. Save results
model.save_hyperparams(trainer.get_final_metrics())
```

For inference on a trained model, use `core/predictor.py`:

```python
from core.predictor import Predictor

predictor = Predictor(model, device="cuda")
predictions = predictor.predict(images, confidence_threshold=0.5)
```

## Model configuration example

A minimal config dict for an SSD model with a ResNet backbone might look like:

```python
config_dict = {
    "experiment": {"dataset_name": "MyDataset"},
    "model": {
        "name": "SSD300_ResNet",
        "num_classes": 11,  # includes background class
        "image_size": [300, 300],
        "backbone": {"type": "ResNet", "out_indices": ["c3", "c4", "c5"]},
        "neck": {"type": "SSDNeck", "source_layer": "c5", "extra_channels": [512, 256, 256]},
        "head": {"type": "SSDHead"},
    },
    "training": {"lr": 1e-3, "batch_size": 8, "epochs": 50, "warm_up_epochs": 2},
    "optimizer": {"type": "AdamW", "params": {"weight_decay": 5e-4}},
    "scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": 48}},
    "metrics": {"monitor_metric": "mAP", "monitor_mode": "max"},
}
```

Swapping `models.detectors.ssd.SSDDetector` for `models.detectors.yolo.Yolo` (and using an `FPN` neck / `YoloHead`) builds the anchor-free YOLO detector instead, reusing the same config structure, trainer, and data pipeline.

## Notes

- Experiment artifacts (checkpoints, metadata, plots) are written to `experiments/<run_id>/` and are git-ignored, along with `data/`, `checkpoints/`, and local scratch files.
- Code comments and some docstrings are in French; the public class/method docstrings referenced above have been translated to English in this README for clarity. -->
