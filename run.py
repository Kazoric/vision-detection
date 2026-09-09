# This script serves as the main entry point for the object detection training pipeline.
# It handles configuration, data loading, model setup, training, validation, and final inference.

import os
import json
import time
import torch

# Core modules imports
from core.config import Config
from core.trainer import Trainer
from core.predictor import Predictor
from core.checkpoint import CheckpointManager
from core.visualizer import Visualizer

# Data and Model imports
from data_loader import get_detection_loaders
from models.detectors import build_detector


# Defines the class names used for the chess dataset.
chess_classes = [
    "background",
    "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook",
    "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook",
]

# Detailed configuration dictionary for the experiment.
yaml_config = {
    "experiment": {
        "dataset_name": "Chess",
        "run_id": "yolo_convnextv2_test",
        "save_checkpoints": True
    },
    "model": {
        "image_size": [640, 640],
        "num_classes": 13,
        "score_thresh": 0.25,
        "iou_thresh": 0.45,
        "backbone": {
            "type": "ConvNeXtV2",
            "out_indices": ["c3", "c4", "c5"],
            "backbone_kwargs": {
                "dims": [40, 80, 160, 320],
                "depths": [2, 2, 6, 2]
            }
        },
        "neck": {
            "type": "FPN",
            "out_channels": 256
        },
        "head": {
            "type": "YoloHead"
        },
        "detector": {
            "type": "Yolo",
            "detector_kwargs": {
                "strides": [8, 16, 32]
            }
        },
    },
    "training": {
        "lr": 0.0001,
        "batch_size": 16,
        "epochs": 10,
        "warm_up_epochs": 0
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "weight_decay": 0.0001,
            "betas": [
                0.9,
                0.999
            ]
        }
    },
    # "scheduler": {
    #     "type": "CosineAnnealingLR",
    #     "params": {
    #         "T_max": 50,
    #         "eta_min": 1e-5,
    #     },
    # },
    "metrics": {
        "monitor_metric": "mAP",
        "monitor_mode": "max",
        "mAP": ("raw_compute_map", {"iou_threshold": 0.5}),
    }
}

def main():
    # ----------------------------------------------------------------------
    # STEP 1: INITIALIZE CONFIGURATION AND DEVICE
    # ----------------------------------------------------------------------
    # Load the configuration from the dictionary.
    config = Config.from_dict(yaml_config)

    # Determine the computation device (CUDA if available, otherwise CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging initial setup information.
    print(f"  Device        : {device}")
    print(f"  Num classes   : {config.model.num_classes}")

    # ----------------------------------------------------------------------
    # STEP 2: LOAD DATASET
    # ----------------------------------------------------------------------
    print("\n=== [STEP 2] Dataset ===")
    # Initialize data loaders for training and validation sets.
    train_loader, val_loader = get_detection_loaders(
        dataset_root=f"data/{config.experiment.dataset_name}",
        batch_size=config.training.batch_size,
        image_size=config.model.image_size,
        use_computed_stats=True,
    )

    # ----------------------------------------------------------------------
    # STEP 3: INITIALIZE MODEL
    # ----------------------------------------------------------------------
    print("\n=== [STEP 3] Model ===")
    # Instantiate the YOLO model architecture based on the configuration.
    model = build_detector(config)

    # Log model specifics like name and total parameters.
    print(f"  Model        : {model.name}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    # ----------------------------------------------------------------------
    # STEP 4: ASSEMBLE TRAINER AND CHECKPOINT MANAGER
    # ----------------------------------------------------------------------
    print("\n=== [STEP 4] Assembly ===")
    # Create the Trainer instance, bundling the model, config, and device.
    trainer = Trainer.from_config(
        model=model,
        config=config,
        device=device,
        metrics_config=config.metrics if config.metrics.configs else None,
    )

    # Setup the CheckpointManager to save and load training state.
    checkpoint = CheckpointManager(
        model=model,
        optimizer=trainer.optimizer,
        run_id=trainer.run_id,
        monitor_metric=config.metrics.monitor_metric,
    )

    # Configure the trainer to save the best model checkpoint.
    if config.experiment.save_checkpoints:
        trainer.on_best_model = checkpoint.save

    # Log the unique run identifier.
    print(f"  Run ID        : {trainer.run_id}")

    # Initialize the Predictor and Visualizer components.
    predictor = Predictor(model=model, device=device)
    visualizer = Visualizer()

    # ----------------------------------------------------------------------
    # RESUME LOGIC
    # ----------------------------------------------------------------------
    RESUME = False
    if RESUME and checkpoint.exists():
        # Load training state if RESUME is True and a checkpoint exists.
        state = checkpoint.load(load_optimizer=True)
        trainer.resume_from(
            epoch=state["epoch"],
            best_metric_value=state["best_metric_value"],
        )

    # ----------------------------------------------------------------------
    # STEP 5: START TRAINING
    # ----------------------------------------------------------------------
    print(f"\n=== [STEP 5] Training ({config.training.epochs} epochs) ===")
    start_time = time.time()
    # Execute the training loop on both training and validation data.
    trainer.train(train_loader, val_loader, epochs=config.training.epochs)
    end_time = time.time() - start_time
    print(f"Training took {end_time:.2f} seconds\n")

    # ----------------------------------------------------------------------
    # STEP 6: FINAL EVALUATION AND METRICS VISUALIZATION
    # ----------------------------------------------------------------------
    print("\n=== [STEP 6] Final Evaluation & Saving ===")

    if checkpoint.exists():
        # Load model state without optimizer for final evaluation.
        checkpoint.load(load_optimizer=False)

    # Generate and save visual reports of training metrics.
    visualizer.plot_metrics(trainer, run_id=trainer.run_id, save=True)

    # ----------------------------------------------------------------------
    # STEP 7: CONTROL INFERENCE (INFERENCE EXAMPLE)
    # ----------------------------------------------------------------------
    print("\n=== [STEP 7] Inference Control ===")

    # Select the first image from the validation dataset for a demonstration.
    real_dataset = val_loader.dataset
    img_tensor, gt = real_dataset[0]
    img_name = real_dataset.img_names[0]
    orig_img_path = os.path.join(real_dataset.img_dir, img_name)

    # Perform prediction on the single image.
    predictions = predictor.predict([img_tensor], confidence_threshold=0.15)

    # Load statistics (like mean and standard deviation) from a stats file.
    stats_file = "data/Chess/stats.json"
    with open(stats_file, "r") as f:
        stats = json.load(f)

    # Visualize the prediction overlaid on the original image, comparing it to ground truth.
    visualizer.plot_predictions(
        image=img_tensor,
        prediction=predictions[0],
        class_names=chess_classes,
        confidence_threshold=0.15,
        orig_image=orig_img_path,
        ground_truth=gt,
        mean=stats["mean"],
        std=stats["std"],
        run_id=trainer.run_id,
        save=True,
        filename="inference_comparison.png",
    )

    # Update metrics configuration to include F1 score calculation.
    config.metrics.configs["detection_f1"] = ("raw_compute_precision_recall_f1", {"iou_threshold": 0.5})

    # Calculate and log final metrics (mAP and F1).
    print("Calculating final metrics (mAP + detection_f1)...")
    final_val_loss, final_val_metrics = trainer.evaluate(val_loader)

    # Compile all final results into a dictionary.
    final_results = trainer.get_final_metrics()
    final_results["final_val_metrics"] = final_val_metrics

    # Save the complete set of hyperparameters and final results.
    trainer.save_hyperparams(extra_results=final_results)

    print(f"\n✅ Pipeline complete — run_id : {trainer.run_id}")

if __name__ == "__main__":
    main()