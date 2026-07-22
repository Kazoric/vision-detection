"""
Script d'exécution principal.

Architecture découplée :
  - Model    : modèle PyTorch (nn.Module) gérant son réseau, son optimiseur et son scheduler
  - Trainer  : gère la boucle d'entraînement et les métriques
  - Predictor: gère l'inférence post-entraînement
  - Checkpoint: gère la sauvegarde/reprise
  - Visualizer: génère les graphiques
"""

import torch

from core.config import Config
from core.trainer import Trainer
from core.predictor import Predictor
from core.checkpoint import CheckpointManager
from core.visualizer import Visualizer

from data_loader import get_detection_loaders
from models.ssd_resnet import SSDResNetModel
# from models.yolo import YOLOModel
# from models.ssd_model import SSDModel
# from models.ssd_vgg16 import SSDVGG16Model

def main():
    # ==============================================================================
    # CLASSES
    # ==============================================================================
    chess_classes = [
        "background",
        "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook",
        "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook",
    ]

    # ==============================================================================
    # STEP 1 — Configuration
    # ==============================================================================
    print("=== [STEP 1] Configuration ===")

    yaml_config = {
        "experiment": {
            "dataset_name": "Chess",
            "run_id": "ssd_resnet_new",
            "save_checkpoints": True,
        },
        "model": {
            "name": "SSD_ResNet",
            "num_classes": len(chess_classes),
        },
        "training": {
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 20,
            "warm_up_epochs": 0,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"weight_decay": 0.0005},
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "params": {"T_max": 20},
        },
        "metrics": {
            "monitor_metric": "mAP",
            "monitor_mode": "max",
            "mAP": ("raw_compute_map", {"iou_threshold": 0.5}),
        },
    }

    config = Config.from_dict(yaml_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device        : {device}")
    print(f"  Run ID        : {config.experiment.run_id}")
    print(f"  Num classes   : {config.model.num_classes}")

    # ==============================================================================
    # STEP 2 — Dataset
    # ==============================================================================
    print("\n=== [STEP 2] Dataset ===")
    train_loader, val_loader = get_detection_loaders(
        dataset_root="data/Chess",
        batch_size=config.training.batch_size,
        image_size=(300, 300),
        use_computed_stats=True,
    )

    # ==============================================================================
    # STEP 3 — Modèle (nn.Module + optimiseur + scheduler)
    # ==============================================================================
    print("\n=== [STEP 3] Modèle ===")
    model = SSDResNetModel(config=config, device=device)

    print(f"  Modèle        : {model.name}")
    print(f"  Paramètres    : {sum(p.numel() for p in model.parameters()):,}")

    # ==============================================================================
    # STEP 4 — Assemblage des composants (Trainer, Checkpoint, Predictor)
    # ==============================================================================
    print("\n=== [STEP 4] Assemblage ===")

    checkpoint = CheckpointManager(
        model=model,
        optimizer=model.optimizer,
        run_id=model.run_id,
        monitor_metric=config.metrics.monitor_metric,
    )

    trainer = Trainer(
        model=model,
        optimizer=model.optimizer,
        device=device,
        scheduler=model.scheduler,
        num_classes=config.model.num_classes,
        metrics_config=config.metrics if config.metrics.configs else None,
        on_best_model=checkpoint.save if config.experiment.save_checkpoints else None,
    )

    predictor = Predictor(model=model, device=device)
    visualizer = Visualizer()

    # --- Reprise optionnelle depuis un checkpoint ---
    RESUME = False
    if RESUME and checkpoint.exists():
        state = checkpoint.load(load_optimizer=True)
        trainer.resume_from(
            epoch=state["epoch"],
            best_metric_value=state["best_metric_value"],
        )

    # ==============================================================================
    # STEP 5 — Entraînement
    # ==============================================================================
    print(f"\n=== [STEP 5] Entraînement ({config.training.epochs} époques) ===")
    trainer.train(train_loader, val_loader, epochs=config.training.epochs)

    # ==============================================================================
    # STEP 6 — Sauvegarde des hyperparamètres et courbes
    # ==============================================================================
    print("\n=== [STEP 6] Sauvegarde ===")
    model.save_hyperparams(extra_results=trainer.get_final_metrics())
    visualizer.plot_metrics(trainer, run_id=model.run_id, save=True)

    # ==============================================================================
    # STEP 7 — Inférence de contrôle
    # ==============================================================================
    print("\n=== [STEP 7] Inférence de contrôle ===")
    import os, json

    real_dataset   = val_loader.dataset
    img_tensor, gt = real_dataset[0]
    img_name       = real_dataset.img_names[0]
    orig_img_path  = os.path.join(real_dataset.img_dir, img_name)

    predictions = predictor.predict([img_tensor], confidence_threshold=0.15)

    stats_file = "data/Chess/stats.json"
    with open(stats_file, "r") as f:
        stats = json.load(f)

    visualizer.plot_predictions(
        image=img_tensor,
        prediction=predictions[0],
        class_names=chess_classes,
        confidence_threshold=0.15,
        orig_image=orig_img_path,
        ground_truth=gt,
        mean=stats["mean"],
        std=stats["std"],
        run_id=model.run_id,
        save=True,
        filename="inference_comparison.png",
    )

    print(f"\n✅ Pipeline complet — run_id : {model.run_id}")


if __name__ == "__main__":
    main()