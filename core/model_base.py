import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR

# Import des composants du framework
from core.trainer import Trainer
from core.predictor import Predictor
from core.checkpoint import CheckpointManager
from core.config import Config
from core.metrics import detection_precision_recall_f1  # Importation de la suite de métriques de détection

class Model(ABC):
    """
    Classe de base abstraite pour les modèles de détection.
    Initialisation et gestion pilotées par une Dataclass de configuration stricte.
    """
    
    def __init__(self, config: Config, device: Optional[str] = None) -> None:
        """
        Initialise le modèle avec une instance de la Dataclass Config et configure le Trainer.
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Extraction propre des attributs typés
        self.num_classes = config.model.num_classes
        self.dataset_name = config.experiment.dataset_name
        self.lr = config.training.lr
        
        # 2. Construction du modèle architecture graphique
        self.model = self.build_model().to(self.device)

        # 3. Gestion du Run ID
        run_id = config.experiment.run_id
        if run_id is None:
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"{self.name}_{self.dataset_name}_{date}"
        self.run_id = run_id

        # 4. Initialisation de l'Optimiseur via Réflexion Python
        opt_type = config.optimizer.type
        opt_params = config.optimizer.params
        optimizer_cls = getattr(optim, opt_type)
        
        self.optimizer_name = opt_type
        self.optimizer_params = opt_params
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, **opt_params)

        # 5. Initialisation du Scheduler
        self.scheduler = None
        self.scheduler_name = None
        self.scheduler_params = {}

        if config.scheduler.type is not None:
            sched_type = config.scheduler.type
            sched_params = config.scheduler.params
            scheduler_cls = getattr(optim.lr_scheduler, sched_type)
            
            # Gestion du Warm-up linéaire
            if config.training.warm_up:
                warm_up_epochs = config.training.warm_up_epochs
                
                self.optimizer.param_groups[0]['lr'] = self.lr
                warmup_scheduler = LinearLR(
                    self.optimizer, start_factor=0.05, end_factor=1.0, total_iters=warm_up_epochs
                )
                main_scheduler = scheduler_cls(self.optimizer, **sched_params)
                
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warm_up_epochs]
                )
                # self.scheduler_name = f"Warmup+{sched_type}"
                # self.scheduler_params = {"warm_up_epochs": warm_up_epochs, "main_params": sched_params}
            else:
                self.scheduler = scheduler_cls(self.optimizer, **sched_params)
            self.scheduler_name = sched_type
            self.scheduler_params = sched_params

        self.metrics = config.metrics
        # print()
        # # 6. Configuration des métriques de détection par défaut si non fournies
        # if self.metrics is None:
        #     # On suit par défaut la suite globale F1-Score, Précision et Rappel
        #     self.metrics = {
        #         "detection_metrics": (
        #             detection_precision_recall_f1,
        #             {"num_classes": self.num_classes, "iou_threshold": 0.5}
        #         )
        #     }
        # else:
        #     self.metrics = metrics

        self.checkpoint = CheckpointManager(
            model=self.model, optimizer=self.optimizer, run_id=self.run_id, model_name=self.name
        )

        # 7. Initialisation du Trainer avec intégration des métriques configurées
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            save=config.experiment.save_checkpoints,
            checkpoint_fn=self.checkpoint.save,
            scheduler=self.scheduler,
            metrics=self.metrics,
            num_classes=self.num_classes
        )

        self.predictor = Predictor(self.model, self.device)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def build_model(self) -> nn.Module:
        pass

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> None:
        self.trainer.train(train_loader, val_loader, epochs)

    def evaluate(self, val_loader: DataLoader) -> float:
        return self.trainer.evaluate(val_loader)

    def predict(self, images: List[torch.Tensor], confidence_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        return self.predictor.predict(images, confidence_threshold=confidence_threshold)
    
    def predict_on_loader(self, dataloader: DataLoader, confidence_threshold: float = 0.5) -> Tuple[List[Any], List[Any]]:
        all_preds = []
        all_targets = []
        for images, targets in dataloader:
            preds = self.predict(images, confidence_threshold=confidence_threshold)
            all_preds.extend(preds)
            all_targets.extend(targets)
        return all_targets, all_preds

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """
        Charge un checkpoint depuis un fichier .pth.
        
        Args:
            path (str): Chemin vers le fichier checkpoint (.pth)
            load_optimizer (bool): Si True, restaure aussi l'état de l'optimiseur
        """
        if not os.path.exists(path):
            print(f"[WARNING] Checkpoint introuvable : {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "epoch" in checkpoint:
            self.trainer.start_epoch = checkpoint["epoch"]
        if "best_val_loss" in checkpoint:
            self.trainer.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"[INFO] Checkpoint chargé depuis : {path}")

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Sauvegarde un checkpoint du modèle et de l'optimiseur.
        """
        save_dir = f"experiments/{self.run_id}"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "best_model.pth")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": val_loss,
        }, path)
        print(f"[INFO] Checkpoint sauvegardé : {path}")

    def save_hyperparams(self) -> None:
        """ Sauvegarde la configuration sous forme de JSON structuré et lisible """
        final_best_metrics = self.trainer.get_final_metrics()

        # 1. On convertit automatiquement toute la structure Config en dictionnaire standard
        meta = asdict(self.config)
        
        # Ajustement cosmétique : Remplacer le run_id initial par le run_id réel s'il a été généré dynamiquement
        meta["experiment"]["run_id"] = self.run_id
        
        # 2. On ajoute le bloc de résultats d'exécution qui n'est pas dans la configuration initiale
        meta["results"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_validation_results": final_best_metrics
        }

        path = os.path.join(f"experiments/{self.run_id}", "meta.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
            
        print(f"[INFO] Métriques et configuration sauvegardées dans : {path}")