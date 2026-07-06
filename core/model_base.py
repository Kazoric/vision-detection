import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR

# Import de tes modules et de tes nouvelles dataclasses
from core.trainer import Trainer
from core.predictor import Predictor
# from core.checkpoint import CheckpointManager
from core.config import Config
# Assure-toi d'importer Config depuis là où tu l'as défini :
# from core.config import Config 

class Model(ABC):
    """
    Classe de base abstraite pour les modèles de détection.
    Initialisation et gestion pilotées par une Dataclass de configuration stricte.
    """
    
    def __init__(self, config: Config, device: Optional[str] = None) -> None:
        """
        Initialise le modèle avec une instance de la Dataclass Config.
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
                self.scheduler_name = f"Warmup+{sched_type}"
                self.scheduler_params = {"warm_up_epochs": warm_up_epochs, "main_params": sched_params}
            else:
                self.scheduler = scheduler_cls(self.optimizer, **sched_params)
                self.scheduler_name = sched_type
                self.scheduler_params = sched_params

        # 6. Initialisation des gestionnaires coeurs (Core)
        # self.checkpoint = CheckpointManager(
        #     model=self.model, optimizer=self.optimizer, run_id=self.run_id, model_name=self.name
        # )

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            save=config.experiment.save_checkpoints,
            # checkpoint_fn=self.checkpoint.save,
            scheduler=self.scheduler,
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

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
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

    def load_checkpoint(self, load_optimizer: bool = True) -> None:
        success = self.checkpoint.load_latest(load_optimizer)
        if success:
            self.trainer.start_epoch = self.checkpoint.start_epoch
            self.trainer.best_val_loss = self.checkpoint.best_val_loss

    def save_hyperparams(self, batch_size: int, num_epochs: int) -> None:
        """ Sauvegarde la configuration sous forme de JSON lisible dans les logs """
        final_best_metrics = self.trainer.get_final_metrics()
        
        # dataclasses.asdict() convertirait l'objet en dict, mais pour rester sans dépendance,
        # on peut simplement stocker les dictionnaires primitifs sous-jacents.
        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "actual_run_id": self.run_id,
            "best_validation_results": final_best_metrics
        }

        path = os.path.join(f"experiments/{self.run_id}", "meta.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"[INFO] Métriques sauvegardées dans : {path}")