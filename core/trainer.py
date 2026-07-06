import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional, List, Dict, Any

class Trainer:
    """
    Classe pour entraîner un modèle de détection d'objets (ex: Faster R-CNN).
    
    Attributes:
        model (nn.Module): Modèle de détection à entraîner
        optimizer (torch.optim.Optimizer): Optimiseur
        device (str): Équipement cible ('cuda' ou 'cpu')
        save (bool): Sauvegarder ou non le meilleur modèle
        save_checkpoint (function): Fonction externe de sauvegarde
        scheduler (LRScheduler): Gestionnaire de Learning Rate
        num_classes (int): Nombre de classes du dataset
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        save: bool = False,
        checkpoint_fn: Optional[Callable[[int, float], None]] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        metrics: Optional[dict] = None,
        num_classes: Optional[int] = None
    ) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save = save
        self.save_checkpoint = checkpoint_fn
        self.scheduler = scheduler
        self.num_classes = num_classes

        # Suivi des pertes moyennes par époque
        self.train_loss = []
        self.valid_loss = []
        self.lr_history = []

        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.best_epoch_metrics: dict = {}

        # Initialisation des métriques (ex: IoU, mAP)
        self.metrics = metrics if metrics else {}
        self.train_metrics = {name: [] for name in self.metrics}
        self.valid_metrics = {name: [] for name in self.metrics}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> None:
        """ Train loop """
        for epoch in range(self.start_epoch, epochs):

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # Mode d'entraînement indispensable pour que le modèle calcule les pertes (Losses)
            self.model.train()
            
            running_loss = 0.0
            
            # Listes pour accumuler les prédictions et cibles à des fins de calcul de métriques en fin d'époque
            all_predictions = []
            all_targets = []

            # Utilisation de tqdm pour la barre de progression
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, targets in pbar:
                
                # ADAPTATION : Transfert sur GPU/CPU pour des listes et dictionnaires
                images = list(img.to(self.device, non_blocking=True) for img in images)
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                
                # Le modèle renvoie un dictionnaire de pertes : loss_classifier, loss_box_reg, etc.
                loss_dict = self.model(images, targets)
                
                # Somme de toutes les pertes du dictionnaire
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                self.optimizer.step()

                running_loss += losses.item()
                pbar.set_postfix({"batch_loss": f"{losses.item():.4f}"})

                # Si on souhaite calculer des métriques d'entraînement, on doit repasser brièvement en eval 
                # pour obtenir les boîtes prédites, mais cela ralentit l'entraînement. 
                # Généralement en détection, on calcule les métriques uniquement sur la validation.
                if self.metrics:
                    all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

            # Calcul de la perte moyenne de l'époque
            epoch_train_loss = running_loss / len(train_loader)
            self.train_loss.append(epoch_train_loss)

            # Gestion facultative des métriques d'entraînement
            metric_outputs = {}
            if self.metrics:
                # Si tu as besoin de prédictions d'entraînement, il faudrait exécuter une passe en eval.
                # Pour l'instant on initialise à 0 ou on calcule si l'infrastructure le permet.
                metric_outputs = {name: 0.0 for name in self.metrics} 

            print(f"{'Train':<12} | Avg Loss: {epoch_train_loss:.4f} | Learning Rate: {current_lr:.4f}")

            # Évaluation
            if val_loader:
                val_loss = self.evaluate(val_loader)

                # Sauvegarde du meilleur modèle basé sur la perte de validation
                if self.save and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    self.best_epoch_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": epoch_train_loss,
                        "val_loss": val_loss,
                        "valid_metrics": {name: self.valid_metrics[name][-1] for name in self.metrics.keys()} if self.metrics else {}
                    }
                    if self.save_checkpoint:
                        self.save_checkpoint(epoch + 1, val_loss)
            
            if self.scheduler is not None:
                self.scheduler.step()

            print()

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Évalue le modèle. 
        Crucial en détection : pour obtenir la perte de validation, le modèle DOIT rester 
        techniquement en mode `.train()` mais enveloppé dans un `torch.no_grad()`.
        """
        # On reste en mode train pour forcer le calcul des pertes sans modifier les poids
        self.model.train() 
        
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in data_loader:
                images = list(img.to(self.device, non_blocking=True) for img in images)
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                # Récupération des pertes de validation
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_loss += losses.item()

                # Pour les métriques de détection (mAP, IoU), il faut les prédictions réelles.
                # On bascule temporairement en eval pour ce sous-batch si les métriques sont activées
                if self.metrics:
                    self.model.eval()
                    preds = self.model(images)
                    self.model.train() # On rebascule immédiatement en train
                    
                    # On stocke sur le CPU pour éviter d'asphyxier la VRAM
                    all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
                    all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

        epoch_val_loss = running_loss / len(data_loader)
        self.valid_loss.append(epoch_val_loss)

        # Calcul des métriques de détection (mAP / IoU) si fournies
        metrics_str = ""
        if self.metrics:
            metric_outputs = self._compute_metrics(all_targets, all_preds)
            for name, value in metric_outputs.items():
                self.valid_metrics[name].append(value)
            metrics_str = " | " + " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())

        print(f"{'Validation':<12} | Avg Loss: {epoch_val_loss:.4f}{metrics_str}")
        
        return epoch_val_loss

    def _compute_metrics(self, targets: List[Dict[str, torch.Tensor]], predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Calcule les métriques de détection.
        targets et predictions sont des listes de dictionnaires contenant 'boxes' et 'labels'.
        """
        metric_outputs = {}
        for name, (func, params) in self.metrics.items():
            # Vos fonctions dans core/metrics.py devront accepter ces listes de dicts
            score = func(targets, predictions, **params)
            metric_outputs[name] = score
        return metric_outputs
    
    def get_final_metrics(self) -> dict:
        return self.best_epoch_metrics