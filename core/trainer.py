import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional, List, Dict, Any

import core.metrics as core_metrics
from core.metrics import compute_dataset_tp_fp

class Trainer:
    """
    Classe pour entraîner un modèle de détection d'objets (ex: Faster R-CNN, SSD, YOLO).
    
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
        num_classes: Optional[int] = None,
        monitor_metric: str = "mAP",
        monitor_mode: str = "max"
    ) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save = save
        self.save_checkpoint = checkpoint_fn
        self.scheduler = scheduler
        self.num_classes = num_classes

        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode.lower()
        assert self.monitor_mode in ["max", "min"], "monitor_mode doit être 'max' ou 'min'"

        # Suivi des pertes moyennes par époque
        self.train_loss = []
        self.valid_loss = []
        self.lr_history = []

        self.best_metric_value = float('-inf') if self.monitor_mode == "max" else float('inf')
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.best_epoch_metrics: dict = {}

        # Initialisation des métriques (ex: IoU, mAP, F1-score)
        self.metrics = metrics if metrics else {}
        self.valid_metrics: Dict[str, list] = {}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> None:
        """ Boucle principale d'entraînement """
        for epoch in range(self.start_epoch, epochs):

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # Mode d'entraînement indispensable pour que le modèle calcule les pertes (Losses)
            self.model.train()
            
            running_loss = 0.0

            # Utilisation de tqdm pour la barre de progression
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, targets in pbar:
                
                # Correction robuste : images est une liste/tuple de tenseurs. On envoie chaque image individuellement.
                images_device = list(img.to(self.device, non_blocking=True) for img in images)
                targets_device = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                
                # Le modèle renvoie un dictionnaire de pertes : loss_classifier, loss_box_reg, etc.
                loss_dict = self.model(images_device, targets_device)
                
                # Somme de toutes les pertes du dictionnaire
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                self.optimizer.step()

                running_loss += losses.item()
                pbar.set_postfix({"batch_loss": f"{losses.item():.4f}"})

            # Calcul de la perte moyenne de l'époque
            epoch_train_loss = running_loss / len(train_loader)
            self.train_loss.append(epoch_train_loss)

            print(f"{'Train':<12} | Avg Loss: {epoch_train_loss:.4f} | Learning Rate: {current_lr:.6f}")

            # Évaluation et calcul des métriques globales de validation
            if val_loader:
                val_loss, val_metrics = self.evaluate(val_loader)

                if self.monitor_metric in ["loss", "val_loss"]:
                    current_metric_val = val_loss
                else:
                    # Valeur par défaut logique en cas d'absence de la clé
                    default_val = float('-inf') if self.monitor_mode == "max" else float('inf')
                    current_metric_val = val_metrics.get(self.monitor_metric, default_val)

                is_better = False
                if self.monitor_mode == "max" and current_metric_val > self.best_metric_value:
                    is_better = True
                elif self.monitor_mode == "min" and current_metric_val < self.best_metric_value:
                    is_better = True

                # Sauvegarde du meilleur modèle basé sur la métrique choisie
                if self.save and is_better:
                    self.best_metric_value = current_metric_val
                    self.best_val_loss = val_loss

                    self.best_epoch_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": epoch_train_loss,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "monitor_metric": self.monitor_metric,
                        "monitor_value": current_metric_val,
                        # "valid_metrics": {name: self.valid_metrics[name][-1] for name in self.valid_metrics.keys()} if self.valid_metrics else {}
                    }
                    if self.save_checkpoint:
                        self.save_checkpoint(epoch + 1, current_metric_val)
            
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
                # Conversion robuste des listes d'images
                images_device = list(img.to(self.device, non_blocking=True) for img in images)
                targets_device = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                # Récupération des pertes de validation
                loss_dict = self.model(images_device, targets_device)
                losses = sum(loss for loss in loss_dict.values())
                running_loss += losses.item()

                # Pour les métriques de détection (mAP, IoU, F1), il faut les prédictions réelles.
                if self.metrics:
                    self.model.eval()
                    # Gestion robuste si le modèle possède une fonction predict custom (ex: notre YOLOModel)
                    if hasattr(self.model, "predict"):
                        preds = self.model.predict(images_device, confidence_threshold=0.15)
                    else:
                        preds = self.model(images_device)
                        
                    self.model.train() # Rebasculer immédiatement en train
                    
                    # Stockage sur le CPU pour préserver la VRAM du GPU
                    all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
                    all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

        epoch_val_loss = running_loss / len(data_loader)
        self.valid_loss.append(epoch_val_loss)

        # Calcul des métriques de détection (mAP / F1-Score / Precision / Recall)
        metrics_str = ""
        if self.metrics:
            metric_outputs = self._compute_metrics(all_targets, all_preds)
            for name, value in metric_outputs.items():
                # Initialisation dynamique des clés de métriques si elles n'existent pas
                if name not in self.valid_metrics:
                    self.valid_metrics[name] = []
                self.valid_metrics[name].append(value)
                
            metrics_str = " | " + " | ".join(f"{name}: {value:.4f}" for name, value in metric_outputs.items())

        print(f"{'Validation':<12} | Avg Loss: {epoch_val_loss:.4f}{metrics_str}")
        
        return epoch_val_loss, metric_outputs


    def _compute_metrics(self, targets: List[Dict[str, torch.Tensor]], predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Calcule les métriques en extrayant les TP/FP une seule fois 
        et en les injectant dans les fonctions cibles.
        """
        metric_outputs = {}
        
        # 1. Calcul UNIQUE du matching géométrique pour tout le lot/dataset
        # On utilise le IoU threshold global de votre configuration (ex: 0.5)
        iou_threshold = getattr(self.metrics, "params", {}).get("iou_threshold", 0.5)
        
        # Cette fonction noyau extrait tout le dictionnaire de données (cls_tp, cls_fp, etc.)
        raw_data = compute_dataset_tp_fp(
            predictions=predictions, 
            targets=targets, 
            num_classes=self.num_classes, 
            iou_threshold=iou_threshold
        )
        
        # 2. Distribution des TP/FP pré-calculés aux fonctions de métriques
        for name, (func, params) in self.metrics.configs.items():
            # ✨ CORRECTION : Si func est une chaîne, on récupère dynamiquement la vraie fonction
            if isinstance(func, str):
                if hasattr(core_metrics, func):
                    func = getattr(core_metrics, func)
                else:
                    raise AttributeError(f"La fonction '{func}' est introuvable dans core.metrics")

            # On fusionne les paramètres
            full_kwargs = {
                **raw_data, 
                "num_classes": self.num_classes,
                **params
            }
            
            # Appel de la fonction avec le dictionnaire de paramètres complet
            score = func(**full_kwargs)
            
            # Déballage standard des résultats
            if isinstance(score, dict):
                for k, v in score.items():
                    metric_outputs[k] = v
            else:
                metric_outputs[name] = score
                
        return metric_outputs
    
    def get_final_metrics(self) -> dict:
        return self.best_epoch_metrics