import torch
import os
import torch.nn as nn
from typing import Optional

class CheckpointManager:
    """Gestionnaire de checkpoints pour sauvegarder/charger les états du modèle."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, run_id: str, model_name: str = "model"):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = f"./experiments/{run_id}/checkpoints"
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.best_mAP = 0
        self.start_epoch = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, epoch: int, current_mAP: float) -> None:
        self.best_mAP = current_mAP 
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'mAP': current_mAP
        }
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
        torch.save(checkpoint, path)
        print(f"[INFO] Meilleur modèle sauvegardé à l'époque {epoch} (mAP: {current_mAP:.4f})")

    def load_latest(self, load_optimizer: bool = True) -> bool:
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
        if not os.path.exists(path):
            print("[INFO] Aucun checkpoint trouvé.")
            return False
        
        checkpoint = torch.load(path, map_location=next(self.model.parameters()).device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        print(f"[INFO] Checkpoint chargé : époques {self.start_epoch}, loss {self.best_val_loss:.4f}")
        return True