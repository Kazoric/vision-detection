import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from typing import Optional, List, Dict
import torch

from core.trainer import Trainer

class Visualizer:
    """
    Classe pour visualiser les métriques d'entraînement et les prédictions de détection d'objets.
    """
    
    def __init__(self) -> None:
        pass

    def plot_metrics(self, trainer: Trainer, run_id: str, save: Optional[bool] = True):
        """
        Trace les courbes de pertes (train/val), le learning rate et les métriques de détection (ex: mAP).
        """
        epochs = range(1, len(trainer.train_loss) + 1)

        # 1. Tracé de la Perte Globale
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, trainer.train_loss, label='Train Loss', marker='o')
        if hasattr(trainer, 'valid_loss') and trainer.valid_loss:
            plt.plot(epochs, trainer.valid_loss, label='Val Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            save_path = f"experiments/{run_id}/plots"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'loss.png'), dpi=300) # Ajout du .png
            plt.close()

        # 2. Tracé de l'historique du Learning Rate
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, trainer.lr_history, label='Learning Rate', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate History')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            save_path = f"experiments/{run_id}/plots"
            plt.savefig(os.path.join(save_path, 'learning_rate.png'), dpi=300)
            plt.close()

        # 3. Tracé des métriques spécifiques (ex: mAP@0.5, mAP@0.5:0.95)
        for metric_name in trainer.train_metrics:
            plt.figure(figsize=(8, 5))

            train_values = trainer.train_metrics[metric_name]
            valid_values = trainer.valid_metrics.get(metric_name, [])

            plt.plot(epochs, train_values, label=f'Train {metric_name}', marker='o')
            if valid_values:
                plt.plot(epochs, valid_values, label=f'Val {metric_name}', marker='x')

            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'Evolution of {metric_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save:
                plt.savefig(os.path.join(f"experiments/{run_id}/plots", f'{metric_name}.png'), dpi=300)
                plt.close()
                
        print(f"[INFO] Toutes les courbes métriques ont été sauvegardées dans : experiments/{run_id}/plots/")

    def plot_predictions(self, image: torch.Tensor, prediction: Dict[str, torch.Tensor], 
                         class_names: List[str], confidence_threshold: float = 0.5, 
                         run_id: Optional[str] = None, save: bool = False, filename: str = "prediction.png"):
        """
        Dessine les boîtes englobantes (Bouding Boxes) prédites sur une image.
        
        Args:
            image (torch.Tensor): Image au format [C, H, W], valeurs entre 0 et 1.
            prediction (Dict): Sortie du Predictor contenant 'boxes', 'labels' et 'scores'.
            class_names (List[str]): Liste des noms de classes (ex: ['background', 'pawn', 'rook'...]).
            confidence_threshold (float): Seuil pour afficher la boîte.
            run_id (str, optional): Identifiant de l'expérience pour la sauvegarde.
            save (bool): Sauvegarder l'image annotée sur le disque.
            filename (str): Nom du fichier image généré.
        """
        # Convertir le tenseur image [C, H, W] en format NumPy [H, W, C] lisible par matplotlib
        img_np = image.permute(1, 2, 0).cpu().numpy()
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_np)
        
        boxes = prediction['boxes'].cpu()
        labels = prediction['labels'].cpu()
        scores = prediction['scores'].cpu()
        
        # Palette de couleurs distinctes pour les classes
        cmap = plt.get_cmap('tab20')
        
        for box, label, score in zip(boxes, labels, scores):
            if score >= confidence_threshold:
                xmin, ymin, xmax, ymax = box.tolist()
                width, height = xmax - xmin, ymax - ymin
                
                # Assigner une couleur basée sur l'ID de la classe
                color = cmap(label.item() % 20)
                
                # 1. Dessiner le rectangle de la boîte englobante
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, 
                                         edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # 2. Ajouter l'étiquette texte (Nom de classe + Score de confiance)
                class_name = class_names[label.item()] if label.item() < len(class_names) else f"Class {label.item()}"
                text_label = f"{class_name}: {score:.2f}"
                
                ax.text(xmin, ymin - 4, text_label, color='white', fontsize=10,
                        bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor='none'))
        
        plt.axis('off')
        plt.tight_layout()
        
        if save and run_id:
            save_dir = f"experiments/{run_id}/predictions"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
            print(f"[INFO] Image annotée sauvegardée : {os.path.join(save_dir, filename)}")
            
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, cm: torch.Tensor, class_names: list, run_id: str, save: Optional[bool] = True):
        """
        Affiche et sauvegarde la matrice de confusion (adaptée si calculée via IoU matching).
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Detection Confusion Matrix (IoU Matched)')
        plt.tight_layout()

        if save:
            save_dir = f"experiments/{run_id}/plots"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches='tight', dpi=300)
            plt.close()