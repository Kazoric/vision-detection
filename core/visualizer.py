import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import numpy as np
from typing import Optional, List, Dict, Any
import torch
from PIL import Image

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
            plt.savefig(os.path.join(save_path, 'loss.png'), dpi=300)
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

        # 3. Tracé des métriques de validation (Precision, Recall, F1, etc.)
        for metric_name, values in trainer.valid_metrics.items():
            if not values:
                continue

            plt.figure(figsize=(8, 5))
            
            # Les métriques de validation peuvent avoir moins de points que le nombre total d'époques
            metric_epochs = range(1, len(values) + 1)
            plt.plot(metric_epochs, values, label=f'Val {metric_name}', marker='x')

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
                         mean: Optional[List[float]] = None, std: Optional[List[float]] = None,
                         orig_image: Optional[Any] = None,
                         ground_truth: Optional[Dict[str, torch.Tensor]] = None,
                         run_id: Optional[str] = None, save: bool = False, filename: str = "prediction.png"):
        """
        Dessine les boîtes englobantes (Bounding Boxes) prédites et réelles (Ground Truth) 
        sur l'image d'origine (haute résolution) ou sur l'image du modèle dénormalisée.
        
        Args:
            image (torch.Tensor): Image au format [C, H, W] normalisée utilisée par le modèle.
            prediction (Dict): Sortie du Predictor contenant 'boxes', 'labels' et 'scores'.
            class_names (List[str]): Liste des noms de classes.
            confidence_threshold (float): Seuil de confiance pour afficher une prédiction.
            mean (List[float], optional): Moyenne de normalisation (pour dénormaliser).
            std (List[float], optional): Écart-type de normalisation (pour dénormaliser).
            orig_image (Any, optional): Image d'origine (chemin str, objet PIL.Image, ou np.ndarray).
            ground_truth (Dict, optional): Dictionnaire contenant les vraies boîtes ('boxes') et labels ('labels').
            run_id (str, optional): Identifiant de l'expérience pour la sauvegarde.
            save (bool): Sauvegarder l'image annotée sur le disque.
            filename (str): Nom du fichier image généré.
        """
        # Récupération des dimensions de l'image d'entrée du modèle (ex: 300x300)
        _, model_h, model_w = image.shape
        
        # Tentative de chargement de l'image d'origine haute résolution
        use_orig = False
        img_to_plot = None
        
        if orig_image is not None:
            try:
                if isinstance(orig_image, str):
                    if os.path.exists(orig_image):
                        img_to_plot = Image.open(orig_image).convert("RGB")
                        img_to_plot = np.array(img_to_plot)
                        use_orig = True
                elif isinstance(orig_image, Image.Image):
                    img_to_plot = np.array(orig_image.convert("RGB"))
                    use_orig = True
                elif isinstance(orig_image, np.ndarray):
                    img_to_plot = orig_image
                    use_orig = True
            except Exception as e:
                print(f"[Warning] Impossible de charger orig_image, repli sur l'image normalisée. Erreur: {e}")

        # Calcul des ratios d'échelle et définition des limites géométriques maximales pour le clipping
        if use_orig and img_to_plot is not None:
            orig_h, orig_w, _ = img_to_plot.shape
            x_scale = orig_w / model_w
            y_scale = orig_h / model_h
            max_w = orig_w
            max_h = orig_h
        else:
            # Comportement par défaut : dénormalisation et affichage de l'image de taille modèle [300x300]
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if mean is None:
                mean = [0.485, 0.456, 0.406]
            if std is None:
                std = [0.229, 0.224, 0.225]
                
            mean_arr = np.array(mean, dtype=np.float32)
            std_arr = np.array(std, dtype=np.float32)
            
            img_to_plot = (img_np * std_arr) + mean_arr
            img_to_plot = np.clip(img_to_plot, 0.0, 1.0)
            x_scale, y_scale = 1.0, 1.0
            max_w = model_w
            max_h = model_h

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_to_plot)
        
        # Palette de couleurs distinctes pour les classes
        cmap = plt.get_cmap('tab20')
        
        # 1. DESSIN DES VRAIES BOÎTES (GROUND TRUTH) - Style : Ligne pointillée (dashed)
        if ground_truth is not None and 'boxes' in ground_truth:
            gt_boxes = ground_truth['boxes'].cpu()
            gt_labels = ground_truth['labels'].cpu()
            
            for box, label in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = box.tolist()
                
                # Mise à l'échelle et clipping des coordonnées de la vérité terrain
                xmin_scaled = np.clip(xmin * x_scale, 0, max_w)
                ymin_scaled = np.clip(ymin * y_scale, 0, max_h)
                xmax_scaled = np.clip(xmax * x_scale, 0, max_w)
                ymax_scaled = np.clip(ymax * y_scale, 0, max_h)
                
                width = xmax_scaled - xmin_scaled
                height = ymax_scaled - ymin_scaled
                
                if width <= 0 or height <= 0:
                    continue
                
                color = cmap(label.item() % 20)
                
                # Dessin de la boîte réelle (Ligne pointillée '--' d'épaisseur 2)
                rect = patches.Rectangle((xmin_scaled, ymin_scaled), width, height, linewidth=2, 
                                         edgecolor=color, linestyle='--', facecolor='none')
                ax.add_patch(rect)
                
                # Étiquette texte pour la vérité terrain (ex: "GT: white-pawn")
                class_name = class_names[label.item()] if label.item() < len(class_names) else f"Class {label.item()}"
                text_label = f"GT: {class_name}"
                
                font_size = max(8, int(8 * (min(x_scale, y_scale) ** 0.35)))
                
                # Positionnement de l'étiquette réelle juste en dessous de la boîte pour éviter les chevauchements
                ax.text(xmin_scaled, ymax_scaled + (12 * y_scale), text_label, color='white', fontsize=font_size,
                        bbox=dict(facecolor=color, alpha=0.5, pad=1.5, edgecolor='none'))

        # 2. DESSIN DES PRÉDICTIONS DU MODÈLE - Style : Ligne pleine (solid)
        boxes = prediction['boxes'].cpu()
        labels = prediction['labels'].cpu()
        scores = prediction['scores'].cpu()
        
        for box, label, score in zip(boxes, labels, scores):
            if score >= confidence_threshold:
                xmin, ymin, xmax, ymax = box.tolist()
                
                # Mise à l'échelle et clipping
                xmin_scaled = np.clip(xmin * x_scale, 0, max_w)
                ymin_scaled = np.clip(ymin * y_scale, 0, max_h)
                xmax_scaled = np.clip(xmax * x_scale, 0, max_w)
                ymax_scaled = np.clip(ymax * y_scale, 0, max_h)
                
                width = xmax_scaled - xmin_scaled
                height = ymax_scaled - ymin_scaled
                
                if width <= 0 or height <= 0:
                    continue
                
                color = cmap(label.item() % 20)
                
                # Dessin de la boîte prédite (Ligne pleine)
                rect = patches.Rectangle((xmin_scaled, ymin_scaled), width, height, linewidth=2.5, 
                                         edgecolor=color, linestyle='-', facecolor='none')
                ax.add_patch(rect)
                
                class_name = class_names[label.item()] if label.item() < len(class_names) else f"Class {label.item()}"
                text_label = f"{class_name}: {score:.2f}"
                
                font_size = max(9, int(9 * (min(x_scale, y_scale) ** 0.35)))
                
                # Positionnement de l'étiquette prédite au-dessus de la boîte
                ax.text(xmin_scaled, ymin_scaled - (4 * y_scale), text_label, color='white', fontsize=font_size,
                        bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor='none'))
        
        plt.axis('off')
        plt.tight_layout()
        
        if save and run_id:
            save_dir = f"experiments/{run_id}/predictions"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=150)
            print(f"[INFO] Image annotée sauvegardée : {os.path.join(save_dir, filename)}")
            
        plt.show()
        plt.close()


    def plot_dataset_item(self, image: torch.Tensor, target: Dict[str, torch.Tensor], 
                          class_names: List[str], mean: Optional[List[float]] = None, 
                          std: Optional[List[float]] = None, orig_image: Optional[Any] = None):
        """
        Affiche directement une image du dataset avec ses annotations réelles (Ground Truth)
        sans faire d'inférence avec un modèle.
        
        Args:
            image (torch.Tensor): Image normalisée issue du Dataset [C, H, W]
            target (Dict): Target du dataset contenant 'boxes' et 'labels'
            class_names (List[str]): Noms des classes du projet d'échecs
            mean/std (List[float], optional): Statistiques de normalisation pour restituer les couleurs d'origine
            orig_image (Any, optional): Image d'origine (chemin, PIL Image, ou np.ndarray) pour une netteté totale
        """
        # On simule un dictionnaire de prédiction vide
        empty_pred = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0, dtype=torch.float32)
        }
        
        # On redirige vers notre plot_predictions robuste en ne transmettant que la vérité terrain (ground_truth)
        self.plot_predictions(
            image=image,
            prediction=empty_pred,
            class_names=class_names,
            mean=mean,
            std=std,
            orig_image=orig_image,
            ground_truth=target
        )
        

    def plot_confusion_matrix(self, cm: torch.Tensor, class_names: list, run_id: str, save: Optional[bool] = True):
        """
        Affiche et sauvegarde la matrice de confusion.
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