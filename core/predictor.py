import torch
from torch import nn
from typing import List, Dict, Any

class Predictor:
    """
    Classe pour effectuer des inférences (prédictions) avec un modèle de détection d'objets.
    
    Attributes:
        model (nn.Module): Modèle de détection entraîné (ex: Faster R-CNN, SSD ou YOLO)
        device (str): Équipement à utiliser pour la prédiction ('cuda' ou 'cpu')
    """
    
    def __init__(self, model: nn.Module, device: str) -> None:
        """
        Initialise l'objet Predictor.
        """
        self.model = model
        self.device = device
        
        # Déplacer le modèle sur l'équipement cible
        self.model.to(self.device)
        
        # Passage en mode évaluation (désactive le Dropout et la BatchNorm)
        self.model.eval()

    def predict(self, inputs: List[torch.Tensor], confidence_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        Effectue des prédictions sur une liste d'images, applique un clipping géométrique 
        aux limites de chaque image d'entrée, et filtre par seuil de confiance.
        
        Args:
            inputs (List[torch.Tensor]): Liste de tenseurs d'images, chaque image étant de forme [C, H, W]
            confidence_threshold (float): Score minimum (entre 0 et 1) pour conserver une détection
        
        Returns:
            List[Dict[str, torch.Tensor]]: Une liste de dictionnaires (un par image) contenant :
                - 'boxes': Les coordonnées [xmin, ymin, xmax, ymax] clippées et retenues
                - 'labels': Les indices des classes correspondantes
                - 'scores': Les scores de confiance associés
        """
        # Sécurité : On force le mode évaluation au début de chaque appel
        self.model.eval()
        
        # Désactivation du calcul des gradients pour économiser de la mémoire et accélérer l'inférence
        with torch.no_grad():
            
            # Transfert des tenseurs d'images sur le device cible
            inputs_device = [img.to(self.device, non_blocking=True) for img in inputs]
            
            # En mode eval(), le modèle renvoie une liste de dictionnaires contenant les prédictions brutes
            outputs = self.model(inputs_device)

            filtered_outputs = []
            
            # Traitement individuel de chaque image du batch
            for i, output in enumerate(outputs):
                # Récupération des dimensions physiques de l'image d'origine [C, H, W]
                _, img_h, img_w = inputs[i].shape
                
                boxes = output['boxes']
                labels = output['labels']
                scores = output['scores']
                
                # 1. Application du CLIPPING (bridage des coordonnées aux dimensions réelles de l'image)
                if boxes.shape[0] > 0:
                    # On clone le tenseur pour éviter de modifier les sorties d'origine par référence
                    boxes = boxes.clone()
                    
                    # xmin (colonne 0) et xmax (colonne 2) clippés entre 0 et la largeur (img_w)
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_w)
                    
                    # ymin (colonne 1) et ymax (colonne 3) clippés entre 0 et la hauteur (img_h)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_h)
                
                # 2. Filtrage par seuil de confiance
                keep = scores >= confidence_threshold
                
                # On applique le masque et on bascule les données nettoyées sur le CPU
                filtered_output = {
                    'boxes': boxes[keep].cpu(),
                    'labels': labels[keep].cpu(),
                    'scores': scores[keep].cpu()
                }
                
                filtered_outputs.append(filtered_output)

            return filtered_outputs