import torch
from torch import nn
from typing import List, Dict, Any

class Predictor:
    """
    Classe pour effectuer des inférences (prédictions) avec un modèle de détection d'objets.
    
    Attributes:
        model (nn.Module): Modèle de détection entraîné (ex: Faster R-CNN)
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
        Effectue des prédictions sur une liste d'images et filtre par seuil de confiance.
        
        Args:
            inputs (List[torch.Tensor]): Liste de tenseurs d'images, chaque image étant de forme [C, H, W]
            confidence_threshold (float): Score minimum (entre 0 et 1) pour conserver une détection
        
        Returns:
            List[Dict[str, torch.Tensor]]: Une liste de dictionnaires (un par image) contenant :
                - 'boxes': Les coordonnées [xmin, ymin, xmax, ymax] des objets retenus
                - 'labels': Les indices des classes correspondantes
                - 'scores': Les scores de confiance associés
        """

        self.model.eval()
        
        # Désactivation du calcul des gradients pour économiser de la mémoire et accélérer l'inférence
        with torch.no_grad():
            
            # ADAPTATION : En détection PyTorch, on passe une liste de tenseurs d'images
            inputs = [img.to(self.device, non_blocking=True) for img in inputs]
            
            # En mode eval(), le modèle renvoie une liste de dictionnaires contenant :
            # [{'boxes': tensor, 'labels': tensor, 'scores': tensor}, ...]
            outputs = self.model(inputs)

            filtered_outputs = []
            
            # Filtrage des prédictions pour chaque image du batch
            for output in outputs:
                scores = output['scores']
                
                # Création d'un masque booléen (True pour les boîtes qui dépassent le seuil)
                keep = scores >= confidence_threshold
                
                # On filtre et on renvoie les résultats sur le CPU 
                # (indispensable pour pouvoir les dessiner facilement avec Matplotlib/OpenCV ensuite)
                filtered_output = {
                    'boxes': output['boxes'][keep].cpu(),
                    'labels': output['labels'][keep].cpu(),
                    'scores': output['scores'][keep].cpu()
                }
                
                filtered_outputs.append(filtered_output)

            return filtered_outputs