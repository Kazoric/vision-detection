import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class SSDResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Charger le ResNet50 de base
        base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 2. Définir les couches que l'on souhaite intercepter
        return_nodes = {
            'layer2': 'feat1',  # Sortie 38x38
            'layer3': 'feat2',  # Sortie 19x19
            'layer4': 'feat3'   # Sortie 10x10
        }
        self.feature_extractor = create_feature_extractor(base_resnet, return_nodes=return_nodes)
        
        # 3. Ajouter les couches supplémentaires (Extra Layers) pour descendre jusqu'à 1x1
        self.extra_layers = nn.ModuleList([
            # De 10x10 (2048 canaux) à 5x5
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) 
            ),
            # De 5x5 à 3x3
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Attention au padding selon la taille exacte
            ),
            # De 3x3 à 1x1
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # Pas de stride, pas de padding -> 1x1
            )
        ])

    def forward(self, x):
        # Extraire les 3 premières feature maps du ResNet
        resnet_feats = self.feature_extractor(x)
        features = [resnet_feats['feat1'], resnet_feats['feat2'], resnet_feats['feat3']]
        
        # Passer séquentiellement dans les couches extras pour obtenir les 3 dernières
        current_feat = features[-1]
        for layer in self.extra_layers:
            current_feat = layer(current_feat)
            features.append(current_feat)
            
        return features # Contient tes 6 feature maps prêtes pour les têtes SSD

# Test du shape des sorties
model = SSDResNetBackbone()
x = torch.randn(1, 3, 300, 300)
outputs = model(x)

for i, feat in enumerate(outputs):
    print(f"Feature Map {i+1} shape: {feat.shape}")