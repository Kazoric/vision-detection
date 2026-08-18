from typing import Any
from .resnet import ResNetBackbone
from .convnext import ConvNeXtBackbone

# Dictionnaire de correspondance String -> Classe PyTorch
BACKBONE_REGISTRY = {
    "ResNet": ResNetBackbone,
    "ConvNeXt" : ConvNeXtBackbone
    # "SSDResNet": SSDResNetModel,
    # Ajoutez d'autres backbones ici au fur et à mesure (ex: "CSPDarknet", "EfficientNet")
}

def build_backbone(name: str, **kwargs) -> Any:
    """
    Instancie un backbone à partir de son nom.
    
    Args:
        name (str): Nom du backbone défini dans le YAML.
        **kwargs: Arguments supplémentaires (pretrained, out_indices, etc.)
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Backbone '{name}' inconnu. Choisir parmi : {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name](**kwargs)