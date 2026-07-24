from typing import Any, Optional
from .fpn import FPN
from .ssd_neck import SSD300Neck

NECK_REGISTRY = {
    "FPN": FPN,
    "SSDNeck": SSD300Neck,
    # "PANet": PANet,
}

def build_neck(name: Optional[str], **kwargs) -> Optional[Any]:
    """
    Instancie un neck s'il est spécifié, sinon renvoie None.
    """
    if name is None or not name:
        return None

    if name not in NECK_REGISTRY:
        raise ValueError(
            f"Neck '{name}' inconnu. Choisir parmi : {list(NECK_REGISTRY.keys())}"
        )
    return NECK_REGISTRY[name](**kwargs)