from typing import Any, Optional
from .yolo_head import YoloHead
from .ssd_head import SSDHead

HEAD_REGISTRY = {
    "YoloHead": YoloHead,
    "SSDHead": SSDHead,
}

def build_head(name: Optional[str], **kwargs) -> Optional[Any]:
    """
    Instancie une tête de détection spécifique.
    """
    if name is None or not name:
        return None

    if name not in HEAD_REGISTRY:
        raise ValueError(
            f"Head '{name}' inconnue. Choisir parmi : {list(HEAD_REGISTRY.keys())}"
        )
    return HEAD_REGISTRY[name](**kwargs)