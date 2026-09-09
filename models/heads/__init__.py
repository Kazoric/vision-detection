from .registry import HEAD_REGISTRY, build_head, register_head

from .yolo_head import YoloHead
from .ssd_head import SSDHead

__all__ = [
    "HEAD_REGISTRY",
    "build_head",
    "register_head",
    "YoloHead",
    "SSDHead",
]