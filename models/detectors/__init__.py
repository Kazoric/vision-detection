from .registry import DETECTOR_REGISTRY, build_detector, register_detector

from .yolo import Yolo
from .ssd import SSDDetector

__all__ = [
    "DETECTOR_REGISTRY",
    "build_detector",
    "register_detector",
    "Yolo",
    "SSDDetector",
]