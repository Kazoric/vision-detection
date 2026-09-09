from typing import Any, Callable, Type

DETECTOR_REGISTRY: dict[str, Type] = {}

def register_detector(name: str) -> Callable[[Type], Type]:
    """
    Decorator allowing a detector class to auto-register itself
    into DETECTOR_REGISTRY under the given name.
 
    Usage:
        @register_detector("yolo")
        class Yolo(nn.Module):
            ...
    """
    def decorator(cls: Type) -> Type:
        if name in DETECTOR_REGISTRY:
            raise ValueError(
                f"Detector name '{name}' is already registered "
                f"by {DETECTOR_REGISTRY[name].__name__}. "
                f"Cannot register it again for {cls.__name__}."
            )
        DETECTOR_REGISTRY[name] = cls
        return cls
    return decorator
 
 
def build_detector(config) -> Any:
    """
    Instantiate a detector from its config (config.model.type must
    match a key registered via @register_detector).
 
    Args:
        config: full Config object, passed as-is to cls.from_config().
    """
    model_type = config.model.type.lower()
    if model_type not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector '{model_type}'. "
            f"Available options: {list(DETECTOR_REGISTRY.keys())}"
        )
    model_cls = DETECTOR_REGISTRY[model_type]
    return model_cls.from_config(config=config)