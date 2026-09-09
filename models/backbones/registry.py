from typing import Any, Callable, Type

BACKBONE_REGISTRY: dict[str, Type] = {}

def register_backbone(name: str) -> Callable[[Type], Type]:
    """
    Decorator allowing a backbone class to auto-register itself
    into BACKBONE_REGISTRY under the given name.
 
    Usage:
        @register_backbone("resnet")
        class ResNetBackbone(nn.Module):
            ...
    """
    def decorator(cls: Type) -> Type:
        if name in BACKBONE_REGISTRY:
            raise ValueError(
                f"backbone name '{name}' is already registered "
                f"by {BACKBONE_REGISTRY[name].__name__}. "
                f"Cannot register it again for {cls.__name__}."
            )
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator
 
def build_backbone(name: str, **kwargs) -> Any:
    """
    Instantiate a backbone from its name.

    Args:
        name (str): backbone name as defined in the YAML config.
        **kwargs: extra arguments (pretrained, out_indices, etc.)
    """
    if name.lower() not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available options: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name.lower()](**kwargs)