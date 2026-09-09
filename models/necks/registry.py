from typing import Any, Callable, Type

NECK_REGISTRY: dict[str, Type] = {}

def register_neck(name: str) -> Callable[[Type], Type]:
    """
    Decorator allowing a neck class to auto-register itself
    into NECK_REGISTRY under the given name.
 
    Usage:
        @register_neck("fpn")
        class FPN(nn.Module):
            ...
    """
    def decorator(cls: Type) -> Type:
        if name in NECK_REGISTRY:
            raise ValueError(
                f"neck name '{name}' is already registered "
                f"by {NECK_REGISTRY[name].__name__}. "
                f"Cannot register it again for {cls.__name__}."
            )
        NECK_REGISTRY[name] = cls
        return cls
    return decorator
 
def build_neck(name: str, **kwargs) -> Any:
    """
    Instantiate a neck from its name.

    Args:
        name (str): neck name as defined in the YAML config.
        **kwargs: extra arguments (pretrained, out_indices, etc.)
    """
    if name.lower() not in NECK_REGISTRY:
        raise ValueError(
            f"Unknown neck '{name}'. "
            f"Available options: {list(NECK_REGISTRY.keys())}"
        )
    return NECK_REGISTRY[name.lower()](**kwargs)