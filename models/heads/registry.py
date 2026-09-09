from typing import Any, Callable, Type

HEAD_REGISTRY: dict[str, Type] = {}

def register_head(name: str) -> Callable[[Type], Type]:
    """
    Decorator allowing a head class to auto-register itself
    into HEAD_REGISTRY under the given name.
 
    Usage:
        @register_head("yolohead")
        class YoloHead(nn.Module):
            ...
    """
    def decorator(cls: Type) -> Type:
        if name in HEAD_REGISTRY:
            raise ValueError(
                f"head name '{name}' is already registered "
                f"by {HEAD_REGISTRY[name].__name__}. "
                f"Cannot register it again for {cls.__name__}."
            )
        HEAD_REGISTRY[name] = cls
        return cls
    return decorator
 
def build_head(name: str, **kwargs) -> Any:
    """
    Instantiate a head from its name.

    Args:
        name (str): head name as defined in the YAML config.
        **kwargs: extra arguments (pretrained, out_indices, etc.)
    """
    if name.lower() not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head '{name}'. "
            f"Available options: {list(HEAD_REGISTRY.keys())}"
        )
    return HEAD_REGISTRY[name.lower()](**kwargs)