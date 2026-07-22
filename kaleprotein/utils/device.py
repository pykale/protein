"""Device helpers for nested tensor containers."""

from collections.abc import Mapping


def move_to_device(value, device):
    """Recursively move tensors in a nested container to ``device``."""
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)

    to = getattr(value, "to", None)
    return to(device) if callable(to) else value


__all__ = ["move_to_device"]
