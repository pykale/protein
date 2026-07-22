"""Model-independent checkpoint parsing and integrity helpers."""

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path


_STATE_DICT_KEYS = ("model", "state_dict", "model_state_dict")


def load_checkpoint_state_dict(checkpoint, *, map_location="cpu", loader=None):
    """Load and extract a state dict from a path or an in-memory checkpoint."""
    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file does not exist: {checkpoint_path}"
            )
        if loader is None:
            loader = _torch_checkpoint_loader
        payload = loader(checkpoint_path, map_location=map_location)
    return extract_checkpoint_state_dict(payload)


def extract_checkpoint_state_dict(checkpoint):
    """Extract raw or commonly wrapped model parameters from a checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a state-dict mapping or contain one under "
            f"one of {_STATE_DICT_KEYS}; got {type(checkpoint).__name__}."
        )

    state_dict = checkpoint
    container_key = next(
        (key for key in _STATE_DICT_KEYS if key in checkpoint), None
    )
    if container_key is not None:
        state_dict = checkpoint[container_key]
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Checkpoint entry {container_key!r} must contain a state-dict "
                f"mapping; got {type(state_dict).__name__}."
            )

    invalid_key = next(
        (key for key in state_dict if not isinstance(key, str)), None
    )
    if invalid_key is not None:
        raise ValueError(
            f"State-dict keys must be strings; found {invalid_key!r} "
            f"({type(invalid_key).__name__})."
        )
    return state_dict


def verify_checksum(path, expected):
    """Verify an optional SHA-256 digest for a checkpoint file."""
    if not expected:
        return
    if not isinstance(expected, str):
        raise TypeError("Checkpoint sha256 must be a hexadecimal string.")

    digest = sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.casefold() != expected.casefold():
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _torch_checkpoint_loader(path, *, map_location):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load checkpoint files. Install the optional "
            "model dependencies or pass a custom checkpoint loader."
        ) from exc
    return torch.load(path, map_location=map_location, weights_only=True)


__all__ = [
    "extract_checkpoint_state_dict",
    "load_checkpoint_state_dict",
    "verify_checksum",
]
