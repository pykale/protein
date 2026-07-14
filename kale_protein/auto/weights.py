import os
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve


_STATE_DICT_KEYS = ("model", "state_dict", "model_state_dict")


def resolve_pretrained_weight(config, downloader=urlretrieve):
    """Resolve a local checkpoint, downloading it atomically only when absent."""
    block = config.get("pretrained", {})
    config_dir = Path(config.get("_config_dir", "."))
    local_dir = config_dir / block.get("local_dir", "weights")
    filename = block.get("filename")
    if not filename:
        raise ValueError(_missing_weight_message(config))

    weight_path = local_dir / filename
    expected_checksum = block.get("sha256")
    if weight_path.is_file():
        verify_checksum(weight_path, expected_checksum)
        return weight_path

    url = block.get("url")
    if not _valid_url(url):
        raise ValueError(_missing_weight_message(config))

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=weight_path.parent,
        prefix=f".{weight_path.name}.",
        suffix=".part",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        downloader(url, temporary_path)
        verify_checksum(temporary_path, expected_checksum)
        os.replace(temporary_path, weight_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return weight_path


def load_checkpoint_state_dict(checkpoint, *, map_location="cpu", loader=None):
    """Load and extract a state dict from a path or an in-memory checkpoint."""
    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        if loader is None:
            loader = _torch_checkpoint_loader
        payload = loader(checkpoint_path, map_location=map_location)
    return extract_checkpoint_state_dict(payload)


def load_pretrained_state_dict(
    config, *, map_location="cpu", downloader=urlretrieve, loader=None
):
    """Resolve a card's pretrained checkpoint and return its model state dict."""
    checkpoint_path = resolve_pretrained_weight(config, downloader=downloader)
    return load_checkpoint_state_dict(
        checkpoint_path, map_location=map_location, loader=loader
    )


def extract_checkpoint_state_dict(checkpoint):
    """Extract raw or commonly wrapped model parameters from a checkpoint object."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a state-dict mapping or contain one under "
            f"one of {_STATE_DICT_KEYS}; got {type(checkpoint).__name__}."
        )

    state_dict = checkpoint
    container_key = next((key for key in _STATE_DICT_KEYS if key in checkpoint), None)
    if container_key is not None:
        state_dict = checkpoint[container_key]
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Checkpoint entry {container_key!r} must contain a state-dict mapping; "
                f"got {type(state_dict).__name__}."
            )

    invalid_key = next((key for key in state_dict if not isinstance(key, str)), None)
    if invalid_key is not None:
        raise ValueError(
            f"State-dict keys must be strings; found {invalid_key!r} "
            f"({type(invalid_key).__name__})."
        )
    return state_dict


def verify_checksum(path, expected):
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
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def _torch_checkpoint_loader(path, *, map_location):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load checkpoint files. Install the optional "
            "model dependencies or pass a custom checkpoint loader."
        ) from exc
    return torch.load(path, map_location=map_location, weights_only=True)


def _valid_url(url):
    return isinstance(url, str) and url.startswith(("https://", "http://"))


def _missing_weight_message(config):
    model_id = config.get("model_id", config.get("name", "this model"))
    return (
        f"No pretrained weight file is available for {model_id}. "
        "Place the expected file in this model card's weights/ folder, provide a valid URL in config.yaml, "
        "or train the model yourself with pretrain=False."
    )


_verify_checksum = verify_checksum
