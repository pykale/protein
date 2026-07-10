"""Small Hugging Face-like loading helpers for local protein examples."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import torch
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_CARD_ROOT = PACKAGE_ROOT / "model_cards"


def model_id_to_dir(model_id: str | Path) -> Path:
    raw = Path(model_id)
    if raw.exists():
        return raw.resolve()
    return MODEL_CARD_ROOT / str(model_id)


def load_model_config(model_id: str | Path) -> dict[str, Any]:
    model_dir = model_id_to_dir(model_id)
    for name in ("config.yaml", "config.yml", "config.json"):
        path = model_dir / name
        if path.exists():
            if path.suffix == ".json":
                config = json.loads(path.read_text(encoding="utf-8"))
            else:
                config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(config, dict):
                raise ValueError(f"Model config must be a mapping: {path}")
            config["_model_dir"] = str(model_dir)
            return config
    raise FileNotFoundError(f"No config.yaml/config.json found for model id: {model_id}")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def resolve_weight_file(
    config: dict[str, Any],
    *,
    pretrain: bool,
    downloader=_download,
) -> Path | None:
    if not pretrain:
        return None

    model_dir = Path(config["_model_dir"])
    weights = config.get("weights") or {}
    filename = weights.get("filename") or "pytorch_model.bin"
    local_path = Path(weights.get("local_path") or Path("weights") / filename)
    if not local_path.is_absolute():
        local_path = model_dir / local_path

    if local_path.exists():
        return local_path

    url = weights.get("url")
    if not url or not str(url).startswith(("https://", "http://")):
        model_name = config.get("model_id") or config.get("model_type") or "this model"
        raise FileNotFoundError(
            f"No pretrained weight file was found for {model_name} at {local_path}. "
            "No valid URL is configured. Train the model locally, place the checkpoint in the "
            "configured weights folder, or set weights.url in config.yaml."
        )

    downloader(url, local_path)
    checksum = weights.get("sha256")
    if checksum and sha256_file(local_path) != checksum:
        local_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for downloaded weights: {local_path}")
    return local_path


def load_state_dict_if_available(module: torch.nn.Module, weight_file: Path | None, *, strict: bool = False) -> None:
    if weight_file is None:
        return
    checkpoint = torch.load(weight_file, map_location="cpu")
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    module.load_state_dict(state_dict, strict=strict)
