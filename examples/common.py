"""Shared utilities for the DrugBAN and MapDiff examples."""

from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import yaml


DownloadFn = Callable[[str, Path], None]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    loaded["_config_dir"] = str(config_path.parent.resolve())
    return loaded


def require_keys(config: dict[str, Any], required: Iterable[str]) -> None:
    missing: list[str] = []
    for dotted_key in required:
        cursor: Any = config
        for part in dotted_key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                missing.append(dotted_key)
                break
            cursor = cursor[part]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(sorted(missing))}")


def set_seed(seed: int = 1024) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def default_downloader(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def is_valid_url(url: str | None) -> bool:
    return bool(url and url.startswith(("https://", "http://")))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pretrained_weight(
    config: dict[str, Any],
    *,
    pretrain: bool,
    downloader: DownloadFn | None = None,
) -> Path | None:
    """Resolve a local checkpoint or download it using config metadata.

    The config follows a Hugging Face-like layout:
    pretrained:
      filename: mapdiff_weight.pt
      local_path: weights/mapdiff_weight.pt
      url: https://...
      sha256: optional_checksum
    """

    if not pretrain:
        return None

    base_dir = Path(config.get("_config_dir", ".")).resolve()
    pretrained = config.get("pretrained") or {}
    filename = pretrained.get("filename") or "model.pt"
    local_path = Path(pretrained.get("local_path") or Path("weights") / filename)
    if not local_path.is_absolute():
        local_path = base_dir / local_path

    if local_path.exists():
        return local_path

    url = pretrained.get("url")
    if not is_valid_url(url):
        model_name = config.get("model", {}).get("name", "this model")
        raise FileNotFoundError(
            f"No pretrained weight file was found for {model_name} at {local_path}. "
            "No valid download URL is configured, so users may need to train the model themselves."
        )

    download = downloader or default_downloader
    download(url, local_path)
    checksum = pretrained.get("sha256")
    if checksum and sha256_file(local_path) != checksum:
        local_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for downloaded weight: {local_path}")
    return local_path


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model": checkpoint}


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def binary_classification_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("at least one prediction is required")

    threshold = 0.5
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "threshold": threshold,
    }
