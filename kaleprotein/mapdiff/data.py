"""Lightweight local MapDiff data module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class MapDiffGraphDataset(Dataset):
    def __init__(self, graph_files: list[Path]):
        self.graph_files = graph_files

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = torch.load(self.graph_files[index], map_location="cpu")
        if isinstance(item, dict):
            return item
        return {name: value for name, value in vars(item).items() if isinstance(value, torch.Tensor)}


def mapdiff_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["x"].size(0) for item in batch)
    x = torch.zeros(len(batch), max_len, batch[0]["x"].size(-1))
    pos = torch.zeros(len(batch), max_len, 3)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.zeros(len(batch), max_len, dtype=torch.long) + 20
    for index, item in enumerate(batch):
        length = item["x"].size(0)
        x[index, :length] = item["x"]
        if "pos" in item:
            pos[index, :length] = item["pos"]
        mask[index, :length] = True
        labels[index, :length] = item["x"].argmax(dim=-1)
    return {"x": x, "pos": pos, "mask": mask, "labels": labels}


class MapDiffDataModule:
    def __init__(self, config: dict[str, Any], *, data_dir: str | Path | None = None):
        self.config = config
        self.model_dir = Path(config["_model_dir"])
        self.data_dir = Path(data_dir) if data_dir else self.model_dir / "data"

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> "MapDiffDataModule":
        return cls(config, **kwargs)

    def dataset(self, split: str = "train") -> MapDiffGraphDataset:
        return MapDiffGraphDataset(sorted((self.data_dir / split).glob("*.pt")))

    def load(self, split: str = "train") -> tuple[dict[str, list[dict[str, torch.Tensor]]], None]:
        dataset = self.dataset(split)
        structures = [dataset[index] for index in range(len(dataset))]
        return {"structure": structures}, None

    def dataloader(self, split: str = "train", *, batch_size: int | None = None, shuffle: bool | None = None) -> DataLoader:
        default_batch = int(self.config.get("training", {}).get("batch_size", 1))
        return DataLoader(
            self.dataset(split),
            batch_size=batch_size or default_batch,
            shuffle=(split == "train") if shuffle is None else shuffle,
            collate_fn=mapdiff_collate,
        )
