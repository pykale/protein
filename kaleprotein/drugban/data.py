"""DrugBAN data module with an AutoProteinData-compatible interface."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from kaleprotein.drugban.tokenization import DrugBANPreprocessor


@dataclass(frozen=True)
class DTIRecord:
    smiles: str
    protein: str
    label: int | None = None


class DrugBANDataset(Dataset):
    def __init__(self, records: list[DTIRecord], preprocessor: DrugBANPreprocessor):
        self.records = records
        self.preprocessor = preprocessor

    @classmethod
    def from_csv(cls, path: str | Path, preprocessor: DrugBANPreprocessor, *, require_label: bool = True) -> "DrugBANDataset":
        records: list[DTIRecord] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = {"SMILES", "Protein"} - fields
            if require_label:
                missing |= {"Y"} - fields
            if missing:
                raise ValueError(f"Missing DTI CSV columns in {path}: {', '.join(sorted(missing))}")
            for row in reader:
                label = int(row["Y"]) if row.get("Y") not in (None, "") else None
                records.append(DTIRecord(row["SMILES"], row["Protein"], label))
        return cls(records, preprocessor)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.preprocessor(record.smiles, record.protein).to_dict()
        if record.label is not None:
            encoded["labels"] = torch.tensor(float(record.label), dtype=torch.float32)
        return encoded


def drugban_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


class DrugBANDataModule:
    """HF-style data object returned by AutoProteinData("DTI/DrugBAN")."""

    def __init__(self, config: dict[str, Any], *, data_dir: str | Path | None = None):
        self.config = config
        self.model_dir = Path(config["_model_dir"])
        self.data_dir = Path(data_dir) if data_dir else self.model_dir / "data"
        self.preprocessor = DrugBANPreprocessor.from_config(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> "DrugBANDataModule":
        return cls(config, **kwargs)

    def dataset(self, split: str = "train", *, path: str | Path | None = None, require_label: bool = True) -> DrugBANDataset:
        csv_path = Path(path) if path else self.data_dir / f"{split}.csv"
        return DrugBANDataset.from_csv(csv_path, self.preprocessor, require_label=require_label)

    def load(self, split: str = "train", *, path: str | Path | None = None, require_label: bool = True) -> tuple[dict[str, list[str]], torch.Tensor | None]:
        dataset = self.dataset(split, path=path, require_label=require_label)
        data = {
            "drug": [record.smiles for record in dataset.records],
            "protein": [record.protein for record in dataset.records],
        }
        labels = [record.label for record in dataset.records]
        if any(label is None for label in labels):
            return data, None
        return data, torch.tensor(labels, dtype=torch.float32)

    def dataloader(
        self,
        split: str = "train",
        *,
        path: str | Path | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        require_label: bool = True,
    ) -> DataLoader:
        dataset = self.dataset(split, path=path, require_label=require_label)
        default_batch = int(self.config.get("training", {}).get("batch_size", 32))
        return DataLoader(
            dataset,
            batch_size=batch_size or default_batch,
            shuffle=(split == "train") if shuffle is None else shuffle,
            collate_fn=drugban_collate,
        )

    def preprocess(self, smiles: str, protein: str) -> dict[str, torch.Tensor]:
        return self.preprocessor(smiles, protein).to_dict()

    def __iter__(self) -> Iterator[DTIRecord]:
        dataset = self.dataset("train")
        return iter(dataset.records)
