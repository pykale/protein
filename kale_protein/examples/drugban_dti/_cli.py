"""Shared, import-safe command-line helpers for the DrugBAN examples."""

from __future__ import annotations

import argparse
import json
import random

import torch

from kale_protein.auto import AutoProteinData


DATASETS = {name: f"DTI/{name}" for name in ("BindingDB", "Human", "BioSNAP")}


class LazyPreprocessedDataset:
    """Apply the Auto preprocessor per item so graph tensors stay batch-local."""

    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.preprocessor.transform_sample(self.dataset[index])


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, default="BindingDB")
    parser.add_argument("--root", help="Root containing bindingdb/, human/, or biosnap/")
    parser.add_argument("--path", help="Explicit CSV path (overrides --root)")
    parser.add_argument("--split", default="full", help="Dataset split directory, or full")
    parser.add_argument(
        "--subset",
        help="CSV basename within the split, e.g. train, test, or target_test",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--seed", type=int, default=42)


def load_dataset(args):
    if not args.root and not args.path:
        raise ValueError("Pass --root with a DrugBAN dataset tree or --path with a DTI CSV")
    return AutoProteinData(
        DATASETS[args.dataset], root=args.root, path=args.path, split=args.split,
        subset=args.subset, limit=args.limit,
    )


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def load_requested_checkpoint(predictor, args):
    if getattr(args, "checkpoint", None):
        predictor.load_checkpoint(args.checkpoint)


def json_ready(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def print_json(value):
    print(json.dumps(json_ready(value), indent=2, sort_keys=True))
