"""Evaluate DrugBAN on a local BindingDB, Human, or BioSNAP dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinModel,
)
from kaleprotein.utils import move_to_device
from examples.drugban_dti._cli import (
    add_data_arguments,
    print_json,
    resolve_device,
    seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)

    # 1. Load, preprocess, collate, and batch normalized DTI records.
    if not args.root and not args.path:
        raise ValueError("Pass --root with a DrugBAN dataset tree or --path with a DTI CSV")
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    loader = AutoProteinDataLoader(
        f"{args.dataset}/DTI",
        config=config,
        root=args.root,
        path=args.path,
        split=args.split,
        subset=args.subset,
        limit=args.limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. Build the complete model and load requested weights.
    device = resolve_device(args.device)
    model = AutoProteinModel(
        "DTI/DrugBAN",
        pretrain=args.pretrain,
        checkpoint=args.checkpoint,
    ).to(device)
    model.eval()

    # 3. Embed, predict, and collect named outputs for every batch.
    probabilities = []
    labels = []
    with torch.no_grad():
        for inputs in loader:
            inputs = move_to_device(inputs, device)
            embeddings = model.embed(**inputs)
            prediction = model.predict(**embeddings)
            probabilities.append(prediction["probabilities"].detach().cpu())
            labels.append(prediction["labels"].detach().cpu())
    if not probabilities:
        raise ValueError("Cannot evaluate an empty DTI dataset.")

    # 4. Evaluate the collected prediction mapping.
    metrics = model.evaluate(
        probabilities=torch.cat(probabilities),
        labels=torch.cat(labels),
        threshold=args.threshold,
    )
    result = {"metrics": metrics}
    print_json(result)
    return metrics


if __name__ == "__main__":
    main()
