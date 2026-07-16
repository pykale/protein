"""Predict DrugBAN interaction probabilities for a local DTI CSV."""

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
from examples._utils import move_to_device
from examples.drugban_dti._cli import (
    add_data_arguments,
    print_json,
    resolve_device,
    seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--pretrain", action="store_true")
    parser.add_argument("--smiles", help="Single molecule SMILES; use with --sequence")
    parser.add_argument("--sequence", help="Single protein sequence; use with --smiles")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    if bool(args.smiles) != bool(args.sequence):
        raise ValueError("--smiles and --sequence must be provided together")

    # 1. Select one pair or a reusable DTI dataset.
    if args.smiles:
        data_id = "UserInput/DTI"
        dataset = [
            {"id": "input_0", "smiles": args.smiles, "sequence": args.sequence}
        ]
        dataset_kwargs = {"dataset": dataset}
    else:
        if not args.root and not args.path:
            raise ValueError(
                "Pass --root with a DrugBAN dataset tree or --path with a DTI CSV"
            )
        data_id = f"{args.dataset}/DTI"
        dataset_kwargs = {
            "root": args.root,
            "path": args.path,
            "split": args.split,
            "subset": args.subset,
            "limit": args.limit,
        }

    # 2. Load, preprocess, collate, and batch inputs.
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    loader = AutoProteinDataLoader(
        data_id,
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **dataset_kwargs,
    )

    # 3. Build the model and load requested weights through AutoProteinModel.
    device = resolve_device(args.device)
    model = AutoProteinModel(
        "DTI/DrugBAN",
        pretrain=args.pretrain,
        checkpoint=args.checkpoint,
    ).to(device)
    model.eval()

    # 4. Embed and predict each prepared batch.
    predictions = []
    with torch.no_grad():
        for inputs in loader:
            inputs = move_to_device(inputs, device)
            embeddings = model.embed(**inputs)
            prediction = model.predict(**embeddings)
            for index, probability in enumerate(
                prediction["probabilities"].detach().cpu()
            ):
                predictions.append(
                    {
                        "id": prediction["sample_ids"][index],
                        "smiles": prediction["molecule_smiles"][index],
                        "probability": float(probability),
                    }
                )
    print_json(predictions)
    return predictions


if __name__ == "__main__":
    main()
