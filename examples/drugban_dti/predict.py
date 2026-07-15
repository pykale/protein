"""Predict DrugBAN interaction probabilities for a local DTI CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaleprotein.auto import AutoProteinConfig, AutoProteinModel, AutoProteinPreprocessor
from examples.drugban_dti._cli import (
    LazyPreprocessedDataset, add_data_arguments, load_dataset,
    load_requested_checkpoint, print_json, resolve_device, seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--smiles", help="Single molecule SMILES; use with --sequence")
    parser.add_argument("--sequence", help="Single protein sequence; use with --smiles")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    if bool(args.smiles) != bool(args.sequence):
        raise ValueError("--smiles and --sequence must be provided together")
    # 1. Load one pair or a reusable DTI dataset.
    dataset = (
        [{"id": "input_0", "smiles": args.smiles, "sequence": args.sequence}]
        if args.smiles
        else load_dataset(args)
    )
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")

    # 2. Preprocess and collate inputs.
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = LazyPreprocessedDataset(dataset, preprocessor)
    # 3. Build the complete model and load requested weights.
    model = AutoProteinModel("DTI/DrugBAN", pretrain=args.pretrain)
    model.to(resolve_device(args.device))
    load_requested_checkpoint(model, args)
    loader = model.make_dataloader(
        processed, batch_size=args.batch_size, num_workers=args.num_workers
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # 4. Embed both modalities and predict probabilities.
            embeddings = model.embed(**batch)
            output = model.predictor(**embeddings)
            for index, probability in enumerate(output["probabilities"].detach().cpu()):
                predictions.append({
                    "id": output["sample_ids"][index],
                    "smiles": output["molecule_smiles"][index],
                    "probability": float(probability),
                })
    print_json(predictions)
    return predictions


if __name__ == "__main__":
    main()
