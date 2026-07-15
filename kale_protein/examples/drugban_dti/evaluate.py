"""Evaluate DrugBAN on a local BindingDB, Human, or BioSNAP dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kale_protein.auto import AutoProteinConfig, AutoProteinModel, AutoProteinPreprocessor
from kale_protein.examples.drugban_dti._cli import (
    LazyPreprocessedDataset, add_data_arguments, load_dataset,
    load_requested_checkpoint, print_json, resolve_device, seed_everything,
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

    # 1. Load and preprocess the held-out split.
    dataset = load_dataset(args)
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = LazyPreprocessedDataset(dataset, preprocessor)

    # 2. Build the full model, load weights, and collate batches.
    model = AutoProteinModel("DTI/DrugBAN", pretrain=args.pretrain)
    model.to(resolve_device(args.device))
    load_requested_checkpoint(model, args)
    loader = model.make_dataloader(
        processed, batch_size=args.batch_size, num_workers=args.num_workers
    )
    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # 3. Embed protein and molecule streams, then predict interactions.
            embeddings = model.embed(**batch)
            output = model.predictor(**embeddings)
            probabilities.append(output["probabilities"].detach().cpu())
            labels.append(output["labels"].detach().cpu())
    # 4. Compute evaluation-only metrics.
    prediction = {
        "probabilities": torch.cat(probabilities),
        "labels": torch.cat(labels),
    }
    metrics = model.evaluate(**prediction, threshold=args.threshold)
    print_json(metrics)
    return metrics


if __name__ == "__main__":
    main()
