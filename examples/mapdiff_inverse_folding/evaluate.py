"""Evaluate a MapDiff checkpoint on PDB or processed CATH graphs."""

import argparse
import json
from pathlib import Path

import torch

from kaleprotein.auto import AutoProteinData, AutoProteinModel, AutoProteinPreprocessor


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--pretrained", action="store_true", help="Resolve and strictly load the configured v1.0.1 release.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    # 1. Load, preprocess, and collate evaluation structures.
    dataset = AutoProteinData("InverseFolding/CATH", source=args.data)
    preprocessor = AutoProteinPreprocessor("protein/structure")
    processed = {"samples": [preprocessor.featurize(record) for record in dataset]}
    # 2. Build one complete model and load the selected checkpoint.
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=args.pretrained).to(args.device)
    if args.checkpoint:
        model.load_compatible_checkpoint(args.checkpoint)
    model.eval()
    batch = model.collator(**processed)
    batch["batch"] = batch["batch"].to(args.device)

    # 3. Encode structural conditions and generate sequences.
    conditioning = model.embed(**batch)
    output = model.predictor.generate(
        **conditioning,
        sampling_config={
            "steps": args.steps,
            "method": args.method,
            "num_samples": args.num_samples,
        },
    )
    # 4. Evaluate generation quality.
    metrics = model.evaluate(**output)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
