"""Evaluate a MapDiff checkpoint on PDB or processed CATH graphs."""

import argparse
import json
from pathlib import Path

import torch

from kaleprotein.auto import (
    AutoProteinCollator,
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPreprocessor,
)
from examples._utils import move_to_device


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

    # 1. Load normalized inverse-folding records.
    data = AutoProteinData("CATH/InverseFolding", source=args.data)

    # 2. Preprocess protein backbones.
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = preprocessor.process(data)

    # 3. Collate data independently from the model.
    collator = AutoProteinCollator.from_config(config)
    batch = move_to_device(collator(**processed), args.device)

    # 4. Build the complete model and load requested weights.
    model = AutoProteinModel(
        "InverseFolding/MapDiff",
        pretrain=args.pretrained,
        checkpoint=args.checkpoint,
    ).to(args.device)
    model.eval()

    # 5. Embed the structural condition and generate sequences.
    with torch.no_grad():
        embeddings = model.embed(**batch)
        generation = model.predictor.generate(
            **embeddings,
            steps=args.steps,
            method=args.method,
            num_samples=args.num_samples,
        )

    # 6. Evaluate or interpret the generation mapping.
    metrics = model.evaluate(**generation)
    interpretation = AutoProteinInterpreter.from_config(config).explain(**generation)
    result = {"metrics": metrics, "interpretation": interpretation}
    print(json.dumps(result, indent=2))
    return metrics


if __name__ == "__main__":
    main()
