"""Interpret MapDiff denoising trajectories for PDB or processed CATH graphs."""

import argparse
import json
from pathlib import Path

import torch

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from examples._utils import move_to_device


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument(
        "--pretrained",
        action="store_true",
        help="Resolve and strictly load the configured v1.0.1 release.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    # 1. Load, preprocess, collate, and batch inverse-folding records.
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    loader = AutoProteinDataLoader(
        "CATH/InverseFolding",
        config=config,
        source=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. Build the model and independent trajectory interpreter.
    model = AutoProteinModel(
        "InverseFolding/MapDiff",
        pretrain=args.pretrained,
        checkpoint=args.checkpoint,
    ).to(args.device)
    interpreter = AutoProteinInterpreter.from_config(config)
    model.eval()

    # 3. Embed, generate, and interpret every prepared batch.
    interpretations = []
    final_sequences = []
    with torch.no_grad():
        for inputs in loader:
            inputs = move_to_device(inputs, args.device)
            embeddings = model.embed(**inputs)
            generation = model.generate(
                **embeddings,
                steps=args.steps,
                method=args.method,
                num_samples=args.num_samples,
            )
            interpretation = interpreter.explain(**generation)
            interpretations.append(interpretation)
            final_sequences.extend(interpretation["final_sequences"])
    if not interpretations:
        raise ValueError("Cannot interpret an empty inverse-folding dataset.")

    result = {
        "batches": interpretations,
        "final_sequences": final_sequences,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
