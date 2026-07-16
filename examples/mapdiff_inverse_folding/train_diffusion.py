"""Train the complete MapDiff denoising diffusion model."""

import argparse
from pathlib import Path

import torch

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinModel,
)
from examples._utils import move_to_device


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint", type=Path, help="Resume from a compatible full checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("mapdiff_diffusion.pt"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    # 1. Load, preprocess, collate, and batch CATH or PDB structures.
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    loader = AutoProteinDataLoader(
        "CATH/InverseFolding",
        config=config,
        source=args.data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # 2. Build the complete model and restore optional checkpoints.
    model = AutoProteinModel(
        "InverseFolding/MapDiff", checkpoint=args.checkpoint
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    # 3. Train the complete categorical diffusion objective.
    for _ in range(args.epochs):
        for inputs in loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = move_to_device(inputs, args.device)
            embeddings = model.embed(**inputs)
            output = model.predict(**embeddings)
            output["loss"].backward()
            optimizer.step()
    # 4. Save the full composed model.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(args.output)
    return args.output


if __name__ == "__main__":
    main()
