"""Pretrain the structure-aware IPA masking prior on processed graphs."""

import argparse
from pathlib import Path

import torch

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinModel,
)
from kaleprotein.utils import move_to_device
from examples.mapdiff_inverse_folding.collators import MapDiffIPACollator


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("mapdiff_ipa_pretrain.pt"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    # 1. Load, preprocess, collate, and batch protein structures.
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    collator = MapDiffIPACollator(config=config)
    loader = AutoProteinDataLoader(
        "CATH/InverseFolding",
        config=config,
        source=args.data,
        collator=collator,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # 2. Build the complete model and select its IPA prior parameters.
    model = AutoProteinModel("InverseFolding/MapDiff").to(args.device)
    optimizer = torch.optim.AdamW(model.network.prior.parameters(), lr=args.learning_rate)
    model.train()

    # 3. Optimize the masking-prior objective.
    for _ in range(args.epochs):
        for inputs in loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = move_to_device(inputs, args.device)
            embeddings = model.embed(**inputs)
            output = model.predict(**embeddings)
            output["loss"].backward()
            optimizer.step()
    # 4. Save a complete checkpoint that AutoProteinModel can restore.
    model.save_checkpoint(args.output)
    return args.output


if __name__ == "__main__":
    main()
