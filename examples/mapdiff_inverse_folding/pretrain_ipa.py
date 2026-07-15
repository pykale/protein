"""Pretrain the structure-aware IPA masking prior on processed graphs."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kaleprotein.auto import AutoProteinData, AutoProteinModel, AutoProteinPreprocessor


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

    # 1. Load and preprocess protein graphs.
    dataset = AutoProteinData("InverseFolding/CATH", source=args.data)
    preprocessor = AutoProteinPreprocessor("protein/structure")
    processed_graphs = [preprocessor.featurize(record)["graph"] for record in dataset]
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=False).to(args.device)
    loader = DataLoader(
        processed_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=model.ipa_collator,
    )
    # 2. Build the complete model and select its IPA prior parameters.
    optimizer = torch.optim.AdamW(model.predictor.network.prior.parameters(), lr=args.learning_rate)
    model.train()

    # 3. Optimize the masking-prior objective.
    for _ in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch["ipa_batch"] = batch["ipa_batch"].to(args.device)
            output = model.prior_pretrain_loss(**batch)
            output["loss"].backward()
            optimizer.step()
    # 4. Save a prior-only checkpoint for diffusion training.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "kale-mapdiff-v1-ipa",
            "prior_state_dict": model.predictor.network.prior.state_dict(),
        },
        args.output,
    )
    return args.output


if __name__ == "__main__":
    main()
