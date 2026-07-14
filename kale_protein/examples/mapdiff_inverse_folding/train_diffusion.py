"""Train the complete MapDiff denoising diffusion model."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kale_protein.auto import AutoProteinData, AutoProteinModel, AutoProteinPreprocessor
from kale_protein.tasks.inverse_folding.collators import CollatorDiff


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint", type=Path, help="Resume from a compatible full checkpoint.")
    parser.add_argument("--ipa-checkpoint", type=Path, help="Load output from pretrain_ipa.py.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("mapdiff_diffusion.pt"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    dataset = AutoProteinData("InverseFolding/CATH", source=args.data)
    preprocessor = AutoProteinPreprocessor("protein/structure")
    processed_graphs = [preprocessor.featurize(record)["graph"] for record in dataset]
    loader = DataLoader(
        processed_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CollatorDiff(),
    )
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=False).to(args.device)
    if args.checkpoint:
        model.load_compatible_checkpoint(args.checkpoint)
    if args.ipa_checkpoint:
        checkpoint = torch.load(args.ipa_checkpoint, map_location="cpu", weights_only=False)
        if "prior_state_dict" not in checkpoint:
            raise RuntimeError("IPA checkpoint is missing prior_state_dict; use output from pretrain_ipa.py.")
        model.network.prior.load_state_dict(checkpoint["prior_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for _ in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.to(args.device))
            output["loss"].backward()
            optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "kale-mapdiff-v1", "state_dict": model.state_dict()}, args.output)
    return args.output


if __name__ == "__main__":
    main()
