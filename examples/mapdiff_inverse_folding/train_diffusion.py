"""Train the complete MapDiff denoising diffusion model."""

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

    # 1. Load, preprocess, and collate CATH or PDB graphs.
    dataset = AutoProteinData("CATH/InverseFolding", source=args.data)
    preprocessor = AutoProteinPreprocessor("protein/structure")
    processed = [preprocessor.featurize(record) for record in dataset]
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=False).to(args.device)
    loader = DataLoader(
        processed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=model.collator,
    )
    # 2. Build the complete model and restore optional checkpoints.
    if args.checkpoint:
        model.load_compatible_checkpoint(args.checkpoint)
    if args.ipa_checkpoint:
        checkpoint = torch.load(args.ipa_checkpoint, map_location="cpu", weights_only=False)
        if "prior_state_dict" not in checkpoint:
            raise RuntimeError("IPA checkpoint is missing prior_state_dict; use output from pretrain_ipa.py.")
        model.predictor.network.prior.load_state_dict(checkpoint["prior_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    # 3. Train the complete categorical diffusion objective.
    for _ in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch["batch"] = batch["batch"].to(args.device)
            output = model(**batch)
            output["loss"].backward()
            optimizer.step()
    # 4. Save the full composed model.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(args.output)
    return args.output


if __name__ == "__main__":
    main()
