"""Generate inverse-folded sequences from a PDB or processed graph file."""

import argparse
import json
from pathlib import Path

import torch

from kale_protein.auto import (
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPreprocessor,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A PDB, processed .pt graph, or directory of graphs.")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path, help="A local lightweight or upstream MapDiff checkpoint.")
    weights.add_argument("--pretrained", action="store_true", help="Resolve and strictly load the configured v1.0.1 release.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    # 1. Load, preprocess, and collate input structures.
    dataset = AutoProteinData("InverseFolding/CATH", source=args.input)
    preprocessor = AutoProteinPreprocessor("protein/structure")
    processed = {"samples": [preprocessor.featurize(record) for record in dataset]}
    # 2. Build one complete model and load the selected checkpoint.
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=args.pretrained).to(args.device)
    if args.checkpoint:
        model.load_compatible_checkpoint(args.checkpoint)
    batch = model.collator(**processed)
    batch["batch"] = batch["batch"].to(args.device)
    # 3. Encode the condition and run the registered generator.
    conditioning = model.embed(**batch)
    output = model.predictor.generate(
        **conditioning,
        steps=args.steps,
        method=args.method,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
    # 4. Serialize generated sequences and their denoising trajectory.
    serializable = {"sequences": output["sequences"], "trajectory": output["trajectory"]}
    text = json.dumps(serializable, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return output


if __name__ == "__main__":
    main()
