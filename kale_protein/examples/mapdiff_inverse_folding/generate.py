"""Generate inverse-folded sequences from a PDB or processed graph file."""

import argparse
import json
from pathlib import Path

import torch

from kale_protein.auto import (
    AutoProteinData,
    AutoProteinGenerator,
    AutoProteinModel,
    AutoProteinPreprocessor,
)
from kale_protein.tasks.inverse_folding.collators import CollatorDiff


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
    dataset = AutoProteinData("InverseFolding/CATH", source=args.input)
    preprocessor = AutoProteinPreprocessor("protein/structure")
    processed_graphs = [preprocessor.featurize(record)["graph"] for record in dataset]
    batch = CollatorDiff()(processed_graphs).to(args.device)
    encoder = AutoProteinModel("InverseFolding/MapDiff", pretrain=args.pretrained).to(args.device)
    generator = AutoProteinGenerator("InverseFolding/MapDiff", pretrain=args.pretrained).to(args.device)
    if args.checkpoint:
        encoder.load_compatible_checkpoint(args.checkpoint)
        generator.model.load_compatible_checkpoint(args.checkpoint)
    conditioning = encoder.embed(batch)
    output = generator.generate(
        conditioning,
        steps=args.steps,
        method=args.method,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
    serializable = {"sequences": output["sequences"], "trajectory": output["trajectory"]}
    text = json.dumps(serializable, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return output


if __name__ == "__main__":
    main()
