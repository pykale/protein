"""Generate inverse-folded sequences from a PDB or processed graph file."""

import argparse
import json
from pathlib import Path

import torch

from kaleprotein.auto import (
    AutoProteinCollator,
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPreprocessor,
)
from examples._utils import move_to_device


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
    data = AutoProteinData("CATH/InverseFolding", source=args.input)
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = preprocessor.process(data)

    # 2. Collate input structures independently from the model.
    collator = AutoProteinCollator.from_config(config)
    batch = move_to_device(collator(**processed), args.device)

    # 3. Build the model and load requested weights through AutoProteinModel.
    model = AutoProteinModel(
        "InverseFolding/MapDiff",
        pretrain=args.pretrained,
        checkpoint=args.checkpoint,
    ).to(args.device)
    model.eval()

    # 4. Encode the condition and run the registered generator.
    with torch.no_grad():
        embeddings = model.embed(**batch)
        generation = model.predictor.generate(
            **embeddings,
            steps=args.steps,
            method=args.method,
            num_samples=args.num_samples,
            temperature=args.temperature,
        )
    # 5. Serialize generated sequences and their denoising trajectory.
    serializable = {
        "sequences": generation["sequences"],
        "trajectory": generation["trajectory"],
    }
    text = json.dumps(serializable, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return generation


if __name__ == "__main__":
    main()
