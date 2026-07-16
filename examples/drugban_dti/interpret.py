"""Map DrugBAN bilinear attention to molecule atoms and protein residues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from examples._utils import move_to_device
from examples.drugban_dti._cli import (
    add_data_arguments,
    print_json,
    resolve_device,
    seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--pretrain", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    if not args.root and not args.path:
        raise ValueError("Pass --root with a DrugBAN dataset tree or --path with a DTI CSV")

    # 1. Load, preprocess, collate, and batch DTI records.
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    loader = AutoProteinDataLoader(
        f"{args.dataset}/DTI",
        config=config,
        root=args.root,
        path=args.path,
        split=args.split,
        subset=args.subset,
        limit=args.limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. Build the model and interpreter.
    device = resolve_device(args.device)
    model = AutoProteinModel(
        "DTI/DrugBAN",
        pretrain=args.pretrain,
        checkpoint=args.checkpoint,
    ).to(device)
    interpreter = AutoProteinInterpreter.from_config(config)
    model.eval()

    # 3. Embed, predict, and interpret each prepared batch.
    samples = []
    with torch.no_grad():
        for inputs in loader:
            inputs = move_to_device(inputs, device)
            embeddings = model.embed(**inputs)
            prediction = model.predict(**embeddings)
            attention = model.extract_attention(**prediction)
            samples.extend(interpreter.explain(**attention)["samples"])
    result = {"samples": samples}
    print_json(result)
    return result


if __name__ == "__main__":
    main()
