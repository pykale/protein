"""Evaluate DrugBAN on a local BindingDB, Human, or BioSNAP dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaleprotein.auto import (
    AutoProteinCollator,
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPreprocessor,
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
    add_data_arguments(parser, batching=False)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)

    # 1. Load normalized DTI records.
    if not args.root and not args.path:
        raise ValueError("Pass --root with a DrugBAN dataset tree or --path with a DTI CSV")
    data = AutoProteinData(
        f"{args.dataset}/DTI",
        root=args.root,
        path=args.path,
        split=args.split,
        subset=args.subset,
        limit=args.limit,
    )

    # 2. Preprocess SMILES and protein sequences.
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = preprocessor.process(data)

    # 3. Collate data independently from the model.
    collator = AutoProteinCollator.from_config(config)
    device = resolve_device(args.device)
    batch = move_to_device(collator(**processed), device)

    # 4. Build the complete model and load requested weights.
    model = AutoProteinModel(
        "DTI/DrugBAN",
        pretrain=args.pretrain,
        checkpoint=args.checkpoint,
    ).to(device)
    model.eval()

    # 5. Embed both modalities and predict interactions.
    with torch.no_grad():
        embeddings = model.embed(**batch)
        prediction = model.predictor(**embeddings)

    # 6. Evaluate or expose attention from the prediction mapping.
    metrics = model.evaluate(**prediction, threshold=args.threshold)
    attention = model.extract_attention(**prediction)
    interpretation = AutoProteinInterpreter.from_config(config).explain(**attention)
    result = {"metrics": metrics, "interpretation": interpretation}
    print_json(result)
    return metrics


if __name__ == "__main__":
    main()
