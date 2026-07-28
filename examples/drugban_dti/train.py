"""Train DrugBAN on a local BindingDB, Human, or BioSNAP split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinModel,
)
from kaleprotein.utils import move_to_device
from examples.drugban_dti import register_model_card
from examples.drugban_dti._cli import (
    add_data_arguments,
    print_json,
    resolve_device,
    seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument(
        "--validation-subset",
        help="Optional validation CSV basename in the same split, e.g. val or target_test",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    register_model_card()
    seed_everything(args.seed)
    if not args.root and not args.path:
        raise ValueError("Pass --root with a DrugBAN dataset tree or --path with a DTI CSV")

    # 1. Load, preprocess, collate, and batch training records.
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
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = None
    if args.validation_subset:
        if args.path:
            raise ValueError(
                "--validation-subset requires --root; a single --path has no sibling split"
            )
        validation_loader = AutoProteinDataLoader(
            f"{args.dataset}/DTI",
            config=config,
            root=args.root,
            split=args.split,
            subset=args.validation_subset,
            limit=args.limit,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # 2. Build the pure model and optimizer.
    device = resolve_device(args.device)
    model = AutoProteinModel("DTI/DrugBAN", pretrain=args.pretrain).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3. Run explicit embedding, prediction, loss, and optimization steps.
    history = []
    for _ in range(args.epochs):
        model.train()
        total_loss = 0.0
        examples = 0
        for inputs in loader:
            inputs = move_to_device(inputs, device)
            optimizer.zero_grad()
            embeddings = model.embed(**inputs)
            prediction = model.predict(**embeddings)
            labels = prediction["labels"].float().view_as(prediction["logits"])
            loss = F.binary_cross_entropy_with_logits(prediction["logits"], labels)
            loss.backward()
            optimizer.step()
            count = labels.numel()
            total_loss += float(loss.detach()) * count
            examples += count
        history.append(total_loss / max(examples, 1))

    result = {
        "loss": history[-1] if history else None,
        "history": history,
        "epochs": args.epochs,
    }

    # 4. Evaluate the optional validation split from prediction dictionaries.
    if validation_loader is not None:
        probabilities = []
        labels = []
        model.eval()
        with torch.no_grad():
            for inputs in validation_loader:
                inputs = move_to_device(inputs, device)
                embeddings = model.embed(**inputs)
                prediction = model.predict(**embeddings)
                probabilities.append(prediction["probabilities"].detach().cpu())
                labels.append(prediction["labels"].detach().cpu())
        result["validation"] = model.evaluate(
            probabilities=torch.cat(probabilities),
            labels=torch.cat(labels),
        )

    # 5. Save the full model checkpoint.
    model.save_checkpoint(args.checkpoint, optimizer=optimizer, extra={"training": result})
    print_json(
        {
            "checkpoint": str(args.checkpoint),
            "training": result,
        }
    )
    return result


if __name__ == "__main__":
    main()
