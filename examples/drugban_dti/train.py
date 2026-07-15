"""Train DrugBAN on a local BindingDB, Human, or BioSNAP split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPreprocessor,
)
from examples.drugban_dti._cli import (
    DATASETS, LazyPreprocessedDataset, add_data_arguments, load_dataset, print_json,
    resolve_device, seed_everything,
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
    seed_everything(args.seed)

    # 1. Load and preprocess task data.
    dataset = load_dataset(args)
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = LazyPreprocessedDataset(dataset, preprocessor)

    # 2. Build the complete model and collate training batches.
    model = AutoProteinModel("DTI/DrugBAN", pretrain=args.pretrain)
    model.to(resolve_device(args.device))
    loader = model.make_dataloader(
        processed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation = None
    if args.validation_subset:
        if args.path:
            raise ValueError("--validation-subset requires --root; a single --path has no sibling split")
        validation_data = AutoProteinData(
            DATASETS[args.dataset],
            root=args.root,
            path=None,
            split=args.split,
            subset=args.validation_subset,
            limit=args.limit,
        )
        validation = LazyPreprocessedDataset(validation_data, preprocessor)
    first_batch = next(iter(loader))

    # 3. Exercise the explicit embedding stage before optimization.
    embeddings = model.embed(**first_batch)

    # 4. Train, validate, and save the full model checkpoint.
    result = model.fit(
        loader, valid_data=validation, epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    model.save_checkpoint(args.checkpoint, extra={"training": result})
    print_json({
        "checkpoint": str(args.checkpoint),
        "training": result,
        "embedding_shapes": {
            name: list(value.shape)
            for name, value in embeddings.items()
            if name.endswith("_embedding")
        },
    })
    return result


if __name__ == "__main__":
    main()
