"""Map DrugBAN bilinear attention to molecule atoms and protein residues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaleprotein.auto import (
    AutoProteinConfig, AutoProteinInterpreter, AutoProteinModel,
    AutoProteinPreprocessor,
)
from examples.drugban_dti._cli import (
    LazyPreprocessedDataset, add_data_arguments, load_dataset,
    load_requested_checkpoint, print_json, resolve_device, seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrain", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)

    # 1. Load and preprocess labeled or unlabeled pairs.
    dataset = load_dataset(args)
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = LazyPreprocessedDataset(dataset, preprocessor)
    # 2. Build the complete model and load weights.
    model = AutoProteinModel("DTI/DrugBAN", pretrain=args.pretrain)
    model.to(resolve_device(args.device))
    load_requested_checkpoint(model, args)
    loader = model.make_dataloader(
        processed, batch_size=args.batch_size, num_workers=args.num_workers
    )
    interpreter = AutoProteinInterpreter.from_config(config)
    samples = []
    for batch in loader:
        # 3. Embed and predict before optional attention interpretation.
        embeddings = model.embed(**batch)
        prediction = model.predictor(**embeddings)
        attention = model.extract_attention(**prediction)
        samples.extend(interpreter.explain(**attention)["samples"])
    result = {"samples": samples}
    print_json(result)
    return result


if __name__ == "__main__":
    main()
