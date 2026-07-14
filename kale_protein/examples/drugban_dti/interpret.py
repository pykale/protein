"""Map DrugBAN bilinear attention to molecule atoms and protein residues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kale_protein.auto import (
    AutoProteinConfig, AutoProteinInterpreter, AutoProteinPredictor,
    AutoProteinPreprocessor,
)
from kale_protein.examples.drugban_dti._cli import (
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
    dataset = load_dataset(args)
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    processed = LazyPreprocessedDataset(dataset, preprocessor)
    predictor = AutoProteinPredictor("DTI/DrugBAN", pretrain=args.pretrain)
    predictor.to(resolve_device(args.device))
    load_requested_checkpoint(predictor, args)
    loader = predictor.make_dataloader(
        processed, batch_size=args.batch_size, num_workers=args.num_workers
    )
    interpreter = AutoProteinInterpreter.from_config(config)
    samples = []
    for batch in loader:
        embeddings = predictor.embed_components(batch)
        predictor(embeddings["target"], embeddings["drug"])
        samples.extend(interpreter.explain(predictor, batch)["samples"])
    result = {"samples": samples}
    print_json(result)
    return result


if __name__ == "__main__":
    main()
