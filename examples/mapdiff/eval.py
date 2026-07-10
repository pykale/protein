from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaleprotein import AutoProteinData, AutoProteinGenerator, AutoProteinModel, AutoProteinPreprocessor


AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KaleProtein MapDiff-style inverse-folding model.")
    parser.add_argument("--config", default="examples/mapdiff/config.yaml")
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    example_dir = config_path.parent

    print("1. load data: processed CATH test graph files")
    data_module = AutoProteinData(config["data_id"], data_dir=example_dir / "data/processed")
    data, _ = data_module.load("test")

    print("2. preprocess: featurize residue structures")
    structure_preprocessor = AutoProteinPreprocessor("protein/structure")
    structure_data = structure_preprocessor.featurize(data)
    print({name: tuple(value.shape) for name, value in structure_data.items()})

    print("3. embed: AutoProteinModel('InverseFolding/MapDiff') builds structure embeddings")
    structure_encoder = AutoProteinModel(config["model_id"], pretrain=args.pretrain)
    structure_embedding = structure_encoder.embed(structure_data)

    print("4. generate/evaluate: AutoProteinGenerator predicts sequences")
    sequence_generator = AutoProteinGenerator(config["model_id"], pretrain=args.pretrain)
    predictions = []
    structure_encoder.eval()
    sequence_generator.eval()
    with torch.no_grad():
        generation = sequence_generator.generate(structure_embedding)
    generated_sequences = generation["sequences"][0]
    for pred, truth, mask in zip(generated_sequences, structure_data["labels"], structure_data["mask"]):
        length = int(mask.sum().item())
        pred_seq = "".join(AMINO_ACIDS[index] for index in pred[:length].tolist())
        true_seq = "".join(AMINO_ACIDS[index] for index in truth[:length].tolist())
        recovery = float((pred[:length] == truth[:length]).float().mean()) if length else 0.0
        predictions.append({"sequence_length": length, "sample_recovery": recovery, "pred_sequence": pred_seq, "true_sequence": true_seq})

    output_dir = example_dir / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_predictions.json").write_text(json.dumps({"predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    print({"predictions": predictions})


if __name__ == "__main__":
    main()
