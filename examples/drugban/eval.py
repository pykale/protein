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

from examples.common import binary_classification_metrics
from kaleprotein import (
    AutoMoleculePreprocessor,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KaleProtein DrugBAN on test CSV data.")
    parser.add_argument("--config", default="examples/drugban/config.yaml")
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    example_dir = config_path.parent

    print("1. load data: test CSV")
    data_module = AutoProteinData(config["data_id"], data_dir=example_dir / "data")
    data, label = data_module.load(
        "test",
        path=example_dir / config["data"]["test_csv"],
    )

    print("2. preprocess: tokenize protein sequences and featurize SMILES")
    preprocessor_protein = AutoProteinPreprocessor("protein/sequence")
    preprocessor_drug = AutoMoleculePreprocessor("molecule/smiles")
    protein_data = preprocessor_protein.tokenize(data)
    drug_data = preprocessor_drug.tokenize(data)
    print({"protein_ids": tuple(protein_data["protein_ids"].shape), "drug_node_ids": tuple(drug_data["drug_node_ids"].shape)})

    print("3. embed: AutoProteinModel('DTI/DrugBAN') returns protein and molecule encoders")
    protein_model, molecule_model = AutoProteinModel(config["model_id"], pretrain=args.pretrain)
    protein_embedding = protein_model.embed(protein_data)
    drug_embedding = molecule_model.embed(drug_data)

    print("4. predict/evaluate: AutoProteinPredictor consumes embeddings")
    interaction_predictor = AutoProteinPredictor(config["model_id"], pretrain=args.pretrain)
    labels = [int(value.item()) for value in label] if label is not None else []
    with torch.no_grad():
        interaction_prediction = interaction_predictor(protein_embedding, drug_embedding, labels=label)
    scores = [float(value) for value in interaction_prediction["probabilities"]]

    metrics = binary_classification_metrics(labels, scores)
    output_dir = example_dir / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
