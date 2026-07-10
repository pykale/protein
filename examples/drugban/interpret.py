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

from kaleprotein import (
    AutoMoleculePreprocessor,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract KaleProtein DrugBAN BAN attention metadata.")
    parser.add_argument("--config", default="examples/drugban/config.yaml")
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    example_dir = config_path.parent

    print("1. load data: prediction CSV")
    data_module = AutoProteinData(config["data_id"], data_dir=example_dir / "data")
    data, label = data_module.load(
        "predict",
        path=example_dir / config["data"]["predict_csv"],
        require_label=False,
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

    print("4. interpret: AutoProteinPredictor returns scores and BAN attention")
    interaction_predictor = AutoProteinPredictor(config["model_id"], pretrain=args.pretrain)
    rows = []
    with torch.no_grad():
        interaction_prediction = interaction_predictor(protein_embedding, drug_embedding, return_attention=True)
    for index, score in enumerate(interaction_prediction["probabilities"].tolist()):
        row = {
            "score": float(score),
            "attention_shape": list(interaction_prediction["attention"][index].shape),
        }
        if label is not None:
            row["label"] = int(label[index].item())
        rows.append(row)

    output_dir = example_dir / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "interpretation.json").write_text(json.dumps({"interpretations": rows}, indent=2) + "\n", encoding="utf-8")
    print({"interpretations": rows})


if __name__ == "__main__":
    main()
