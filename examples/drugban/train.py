from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Train KaleProtein DrugBAN through AutoProteinModel/AutoProteinData.")
    parser.add_argument("--config", default="examples/drugban/config.yaml")
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    example_dir = config_path.parent

    print("1. load data: CSV train data and labels")
    data_module = AutoProteinData(config["data_id"], data_dir=example_dir / "data")
    data, label = data_module.load(
        "train",
        path=example_dir / config["data"]["train_csv"],
    )
    if label is None:
        raise ValueError("DrugBAN training requires labels in the Y column.")

    print("2. preprocess: tokenize protein sequences and featurize SMILES")
    preprocessor_protein = AutoProteinPreprocessor("protein/sequence")
    preprocessor_drug = AutoMoleculePreprocessor("molecule/smiles")
    protein_data = preprocessor_protein.tokenize(data)
    drug_data = preprocessor_drug.tokenize(data)
    print({"protein_ids": tuple(protein_data["protein_ids"].shape), "drug_node_ids": tuple(drug_data["drug_node_ids"].shape)})

    print("3. embed/model: AutoProteinModel('DTI/DrugBAN') returns protein and molecule encoders")
    protein_model, molecule_model = AutoProteinModel(config["model_id"], pretrain=args.pretrain)
    interaction_predictor = AutoProteinPredictor(config["model_id"], pretrain=args.pretrain)
    optimizer = torch.optim.Adam(
        list(protein_model.parameters()) + list(molecule_model.parameters()) + list(interaction_predictor.parameters()),
        lr=float(config["solver"]["lr"]),
    )

    print("4. train: embed, predict interactions, optimize loss, save checkpoint")
    protein_model.train()
    molecule_model.train()
    interaction_predictor.train()
    last_loss = 0.0
    for _ in range(int(config["solver"]["max_epoch"])):
        optimizer.zero_grad()
        protein_embedding = protein_model.embed(protein_data)
        drug_embedding = molecule_model.embed(drug_data)
        interaction_prediction = interaction_predictor(protein_embedding, drug_embedding, labels=label)
        interaction_prediction["loss"].backward()
        optimizer.step()
        last_loss = float(interaction_prediction["loss"].detach())

    output_dir = example_dir / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protein_model": protein_model.state_dict(),
            "molecule_model": molecule_model.state_dict(),
            "interaction_predictor": interaction_predictor.state_dict(),
        },
        output_dir / "drugban_finetuned.pt",
    )
    print({"train_loss": last_loss, "checkpoint": str(output_dir / "drugban_finetuned.pt")})


if __name__ == "__main__":
    main()
