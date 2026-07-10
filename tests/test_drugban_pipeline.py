from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kaleprotein import (
    AutoMoleculePreprocessor,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)
from kaleprotein.drugban.data import DrugBANDataset
from kaleprotein.drugban.tokenization import DrugBANPreprocessor


def test_auto_protein_model_builds_local_drugban() -> None:
    protein_model, molecule_model = AutoProteinModel("DTI/DrugBAN")
    predictor = AutoProteinPredictor("DTI/DrugBAN")
    protein_data = AutoProteinPreprocessor("protein/sequence").tokenize({"protein": ["ACDE"]})
    drug_data = AutoMoleculePreprocessor("molecule/smiles").tokenize({"drug": ["CCO"]})

    protein_embedding = protein_model.embed(protein_data)
    drug_embedding = molecule_model.embed(drug_data)
    output = predictor(protein_embedding, drug_embedding, return_attention=True)

    assert output["logits"].shape == (1,)
    assert output["probabilities"].shape == (1,)
    assert output["attention"].ndim == 4


def test_auto_protein_data_loads_csv_and_batches(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("SMILES,Protein,Y\nCCO,ACDE,1\nCCN,AAAA,0\n", encoding="utf-8")
    data = AutoProteinData("DTI/PDBBind", data_dir=data_dir)

    batch = next(iter(data.dataloader("train", batch_size=2)))

    assert batch["drug_node_ids"].shape[0] == 2
    assert batch["drug_adj"].shape[1:] == (290, 290)
    assert batch["protein_ids"].shape[1] == 1200
    assert batch["labels"].tolist() == [1.0, 0.0]


def test_drugban_csv_requires_training_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "dti.csv"
    csv_path.write_text("SMILES,Protein\nCCO,ACDE\n", encoding="utf-8")
    preprocessor = DrugBANPreprocessor(max_drug_nodes=8, max_protein_length=8)

    with pytest.raises(ValueError, match="Y"):
        DrugBANDataset.from_csv(csv_path, preprocessor)


def test_drugban_pretrain_true_errors_without_weight() -> None:
    with pytest.raises(FileNotFoundError, match="Train the model locally"):
        AutoProteinModel("DTI/DrugBAN", pretrain=True)


def test_drugban_one_training_step(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("SMILES,Protein,Y\nCCO,ACDE,1\nCCN,AAAA,0\n", encoding="utf-8")
    data, label = AutoProteinData("DTI/PDBBind", data_dir=data_dir).load("train")
    protein_data = AutoProteinPreprocessor("protein/sequence").tokenize(data)
    drug_data = AutoMoleculePreprocessor("molecule/smiles").tokenize(data)
    protein_model, molecule_model = AutoProteinModel("DTI/DrugBAN")
    predictor = AutoProteinPredictor("DTI/DrugBAN")
    opt = torch.optim.Adam(
        list(protein_model.parameters()) + list(molecule_model.parameters()) + list(predictor.parameters()),
        lr=1e-4,
    )

    opt.zero_grad()
    protein_embedding = protein_model.embed(protein_data)
    drug_embedding = molecule_model.embed(drug_data)
    output = predictor(protein_embedding, drug_embedding, labels=label)
    output["loss"].backward()
    opt.step()

    assert torch.isfinite(output["loss"])
