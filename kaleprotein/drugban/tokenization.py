"""Tokenizers and lightweight graph preprocessing for DrugBAN."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kaleprotein.hub import load_json


DEFAULT_PROTEIN_VOCAB = {
    "<pad>": 0,
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}


DEFAULT_SMILES_VOCAB = {
    "<pad>": 0,
    "<unk>": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "S": 5,
    "P": 6,
    "F": 7,
    "Cl": 8,
    "Br": 9,
    "I": 10,
    "B": 11,
    "c": 12,
    "n": 13,
    "o": 14,
    "s": 15,
    "(": 16,
    ")": 17,
    "=": 18,
    "#": 19,
    "-": 20,
    "[": 21,
    "]": 22,
    "+": 23,
    "@": 24,
    "1": 25,
    "2": 26,
    "3": 27,
    "4": 28,
    "5": 29,
    "6": 30,
}


SMILES_PATTERN = re.compile(r"Cl|Br|[A-Za-z]|\d|[^A-Za-z\d]")


@dataclass
class DrugBANBatchEncoding:
    drug_node_ids: torch.Tensor
    drug_adj: torch.Tensor
    drug_mask: torch.Tensor
    protein_ids: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "drug_node_ids": self.drug_node_ids,
            "drug_adj": self.drug_adj,
            "drug_mask": self.drug_mask,
            "protein_ids": self.protein_ids,
        }


class DrugBANPreprocessor:
    """Preprocess SMILES/protein pairs into tensors consumed by DrugBANForDTI."""

    def __init__(
        self,
        *,
        smiles_vocab: dict[str, int] | None = None,
        protein_vocab: dict[str, int] | None = None,
        max_drug_nodes: int = 290,
        max_protein_length: int = 1200,
    ):
        self.smiles_vocab = smiles_vocab or DEFAULT_SMILES_VOCAB
        self.protein_vocab = protein_vocab or DEFAULT_PROTEIN_VOCAB
        self.max_drug_nodes = max_drug_nodes
        self.max_protein_length = max_protein_length

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DrugBANPreprocessor":
        model_dir = Path(config["_model_dir"])
        preprocessing = config.get("preprocessing", {})
        maps = config.get("maps", {})
        smiles_vocab = DEFAULT_SMILES_VOCAB
        protein_vocab = DEFAULT_PROTEIN_VOCAB
        if maps.get("smiles_vocab"):
            smiles_vocab = load_json(model_dir / maps["smiles_vocab"])
        if maps.get("protein_vocab"):
            protein_vocab = load_json(model_dir / maps["protein_vocab"])
        return cls(
            smiles_vocab=smiles_vocab,
            protein_vocab=protein_vocab,
            max_drug_nodes=int(preprocessing.get("max_drug_nodes", 290)),
            max_protein_length=int(preprocessing.get("max_protein_length", 1200)),
        )

    def tokenize_smiles(self, smiles: str) -> list[str]:
        return SMILES_PATTERN.findall(smiles)

    def encode_smiles_graph(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.tokenize_smiles(smiles)[: self.max_drug_nodes]
        node_ids = torch.zeros(self.max_drug_nodes, dtype=torch.long)
        mask = torch.zeros(self.max_drug_nodes, dtype=torch.bool)
        adj = torch.eye(self.max_drug_nodes, dtype=torch.float32)
        unk = self.smiles_vocab.get("<unk>", 1)
        for index, token in enumerate(tokens):
            node_ids[index] = self.smiles_vocab.get(token, unk)
            mask[index] = True
            if index > 0:
                adj[index, index - 1] = 1.0
                adj[index - 1, index] = 1.0
        return node_ids, adj, mask

    def encode_protein(self, sequence: str) -> torch.Tensor:
        encoded = torch.zeros(self.max_protein_length, dtype=torch.long)
        for index, residue in enumerate(sequence[: self.max_protein_length]):
            encoded[index] = self.protein_vocab.get(residue.upper(), 0)
        return encoded

    def __call__(self, smiles: str, protein: str) -> DrugBANBatchEncoding:
        drug_node_ids, drug_adj, drug_mask = self.encode_smiles_graph(smiles)
        protein_ids = self.encode_protein(protein)
        return DrugBANBatchEncoding(drug_node_ids, drug_adj, drug_mask, protein_ids)


def _as_list(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values]
    return list(values)


class DrugBANProteinPreprocessor:
    """Sequence preprocessor used by AutoProteinPreprocessor("protein/sequence")."""

    def __init__(self, *, max_length: int = 1200, protein_vocab: dict[str, int] | None = None):
        self.preprocessor = DrugBANPreprocessor(protein_vocab=protein_vocab, max_protein_length=max_length)

    def tokenize(self, data: Any) -> dict[str, torch.Tensor]:
        sequences = data["protein"] if isinstance(data, dict) else data
        encoded = [self.preprocessor.encode_protein(sequence) for sequence in _as_list(sequences)]
        return {"protein_ids": torch.stack(encoded)}


class DrugBANMoleculePreprocessor:
    """SMILES preprocessor used by AutoMoleculePreprocessor("molecule/smiles")."""

    def __init__(self, *, max_nodes: int = 290, smiles_vocab: dict[str, int] | None = None):
        self.preprocessor = DrugBANPreprocessor(smiles_vocab=smiles_vocab, max_drug_nodes=max_nodes)

    def tokenize(self, data: Any) -> dict[str, torch.Tensor]:
        return self.featurize(data)

    def featurize(self, data: Any) -> dict[str, torch.Tensor]:
        smiles_values = data["drug"] if isinstance(data, dict) else data
        node_ids, adj, mask = [], [], []
        for smiles in _as_list(smiles_values):
            curr_node_ids, curr_adj, curr_mask = self.preprocessor.encode_smiles_graph(smiles)
            node_ids.append(curr_node_ids)
            adj.append(curr_adj)
            mask.append(curr_mask)
        return {
            "drug_node_ids": torch.stack(node_ids),
            "drug_adj": torch.stack(adj),
            "drug_mask": torch.stack(mask),
        }
