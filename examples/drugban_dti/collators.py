"""DrugBAN-specific collation kept outside the model implementation."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from kaleprotein.core.preprocessing.molecule import RDKitGraphProcessor


PROTEIN_ALPHABET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14,
    "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20,
    "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}


def encode_protein(sequence: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = str(sequence).upper()[:max_length]
    tokens = torch.zeros(max_length, dtype=torch.long)
    mask = torch.zeros(max_length, dtype=torch.bool)
    for index, residue in enumerate(sequence):
        tokens[index] = PROTEIN_ALPHABET.get(residue, 0)
        mask[index] = residue in PROTEIN_ALPHABET
    return tokens, mask


class DrugBANCollator:
    """Collate preprocessed DTI samples into DrugBAN tensor mappings."""

    def __init__(self, config, **kwargs):
        self.config = config
        streams = config.get_streams()
        drug_kwargs = dict(streams["drug"].processor_kwargs)
        self.max_drug_nodes = int(drug_kwargs.get("max_nodes", 290))
        self.max_protein_length = int(
            streams["target"].processor_kwargs.get("max_length", 1000)
        )
        self.molecule_processor = RDKitGraphProcessor(
            input_key=streams["drug"].input_key,
            max_nodes=self.max_drug_nodes,
        )
        kernels = tuple(config.get_embedders()["target"].kwargs.get("kernels", (3, 6, 9)))
        self.minimum_protein_length = sum(int(kernel) - 1 for kernel in kernels) + 1

    def __call__(self, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        samples = list(samples)
        if not samples:
            raise ValueError("DrugBAN cannot collate an empty batch.")
        drug_items = []
        target_items = []
        labels = []
        ids = []
        for sample in samples:
            if not isinstance(sample, dict):
                raise TypeError(
                    f"DrugBAN samples must be dictionaries, got {type(sample).__name__}."
                )
            drug = sample.get("drug")
            if not (isinstance(drug, dict) and "node_features" in drug):
                drug = self.molecule_processor.transform(sample)
            target = sample.get("target")
            if not isinstance(target, dict):
                target = {"sequence": sample.get("sequence", "")}
            drug_items.append(drug)
            target_items.append(target)
            if "label" in sample:
                labels.append(float(sample["label"]))
            ids.append(sample.get("id"))
        batch = {
            "drug": self.collate_drugs(drug_items),
            "target": self.collate_proteins(target_items),
            "ids": ids,
        }
        if labels:
            if len(labels) != len(samples):
                raise ValueError("Every sample in a labeled DrugBAN batch must have a label.")
            batch["label"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    def collate_drugs(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        graphs = list(items)
        if not graphs:
            raise ValueError("Cannot collate an empty molecule batch.")
        max_nodes = max(int(torch.as_tensor(graph["node_features"]).shape[0]) for graph in graphs)
        feature_dims = {
            int(torch.as_tensor(graph["node_features"]).shape[-1]) for graph in graphs
        }
        if not feature_dims <= {74, 75}:
            raise ValueError(
                "DrugBAN expects canonical 74-feature graphs or legacy 75-feature "
                f"graphs, received dimensions {sorted(feature_dims)}."
            )
        features = torch.zeros((len(graphs), max_nodes, 75), dtype=torch.float32)
        features[:, :, 74] = 1.0
        adjacency = torch.eye(max_nodes, dtype=torch.float32).expand(len(graphs), -1, -1).clone()
        masks = torch.zeros((len(graphs), max_nodes), dtype=torch.bool)
        smiles = []
        atom_symbols = []
        for index, graph in enumerate(graphs):
            graph_features = torch.as_tensor(graph["node_features"], dtype=torch.float32)
            count = graph_features.shape[0]
            graph_mask = torch.as_tensor(
                graph.get("node_mask", torch.ones(count, dtype=torch.bool)), dtype=torch.bool
            )
            graph_adjacency = graph.get("adjacency")
            if graph_adjacency is None:
                graph_adjacency = _adjacency_from_edge_index(graph.get("edge_index"), count)
            features[index, :count, 74] = 0.0
            features[index, :count, : graph_features.shape[-1]] = graph_features
            adjacency[index, :count, :count] = torch.as_tensor(graph_adjacency, dtype=torch.float32)
            masks[index, :count] = graph_mask
            smiles.append(graph.get("smiles"))
            atom_symbols.append(list(graph.get("atom_symbols", [])))
        return {
            "node_features": features,
            "adjacency": adjacency,
            "node_mask": masks,
            "smiles": smiles,
            "atom_symbols": atom_symbols,
        }

    def collate_proteins(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        proteins = list(items)
        if not proteins:
            raise ValueError("Cannot collate an empty protein batch.")
        sequences = [protein.get("sequence") for protein in proteins]
        if all(sequence is not None for sequence in sequences):
            encoded = [encode_protein(sequence, self.max_protein_length) for sequence in sequences]
            return {
                "tokens": torch.stack([item[0] for item in encoded]),
                "attention_mask": torch.stack([item[1] for item in encoded]),
                "sequences": [str(sequence)[: self.max_protein_length] for sequence in sequences],
            }
        raw_tokens = [torch.as_tensor(protein["tokens"], dtype=torch.long).flatten() for protein in proteins]
        length = max(self.minimum_protein_length, max(tokens.numel() for tokens in raw_tokens))
        tokens = torch.zeros((len(proteins), length), dtype=torch.long)
        mask = torch.zeros((len(proteins), length), dtype=torch.bool)
        for index, item in enumerate(proteins):
            item_tokens = raw_tokens[index][:length]
            item_mask = torch.as_tensor(
                item.get("attention_mask", torch.ones(item_tokens.numel())), dtype=torch.bool
            ).flatten()[: item_tokens.numel()]
            tokens[index, : item_tokens.numel()] = item_tokens
            mask[index, : item_mask.numel()] = item_mask
        return {"tokens": tokens, "attention_mask": mask, "sequences": sequences}


def _adjacency_from_edge_index(edge_index, num_nodes):
    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    if edge_index is None:
        return adjacency
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    if edges.numel():
        adjacency[edges[0], edges[1]] = 1.0
    return adjacency


__all__ = ["DrugBANCollator", "encode_protein"]
