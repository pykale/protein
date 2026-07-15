"""Complete DrugBAN model assembled from reusable KaleProtein components.

The tensor conventions follow peizhenbai/DrugBAN while remaining independent
of DGL and the upstream repository at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from kale_protein.auto import AutoProteinEmbedder, AutoProteinPredictor
from kale_protein.core.data.modalities.molecule.processors import RDKitGraphProcessor
from kale_protein.core.weights import load_checkpoint_state_dict, resolve_pretrained_weight


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


def _training(config, key: str, default: Any) -> Any:
    value = config.get("training", {})
    return value.get(key, default) if isinstance(value, dict) else default


def _device_of(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


class DrugBANCollator:
    """Collate raw or preprocessed DTI samples into tensor graph batches."""

    def __init__(self, config):
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


class DrugBANModel(nn.Module):
    """Full DrugBAN composition root and sole checkpoint owner."""

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.pretrain = pretrain
        self.collator = DrugBANCollator(config)
        embedders = config.get_embedders()
        self.protein_embedder = AutoProteinEmbedder.from_config(
            embedders["target"], config=config
        )
        self.molecule_embedder = AutoProteinEmbedder.from_config(
            embedders["drug"], config=config
        )
        self.predictor = AutoProteinPredictor.from_config(
            config.get_predictor(), config=config
        )
        self.weight_path = None
        if pretrain:
            self.weight_path = resolve_pretrained_weight(config)
            self.load_checkpoint(self.weight_path)

    def embed(self, data=None, *, drug=None, target=None, label=None, ids=None, **batch_fields):
        """Return a flat mapping that can be expanded into the task predictor."""

        if data is None:
            data = {"drug": drug, "target": target, **batch_fields}
            if label is not None:
                data["label"] = label
            if ids is not None:
                data["ids"] = ids
        batch = _move_to_device(_prepare_batch(data, self.collator), _device_of(self))
        protein = self.protein_embedder.embed(batch["target"])
        molecule = self.molecule_embedder.embed(batch["drug"])
        embeddings = {
            "protein_embedding": protein["embedding"],
            "protein_mask": protein.get("mask"),
            "protein_sequences": protein.get("sequences"),
            "molecule_embedding": molecule["embedding"],
            "molecule_mask": molecule.get("mask"),
            "molecule_smiles": molecule.get("smiles"),
            "molecule_atom_symbols": molecule.get("atom_symbols"),
        }
        if "label" in batch:
            embeddings["labels"] = batch["label"]
        if "ids" in batch:
            embeddings["sample_ids"] = batch["ids"]
        return embeddings

    embed_components = embed

    def forward(self, batch_or_protein_embedding=None, molecule_embedding=None, **inputs):
        if molecule_embedding is not None:
            return self.predictor(
                protein_embedding=batch_or_protein_embedding,
                molecule_embedding=molecule_embedding,
                **inputs,
            )
        if _is_embedding_pair(batch_or_protein_embedding):
            return self.predictor(**batch_or_protein_embedding)
        if _is_embedding_pair(inputs):
            return self.predictor(**inputs)
        embeddings = self.embed(batch_or_protein_embedding, **inputs)
        output = self.predictor(**embeddings)
        output["protein_embedding"] = embeddings["protein_embedding"]
        output["molecule_embedding"] = embeddings["molecule_embedding"]
        return output

    def make_dataloader(self, data, batch_size=None, shuffle=False, num_workers=0):
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, dict):
            data = [data]
        return DataLoader(
            data,
            batch_size=batch_size or int(_training(self.config, "batch_size", 64)),
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
        )

    def predict(self, batch_or_dataset, batch_size=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(batch_or_dataset, dict):
                result = _detach_output(self(batch_or_dataset))
            else:
                result = _merge_outputs(
                    [_detach_output(self(batch)) for batch in self.make_dataloader(batch_or_dataset, batch_size=batch_size)]
                )
        self.train(was_training)
        return result

    def fit(
        self,
        train_data,
        valid_data=None,
        epochs=None,
        learning_rate=None,
        batch_size=None,
        optimizer=None,
    ):
        epochs = int(epochs or _training(self.config, "epochs", 100))
        learning_rate = float(learning_rate or _training(self.config, "learning_rate", 5e-5))
        loader = self.make_dataloader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = []
        self.train()
        for _ in range(epochs):
            total_loss = 0.0
            examples = 0
            for batch in loader:
                if "label" not in batch:
                    raise ValueError("DrugBAN fit() requires a label for every training sample.")
                batch = _move_to_device(batch, _device_of(self))
                optimizer.zero_grad()
                output = self(**batch)
                labels = batch["label"].float().view_as(output["logits"])
                loss = F.binary_cross_entropy_with_logits(output["logits"], labels)
                loss.backward()
                optimizer.step()
                count = labels.numel()
                total_loss += float(loss.detach()) * count
                examples += count
            history.append(total_loss / max(examples, 1))
        result = {"loss": history[-1] if history else None, "history": history, "epochs": epochs}
        if valid_data is not None:
            result["validation"] = self.evaluate(valid_data, batch_size=batch_size)
        return result

    def evaluate(
        self,
        data=None,
        batch_size=None,
        threshold=0.5,
        probabilities=None,
        labels=None,
        **prediction,
    ):
        from kale_protein.core.evaluation.tasks.dti.metrics import compute_metrics

        if probabilities is None:
            if data is None:
                raise ValueError("DrugBAN evaluation requires prediction fields or input data.")
            output = self.predict(data, batch_size=batch_size)
            probabilities = output["probabilities"]
            labels = _labels_from_data(data, self.collator, batch_size)
        elif labels is None:
            labels = prediction.get("label")
        if labels is None:
            raise ValueError("DrugBAN evaluation requires labels.")
        return compute_metrics(labels, probabilities, threshold=threshold)

    def extract_attention(self, batch_or_dataset=None, batch_size=None, **prediction):
        output = prediction
        if "attention" not in output:
            if batch_or_dataset is None:
                raise ValueError("Attention extraction requires prediction fields or input data.")
            output = self.predict(batch_or_dataset, batch_size=batch_size)
        return {
            key: output.get(key)
            for key in (
                "attention", "molecule_mask", "protein_mask", "molecule_atom_symbols",
                "protein_sequences", "molecule_smiles", "sample_ids",
            )
        }

    def save_checkpoint(self, path, optimizer=None, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "kale-drugban-v2",
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path, strict=True, map_location="cpu"):
        state = load_checkpoint_state_dict(path, map_location=map_location)
        return self.load_state_dict(_adapt_checkpoint_keys(state), strict=strict)


def _prepare_batch(data, collator):
    if isinstance(data, dict) and "drug" in data and "target" in data:
        drug = data["drug"]
        target = data["target"]
        if (
            isinstance(drug, dict)
            and "node_features" in drug
            and torch.as_tensor(drug["node_features"]).ndim == 3
            and isinstance(target, dict)
            and "tokens" in target
            and torch.as_tensor(target["tokens"]).ndim == 2
        ):
            return data
    if isinstance(data, dict):
        return collator([data])
    return collator(data)


def _is_embedding_pair(value):
    return (
        isinstance(value, dict)
        and "protein_embedding" in value
        and "molecule_embedding" in value
    )


def _detach_output(output):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in output.items()
    }


def _merge_outputs(outputs):
    if not outputs:
        return {"logits": torch.empty(0), "probabilities": torch.empty(0)}
    merged = {}
    for key in outputs[0]:
        values = [output.get(key) for output in outputs]
        if all(torch.is_tensor(value) for value in values):
            try:
                merged[key] = torch.cat(values, dim=0)
            except RuntimeError:
                merged[key] = values
        elif all(isinstance(value, list) for value in values):
            merged[key] = [item for value in values for item in value]
        else:
            merged[key] = values[0] if len(values) == 1 else values
    return merged


def _labels_from_data(data, collator, batch_size):
    if isinstance(data, dict):
        if "label" in data:
            return data["label"] if torch.is_tensor(data["label"]) else [float(data["label"])]
        raise ValueError("DrugBAN evaluate() requires labels.")
    labels = []
    loader = data if isinstance(data, DataLoader) else DataLoader(
        data, batch_size=batch_size or 64, shuffle=False, collate_fn=collator
    )
    for batch in loader:
        if "label" not in batch:
            raise ValueError("DrugBAN evaluate() requires labels.")
        labels.append(batch["label"])
    return torch.cat(labels) if labels else torch.empty(0)


def _adapt_checkpoint_keys(state):
    """Map the previous monolithic DrugBAN tree into component prefixes."""

    adapted = {}
    prefixes = (
        ("molecule_encoder.", "molecule_embedder."),
        ("protein_encoder.", "protein_embedder."),
        ("ban.", "predictor.ban."),
        ("decoder.", "predictor.decoder."),
        ("drug_extractor.", "molecule_embedder."),
        ("protein_extractor.", "protein_embedder."),
        ("bcn.", "predictor.ban."),
        ("mlp_classifier.", "predictor.decoder."),
    )
    for raw_key, value in state.items():
        key = raw_key.removeprefix("module.").removeprefix("model.")
        key = key.removeprefix("network.")
        if key.startswith(("protein_model.", "molecule_model.")):
            continue
        for old, new in prefixes:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        adapted[key] = value
    return adapted


__all__ = ["DrugBANCollator", "DrugBANModel", "encode_protein"]
