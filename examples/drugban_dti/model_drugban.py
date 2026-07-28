"""Complete DrugBAN model assembled from reusable KaleProtein components.

The model consumes already-collated tensor mappings. Dataset preprocessing,
collation, loading, and training orchestration live outside this module.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from kaleprotein.auto import AutoProteinEmbedder, AutoProteinPredictor
from kaleprotein.utils import load_checkpoint_state_dict


class DrugBANModel(nn.Module):
    """DrugBAN tensor computation and checkpoint state adapter."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
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

    def embed(self, *, drug, target, label=None, ids=None, **batch_fields):
        """Embed an already-collated DrugBAN tensor mapping."""

        protein = self.protein_embedder.embed(target)
        molecule = self.molecule_embedder.embed(drug)
        embeddings = {
            "protein_embedding": protein["embedding"],
            "protein_mask": protein.get("mask"),
            "protein_sequences": protein.get("sequences"),
            "molecule_embedding": molecule["embedding"],
            "molecule_mask": molecule.get("mask"),
            "molecule_smiles": molecule.get("smiles"),
            "molecule_atom_symbols": molecule.get("atom_symbols"),
            **batch_fields,
        }
        if label is not None:
            embeddings["labels"] = label
        if ids is not None:
            embeddings["sample_ids"] = ids
        return embeddings

    embed_components = embed

    def predict(self, **embeddings):
        return self.predictor(**embeddings)

    def forward(self, protein_embedding=None, molecule_embedding=None, **inputs):
        if protein_embedding is not None and molecule_embedding is not None:
            return self.predict(
                protein_embedding=protein_embedding,
                molecule_embedding=molecule_embedding,
                **inputs,
            )
        return self.predict(**self.embed(**inputs))

    def evaluate(self, *, probabilities, labels, threshold=0.5, **prediction):
        from kaleprotein.evaluate import compute_metrics

        return compute_metrics(labels, probabilities, threshold=threshold)

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


__all__ = ["DrugBANModel"]
