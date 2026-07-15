"""Complete MapDiff model assembled through KaleProtein Auto components."""

from __future__ import annotations

import json
import weakref
from pathlib import Path

import torch
from torch import nn

from kale_protein.auto import AutoProteinEmbedder, AutoProteinPredictor
from kale_protein.core.tasks.inverse_folding.collators import CollatorDiff
from kale_protein.core.tasks.inverse_folding.datasets import (
    DiffusionBatch,
    GraphBatch,
    coerce_protein_graph,
)
from kale_protein.core.weights import load_checkpoint_state_dict, resolve_pretrained_weight

from .configuration import MapDiffConfig
from .diffusion import MapDiffDiffusion
from .upstream_compat import UpstreamMapDiff


def _model_kwargs(config):
    model = dict(config.get("model", {}))
    diffusion = dict(config.get("diffusion", {}))
    return {
        "hidden_dim": model.get("hidden_dim", 128),
        "egnn_layers": model.get("egnn_layers", 4),
        "ipa_layers": model.get("ipa_layers", 3),
        "ipa_heads": model.get("ipa_heads", 4),
        "ipa_head_dim": model.get("ipa_head_dim", 24),
        "ipa_points": model.get("ipa_points", 4),
        "dropout": model.get("dropout", 0.0),
        "timesteps": diffusion.get("timesteps", config.get("sampling", {}).get("steps", 100)),
        "marginal": diffusion.get("marginal"),
        "prior_loss_weight": diffusion.get("prior_loss_weight", 1.0),
        "min_mask_ratio": diffusion.get("min_mask_ratio", 0.35),
        "mask_ratio_deviation": diffusion.get("mask_ratio_deviation", 0.25),
    }


def _upstream_kwargs(config):
    upstream = dict(config.get("upstream", {}))
    marginal = None
    marginal_map = upstream.get("marginal_map")
    if marginal_map:
        path = Path(config.get("_config_dir", ".")) / marginal_map
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read MapDiff marginal map {path}: {exc}") from exc
        marginal = payload.get("probabilities")
        if payload.get("alphabet") != "ARNDCQEGHILKMFPSTWYV" or not isinstance(marginal, list):
            raise ValueError(f"Invalid MapDiff marginal map schema in {path}.")
    return {
        "hidden_dim": upstream.get("hidden_dim", 128),
        "egnn_depth": upstream.get("depth", upstream.get("egnn_depth", 6)),
        "ipa_depth": upstream.get("ipa_depth", 6),
        "ipa_heads": upstream.get("ipa_heads", 4),
        "qk_points": upstream.get("qk_points", 4),
        "v_points": upstream.get("v_points", 8),
        "egnn_dropout": upstream.get("drop_out", upstream.get("egnn_dropout", 0.1)),
        "ipa_dropout": upstream.get("ipa_drop_out", upstream.get("ipa_dropout", 0.2)),
        "timesteps": upstream.get("timesteps", 500),
        "min_mask_ratio": upstream.get("min_mask_ratio", 0.4),
        "mask_ratio_deviation": upstream.get(
            "dev_mask_ratio", upstream.get("mask_ratio_deviation", 0.2)
        ),
        **({"marginal": marginal} if marginal is not None else {}),
    }


@AutoProteinPredictor.register("inverse_folding/mapdiff_generator")
class MapDiffGenerator(nn.Module):
    """MapDiff denoising generator; checkpoint loading remains model-owned."""

    def __init__(self, config, architecture=None, **kwargs):
        super().__init__()
        self.config = config
        self.architecture = architecture or config.get("model", {}).get(
            "architecture", "kale-mapdiff-v1"
        )
        self.network = self._build_network(self.architecture)
        self.last_trajectory = None

    def _build_network(self, architecture):
        if architecture == "kale-mapdiff-v1":
            return MapDiffDiffusion(**_model_kwargs(self.config))
        if architecture == "upstream-mapdiff-v1":
            return UpstreamMapDiff(**_upstream_kwargs(self.config))
        raise ValueError(
            f"Unknown MapDiff architecture {architecture!r}; expected "
            "'kale-mapdiff-v1' or 'upstream-mapdiff-v1'."
        )

    def select_architecture(self, architecture):
        if architecture == self.architecture:
            return
        device = next(self.parameters()).device
        self.network = self._build_network(architecture).to(device)
        self.architecture = architecture

    @staticmethod
    def as_batch(value):
        if isinstance(value, DiffusionBatch):
            return value
        if isinstance(value, GraphBatch):
            raise ValueError("MapDiff needs the paired IPA view; collate records with CollatorDiff.")
        if isinstance(value, dict) and isinstance(value.get("batch"), DiffusionBatch):
            return value["batch"]
        structure = value.get("structure", value) if isinstance(value, dict) else value
        graph = structure["graph"] if isinstance(structure, dict) and "graph" in structure else coerce_protein_graph(structure)
        return CollatorDiff()([graph])

    def embed(self, value):
        batch = self.as_batch(value)
        if self.architecture == "upstream-mapdiff-v1":
            conditioning = self.network.feature_adapter(batch)
            hidden = self.network.embed(batch, prepared=conditioning)
            coordinates = batch.graph.pos
        else:
            conditioning = self.network.denoiser.encode_condition(batch.graph)
            graph_t = torch.zeros(batch.graph.num_graphs, device=batch.graph.x.device)
            hidden, coordinates = self.network.denoiser.encode(
                batch.graph,
                torch.zeros_like(batch.graph.x),
                graph_t,
                conditioning=conditioning,
            )
        return {
            "hidden": hidden,
            "conditioning": conditioning,
            "coordinates": coordinates,
            "graph": batch.graph,
            "batch": batch,
        }

    def forward(self, conditioning):
        batch = self.as_batch(conditioning)
        encoded = conditioning.get("conditioning") if isinstance(conditioning, dict) else None
        if self.architecture == "upstream-mapdiff-v1":
            return self.network(batch, prepared=encoded)
        return self.network(batch, conditioning=encoded)

    def generate(self, conditioning, sampling_config=None, **kwargs):
        sampling = dict(self.config.get("sampling", {}))
        sampling.update(sampling_config or {})
        sampling.update(kwargs)
        batch = self.as_batch(conditioning)
        encoded = conditioning.get("conditioning") if isinstance(conditioning, dict) else None
        arguments = {
            "steps": sampling.get("steps", 50),
            "method": sampling.get("method", "ddim"),
            "temperature": sampling.get("temperature", 1.0),
            "num_samples": sampling.get("num_samples", 1),
        }
        if self.architecture == "upstream-mapdiff-v1":
            output = self.network.sample(batch, prepared=encoded, **arguments)
        else:
            output = self.network.sample(batch, conditioning=encoded, **arguments)
        self.last_trajectory = output["trajectory"]
        return output

    def prior_pretrain_loss(self, ipa_batch):
        return self.network.prior_pretrain_loss(ipa_batch)


@AutoProteinEmbedder.register("structure/mapdiff_condition")
class MapDiffConditionEmbedder(nn.Module):
    """Non-owning structure-condition encoder view over a MapDiff generator."""

    def __init__(self, config, predictor, **kwargs):
        super().__init__()
        self.config = config
        self.__dict__["_predictor_ref"] = weakref.ref(predictor)

    @property
    def predictor(self):
        predictor = self._predictor_ref()
        if predictor is None:
            raise RuntimeError("The MapDiff generator backing this embedder no longer exists.")
        return predictor

    def embed(self, value):
        return self.predictor.embed(value)

    forward = embed


class MapDiffModel(nn.Module):
    """Full MapDiff composition root and sole checkpoint owner."""

    config_class = MapDiffConfig

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.pretrain = pretrain
        model_architecture = config.get("model", {}).get("architecture", "kale-mapdiff-v1")
        pretrained_architecture = config.get("pretrained", {}).get(
            "architecture", model_architecture
        )
        architecture = pretrained_architecture if pretrain else model_architecture
        self.predictor = AutoProteinPredictor.from_config(
            config.get_predictor(), config=config, architecture=architecture
        )
        self.embedder = AutoProteinEmbedder.from_config(
            config.get_embedders()["structure"], config=config, predictor=self.predictor
        )
        self.weight_path = None
        if pretrain:
            self.weight_path = resolve_pretrained_weight(config)
            self.load_compatible_checkpoint(self.weight_path)

    @property
    def architecture(self):
        return self.predictor.architecture

    @property
    def network(self):
        """Compatibility view; the registered predictor owns this network."""

        return self.predictor.network

    def embed(self, value):
        return self.embedder.embed(value)

    def forward(self, value):
        return self.predictor(self.embed(value))

    def generate(self, value, sampling_config=None, **kwargs):
        conditioning = value if _is_conditioning(value) else self.embed(value)
        return self.predictor.generate(conditioning, sampling_config=sampling_config, **kwargs)

    sample = generate

    def prior_pretrain_loss(self, ipa_batch):
        return self.predictor.prior_pretrain_loss(ipa_batch)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format": "kale-mapdiff-v2", "state_dict": self.state_dict()}, path
        )
        return path

    def load_compatible_checkpoint(self, path):
        path = Path(path)
        state = load_checkpoint_state_dict(path, map_location="cpu")
        return self._load_compatible_state_dict(state, path)

    def _load_compatible_state_dict(self, state, path):
        if not state:
            raise RuntimeError(
                f"MapDiff checkpoint compatibility error for {path}: no state dictionary was found."
            )
        normalized = _network_state_dict(state)
        upstream_layout = any(key.startswith("model.mpnn_layes.") for key in normalized)
        architecture = "upstream-mapdiff-v1" if upstream_layout else "kale-mapdiff-v1"
        self.predictor.select_architecture(architecture)
        expected = self.network.state_dict()
        missing = sorted(set(expected) - set(normalized))
        unexpected = sorted(set(normalized) - set(expected))
        shape_mismatch = sorted(
            key
            for key in set(expected) & set(normalized)
            if getattr(expected[key], "shape", None) != getattr(normalized[key], "shape", None)
        )
        if missing or unexpected or shape_mismatch:
            details = []
            if missing:
                details.append(f"missing keys {missing[:5]}")
            if unexpected:
                details.append(f"unexpected keys {unexpected[:5]}")
            if shape_mismatch:
                details.append(f"shape mismatches {shape_mismatch[:5]}")
            raise RuntimeError(
                f"MapDiff checkpoint compatibility error for {path}: {'; '.join(details)}. "
                f"The selected {architecture} parameter tree requires an exact state-key and shape match."
            )
        self.network.load_state_dict(normalized, strict=True)
        return self


def _network_state_dict(state):
    normalized = {}
    for raw_key, value in state.items():
        key = raw_key.removeprefix("module.")
        for prefix in ("predictor.network.", "network."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value
    return normalized


def _is_conditioning(value):
    return isinstance(value, dict) and "conditioning" in value and "batch" in value


__all__ = [
    "MapDiffConditionEmbedder",
    "MapDiffGenerator",
    "MapDiffModel",
]
