"""Auto-compatible, self-contained MapDiff inverse-folding model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from kale_protein.auto.weights import load_checkpoint_state_dict, load_pretrained_state_dict
from kale_protein.tasks.inverse_folding.collators import CollatorDiff
from kale_protein.tasks.inverse_folding.datasets import DiffusionBatch, GraphBatch, coerce_protein_graph

from kale_protein.examples.mapdiff_inverse_folding.configuration import MapDiffConfig
from kale_protein.examples.mapdiff_inverse_folding.diffusion import MapDiffDiffusion
from kale_protein.examples.mapdiff_inverse_folding.upstream_compat import UpstreamMapDiff


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
        "mask_ratio_deviation": upstream.get("dev_mask_ratio", upstream.get("mask_ratio_deviation", 0.2)),
        **({"marginal": marginal} if marginal is not None else {}),
    }


class MapDiffModel(nn.Module):
    config_class = MapDiffConfig

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.pretrain = pretrain
        model_architecture = config.get("model", {}).get("architecture", "kale-mapdiff-v1")
        pretrained_architecture = config.get("pretrained", {}).get("architecture", model_architecture)
        self.architecture = pretrained_architecture if pretrain else model_architecture
        self.network = self._build_network(self.architecture)
        self.weight_path = None
        if pretrain:
            pretrained = config.get("pretrained", {})
            self.weight_path = (
                Path(config.get("_config_dir", "."))
                / pretrained.get("local_dir", "weights")
                / pretrained.get("filename", "")
            )
            state = load_pretrained_state_dict(config, map_location="cpu")
            self._load_compatible_state_dict(state, self.weight_path)

    def _build_network(self, architecture):
        if architecture == "kale-mapdiff-v1":
            return MapDiffDiffusion(**_model_kwargs(self.config))
        if architecture == "upstream-mapdiff-v1":
            return UpstreamMapDiff(**_upstream_kwargs(self.config))
        raise ValueError(
            f"Unknown MapDiff architecture {architecture!r}; expected "
            "'kale-mapdiff-v1' or 'upstream-mapdiff-v1'."
        )

    def _select_architecture(self, architecture):
        if architecture == self.architecture:
            return
        device = next(self.parameters()).device
        self.network = self._build_network(architecture).to(device)
        self.architecture = architecture

    def component(self, stream_name):
        if stream_name != "structure":
            raise ValueError("MapDiff exposes only the learned 'structure' component.")
        return self

    @staticmethod
    def _as_batch(value):
        if isinstance(value, DiffusionBatch):
            return value
        if isinstance(value, GraphBatch):
            raise ValueError("MapDiff needs the paired IPA view; collate records with CollatorDiff.")
        if isinstance(value, dict) and isinstance(value.get("batch"), DiffusionBatch):
            return value["batch"]
        structure = value.get("structure", value) if isinstance(value, dict) else value
        if isinstance(structure, dict) and "graph" in structure:
            graph = structure["graph"]
        else:
            graph = coerce_protein_graph(structure)
        return CollatorDiff()([graph])

    def embed(self, stream_data):
        batch = self._as_batch(stream_data)
        if self.architecture == "upstream-mapdiff-v1":
            hidden = self.network.embed(batch)
            coordinates = batch.graph.pos
        else:
            graph_t = torch.zeros(batch.graph.num_graphs, device=batch.graph.x.device)
            hidden, coordinates = self.network.denoiser.encode(batch.graph, torch.zeros_like(batch.graph.x), graph_t)
        return {
            "hidden": hidden,
            "coordinates": coordinates,
            "graph": batch.graph,
            "batch": batch,
            "conditioning": hidden,
        }

    def forward(self, batch):
        return self.network(self._as_batch(batch))

    def prior_pretrain_loss(self, ipa_batch):
        return self.network.prior_pretrain_loss(ipa_batch)

    def sample(self, batch, sampling_config=None, **kwargs):
        batch = self._as_batch(batch)
        sampling = dict(self.config.get("sampling", {}))
        sampling.update(sampling_config or {})
        sampling.update(kwargs)
        return self.network.sample(
            batch,
            steps=sampling.get("steps", 50),
            method=sampling.get("method", "ddim"),
            temperature=sampling.get("temperature", 1.0),
            num_samples=sampling.get("num_samples", 1),
        )

    def load_compatible_checkpoint(self, path):
        path = Path(path)
        state = load_checkpoint_state_dict(path, map_location="cpu")
        return self._load_compatible_state_dict(state, path)

    def _load_compatible_state_dict(self, state, path):
        if not state:
            raise RuntimeError(f"MapDiff checkpoint compatibility error for {path}: no state dictionary was found.")
        normalized = {key.removeprefix("module."): value for key, value in state.items()}
        upstream_layout = any(key.startswith("model.mpnn_layes.") for key in normalized)
        if not upstream_layout:
            upstream_layout = any(key.startswith("network.model.mpnn_layes.") for key in normalized)
        target_architecture = "upstream-mapdiff-v1" if upstream_layout else "kale-mapdiff-v1"
        self._select_architecture(target_architecture)
        if target_architecture == "upstream-mapdiff-v1":
            normalized = {key.removeprefix("network."): value for key, value in normalized.items()}
            expected = self.network.state_dict()
        else:
            normalized = {
                key if key.startswith("network.") else f"network.{key}": value
                for key, value in normalized.items()
            }
            expected = self.state_dict()
        missing = sorted(set(expected) - set(normalized))
        unexpected = sorted(set(normalized) - set(expected))
        shape_mismatch = sorted(
            key for key in set(expected) & set(normalized)
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
                f"The selected {target_architecture} parameter tree requires an exact state-key and shape match."
            )
        if target_architecture == "upstream-mapdiff-v1":
            self.network.load_state_dict(normalized, strict=True)
        else:
            self.load_state_dict(normalized, strict=True)
        return self


class MapDiffGenerator(nn.Module):
    config_class = MapDiffConfig

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.model = MapDiffModel(config, pretrain=pretrain)
        self.pretrain = pretrain
        self.weight_path = self.model.weight_path
        self.last_trajectory = None

    def forward(self, structure_embedding, **kwargs):
        return self.generate(structure_embedding, **kwargs)

    def generate(self, structure_embedding, **kwargs):
        output = self.model.sample(structure_embedding, **kwargs)
        self.last_trajectory = output["trajectory"]
        return output
