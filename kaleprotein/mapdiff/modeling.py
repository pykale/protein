"""Small self-contained MapDiff-style inverse-folding model.

This is intentionally compact but keeps the same user-facing pipeline: structure
features in, residue logits out, optional pretrained weights from config.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from kaleprotein.hub import load_state_dict_if_available, resolve_weight_file


class MapDiffStructurePreprocessor:
    """Structure preprocessor used by AutoProteinPreprocessor("protein/structure")."""

    def featurize(self, data: dict[str, Any] | list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        structures = data["structure"] if isinstance(data, dict) else data
        max_len = max(item["x"].size(0) for item in structures)
        x = torch.zeros(len(structures), max_len, structures[0]["x"].size(-1))
        pos = torch.zeros(len(structures), max_len, 3)
        mask = torch.zeros(len(structures), max_len, dtype=torch.bool)
        labels = torch.zeros(len(structures), max_len, dtype=torch.long) + 20
        for index, item in enumerate(structures):
            length = item["x"].size(0)
            x[index, :length] = item["x"]
            if "pos" in item:
                pos[index, :length] = item["pos"]
            mask[index, :length] = True
            labels[index, :length] = item["x"].argmax(dim=-1)
        return {"x": x, "pos": pos, "mask": mask, "labels": labels}


class MapDiffForInverseFolding(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        model = config["model"]
        hidden = int(model.get("hidden_dim", 128))
        self.input = nn.Linear(int(model.get("input_dim", 23)), hidden)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(int(model.get("depth", 4))))
        self.output = nn.Linear(hidden, int(model.get("num_residue_types", 20)))

    @classmethod
    def from_config(cls, config: dict[str, Any], *, pretrain: bool = False, downloader=None, strict: bool = False) -> "MapDiffForInverseFolding":
        model = cls(config)
        weight_file = resolve_weight_file(config, pretrain=pretrain, downloader=downloader) if downloader else resolve_weight_file(config, pretrain=pretrain)
        load_state_dict_if_available(model, weight_file, strict=strict)
        return model

    def embed(self, structure_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = structure_data["x"]
        pos = structure_data.get("pos")
        if pos is not None:
            features = torch.cat([x, pos], dim=-1)
        else:
            pad = torch.zeros(*x.shape[:-1], 3, device=x.device, dtype=x.dtype)
            features = torch.cat([x, pad], dim=-1)
        hidden = self.input(features)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return {
            "hidden_states": hidden,
            "mask": structure_data.get("mask"),
            "labels": structure_data.get("labels"),
        }

    def generate(
        self,
        structure_embedding: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        logits = self.output(structure_embedding["hidden_states"]) / max(temperature, 1e-6)
        if num_samples <= 1:
            samples = logits.argmax(dim=-1).unsqueeze(0)
        else:
            probs = torch.softmax(logits, dim=-1)
            flat = probs.reshape(-1, probs.size(-1))
            draws = [torch.multinomial(flat, num_samples=1).view(probs.shape[:-1]) for _ in range(num_samples)]
            samples = torch.stack(draws)
        return {"logits": logits, "sequences": samples}

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        structure_embedding = self.embed({"x": x, "pos": pos, "mask": mask, "labels": labels})
        logits = self.output(structure_embedding["hidden_states"])
        output = {"logits": logits, "predictions": logits.argmax(dim=-1)}
        if labels is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=20)
            output["loss"] = loss
        return output


class MapDiffGenerator(nn.Module):
    def __init__(self, model: MapDiffForInverseFolding):
        super().__init__()
        self.model = model

    @classmethod
    def from_config(cls, config: dict[str, Any], *, pretrain: bool = False, downloader=None, strict: bool = False) -> "MapDiffGenerator":
        return cls(MapDiffForInverseFolding.from_config(config, pretrain=pretrain, downloader=downloader, strict=strict))

    def generate(
        self,
        structure_embedding: dict[str, torch.Tensor],
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        return self.model.generate(structure_embedding, num_samples=num_samples, temperature=temperature)
