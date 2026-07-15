"""Compact residue-level EGNN used by the self-contained MapDiff card."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    timestep = timestep.float().reshape(-1, 1)
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timestep.device).float() / max(half - 1, 1)
    )
    angles = timestep * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class EquivariantGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 9, dropout: float = 0.0):
        super().__init__()
        edge_input = hidden_dim * 2 + edge_dim + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.edge_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.coord_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, coords, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return self.norm(hidden + self.node_mlp(torch.cat([hidden, torch.zeros_like(hidden)], dim=-1))), coords
        src, dst = edge_index
        delta = coords[src] - coords[dst]
        distance_sq = delta.square().sum(dim=-1, keepdim=True)
        message = self.edge_mlp(torch.cat([hidden[src], hidden[dst], edge_attr, distance_sq], dim=-1))
        message = message * self.edge_gate(message)

        aggregate = hidden.new_zeros(hidden.shape)
        aggregate.index_add_(0, dst, message)
        degree = hidden.new_zeros(hidden.shape[0], 1)
        degree.index_add_(0, dst, torch.ones_like(distance_sq))
        aggregate = aggregate / degree.clamp_min(1.0)

        unit_delta = delta / distance_sq.sqrt().clamp_min(1e-6)
        coordinate_message = unit_delta * self.coord_gate(message).tanh() * 0.1
        coordinate_update = coords.new_zeros(coords.shape)
        coordinate_update.index_add_(0, dst, coordinate_message)
        coordinate_update = coordinate_update / degree.clamp_min(1.0)
        coords = coords + coordinate_update
        hidden = self.norm(hidden + self.node_mlp(torch.cat([hidden, aggregate], dim=-1)))
        return hidden, coords


class EGNNSequenceDenoiser(nn.Module):
    """Predict clean residue logits from a noisy sequence and backbone graph."""

    def __init__(self, hidden_dim=128, num_layers=4, edge_dim=9, dropout=0.0, vocab_size=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sequence_projection = nn.Linear(vocab_size, hidden_dim)
        self.geometry_projection = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(
            [EquivariantGraphLayer(hidden_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size))

    @staticmethod
    def _backbone_geometry(atom_pos):
        pairs = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3))
        return torch.stack(
            [torch.linalg.vector_norm(atom_pos[:, first] - atom_pos[:, second], dim=-1) for first, second in pairs],
            dim=-1,
        )

    def encode_condition(self, graph):
        return self.geometry_projection(self._backbone_geometry(graph.atom_pos))

    def encode(self, graph, noisy_x, timesteps, conditioning=None):
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]
        node_time = timesteps[graph.batch]
        hidden = self.sequence_projection(noisy_x)
        hidden = hidden + (
            self.encode_condition(graph) if conditioning is None else conditioning
        )
        hidden = hidden + self.time_projection(sinusoidal_timestep_embedding(node_time, self.hidden_dim))
        coords = graph.pos
        for layer in self.layers:
            hidden, coords = layer(hidden, coords, graph.edge_index, graph.edge_attr)
        return hidden, coords

    def forward(self, graph, noisy_x, timesteps, conditioning=None):
        hidden, _ = self.encode(graph, noisy_x, timesteps, conditioning=conditioning)
        return self.output(hidden)
