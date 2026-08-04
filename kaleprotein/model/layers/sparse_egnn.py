"""Dependency-light sparse equivariant graph layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphLayerNorm(nn.Module):
    """Normalize all features within each graph in a sparse batch."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, values, batch=None):
        if batch is None:
            centered = values - values.mean()
            output = centered / centered.square().mean().add(self.eps).sqrt()
        else:
            output = values.new_empty(values.shape)
            for graph_index in batch.unique():
                selected = batch == graph_index
                graph_values = values[selected]
                centered = graph_values - graph_values.mean()
                output[selected] = (
                    centered
                    / centered.square().mean().add(self.eps).sqrt()
                )
        return output * self.weight + self.bias


class CoordinateNorm(nn.Module):
    """Normalize relative coordinates with a learned scalar magnitude."""

    def __init__(self, eps=1e-8, scale_init=1e-2):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.full((1,), scale_init))

    def forward(self, coordinates):
        return F.normalize(coordinates, dim=-1, eps=self.eps) * self.scale


class SparseEGNNLayer(nn.Module):
    """Sparse EGNN message passing with node, edge, coordinate, and graph updates."""

    def __init__(
        self,
        feature_dim=128,
        edge_dim=128,
        message_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.feats_dim = feature_dim
        self.edge_input_dim = edge_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + feature_dim * 2, edge_dim * 2 + feature_dim * 4),
            self.dropout,
            nn.SiLU(),
            nn.Linear(edge_dim * 2 + feature_dim * 4, edge_dim),
            nn.SiLU(),
        )
        message_input_dim = edge_dim + 1 + feature_dim * 2
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, message_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(message_input_dim * 2, message_dim),
            nn.SiLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.node_norm = GraphLayerNorm(feature_dim)
        self.edge_norm = GraphLayerNorm(edge_dim)
        self.coors_norm = CoordinateNorm(scale_init=1e-2)
        self.node_mlp = nn.Sequential(
            nn.Linear(feature_dim + message_dim, feature_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.coors_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim * 4),
            self.dropout,
            nn.SiLU(),
            nn.Linear(message_dim * 4, 1),
        )

    def forward(self, values, edge_index, edge_features, batch):
        coordinates, features = values[:, :3], values[:, 3:]
        if edge_index.shape[1] == 0:
            return values, edge_features
        source, target = edge_index
        relative = coordinates[source] - coordinates[target]
        distance = relative.square().sum(dim=-1, keepdim=True)
        messages = self.message_mlp(
            torch.cat(
                [
                    features[target],
                    features[source],
                    edge_features,
                    distance,
                ],
                dim=-1,
            )
        )
        coordinate_update = scatter_sum(
            self.coors_mlp(messages) * self.coors_norm(relative),
            target,
            values.shape[0],
        )
        coordinates = coordinates + coordinate_update
        aggregate = scatter_sum(messages, target, values.shape[0])
        normalized = self.node_norm(features, batch)
        hidden = features + self.node_mlp(
            torch.cat([normalized, aggregate], dim=-1)
        )

        hidden_edge = torch.cat(
            [hidden[source], edge_features, hidden[target]], dim=-1
        )
        edge_batch = batch[source]
        edge_features = self.edge_norm(
            self.dropout(self.edge_mlp(hidden_edge)) + edge_features,
            edge_batch,
        )
        graph_count = int(batch.max().item()) + 1
        graph_mean = scatter_mean(hidden, batch, graph_count)[batch]
        hidden = hidden * self.global_mlp(
            torch.cat([hidden, graph_mean], dim=-1)
        )
        return torch.cat([coordinates, hidden], dim=-1), edge_features


def scatter_sum(values, index, size):
    output = values.new_zeros((size,) + values.shape[1:])
    output.index_add_(0, index, values)
    return output


def scatter_mean(values, index, size):
    output = scatter_sum(values, index, size)
    counts = values.new_zeros(size)
    counts.index_add_(
        0,
        index,
        values.new_ones(index.shape[0]),
    )
    return output / counts.clamp_min(1).reshape(
        (-1,) + (1,) * (values.ndim - 1)
    )


__all__ = [
    "CoordinateNorm",
    "GraphLayerNorm",
    "SparseEGNNLayer",
    "scatter_mean",
    "scatter_sum",
]
