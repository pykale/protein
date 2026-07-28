"""Reusable invariant point attention and transition layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Linear):
    """Linear layer kept explicit for OpenFold-compatible parameter trees."""


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.channels = (channels,)
        self.eps = eps

    def forward(self, values):
        return F.layer_norm(
            values, self.channels, self.weight, self.bias, self.eps
        )


@dataclass
class RigidFrames:
    rotation: torch.Tensor
    translation: torch.Tensor

    @classmethod
    def from_three_points(
        cls, negative_x, origin, xy_plane, eps=1e-8
    ):
        first_axis = F.normalize(origin - negative_x, dim=-1, eps=eps)
        second_axis = xy_plane - origin
        second_axis = second_axis - first_axis * (
            first_axis * second_axis
        ).sum(dim=-1, keepdim=True)
        second_axis = F.normalize(second_axis, dim=-1, eps=eps)
        third_axis = torch.cross(first_axis, second_axis, dim=-1)
        return cls(
            torch.stack([first_axis, second_axis, third_axis], dim=-1),
            origin.float(),
        )

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        return RigidFrames(
            self.rotation[index + (slice(None), slice(None))],
            self.translation[index + (slice(None),)],
        )

    def apply(self, points):
        return (
            torch.matmul(self.rotation, points.unsqueeze(-1)).squeeze(-1)
            + self.translation
        )

    def invert_apply(self, points):
        return torch.matmul(
            self.rotation.transpose(-1, -2),
            (points - self.translation).unsqueeze(-1),
        ).squeeze(-1)


class InvariantPointAttention(nn.Module):
    """Scalar, pair, and point attention invariant to rigid transformations."""

    def __init__(
        self,
        c_s=128,
        c_z=128,
        c_hidden=32,
        no_heads=4,
        no_qk_points=4,
        no_v_points=8,
    ):
        super().__init__()
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = 1e5
        self.eps = 1e-8
        hidden_heads = c_hidden * no_heads
        self.linear_q = Linear(c_s, hidden_heads)
        self.linear_kv = Linear(c_s, hidden_heads * 2)
        self.linear_q_points = Linear(
            c_s, no_heads * no_qk_points * 3
        )
        self.linear_kv_points = Linear(
            c_s, no_heads * (no_qk_points + no_v_points) * 3
        )
        self.linear_b = Linear(c_z, no_heads)
        self.head_weights = nn.Parameter(
            torch.full((no_heads,), 0.5413248546)
        )
        output_dim = no_heads * (
            c_z + c_hidden + no_v_points * 4
        )
        self.linear_out = Linear(output_dim, c_s)

    def forward(self, single, pair, frames, mask, attn_drop_rate=0.0):
        query = self.linear_q(single).view(
            *single.shape[:-1], self.no_heads, self.c_hidden
        )
        key_value = self.linear_kv(single).view(
            *single.shape[:-1], self.no_heads, self.c_hidden * 2
        )
        key, value = torch.split(key_value, self.c_hidden, dim=-1)

        query_points = self.linear_q_points(single)
        query_points = torch.stack(
            torch.split(
                query_points, query_points.shape[-1] // 3, dim=-1
            ),
            dim=-1,
        )
        query_points = frames[..., None].apply(query_points)
        query_points = query_points.view(
            *query_points.shape[:-2],
            self.no_heads,
            self.no_qk_points,
            3,
        )

        key_value_points = self.linear_kv_points(single)
        key_value_points = torch.stack(
            torch.split(
                key_value_points,
                key_value_points.shape[-1] // 3,
                dim=-1,
            ),
            dim=-1,
        )
        key_value_points = frames[..., None].apply(key_value_points)
        key_value_points = key_value_points.view(
            *key_value_points.shape[:-2], self.no_heads, -1, 3
        )
        key_points, value_points = torch.split(
            key_value_points,
            [self.no_qk_points, self.no_v_points],
            dim=-2,
        )

        pair_bias = self.linear_b(pair)
        attention = torch.matmul(
            _permute_final_dims(query, (1, 0, 2)),
            _permute_final_dims(key, (1, 2, 0)),
        ) * math.sqrt(1.0 / (3 * self.c_hidden))
        attention = attention + math.sqrt(1.0 / 3) * _permute_final_dims(
            pair_bias, (2, 0, 1)
        )
        point_distance = (
            query_points.unsqueeze(-4) - key_points.unsqueeze(-5)
        ).square().sum(dim=-1)
        point_weight = F.softplus(self.head_weights).view(
            1, self.no_heads, 1
        )
        point_weight = point_weight * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        point_score = (
            point_distance * point_weight
        ).sum(dim=-1) * -0.5
        attention = attention + _permute_final_dims(
            point_score, (2, 0, 1)
        )
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        attention = torch.softmax(
            attention + self.inf * (square_mask[:, None] - 1), dim=-1
        )

        scalar_output = torch.matmul(
            attention, value.transpose(-2, -3)
        ).transpose(-2, -3)
        scalar_output = _flatten_final_dims(scalar_output, 2)
        point_output = torch.sum(
            attention[..., None, :, :, None]
            * _permute_final_dims(
                value_points, (1, 3, 0, 2)
            )[..., None, :, :],
            dim=-2,
        )
        point_output = _permute_final_dims(
            point_output, (2, 0, 3, 1)
        )
        point_output = frames[..., None, None].invert_apply(point_output)
        point_norm = _flatten_final_dims(
            torch.sqrt(
                point_output.square().sum(dim=-1) + self.eps
            ),
            2,
        )
        point_output = point_output.reshape(
            *point_output.shape[:-3], -1, 3
        )
        pair_output = torch.matmul(
            attention.transpose(-2, -3), pair
        )
        pair_output = _flatten_final_dims(pair_output, 2)
        return self.linear_out(
            torch.cat(
                (
                    scalar_output,
                    *torch.unbind(point_output, dim=-1),
                    point_norm,
                    pair_output,
                ),
                dim=-1,
            )
        )


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear_1 = Linear(channels, channels)
        self.linear_2 = Linear(channels, channels)
        self.linear_3 = Linear(channels, channels)

    def forward(self, single):
        residual = single
        single = F.relu(self.linear_1(single))
        single = F.relu(self.linear_2(single))
        return self.linear_3(single) + residual


class StructureModuleTransition(nn.Module):
    def __init__(self, channels, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StructureModuleTransitionLayer(channels)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(channels)

    def forward(self, single):
        for layer in self.layers:
            single = layer(single)
        return self.layer_norm(self.dropout(single))


class EdgeTransition(nn.Module):
    def __init__(
        self,
        node_embed_size=128,
        edge_embed_in=128,
        edge_embed_out=128,
        num_layers=2,
    ):
        super().__init__()
        bias_size = node_embed_size // 2
        self.initial_embed = Linear(node_embed_size, bias_size)
        hidden_size = bias_size * 2 + edge_embed_in
        trunk = []
        for _ in range(num_layers):
            trunk.extend([Linear(hidden_size, hidden_size), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk)
        self.final_layer = Linear(hidden_size, edge_embed_out)
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, single, pair):
        single = self.initial_embed(single)
        batch, length = single.shape[:2]
        bias = torch.cat(
            [
                single[:, :, None].expand(-1, -1, length, -1),
                single[:, None].expand(-1, length, -1, -1),
            ],
            dim=-1,
        )
        pair = torch.cat([pair, bias], dim=-1).reshape(
            batch * length * length, -1
        )
        pair = self.final_layer(self.trunk(pair) + pair)
        return self.layer_norm(pair).reshape(
            batch, length, length, -1
        )


def _permute_final_dims(tensor, indices):
    start = -len(indices)
    leading = list(range(len(tensor.shape[:start])))
    return tensor.permute(
        leading + [start + index for index in indices]
    )


def _flatten_final_dims(tensor, dimensions):
    return tensor.reshape(
        tensor.shape[:-dimensions] + (-1,)
    )


__all__ = [
    "EdgeTransition",
    "InvariantPointAttention",
    "LayerNorm",
    "Linear",
    "RigidFrames",
    "StructureModuleTransition",
]
