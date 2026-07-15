"""Invariant point-attention masking prior for MapDiff."""

from __future__ import annotations

import math

import torch
from torch import nn


def backbone_frames(atom_pos: torch.Tensor):
    """Return CA translations and orthonormal local frames from N/CA/C."""

    n, ca, c = atom_pos[:, :, 0], atom_pos[:, :, 1], atom_pos[:, :, 2]
    e1 = torch.nn.functional.normalize(c - ca, dim=-1, eps=1e-6)
    n_direction = torch.nn.functional.normalize(n - ca, dim=-1, eps=1e-6)
    e3 = torch.nn.functional.normalize(torch.cross(e1, n_direction, dim=-1), dim=-1, eps=1e-6)
    e2 = torch.cross(e3, e1, dim=-1)
    return ca, torch.stack([e1, e2, e3], dim=-2)


class InvariantPointAttention(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, head_dim=24, points=4):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.points = points
        scalar_size = heads * head_dim
        point_size = heads * points * 3
        self.to_q = nn.Linear(hidden_dim, scalar_size, bias=False)
        self.to_k = nn.Linear(hidden_dim, scalar_size, bias=False)
        self.to_v = nn.Linear(hidden_dim, scalar_size, bias=False)
        self.to_q_points = nn.Linear(hidden_dim, point_size)
        self.to_k_points = nn.Linear(hidden_dim, point_size)
        self.to_v_points = nn.Linear(hidden_dim, point_size)
        self.point_weights = nn.Parameter(torch.ones(heads))
        self.output = nn.Linear(scalar_size + point_size, hidden_dim)

    def _global_points(self, projection, hidden, translation, frame):
        batch, length = hidden.shape[:2]
        local = projection(hidden).reshape(batch, length, self.heads, self.points, 3)
        global_points = torch.einsum("blhpc,blcd->blhpd", local, frame)
        return global_points + translation[:, :, None, None]

    def forward(self, hidden, atom_pos, mask):
        batch, length = hidden.shape[:2]
        translation, frame = backbone_frames(atom_pos)
        q = self.to_q(hidden).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(hidden).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(hidden).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        q_points = self._global_points(self.to_q_points, hidden, translation, frame).permute(0, 2, 1, 3, 4)
        k_points = self._global_points(self.to_k_points, hidden, translation, frame).permute(0, 2, 1, 3, 4)
        v_points = self._global_points(self.to_v_points, hidden, translation, frame).permute(0, 2, 1, 3, 4)

        scalar_score = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        point_distance = (q_points[:, :, :, None] - k_points[:, :, None]).square().sum(dim=(-1, -2))
        point_weight = torch.nn.functional.softplus(self.point_weights)[None, :, None, None]
        scores = scalar_score - 0.5 * point_weight * point_distance
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        scores = scores.masked_fill(~pair_mask, -torch.finfo(scores.dtype).max)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * pair_mask

        scalar_output = torch.einsum("bhij,bhjd->bhid", attention, v)
        point_output = torch.einsum("bhij,bhjpc->bhipc", attention, v_points)
        point_output = point_output - translation[:, None, :, None]
        point_local = torch.einsum("bhipd,bidc->bhipc", point_output, frame.transpose(-1, -2))
        scalar_output = scalar_output.permute(0, 2, 1, 3).reshape(batch, length, -1)
        point_local = point_local.permute(0, 2, 1, 3, 4).reshape(batch, length, -1)
        return self.output(torch.cat([scalar_output, point_local], dim=-1)) * mask[:, :, None]


class IPABlock(nn.Module):
    def __init__(self, hidden_dim, heads, head_dim, points, dropout):
        super().__init__()
        self.attention = InvariantPointAttention(hidden_dim, heads, head_dim, points)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.transition_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, atom_pos, mask):
        hidden = self.attention_norm(hidden + self.attention(hidden, atom_pos, mask))
        hidden = self.transition_norm(hidden + self.transition(hidden))
        return hidden * mask[:, :, None]


class IPAMaskPrior(nn.Module):
    """Structure-aware masked residue predictor used to refine diffusion logits."""

    def __init__(self, hidden_dim=128, num_layers=3, heads=4, head_dim=24, points=4, dropout=0.0, vocab_size=20):
        super().__init__()
        self.residue_embedding = nn.Linear(vocab_size, hidden_dim)
        self.mask_embedding = nn.Embedding(4, hidden_dim)
        self.blocks = nn.ModuleList(
            [IPABlock(hidden_dim, heads, head_dim, points, dropout) for _ in range(num_layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size))

    def forward(self, x, atom_pos, x_mask, x_pad):
        hidden = self.residue_embedding(x) + self.mask_embedding(x_mask.clamp(0, 3))
        hidden = hidden * x_pad[:, :, None]
        for block in self.blocks:
            hidden = block(hidden, atom_pos, x_pad)
        return self.output(hidden)
