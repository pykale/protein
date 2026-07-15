"""MapDiff v1.0.1 release architecture and plain-PyTorch runtime adapter.

Adapted from MapDiff, copyright 2024 Peizhen Bai, under the MIT License.
The invariant-point-attention design follows the Apache-2.0 OpenFold-derived
implementation vendored by MapDiff. See README.md for complete attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from kale_protein.core.data.tasks.inverse_folding.datasets import AA_ALPHABET, DiffusionBatch, IPABatch


UPSTREAM_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
UPSTREAM_MARGINAL = (
    0.0834947526, 0.0519946255, 0.0415460430, 0.0584252477, 0.0132485991,
    0.0369899236, 0.0691716149, 0.0725991055, 0.0238895807, 0.0592938587,
    0.0962855667, 0.0578796901, 0.0165696796, 0.0400275066, 0.0450712629,
    0.0606069826, 0.0535279624, 0.0130431838, 0.0341296867, 0.0722051188,
)
_UPSTREAM_FROM_STANDARD = [AA_ALPHABET.index(aa) for aa in UPSTREAM_ALPHABET]
_STANDARD_FROM_UPSTREAM = [UPSTREAM_ALPHABET.index(aa) for aa in AA_ALPHABET]


def to_upstream_order(values):
    return values[..., _UPSTREAM_FROM_STANDARD]


def to_standard_order(values):
    return values[..., _STANDARD_FROM_UPSTREAM]


class GraphLayerNorm(nn.Module):
    """Parameter-compatible replacement for PyG LayerNorm(mode='graph')."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x, batch=None):
        if batch is None:
            centered = x - x.mean()
            output = centered / centered.square().mean().add(self.eps).sqrt()
        else:
            output = torch.empty_like(x)
            for graph_index in torch.unique(batch):
                selected = batch == graph_index
                graph_x = x[selected]
                centered = graph_x - graph_x.mean()
                output[selected] = centered / centered.square().mean().add(self.eps).sqrt()
        return output * self.weight + self.bias


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1e-2):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.full((1,), scale_init))

    def forward(self, coordinates):
        return F.normalize(coordinates, dim=-1, eps=self.eps) * self.scale


def _scatter_sum(values, index, size):
    output = values.new_zeros((size,) + values.shape[1:])
    output.index_add_(0, index, values)
    return output


def _scatter_mean(values, index, size):
    output = _scatter_sum(values, index, size)
    counts = values.new_zeros(size)
    counts.index_add_(0, index, torch.ones(index.shape[0], device=values.device, dtype=values.dtype))
    return output / counts.clamp_min(1).reshape((-1,) + (1,) * (values.ndim - 1))


class UpstreamEGNNLayer(nn.Module):
    """State-key-compatible refactor of MapDiff's EGNN_Sparse layer."""

    def __init__(self, feats_dim=128, edge_attr_dim=128, m_dim=128, dropout=0.1):
        super().__init__()
        self.feats_dim = feats_dim
        self.edge_input_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim + feats_dim * 2, edge_attr_dim * 2 + feats_dim * 4),
            self.dropout,
            nn.SiLU(),
            nn.Linear(edge_attr_dim * 2 + feats_dim * 4, edge_attr_dim),
            nn.SiLU(),
        )
        message_input_dim = edge_attr_dim + 1 + feats_dim * 2
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, message_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(message_input_dim * 2, m_dim),
            nn.SiLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * feats_dim, 2 * feats_dim),
            nn.ReLU(),
            nn.Linear(2 * feats_dim, 2 * feats_dim),
            nn.ReLU(),
            nn.Linear(2 * feats_dim, feats_dim),
            nn.Sigmoid(),
        )
        self.node_norm = GraphLayerNorm(feats_dim)
        self.edge_norm = GraphLayerNorm(edge_attr_dim)
        self.coors_norm = CoorsNorm(scale_init=1e-2)
        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            self.dropout,
            nn.SiLU(),
            nn.Linear(m_dim * 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        coordinates, features = x[:, :3], x[:, 3:]
        if edge_index.shape[1] == 0:
            return x, edge_attr
        src, dst = edge_index
        relative = coordinates[src] - coordinates[dst]
        distance = relative.square().sum(dim=-1, keepdim=True)
        messages = self.message_mlp(torch.cat([features[dst], features[src], edge_attr, distance], dim=-1))
        coordinate_update = _scatter_sum(self.coors_mlp(messages) * self.coors_norm(relative), dst, x.shape[0])
        coordinates = coordinates + coordinate_update
        aggregate = _scatter_sum(messages, dst, x.shape[0])
        normalized = self.node_norm(features, batch)
        hidden = features + self.node_mlp(torch.cat([normalized, aggregate], dim=-1))

        hidden_edge = torch.cat([hidden[src], edge_attr, hidden[dst]], dim=-1)
        edge_batch = batch[src]
        edge_attr = self.edge_norm(self.dropout(self.edge_mlp(hidden_edge)) + edge_attr, edge_batch)
        graph_mean = _scatter_mean(hidden, batch, int(batch.max().item()) + 1)[batch]
        hidden = hidden * self.global_mlp(torch.cat([hidden, graph_mean], dim=-1))
        return torch.cat([coordinates, hidden], dim=-1), edge_attr


class NodeEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.node_feature_dim = [20, 1, 1, 4, 5]
        self.atom_embedding_list = nn.ModuleList([nn.Linear(dim, emb_dim) for dim in self.node_feature_dim])

    def forward(self, x):
        output = 0
        offset = 0
        for size, layer in zip(self.node_feature_dim, self.atom_embedding_list):
            output = output + layer(x[:, offset:offset + size])
            offset += size
        return output


class EdgeEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.edge_feature_dims = [65, 1, 15, 12]
        self.atom_embedding_list = nn.ModuleList([nn.Linear(dim, emb_dim) for dim in self.edge_feature_dims])

    def forward(self, x):
        output = 0
        offset = 0
        for size, layer in zip(self.edge_feature_dims, self.atom_embedding_list):
            output = output + layer(x[:, offset:offset + size])
            offset += size
        return output


class UpstreamEGNN(nn.Module):
    def __init__(self, hidden_channels=128, n_layers=6, dropout=0.1, embedding_dim=128):
        super().__init__()
        self.dropout = dropout
        self.update_edge = True
        self.embed_ss = -3
        self.n_layers = n_layers
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, embedding_dim))
        self.ss_mlp = nn.Sequential(nn.Linear(8, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, embedding_dim))
        for _ in range(n_layers):
            self.mpnn_layes.append(UpstreamEGNNLayer(embedding_dim, embedding_dim, hidden_channels, dropout))
            self.time_mlp_list.append(nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, embedding_dim * 2)))
            self.ff_list.append(
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Dropout(dropout),
                    nn.SiLU(),
                    GraphLayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                )
            )
        self.node_embedding = NodeEncoder(embedding_dim)
        self.edge_embedding = EdgeEncoder(embedding_dim)
        self.lin = nn.Linear(embedding_dim, 20)

    def forward(self, data, time):
        time_embedding = self.time_mlp(time)
        secondary = self.ss_mlp(data.ss)
        features = self.node_embedding(torch.cat([data.x, data.extra_x], dim=1))
        edge_attr = self.edge_embedding(data.edge_attr)
        features = features + secondary
        x = torch.cat([data.pos, features], dim=1)
        for index, layer in enumerate(self.mpnn_layes):
            x, edge_attr = layer(x, data.edge_index, edge_attr, data.batch)
            coordinates, features = x[:, :3], x[:, 3:]
            scale, shift = self.time_mlp_list[index](time_embedding).chunk(2, dim=1)
            features = features * (scale[data.batch] + 1) + shift[data.batch]
            features = self.ff_list[index](features)
            x = torch.cat([coordinates, features], dim=-1)
        return self.lin(F.dropout(x[:, 3:], p=self.dropout, training=self.training))


class Linear(nn.Linear):
    """OpenFold-compatible Linear; initialization does not affect loaded weights."""


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.channels = (channels,)
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.channels, self.weight, self.bias, self.eps)


def _permute_final_dims(tensor, indices):
    start = -len(indices)
    leading = list(range(len(tensor.shape[:start])))
    return tensor.permute(leading + [start + index for index in indices])


def _flatten_final_dims(tensor, dimensions):
    return tensor.reshape(tensor.shape[:-dimensions] + (-1,))


@dataclass
class RigidFrames:
    rotation: torch.Tensor
    translation: torch.Tensor

    @classmethod
    def from_three_points(cls, negative_x, origin, xy_plane, eps=1e-8):
        e0 = origin - negative_x
        e1 = xy_plane - origin
        e0 = F.normalize(e0, dim=-1, eps=eps)
        e1 = e1 - e0 * (e0 * e1).sum(dim=-1, keepdim=True)
        e1 = F.normalize(e1, dim=-1, eps=eps)
        e2 = torch.cross(e0, e1, dim=-1)
        return cls(torch.stack([e0, e1, e2], dim=-1), origin.float())

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        return RigidFrames(
            self.rotation[index + (slice(None), slice(None))],
            self.translation[index + (slice(None),)],
        )

    def apply(self, points):
        return torch.matmul(self.rotation, points.unsqueeze(-1)).squeeze(-1) + self.translation

    def invert_apply(self, points):
        return torch.matmul(self.rotation.transpose(-1, -2), (points - self.translation).unsqueeze(-1)).squeeze(-1)


class InvariantPointAttention(nn.Module):
    def __init__(self, c_s=128, c_z=128, c_hidden=32, no_heads=4, no_qk_points=4, no_v_points=8):
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
        self.linear_q_points = Linear(c_s, no_heads * no_qk_points * 3)
        self.linear_kv_points = Linear(c_s, no_heads * (no_qk_points + no_v_points) * 3)
        self.linear_b = Linear(c_z, no_heads)
        self.head_weights = nn.Parameter(torch.full((no_heads,), 0.5413248546))
        output_dim = no_heads * (c_z + c_hidden + no_v_points * 4)
        self.linear_out = Linear(output_dim, c_s)

    def forward(self, single, pair, frames, mask, attn_drop_rate=0.0):
        q = self.linear_q(single).view(*single.shape[:-1], self.no_heads, self.c_hidden)
        kv = self.linear_kv(single).view(*single.shape[:-1], self.no_heads, self.c_hidden * 2)
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        q_points = self.linear_q_points(single)
        q_points = torch.stack(torch.split(q_points, q_points.shape[-1] // 3, dim=-1), dim=-1)
        q_points = frames[..., None].apply(q_points)
        q_points = q_points.view(*q_points.shape[:-2], self.no_heads, self.no_qk_points, 3)
        kv_points = self.linear_kv_points(single)
        kv_points = torch.stack(torch.split(kv_points, kv_points.shape[-1] // 3, dim=-1), dim=-1)
        kv_points = frames[..., None].apply(kv_points)
        kv_points = kv_points.view(*kv_points.shape[:-2], self.no_heads, -1, 3)
        k_points, v_points = torch.split(kv_points, [self.no_qk_points, self.no_v_points], dim=-2)

        pair_bias = self.linear_b(pair)
        attention = torch.matmul(
            _permute_final_dims(q, (1, 0, 2)),
            _permute_final_dims(k, (1, 2, 0)),
        ) * math.sqrt(1.0 / (3 * self.c_hidden))
        attention = attention + math.sqrt(1.0 / 3) * _permute_final_dims(pair_bias, (2, 0, 1))
        point_distance = (q_points.unsqueeze(-4) - k_points.unsqueeze(-5)).square().sum(dim=-1)
        point_weight = F.softplus(self.head_weights).view(1, self.no_heads, 1)
        point_weight = point_weight * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        point_score = (point_distance * point_weight).sum(dim=-1) * -0.5
        attention = attention + _permute_final_dims(point_score, (2, 0, 1))
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        attention = torch.softmax(attention + self.inf * (square_mask[:, None] - 1), dim=-1)

        scalar_output = torch.matmul(attention, v.transpose(-2, -3)).transpose(-2, -3)
        scalar_output = _flatten_final_dims(scalar_output, 2)
        point_output = torch.sum(
            attention[..., None, :, :, None]
            * _permute_final_dims(v_points, (1, 3, 0, 2))[..., None, :, :],
            dim=-2,
        )
        point_output = _permute_final_dims(point_output, (2, 0, 3, 1))
        point_output = frames[..., None, None].invert_apply(point_output)
        point_norm = _flatten_final_dims(torch.sqrt(point_output.square().sum(dim=-1) + self.eps), 2)
        point_output = point_output.reshape(*point_output.shape[:-3], -1, 3)
        pair_output = torch.matmul(attention.transpose(-2, -3), pair)
        pair_output = _flatten_final_dims(pair_output, 2)
        return self.linear_out(
            torch.cat((scalar_output, *torch.unbind(point_output, dim=-1), point_norm, pair_output), dim=-1)
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
        self.layers = nn.ModuleList([StructureModuleTransitionLayer(channels) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(channels)

    def forward(self, single):
        for layer in self.layers:
            single = layer(single)
        return self.layer_norm(self.dropout(single))


class EdgeTransition(nn.Module):
    def __init__(self, node_embed_size=128, edge_embed_in=128, edge_embed_out=128, num_layers=2):
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
            [single[:, :, None].expand(-1, -1, length, -1), single[:, None].expand(-1, length, -1, -1)],
            dim=-1,
        )
        pair = torch.cat([pair, bias], dim=-1).reshape(batch * length * length, -1)
        pair = self.final_layer(self.trunk(pair) + pair)
        return self.layer_norm(pair).reshape(batch, length, length, -1)


class NodeMaskEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.aa_in_proj = nn.Linear(20, emb_dim)
        self.d_in_proj = nn.Linear(6, emb_dim)
        self.mask_emb = nn.Embedding(1, emb_dim)
        self.pad_emb = nn.Embedding(1, emb_dim, padding_idx=0)
        self.pe_encoding = nn.Parameter(self._positional_encoding(1200), requires_grad=False)

    def _positional_encoding(self, maximum):
        encoding = torch.zeros(maximum, self.emb_dim)
        position = torch.arange(maximum, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, self.emb_dim, 2).float() * (-math.log(10000.0) / self.emb_dim))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        return encoding.unsqueeze(0)

    def forward(self, amino_acids, atom_pos, node_mask, sequence_mask):
        single = self.aa_in_proj(amino_acids)
        single = torch.where(node_mask[..., None] == 1, self.mask_emb.weight.reshape(1, 1, -1), single)
        single = torch.where(sequence_mask[..., None] == 0, self.pad_emb.weight.reshape(1, 1, -1), single)
        return single + self.d_in_proj(_cal_dihedrals(atom_pos).float()) + self.pe_encoding[:, :single.shape[1]]


class EdgePairEncoder(nn.Module):
    def __init__(self, emb_dim=128, dist_bins=24, dist_bin_width=0.5, rel_pos_k=32):
        super().__init__()
        self.dist_bins = dist_bins
        self.dist_bin_width = dist_bin_width
        self.rel_pos_k = rel_pos_k
        self.rbf_in_proj = nn.Linear(dist_bins * 16, emb_dim)
        self.relpos_in_proj = nn.Linear(rel_pos_k * 2 + 1, emb_dim)

    def forward(self, atom_pos):
        batch, length = atom_pos.shape[:2]
        backbone = atom_pos[:, :, :4].reshape(batch, length * 4, 3)
        distance = torch.cdist(backbone, backbone)
        centers = torch.linspace(0, self.dist_bins * self.dist_bin_width, self.dist_bins, device=atom_pos.device)
        rbf = torch.exp(-((distance[..., None] - centers) / self.dist_bin_width) ** 2)
        rbf = rbf.reshape(batch, length, 4, length, 4, self.dist_bins)
        rbf = rbf.permute(0, 1, 3, 2, 4, 5).reshape(batch, length, length, -1)
        relative = torch.arange(length, device=atom_pos.device)[:, None] - torch.arange(length, device=atom_pos.device)[None]
        relative = (relative.clamp(-self.rel_pos_k, self.rel_pos_k) + self.rel_pos_k).long()
        relative = F.one_hot(relative, self.rel_pos_k * 2 + 1).float().unsqueeze(0).expand(batch, -1, -1, -1)
        return self.rbf_in_proj(rbf) + self.relpos_in_proj(relative)


class IPANetModel(nn.Module):
    def __init__(self, ipa_dim=128, ipa_pairwise_dim=128, ipa_heads=4, ipa_depth=6, qk_points=4, v_points=8, dropout=0.2):
        super().__init__()
        self.ipa_layers = nn.ModuleList()
        for index in range(ipa_depth):
            edge_transition = None if index == ipa_depth - 1 else EdgeTransition(ipa_dim, ipa_pairwise_dim, ipa_pairwise_dim, 2)
            self.ipa_layers.append(
                nn.ModuleList(
                    [
                        InvariantPointAttention(ipa_dim, ipa_pairwise_dim, ipa_dim // ipa_heads, ipa_heads, qk_points, v_points),
                        nn.Dropout(dropout),
                        LayerNorm(ipa_dim),
                        nn.Linear(ipa_dim, ipa_dim * 4),
                        StructureModuleTransition(ipa_dim * 4, 1, 0.1),
                        nn.Linear(ipa_dim * 4, ipa_dim),
                        edge_transition,
                    ]
                )
            )

    def forward(self, single, pair, frames, sequence_mask):
        for ipa, dropout, norm, pre, transition, post, edge_transition in self.ipa_layers:
            single = norm(dropout(single + ipa(single, pair, frames, sequence_mask)))
            single = post(transition(pre(single)))
            if edge_transition is not None:
                pair = edge_transition(single, pair)
        return single


class UpstreamIPAPredictor(nn.Module):
    def __init__(self, dropout=0.2, hidden_dim=128, ipa_depth=6, ipa_heads=4, qk_points=4, v_points=8):
        super().__init__()
        self.node_encoder = NodeMaskEncoder(hidden_dim)
        self.edge_pair_encoder = EdgePairEncoder(hidden_dim)
        self.s_dropout = nn.Dropout(dropout)
        self.z_dropout = nn.Dropout(dropout)
        self.node_predictor = nn.Linear(hidden_dim, 20)
        self.ipa = IPANetModel(hidden_dim, hidden_dim, ipa_heads, ipa_depth, qk_points, v_points, dropout)

    def forward(self, amino_acids, atom_pos, amino_acid_mask, sequence_mask):
        frames = RigidFrames.from_three_points(atom_pos[:, :, 0], atom_pos[:, :, 1], atom_pos[:, :, 2])
        single = self.s_dropout(self.node_encoder(amino_acids, atom_pos, amino_acid_mask, sequence_mask))
        pair = self.z_dropout(self.edge_pair_encoder(atom_pos))
        single = self.ipa(single, pair, frames, sequence_mask.long())
        return self.node_predictor(single)


class UpstreamNoiseSchedule(nn.Module):
    def __init__(self, timesteps=500):
        super().__init__()
        steps = torch.linspace(0, timesteps + 2, timesteps + 2, dtype=torch.float64)
        alpha_bar = torch.cos(0.5 * math.pi * ((steps / (timesteps + 2) + 0.008) / 1.008)).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        self.register_buffer("betas", betas.float())

    @property
    def alphas_bar(self):
        return torch.cumprod(1 - self.betas.clamp(0, 0.9999), dim=0)


def _cal_dihedrals(atom_pos, eps=1e-7):
    coordinates = atom_pos[:, :, :3].reshape(atom_pos.shape[0], -1, 3)
    unit = F.normalize(coordinates[:, 1:] - coordinates[:, :-1], dim=-1)
    u2, u1, u0 = unit[:, :-2], unit[:, 1:-1], unit[:, 2:]
    n2 = F.normalize(torch.cross(u2, u1, dim=-1), dim=-1)
    n1 = F.normalize(torch.cross(u1, u0, dim=-1), dim=-1)
    cosine = (n2 * n1).sum(-1).clamp(-1 + eps, 1 - eps)
    angles = torch.sign((u2 * n1).sum(-1)) * torch.acos(cosine)
    angles = F.pad(angles, (1, 2)).reshape(atom_pos.shape[0], atom_pos.shape[1], 3)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def _virtual_cb(atom_pos):
    c, n, ca = atom_pos[:, 2], atom_pos[:, 0], atom_pos[:, 1]
    bc = F.normalize(n - ca, dim=-1)
    normal = F.normalize(torch.cross(n - c, bc, dim=-1), dim=-1)
    middle = torch.cross(normal, bc, dim=-1)
    length, planar, dihedral = 1.522, 1.927, -2.143
    return ca + (
        bc * (length * math.cos(planar))
        + middle * (length * math.sin(planar) * math.cos(dihedral))
        + normal * (-length * math.sin(planar) * math.sin(dihedral))
    )


def _dihedral(a, b, c, d):
    b0 = a - b
    b1 = F.normalize(c - b, dim=-1)
    b2 = d - c
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


class UpstreamFeatureAdapter:
    """Construct the 31-node/93-edge and five-atom release tensor views."""

    def __call__(self, batch: DiffusionBatch):
        graph = batch.graph
        device = graph.x.device
        all_edges, all_edge_features, all_extra, all_secondary = [], [], [], []
        padded_atoms = graph.atom_pos.new_zeros((graph.num_graphs, int((graph.ptr[1:] - graph.ptr[:-1]).max()), 5, 3))
        padded_mask = torch.zeros(padded_atoms.shape[:2], dtype=torch.bool, device=device)
        offset = 0
        for graph_index in range(graph.num_graphs):
            start, end = graph.ptr[graph_index:graph_index + 2].tolist()
            atoms4 = graph.atom_pos[start:end]
            length = atoms4.shape[0]
            atoms5 = torch.stack([atoms4[:, 0], atoms4[:, 1], atoms4[:, 2], _virtual_cb(atoms4), atoms4[:, 3]], dim=1)
            padded_atoms[graph_index, :length] = atoms5
            padded_mask[graph_index, :length] = True
            ca, n_coord, c_coord = atoms4[:, 1], atoms4[:, 0], atoms4[:, 2]
            distance_matrix = torch.cdist(ca, ca)
            src_list, dst_list = [], []
            for residue in range(length):
                candidates = torch.where((distance_matrix[residue] < 30) & (torch.arange(length, device=device) != residue))[0]
                if candidates.numel() == 0 and length > 1:
                    candidates = distance_matrix[residue].topk(2, largest=False).indices[1:]
                if candidates.numel() > 10:
                    candidates = candidates[distance_matrix[residue, candidates].argsort()[:10]]
                src_list.extend([residue] * candidates.numel())
                dst_list.extend(candidates.tolist())
            if src_list:
                src = torch.tensor(src_list, device=device, dtype=torch.long)
                dst = torch.tensor(dst_list, device=device, dtype=torch.long)
                edge_distance = torch.linalg.vector_norm(ca[src] - ca[dst], dim=-1)
                sequence_distance = (src - dst).abs().clamp_max(64)
                sequence_feature = F.one_hot(sequence_distance, 65).float()
                scales = torch.tensor([1.5 ** value for value in range(15)], device=device)
                distance_feature = torch.exp(-((edge_distance[:, None] / 4) ** 2) / scales)
                contact = (edge_distance <= 8).float()[:, None]
                u = F.normalize(n_coord - ca, dim=-1)
                tangent = F.normalize(c_coord - ca, dim=-1)
                normal = F.normalize(torch.cross(u, tangent, dim=-1), dim=-1)
                v = torch.cross(normal, u, dim=-1)
                basis = torch.stack([normal[dst], u[dst], v[dst]], dim=1)
                p = torch.bmm(basis, (ca[src] - ca[dst]).unsqueeze(-1)).squeeze(-1)
                q = torch.bmm(basis, normal[src].unsqueeze(-1)).squeeze(-1)
                k = torch.bmm(basis, u[src].unsqueeze(-1)).squeeze(-1)
                t = torch.bmm(basis, v[src].unsqueeze(-1)).squeeze(-1)
                # Preserve the release preprocessing order, including its
                # historical distance/contact ordering quirk.
                edge_features = torch.cat([sequence_feature, distance_feature, contact, p, q, k, t], dim=-1)
                all_edges.append(torch.stack([src + offset, dst + offset]))
                all_edge_features.append(edge_features)
                mu = atoms4.new_zeros(length, 5)
                for residue in range(length):
                    selected = src == residue
                    if selected.any():
                        distances = edge_distance[selected]
                        differences = ca[src[selected]] - ca[dst[selected]]
                        for scale_index, sigma in enumerate((1.0, 2.0, 5.0, 10.0, 30.0)):
                            weights = torch.softmax(-(distances.square()) / sigma, dim=0)
                            mean = (weights[:, None] * differences).sum(0)
                            denominator = (weights * distances).sum().clamp_min(1e-8)
                            mu[residue, scale_index] = mean.norm() / denominator
            else:
                mu = atoms4.new_zeros(length, 5)

            angles = atoms4.new_zeros(length, 4)
            if length > 1:
                phi = _dihedral(c_coord[:-1], n_coord[:-1], ca[:-1], n_coord[1:])
                psi = _dihedral(n_coord[:-1], ca[:-1], c_coord[:-1], n_coord[1:])
                angles[:-1, 0], angles[:-1, 1] = torch.sin(phi), torch.cos(phi)
                angles[:-1, 2], angles[:-1, 3] = torch.sin(psi), torch.cos(psi)
            all_extra.append(torch.cat([atoms4.new_zeros(length, 2), angles, mu], dim=-1))
            all_secondary.append(atoms4.new_zeros(length, 8))
            offset += length

        edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.cat(all_edge_features) if all_edge_features else graph.x.new_zeros((0, 93))
        graph_view = SimpleNamespace(
            x=None,
            pos=graph.pos,
            extra_x=torch.cat(all_extra),
            edge_index=edge_index,
            edge_attr=edge_attr,
            ss=torch.cat(all_secondary),
            batch=graph.batch,
        )
        return graph_view, padded_atoms, padded_mask


class UpstreamMapDiff(nn.Module):
    """Exact v1.0.1 parameter tree plus release-compatible denoising runtime."""

    def __init__(
        self,
        hidden_dim=128,
        egnn_depth=6,
        ipa_depth=6,
        ipa_heads=4,
        qk_points=4,
        v_points=8,
        egnn_dropout=0.1,
        ipa_dropout=0.2,
        timesteps=500,
        min_mask_ratio=0.4,
        mask_ratio_deviation=0.2,
        marginal=UPSTREAM_MARGINAL,
    ):
        super().__init__()
        self.model = UpstreamEGNN(hidden_dim, egnn_depth, egnn_dropout, hidden_dim)
        self.prior_model = UpstreamIPAPredictor(ipa_dropout, hidden_dim, ipa_depth, ipa_heads, qk_points, v_points)
        self.noise_schedule = UpstreamNoiseSchedule(timesteps)
        self.timesteps = timesteps
        self.min_mask_ratio = min_mask_ratio
        self.mask_ratio_deviation = mask_ratio_deviation
        self.register_buffer("marginal", torch.tensor(marginal, dtype=torch.float32), persistent=False)
        self.feature_adapter = UpstreamFeatureAdapter()

    def _matrix(self, alpha):
        eye = torch.eye(20, device=alpha.device, dtype=alpha.dtype)
        stationary = self.marginal.to(alpha).expand(20, -1)
        return alpha[..., None, None] * eye + (1 - alpha[..., None, None]) * stationary

    def _q_sample(self, x0, node_t):
        alpha = self.noise_schedule.alphas_bar[node_t]
        probability = torch.bmm(x0[:, None], self._matrix(alpha)).squeeze(1)
        tokens = torch.multinomial(probability.clamp_min(1e-8), 1).squeeze(-1)
        return F.one_hot(tokens, 20).float()

    def _predict(self, batch, prepared, noisy, graph_t):
        graph_view, atom_pos, sequence_mask = prepared
        graph_view.x = noisy
        base_logits = self.model(graph_view, graph_t.float().reshape(-1, 1))
        base_probability = torch.softmax(base_logits, dim=-1)
        entropy = -(base_probability * torch.log(base_probability.clamp_min(1e-8))).mean(dim=-1)
        padded_x = atom_pos.new_zeros((*atom_pos.shape[:2], 20))
        padded_node_mask = torch.zeros(atom_pos.shape[:2], dtype=torch.long, device=atom_pos.device)
        base_tokens = base_logits.argmax(dim=-1)
        for graph_index in range(batch.graph.num_graphs):
            start, end = batch.graph.ptr[graph_index:graph_index + 2].tolist()
            length = end - start
            padded_x[graph_index, :length] = F.one_hot(base_tokens[start:end], 20).float()
            noise = 1 - self.noise_schedule.alphas_bar[graph_t[graph_index]]
            ratio = self.min_mask_ratio + self.mask_ratio_deviation * torch.sin(noise * math.pi / 2)
            count = min(length, max(1, math.ceil(length * float(ratio))))
            selected = entropy[start:end].topk(count).indices
            padded_node_mask[graph_index, selected] = 1
        prior_padded = self.prior_model(padded_x, atom_pos, padded_node_mask, sequence_mask)
        prior_logits = prior_padded[sequence_mask]
        prior_probability = torch.softmax(prior_logits, dim=-1)
        prior_entropy = -(prior_probability * torch.log(prior_probability.clamp_min(1e-8))).mean(dim=-1)
        weights = torch.softmax(-torch.stack([entropy, prior_entropy], dim=0), dim=0)
        fused = weights[0, :, None] * base_logits + weights[1, :, None] * prior_logits
        return fused, base_logits, prior_logits, padded_node_mask

    def forward(self, batch: DiffusionBatch, prepared=None):
        prepared = self.feature_adapter(batch) if prepared is None else prepared
        target_x = to_upstream_order(batch.graph.x)
        graph_t = torch.randint(0, self.timesteps + 1, (batch.graph.num_graphs,), device=target_x.device)
        noisy = self._q_sample(target_x, graph_t[batch.graph.batch])
        fused, base, prior, node_mask = self._predict(batch, prepared, noisy, graph_t)
        labels = target_x.argmax(dim=-1)
        selected = node_mask[prepared[2]] == 1
        base_loss = F.cross_entropy(base, labels)
        prior_loss = F.cross_entropy(prior[selected], labels[selected])
        return {
            "loss": base_loss + prior_loss,
            "base_loss": base_loss,
            "prior_loss": prior_loss,
            "logits": to_standard_order(fused),
            "noisy_x": to_standard_order(noisy),
            "timesteps": graph_t,
        }

    def prior_pretrain_loss(self, ipa: IPABatch):
        atom4 = ipa.atom_pos
        atom5 = torch.stack([atom4[:, :, 0], atom4[:, :, 1], atom4[:, :, 2], _virtual_cb(atom4.reshape(-1, 4, 3)).reshape(*atom4.shape[:2], 3), atom4[:, :, 3]], dim=2)
        logits = self.prior_model(to_upstream_order(ipa.x), atom5, ipa.x_mask, ipa.x_pad)
        labels_standard = ipa.label.clamp_max(19)
        mapping = torch.tensor([UPSTREAM_ALPHABET.index(AA_ALPHABET[index]) for index in range(20)], device=ipa.label.device)
        labels = mapping[labels_standard]
        selected = (ipa.x_mask > 0) & ipa.x_pad
        return {"loss": F.cross_entropy(logits[selected], labels[selected]), "logits": to_standard_order(logits)}

    def embed(self, batch, prepared=None):
        prepared = self.feature_adapter(batch) if prepared is None else prepared
        graph_view = prepared[0]
        graph_view.x = to_upstream_order(batch.graph.x.new_zeros(batch.graph.x.shape))
        timestep = torch.zeros(batch.graph.num_graphs, 1, device=batch.graph.x.device)
        return to_standard_order(self.model(graph_view, timestep))

    @staticmethod
    def _decode(tokens, graph):
        output = []
        for index in range(graph.num_graphs):
            start, end = graph.ptr[index:index + 2].tolist()
            output.append("".join(UPSTREAM_ALPHABET[token] for token in tokens[start:end].tolist()))
        return output

    def _reverse_probability(self, z_t, predicted_x0, node_t, node_s, method):
        alphas = self.noise_schedule.alphas_bar
        alpha_t, alpha_s = alphas[node_t], alphas[node_s]
        q_s, q_t_bar = self._matrix(alpha_s), self._matrix(alpha_t)
        if method == "ddim":
            q_step = q_s / q_t_bar.clamp_min(1e-8)
            q_step = q_step / q_step.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            q_step = self._matrix(1 - self.noise_schedule.betas[node_t])
        left = torch.bmm(z_t[:, None], q_step.transpose(-1, -2))
        numerator = left * q_s
        denominator = torch.bmm(q_t_bar, z_t[:, :, None]).clamp_min(1e-8)
        posterior_over_x0 = numerator / denominator
        probability = (predicted_x0[:, :, None] * posterior_over_x0).sum(dim=1)
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def sample(
        self,
        batch,
        steps=100,
        method="ddim",
        temperature=1.0,
        num_samples=1,
        prepared=None,
    ):
        if method not in {"ddim", "ddpm"}:
            raise ValueError("sampling method must be 'ddim' or 'ddpm'.")
        prepared = self.feature_adapter(batch) if prepared is None else prepared
        schedule = torch.linspace(self.timesteps, 0, min(steps, self.timesteps) + 1, device=batch.graph.x.device)
        schedule = torch.unique_consecutive(schedule.round().long())
        all_sequences, all_trajectories = [], []
        final_logits = None
        for _ in range(num_samples):
            tokens = torch.multinomial(self.marginal.expand(batch.graph.num_nodes, -1), 1).squeeze(-1)
            noisy = F.one_hot(tokens, 20).float()
            trajectory = [{"timestep": int(schedule[0]), "sequences": self._decode(tokens, batch.graph)}]
            for current, following in zip(schedule[:-1], schedule[1:]):
                graph_t = torch.full((batch.graph.num_graphs,), int(current), dtype=torch.long, device=noisy.device)
                fused, _, _, _ = self._predict(batch, prepared, noisy, graph_t)
                predicted = torch.softmax(fused / max(float(temperature), 1e-4), dim=-1)
                if int(following) == 0:
                    tokens = predicted.argmax(dim=-1)
                else:
                    node_t = graph_t[batch.graph.batch]
                    node_s = torch.full_like(node_t, int(following))
                    reverse = self._reverse_probability(noisy, predicted, node_t, node_s, method)
                    tokens = torch.multinomial(reverse.clamp_min(1e-8), 1).squeeze(-1)
                noisy = F.one_hot(tokens, 20).float()
                trajectory.append({"timestep": int(following), "sequences": self._decode(tokens, batch.graph)})
                final_logits = to_standard_order(fused)
            all_sequences.extend(trajectory[-1]["sequences"])
            all_trajectories.append(trajectory)
        standard_tokens = torch.tensor([AA_ALPHABET.index(UPSTREAM_ALPHABET[token]) for token in tokens.tolist()], device=tokens.device)
        return {
            "sequences": all_sequences,
            "token_ids": standard_tokens,
            "logits": final_logits,
            "trajectory": all_trajectories[0],
            "trajectories": all_trajectories,
            "sampling_method": method,
            "checkpoint_architecture": "upstream-mapdiff-v1",
        }


__all__ = ["UpstreamMapDiff", "UPSTREAM_ALPHABET", "UPSTREAM_MARGINAL"]
