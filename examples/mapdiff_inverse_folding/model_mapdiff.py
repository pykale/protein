"""Complete MapDiff v1.0.1 model and KaleProtein Auto composition.

Adapted from MapDiff, copyright 2024 Peizhen Bai, under the MIT License.
The invariant-point-attention design follows the Apache-2.0 OpenFold-derived
implementation vendored by MapDiff. See README.md for complete attribution.
"""

from __future__ import annotations

import math
import json
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from kaleprotein.auto import AutoProteinEmbedder, AutoProteinPredictor
from kaleprotein.loaddata.records import AMINO_ACID_ALPHABET
from kaleprotein.model.layers import (
    EdgeTransition,
    GraphLayerNorm,
    InvariantPointAttention,
    RigidFrames,
    SparseEGNNLayer,
    StructureModuleTransition,
)
from kaleprotein.model.layers.invariant_point_attention import LayerNorm
from kaleprotein.utils import load_checkpoint_state_dict

from .configuration import MapDiffConfig


AA_ALPHABET = AMINO_ACID_ALPHABET
MAPDIFF_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
MAPDIFF_MARGINAL = (
    0.0834947526, 0.0519946255, 0.0415460430, 0.0584252477, 0.0132485991,
    0.0369899236, 0.0691716149, 0.0725991055, 0.0238895807, 0.0592938587,
    0.0962855667, 0.0578796901, 0.0165696796, 0.0400275066, 0.0450712629,
    0.0606069826, 0.0535279624, 0.0130431838, 0.0341296867, 0.0722051188,
)
_MAPDIFF_FROM_STANDARD = [
    AA_ALPHABET.index(amino_acid) for amino_acid in MAPDIFF_ALPHABET
]
_STANDARD_FROM_MAPDIFF = [
    MAPDIFF_ALPHABET.index(amino_acid) for amino_acid in AA_ALPHABET
]


def to_mapdiff_order(values):
    return values[..., _MAPDIFF_FROM_STANDARD]


def to_standard_order(values):
    return values[..., _STANDARD_FROM_MAPDIFF]


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

    def embed_sequence(self, residue_one_hot):
        return self.atom_embedding_list[0](residue_one_hot)

    def embed_condition(self, extra_residue_features):
        output = 0
        offset = 0
        for size, layer in zip(
            self.node_feature_dim[1:], self.atom_embedding_list[1:]
        ):
            output = output + layer(
                extra_residue_features[:, offset : offset + size]
            )
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


class MapDiffEGNN(nn.Module):
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
            self.mpnn_layes.append(
                SparseEGNNLayer(
                    embedding_dim,
                    embedding_dim,
                    hidden_channels,
                    dropout,
                )
            )
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

    def encode_condition(
        self,
        extra_residue_features,
        edge_features,
        secondary_structure,
    ):
        return {
            "structure_embedding": (
                self.node_embedding.embed_condition(extra_residue_features)
                + self.ss_mlp(secondary_structure)
            ),
            "edge_embedding": self.edge_embedding(edge_features),
        }

    def forward(
        self,
        noisy_residue_one_hot,
        ca_coordinates,
        edge_index,
        graph_index,
        graph_time,
        structure_embedding,
        edge_embedding,
    ):
        time_embedding = self.time_mlp(graph_time)
        features = (
            self.node_embedding.embed_sequence(noisy_residue_one_hot)
            + structure_embedding
        )
        x = torch.cat([ca_coordinates, features], dim=1)
        edge_embedding = edge_embedding
        for index, layer in enumerate(self.mpnn_layes):
            x, edge_embedding = layer(
                x, edge_index, edge_embedding, graph_index
            )
            coordinates, features = x[:, :3], x[:, 3:]
            scale, shift = self.time_mlp_list[index](time_embedding).chunk(2, dim=1)
            features = (
                features * (scale[graph_index] + 1)
                + shift[graph_index]
            )
            features = self.ff_list[index](features)
            x = torch.cat([coordinates, features], dim=-1)
        return self.lin(F.dropout(x[:, 3:], p=self.dropout, training=self.training))


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


class MapDiffIPAPrior(nn.Module):
    def __init__(self, dropout=0.2, hidden_dim=128, ipa_depth=6, ipa_heads=4, qk_points=4, v_points=8):
        super().__init__()
        self.node_encoder = NodeMaskEncoder(hidden_dim)
        self.edge_pair_encoder = EdgePairEncoder(hidden_dim)
        self.s_dropout = nn.Dropout(dropout)
        self.z_dropout = nn.Dropout(dropout)
        self.node_predictor = nn.Linear(hidden_dim, 20)
        self.ipa = IPANetModel(hidden_dim, hidden_dim, ipa_heads, ipa_depth, qk_points, v_points, dropout)

    def forward(
        self,
        amino_acids,
        atom_pos,
        amino_acid_mask,
        sequence_mask,
        pair_embedding=None,
    ):
        frames = RigidFrames.from_three_points(atom_pos[:, :, 0], atom_pos[:, :, 1], atom_pos[:, :, 2])
        single = self.s_dropout(self.node_encoder(amino_acids, atom_pos, amino_acid_mask, sequence_mask))
        pair = (
            self.edge_pair_encoder(atom_pos)
            if pair_embedding is None
            else pair_embedding
        )
        pair = self.z_dropout(pair)
        single = self.ipa(single, pair, frames, sequence_mask.long())
        return self.node_predictor(single)


class MapDiffNoiseSchedule(nn.Module):
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


class MapDiffNetwork(nn.Module):
    """Published MapDiff parameter tree and complete training/sampling runtime."""

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
        candidate_rate=0.7,
        ipa_mask_rate=0.8,
        ipa_replace_rate=0.1,
        ipa_keep_rate=0.1,
        marginal=MAPDIFF_MARGINAL,
    ):
        super().__init__()
        self.model = MapDiffEGNN(
            hidden_dim, egnn_depth, egnn_dropout, hidden_dim
        )
        self.prior_model = MapDiffIPAPrior(
            ipa_dropout,
            hidden_dim,
            ipa_depth,
            ipa_heads,
            qk_points,
            v_points,
        )
        self.noise_schedule = MapDiffNoiseSchedule(timesteps)
        self.timesteps = timesteps
        self.min_mask_ratio = min_mask_ratio
        self.mask_ratio_deviation = mask_ratio_deviation
        self.candidate_rate = candidate_rate
        self.ipa_mask_rate = ipa_mask_rate
        self.ipa_replace_rate = ipa_replace_rate
        self.ipa_keep_rate = ipa_keep_rate
        if not 0 < self.candidate_rate <= 1:
            raise ValueError("MapDiff candidate_rate must be in (0, 1].")
        corruption_rates = (
            self.ipa_mask_rate,
            self.ipa_replace_rate,
            self.ipa_keep_rate,
        )
        if any(rate < 0 for rate in corruption_rates) or not math.isclose(
            sum(corruption_rates),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "MapDiff IPA mask, replace, and keep rates must be "
                "non-negative and sum to 1."
            )
        self.register_buffer(
            "marginal",
            torch.tensor(marginal, dtype=torch.float32),
            persistent=False,
        )

    def encode_condition(
        self,
        *,
        extra_residue_features,
        edge_features,
        secondary_structure,
        ipa_atom_positions,
    ):
        condition = self.model.encode_condition(
            extra_residue_features,
            edge_features,
            secondary_structure,
        )
        condition["ipa_pair_embedding"] = (
            self.prior_model.edge_pair_encoder(ipa_atom_positions)
        )
        return condition

    def _matrix(self, alpha):
        eye = torch.eye(20, device=alpha.device, dtype=alpha.dtype)
        stationary = self.marginal.to(alpha).expand(20, -1)
        return alpha[..., None, None] * eye + (1 - alpha[..., None, None]) * stationary

    def _q_sample(self, x0, node_t):
        alpha = self.noise_schedule.alphas_bar[node_t]
        probability = torch.bmm(x0[:, None], self._matrix(alpha)).squeeze(1)
        tokens = torch.multinomial(probability.clamp_min(1e-8), 1).squeeze(-1)
        return F.one_hot(tokens, 20).float()

    def _predict(
        self,
        *,
        noisy_residue_one_hot,
        graph_time,
        ca_coordinates,
        edge_index,
        graph_index,
        graph_ptr,
        ipa_atom_positions,
        ipa_sequence_mask,
        structure_embedding,
        edge_embedding,
        ipa_pair_embedding,
    ):
        base_logits = self.model(
            noisy_residue_one_hot,
            ca_coordinates,
            edge_index,
            graph_index,
            graph_time.float().reshape(-1, 1),
            structure_embedding,
            edge_embedding,
        )
        base_probability = torch.softmax(base_logits, dim=-1)
        entropy = -(
            base_probability
            * torch.log(base_probability.clamp_min(1e-8))
        ).mean(dim=-1)
        padded_x = ipa_atom_positions.new_zeros(
            (*ipa_atom_positions.shape[:2], 20)
        )
        padded_node_mask = torch.zeros(
            ipa_atom_positions.shape[:2],
            dtype=torch.long,
            device=ipa_atom_positions.device,
        )
        base_tokens = base_logits.argmax(dim=-1)
        graph_count = graph_ptr.numel() - 1
        for batch_id in range(graph_count):
            start, end = graph_ptr[batch_id : batch_id + 2].tolist()
            length = end - start
            padded_x[batch_id, :length] = F.one_hot(
                base_tokens[start:end], 20
            ).float()
            noise = (
                1
                - self.noise_schedule.alphas_bar[
                    graph_time[batch_id]
                ]
            )
            ratio = (
                self.min_mask_ratio
                + self.mask_ratio_deviation
                * torch.sin(noise * math.pi / 2)
            )
            count = min(length, max(1, math.ceil(length * float(ratio))))
            selected = entropy[start:end].topk(count).indices
            padded_node_mask[batch_id, selected] = 1
        prior_padded = self.prior_model(
            padded_x,
            ipa_atom_positions,
            padded_node_mask,
            ipa_sequence_mask,
            pair_embedding=ipa_pair_embedding,
        )
        prior_logits = prior_padded[ipa_sequence_mask]
        prior_probability = torch.softmax(prior_logits, dim=-1)
        prior_entropy = -(
            prior_probability
            * torch.log(prior_probability.clamp_min(1e-8))
        ).mean(dim=-1)
        weights = torch.softmax(
            -torch.stack([entropy, prior_entropy], dim=0), dim=0
        )
        fused = (
            weights[0, :, None] * base_logits
            + weights[1, :, None] * prior_logits
        )
        return fused, base_logits, prior_logits, padded_node_mask

    def forward(self, training_stage="diffusion", **inputs):
        if training_stage == "ipa":
            return self.prior_pretrain_loss(**inputs)
        if training_stage != "diffusion":
            raise ValueError(
                "MapDiff training_stage must be 'ipa' or 'diffusion'."
            )
        return self.diffusion_loss(**inputs)

    def diffusion_loss(
        self,
        *,
        target_residue_one_hot,
        ca_coordinates,
        edge_index,
        graph_index,
        graph_ptr,
        ipa_atom_positions,
        ipa_sequence_mask,
        structure_embedding,
        edge_embedding,
        ipa_pair_embedding,
        **metadata,
    ):
        target_x = to_mapdiff_order(target_residue_one_hot)
        graph_count = graph_ptr.numel() - 1
        graph_time = torch.randint(
            0,
            self.timesteps + 1,
            (graph_count,),
            device=target_x.device,
        )
        noisy = self._q_sample(target_x, graph_time[graph_index])
        fused, base, prior, node_mask = self._predict(
            noisy_residue_one_hot=noisy,
            graph_time=graph_time,
            ca_coordinates=ca_coordinates,
            edge_index=edge_index,
            graph_index=graph_index,
            graph_ptr=graph_ptr,
            ipa_atom_positions=ipa_atom_positions,
            ipa_sequence_mask=ipa_sequence_mask,
            structure_embedding=structure_embedding,
            edge_embedding=edge_embedding,
            ipa_pair_embedding=ipa_pair_embedding,
        )
        labels = target_x.argmax(dim=-1)
        selected = node_mask[ipa_sequence_mask] == 1
        base_loss = F.cross_entropy(base, labels)
        prior_loss = F.cross_entropy(prior[selected], labels[selected])
        return {
            "loss": base_loss + prior_loss,
            "base_loss": base_loss,
            "prior_loss": prior_loss,
            "logits": to_standard_order(fused),
            "noisy_residue_one_hot": to_standard_order(noisy),
            "timesteps": graph_time,
        }

    def prior_pretrain_loss(
        self,
        *,
        ipa_residue_one_hot,
        ipa_atom_positions,
        ipa_sequence_mask,
        ipa_target_tokens,
        ipa_pair_embedding,
        **metadata,
    ):
        corrupted, mask_kind = self._mask_ipa_inputs(
            ipa_residue_one_hot,
            ipa_target_tokens,
            ipa_sequence_mask,
        )
        logits = self.prior_model(
            to_mapdiff_order(corrupted),
            ipa_atom_positions,
            mask_kind,
            ipa_sequence_mask,
            pair_embedding=ipa_pair_embedding,
        )
        standard_to_mapdiff = torch.tensor(
            _STANDARD_FROM_MAPDIFF,
            device=ipa_target_tokens.device,
        )
        labels = standard_to_mapdiff[
            ipa_target_tokens.clamp_max(19)
        ]
        selected = (mask_kind > 0) & ipa_sequence_mask
        return {
            "loss": F.cross_entropy(
                logits[selected], labels[selected]
            ),
            "logits": to_standard_order(logits),
            "ipa_mask_kind": mask_kind,
        }

    def _mask_ipa_inputs(
        self,
        residue_one_hot,
        target_tokens,
        sequence_mask,
    ):
        corrupted = residue_one_hot.clone()
        mask_kind = torch.zeros_like(
            target_tokens, dtype=torch.long
        )
        for row in range(residue_one_hot.shape[0]):
            length = int(sequence_mask[row].sum())
            candidate_count = max(
                1, int(length * self.candidate_rate)
            )
            candidates = torch.randperm(
                length, device=residue_one_hot.device
            )[:candidate_count]
            candidates = candidates[
                torch.randperm(
                    candidate_count,
                    device=residue_one_hot.device,
                )
            ]
            mask_count = max(
                1, int(candidate_count * self.ipa_mask_rate)
            )
            remaining_count = candidate_count - mask_count
            replacement_pool_rate = (
                self.ipa_replace_rate + self.ipa_keep_rate
            )
            replace_count = min(
                int(
                    remaining_count
                    * self.ipa_replace_rate
                    / replacement_pool_rate
                )
                if replacement_pool_rate
                else 0,
                remaining_count,
            )
            mask_ids = candidates[:mask_count]
            replace_ids = candidates[
                mask_count : mask_count + replace_count
            ]
            keep_ids = candidates[mask_count + replace_count :]
            mask_kind[row, mask_ids] = 1
            mask_kind[row, replace_ids] = 2
            mask_kind[row, keep_ids] = 3
            if replace_ids.numel():
                original = target_tokens[row, replace_ids]
                replacements = torch.randint(
                    0,
                    19,
                    original.shape,
                    device=original.device,
                )
                replacements = (
                    replacements + (replacements >= original).long()
                )
                corrupted[row, replace_ids] = F.one_hot(
                    replacements, 20
                ).float()
        return corrupted, mask_kind

    @staticmethod
    def _decode(tokens, graph_ptr):
        output = []
        for index in range(graph_ptr.numel() - 1):
            start, end = graph_ptr[index : index + 2].tolist()
            output.append(
                "".join(
                    MAPDIFF_ALPHABET[token]
                    for token in tokens[start:end].tolist()
                )
            )
        return output

    def _reverse_probability(self, z_t, predicted_x0, node_t, node_s, method):
        alphas = self.noise_schedule.alphas_bar
        alpha_t, alpha_s = alphas[node_t], alphas[node_s]
        q_s, q_t_bar = self._matrix(alpha_s), self._matrix(alpha_t)
        if method == "ddim":
            q_step = q_s / q_t_bar.clamp_min(1e-8)
            q_step = q_step / q_step.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            q_step = self._ddpm_step_matrix(node_t, node_s)
        left = torch.bmm(z_t[:, None], q_step.transpose(-1, -2))
        numerator = left * q_s
        denominator = torch.bmm(q_t_bar, z_t[:, :, None]).clamp_min(1e-8)
        posterior_over_x0 = numerator / denominator
        probability = (predicted_x0[:, :, None] * posterior_over_x0).sum(dim=1)
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _ddpm_step_matrix(self, node_t, node_s):
        alphas = self.noise_schedule.alphas_bar
        relative_alpha = (
            alphas[node_t] / alphas[node_s].clamp_min(1e-8)
        )
        return self._matrix(relative_alpha.clamp(0, 1))

    @torch.no_grad()
    def sample(
        self,
        *,
        ca_coordinates,
        edge_index,
        graph_index,
        graph_ptr,
        ipa_atom_positions,
        ipa_sequence_mask,
        structure_embedding,
        edge_embedding,
        ipa_pair_embedding,
        steps=100,
        method="ddim",
        temperature=1.0,
        num_samples=1,
        **metadata,
    ):
        if method not in {"ddim", "ddpm"}:
            raise ValueError("sampling method must be 'ddim' or 'ddpm'.")
        if steps < 1 or num_samples < 1:
            raise ValueError("steps and num_samples must both be positive.")
        graph_count = graph_ptr.numel() - 1
        node_count = ca_coordinates.shape[0]
        schedule = torch.linspace(
            self.timesteps,
            0,
            min(steps, self.timesteps) + 1,
            device=ca_coordinates.device,
        )
        schedule = torch.unique_consecutive(schedule.round().long())
        all_sequences = []
        all_trajectories = []
        all_logits = []
        all_token_ids = []
        mapdiff_to_standard = torch.tensor(
            [
                AA_ALPHABET.index(amino_acid)
                for amino_acid in MAPDIFF_ALPHABET
            ],
            device=ca_coordinates.device,
        )
        for _ in range(num_samples):
            tokens = torch.multinomial(
                self.marginal.expand(node_count, -1), 1
            ).squeeze(-1)
            noisy = F.one_hot(tokens, 20).float()
            trajectory = [
                {
                    "timestep": int(schedule[0].item()),
                    "sequences": self._decode(tokens, graph_ptr),
                }
            ]
            final_logits = None
            for current, following in zip(schedule[:-1], schedule[1:]):
                graph_time = torch.full(
                    (graph_count,),
                    int(current.item()),
                    dtype=torch.long,
                    device=noisy.device,
                )
                fused, _, _, _ = self._predict(
                    noisy_residue_one_hot=noisy,
                    graph_time=graph_time,
                    ca_coordinates=ca_coordinates,
                    edge_index=edge_index,
                    graph_index=graph_index,
                    graph_ptr=graph_ptr,
                    ipa_atom_positions=ipa_atom_positions,
                    ipa_sequence_mask=ipa_sequence_mask,
                    structure_embedding=structure_embedding,
                    edge_embedding=edge_embedding,
                    ipa_pair_embedding=ipa_pair_embedding,
                )
                predicted = torch.softmax(
                    fused / max(float(temperature), 1e-4), dim=-1
                )
                if int(following.item()) == 0:
                    tokens = predicted.argmax(dim=-1)
                else:
                    node_time = graph_time[graph_index]
                    following_time = torch.full_like(
                        node_time, int(following.item())
                    )
                    reverse = self._reverse_probability(
                        noisy,
                        predicted,
                        node_time,
                        following_time,
                        method,
                    )
                    tokens = torch.multinomial(
                        reverse.clamp_min(1e-8), 1
                    ).squeeze(-1)
                noisy = F.one_hot(tokens, 20).float()
                trajectory.append(
                    {
                        "timestep": int(following.item()),
                        "sequences": self._decode(tokens, graph_ptr),
                    }
                )
                final_logits = to_standard_order(fused)
            all_sequences.extend(trajectory[-1]["sequences"])
            all_trajectories.append(trajectory)
            all_logits.append(final_logits)
            all_token_ids.append(mapdiff_to_standard[tokens])
        return {
            "sequences": all_sequences,
            "token_ids": torch.cat(all_token_ids),
            "logits": torch.cat(all_logits),
            "trajectory": all_trajectories[0],
            "trajectories": all_trajectories,
            "sampling_method": method,
            "checkpoint_architecture": "mapdiff-v1.0.1",
        }


def _network_kwargs(config):
    settings = dict(config.get("model", {}))
    masking = dict(config.get("ipa_masking", {}))
    marginal = None
    marginal_map = settings.get("marginal_map")
    if marginal_map:
        path = Path(config.get("_config_dir", ".")) / marginal_map
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Cannot read MapDiff marginal map {path}: {error}"
            ) from error
        marginal = payload.get("probabilities")
        if (
            payload.get("alphabet") != MAPDIFF_ALPHABET
            or not isinstance(marginal, list)
            or len(marginal) != 20
        ):
            raise ValueError(
                f"Invalid MapDiff marginal map schema in {path}."
            )
    return {
        "hidden_dim": settings.get("hidden_dim", 128),
        "egnn_depth": settings.get("egnn_depth", 6),
        "ipa_depth": settings.get("ipa_depth", 6),
        "ipa_heads": settings.get("ipa_heads", 4),
        "qk_points": settings.get("qk_points", 4),
        "v_points": settings.get("v_points", 8),
        "egnn_dropout": settings.get("egnn_dropout", 0.1),
        "ipa_dropout": settings.get("ipa_dropout", 0.2),
        "timesteps": settings.get("timesteps", 500),
        "min_mask_ratio": settings.get("min_mask_ratio", 0.4),
        "mask_ratio_deviation": settings.get(
            "mask_ratio_deviation", 0.2
        ),
        "candidate_rate": masking.get("candidate_rate", 0.7),
        "ipa_mask_rate": masking.get("mask_rate", 0.8),
        "ipa_replace_rate": masking.get("replace_rate", 0.1),
        "ipa_keep_rate": masking.get("keep_rate", 0.1),
        **({"marginal": marginal} if marginal is not None else {}),
    }


@AutoProteinPredictor.register("inverse_folding/mapdiff_generator")
class MapDiffGenerator(nn.Module):
    """Complete MapDiff IPA prior and categorical diffusion generator."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.network = MapDiffNetwork(**_network_kwargs(config))
        self.last_trajectory = None

    @property
    def architecture(self):
        return "mapdiff-v1.0.1"

    def embed(
        self,
        *,
        extra_residue_features,
        edge_features,
        secondary_structure,
        ipa_atom_positions,
        **inputs,
    ):
        condition = self.network.encode_condition(
            extra_residue_features=extra_residue_features,
            edge_features=edge_features,
            secondary_structure=secondary_structure,
            ipa_atom_positions=ipa_atom_positions,
        )
        return {
            **inputs,
            "extra_residue_features": extra_residue_features,
            "edge_features": edge_features,
            "secondary_structure": secondary_structure,
            "ipa_atom_positions": ipa_atom_positions,
            **condition,
        }

    def forward(self, **embeddings):
        output = self.network(**embeddings)
        return _attach_metadata(
            output,
            embeddings.get("reference_sequences"),
            embeddings.get("sample_ids"),
        )

    def generate(
        self,
        *,
        sampling_config=None,
        reference_sequences=None,
        sample_ids=None,
        **embeddings,
    ):
        sampling = dict(self.config.get("sampling", {}))
        sampling.update(sampling_config or {})
        for key in ("steps", "method", "temperature", "num_samples"):
            if key in embeddings:
                sampling[key] = embeddings.pop(key)
        arguments = {
            "steps": sampling.get("steps", 100),
            "method": sampling.get("method", "ddim"),
            "temperature": sampling.get("temperature", 1.0),
            "num_samples": sampling.get("num_samples", 1),
        }
        output = self.network.sample(**embeddings, **arguments)
        self.last_trajectory = output["trajectory"]
        repeats = int(arguments["num_samples"])
        return _attach_metadata(
            output,
            (
                list(reference_sequences) * repeats
                if reference_sequences is not None
                else None
            ),
            (
                list(sample_ids) * repeats
                if sample_ids is not None
                else None
            ),
        )


@AutoProteinEmbedder.register("structure/mapdiff_condition")
class MapDiffConditionEmbedder(nn.Module):
    """Non-owning condition-encoder view over the complete generator."""

    def __init__(self, config, predictor, **kwargs):
        super().__init__()
        self.config = config
        self.__dict__["_predictor_ref"] = weakref.ref(predictor)

    @property
    def predictor(self):
        predictor = self._predictor_ref()
        if predictor is None:
            raise RuntimeError(
                "The MapDiff generator backing this embedder no longer exists."
            )
        return predictor

    def embed(self, **inputs):
        return self.predictor.embed(**inputs)

    forward = embed


class MapDiffModel(nn.Module):
    """Complete MapDiff model with strict release-checkpoint loading."""

    config_class = MapDiffConfig

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.predictor = AutoProteinPredictor.from_config(
            config.get_predictor(), config=config
        )
        self.embedder = AutoProteinEmbedder.from_config(
            config.get_embedders()["structure"],
            config=config,
            predictor=self.predictor,
        )

    @property
    def architecture(self):
        return self.predictor.architecture

    @property
    def network(self):
        return self.predictor.network

    def embed(self, **inputs):
        return self.embedder.embed(**inputs)

    def predict(self, **embeddings):
        return self.predictor(**embeddings)

    def forward(self, **inputs):
        return self.predict(**self.embed(**inputs))

    def generate(self, *, sampling_config=None, **embeddings):
        return self.predictor.generate(
            **embeddings,
            sampling_config=sampling_config,
        )

    sample = generate

    def evaluate(
        self,
        sequences,
        reference_sequences,
        logits=None,
        **generation,
    ):
        from kaleprotein.evaluate.diversity import Diversity
        from kaleprotein.evaluate.perplexity import Perplexity
        from kaleprotein.evaluate.sequence_recovery import SequenceRecovery

        output = {
            "sequences": sequences,
            "logits": logits,
            **generation,
        }
        metrics = {
            "sequence_recovery": SequenceRecovery()(
                output, reference_sequences
            ),
            "diversity": Diversity()(output),
        }
        if logits is not None:
            metrics["perplexity"] = Perplexity()(
                output, reference_sequences
            )
        return metrics

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "mapdiff-v1.0.1",
                "state_dict": self.state_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path):
        path = Path(path)
        state = load_checkpoint_state_dict(
            path, map_location="cpu"
        )
        if not state:
            raise RuntimeError(
                f"MapDiff checkpoint compatibility error for {path}: "
                "no state dictionary was found."
            )
        normalized = _network_state_dict(state)
        expected = self.network.state_dict()
        missing = sorted(set(expected) - set(normalized))
        unexpected = sorted(set(normalized) - set(expected))
        shape_mismatch = sorted(
            key
            for key in set(expected) & set(normalized)
            if expected[key].shape != normalized[key].shape
        )
        if missing or unexpected or shape_mismatch:
            details = []
            if missing:
                details.append(f"missing keys {missing[:5]}")
            if unexpected:
                details.append(f"unexpected keys {unexpected[:5]}")
            if shape_mismatch:
                details.append(
                    f"shape mismatches {shape_mismatch[:5]}"
                )
            raise RuntimeError(
                f"MapDiff checkpoint compatibility error for {path}: "
                f"{'; '.join(details)}. The mapdiff-v1.0.1 parameter "
                "tree requires an exact state-key and shape match."
            )
        self.network.load_state_dict(normalized, strict=True)
        return self


def _network_state_dict(state):
    normalized = {}
    for raw_key, value in state.items():
        key = raw_key.removeprefix("module.")
        for prefix in ("predictor.network.", "network."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        normalized[key] = value
    return normalized


def _attach_metadata(output, reference_sequences, sample_ids):
    output = dict(output)
    if reference_sequences is not None:
        output["reference_sequences"] = list(reference_sequences)
    if sample_ids is not None:
        output["sample_ids"] = list(sample_ids)
    return output


__all__ = [
    "MapDiffConditionEmbedder",
    "MapDiffGenerator",
    "MapDiffModel",
    "MapDiffNetwork",
]
