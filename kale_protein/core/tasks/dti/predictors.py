"""Reusable drug-target interaction predictors and heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - older supported PyTorch
    from torch.nn.utils import weight_norm

from kale_protein.core.registry import PREDICTOR_REGISTRY


class FCNet(nn.Module):
    def __init__(self, dimensions, activation="ReLU", dropout=0.0):
        super().__init__()
        layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(input_dim, output_dim), dim=None))
            if activation:
                layers.append(getattr(nn, activation)())
        self.main = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.main(inputs)


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, inputs):
        if self.training and inputs.ndim == 2 and inputs.shape[0] == 1:
            return F.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(inputs)


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, activation="ReLU", dropout=0.2, k=3):
        super().__init__()
        self.k = k
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_net = FCNet([v_dim, h_dim * k], activation=activation, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * k], activation=activation, dropout=dropout)
        self.p_net = nn.AvgPool1d(k, stride=k) if k > 1 else None
        self.h_mat = nn.Parameter(torch.empty(1, h_out, 1, h_dim * k).normal_())
        self.h_bias = nn.Parameter(torch.empty(1, h_out, 1, 1).normal_())
        self.batch_norm = SafeBatchNorm1d(h_dim)

    def attention_pooling(self, visual, question, attention):
        fused = torch.einsum("bvk,bvq,bqk->bk", visual, attention, question)
        if self.p_net is not None:
            fused = self.p_net(fused.unsqueeze(1)).squeeze(1) * self.k
        return fused

    def forward(self, visual, question, visual_mask=None, question_mask=None, softmax=False):
        projected_visual = self.v_net(visual)
        projected_question = self.q_net(question)
        attention = torch.einsum(
            "xhyk,bvk,bqk->bhvq", self.h_mat, projected_visual, projected_question
        ) + self.h_bias
        pair_mask = None
        if visual_mask is not None and question_mask is not None:
            pair_mask = visual_mask[:, None, :, None] & question_mask[:, None, None, :]
            attention = attention.masked_fill(~pair_mask, float("-inf") if softmax else 0.0)
        if softmax:
            attention = torch.nan_to_num(torch.softmax(attention.flatten(2), dim=-1)).view_as(attention)
            if pair_mask is not None:
                attention = attention * pair_mask
        fused = sum(
            self.attention_pooling(projected_visual, projected_question, attention[:, head])
            for head in range(self.h_out)
        )
        return self.batch_norm(fused), attention


class MLPDecoder(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=128, binary=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = SafeBatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = SafeBatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = SafeBatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, inputs):
        hidden = self.bn1(F.relu(self.fc1(inputs)))
        hidden = self.bn2(F.relu(self.fc2(hidden)))
        hidden = self.bn3(F.relu(self.fc3(hidden)))
        return self.fc4(hidden)


def _embedding_parts(value):
    if isinstance(value, dict):
        embedding = value["embedding"]
        mask = value.get("mask")
    else:
        embedding = value
        mask = None
    if mask is None:
        mask = torch.ones(embedding.shape[:2], dtype=torch.bool, device=embedding.device)
    return embedding, mask.bool()


@PREDICTOR_REGISTRY.register("dti/ban")
class BANInteractionPredictor(nn.Module):
    """Bilinear-attention fusion and binary DTI classification head."""

    def __init__(
        self,
        config=None,
        molecule_dim=128,
        protein_dim=128,
        hidden_dim=256,
        heads=2,
        dropout=0.2,
        k=3,
        decoder_hidden_dim=512,
        decoder_output_dim=128,
        binary_dim=1,
        **kwargs,
    ):
        super().__init__()
        self.ban = weight_norm(
            BANLayer(
                v_dim=molecule_dim,
                q_dim=protein_dim,
                h_dim=hidden_dim,
                h_out=heads,
                dropout=dropout,
                k=k,
            ),
            name="h_mat",
            dim=None,
        )
        self.decoder = MLPDecoder(
            in_dim=hidden_dim,
            hidden_dim=decoder_hidden_dim,
            out_dim=decoder_output_dim,
            binary=binary_dim,
        )

    def forward(self, protein_embedding, molecule_embedding=None):
        if molecule_embedding is None:
            if not isinstance(protein_embedding, dict):
                raise TypeError("dti/ban expects protein and molecule embeddings.")
            molecule_embedding = protein_embedding["drug"]
            protein_embedding = protein_embedding["target"]
        protein, protein_mask = _embedding_parts(protein_embedding)
        molecule, molecule_mask = _embedding_parts(molecule_embedding)
        fused, attention = self.ban(
            molecule, protein, molecule_mask, protein_mask
        )
        logits = self.decoder(fused).squeeze(-1)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "attention": attention,
            "drug_mask": molecule_mask,
            "protein_mask": protein_mask,
            "fused_embedding": fused,
            "atom_symbols": molecule_embedding.get("atom_symbols") if isinstance(molecule_embedding, dict) else None,
            "smiles": molecule_embedding.get("smiles") if isinstance(molecule_embedding, dict) else None,
            "sequences": protein_embedding.get("sequences") if isinstance(protein_embedding, dict) else None,
        }


__all__ = ["BANInteractionPredictor", "BANLayer", "MLPDecoder"]
