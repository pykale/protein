"""Self-contained PyTorch DrugBAN refactor."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from kaleprotein.drugban.tokenization import DrugBANPreprocessor
from kaleprotein.hub import load_state_dict_if_available, resolve_weight_file


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1).clamp_min(1.0)
        norm_adj = adj / degree.unsqueeze(-1)
        hidden = torch.bmm(norm_adj, node_features)
        return F.relu(self.linear(self.dropout(hidden)))


class MolecularGCN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        dims = [embedding_dim] + hidden_dims
        self.layers = nn.ModuleList(GraphConvolution(dims[i], dims[i + 1], dropout) for i in range(len(dims) - 1))
        self.output_dim = hidden_dims[-1]

    def forward(self, node_ids: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(node_ids)
        for layer in self.layers:
            hidden = layer(hidden, adj)
        return hidden


class ProteinCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, filters: list[int], kernels: list[int], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        channels = [embedding_dim] + filters
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernels[i], padding=kernels[i] // 2),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(len(filters))
        )
        self.output_dim = filters[-1]

    def forward(self, protein_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(protein_ids).transpose(1, 2)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden.transpose(1, 2)


class BilinearAttention(nn.Module):
    """Bilinear attention layer shaped like DrugBAN's interaction block."""

    def __init__(self, drug_dim: int, protein_dim: int, hidden_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.drug_proj = nn.Linear(drug_dim, hidden_dim * heads)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim * heads)
        self.output = nn.Linear(hidden_dim * heads, hidden_dim)

    def forward(
        self,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = drug_features.size(0)
        drug = self.drug_proj(drug_features).view(batch_size, drug_features.size(1), self.heads, -1).transpose(1, 2)
        protein = self.protein_proj(protein_features).view(batch_size, protein_features.size(1), self.heads, -1).transpose(1, 2)
        scores = torch.matmul(drug, protein.transpose(-1, -2)) / (drug.size(-1) ** 0.5)
        mask = drug_mask[:, None, :, None] & protein_mask[:, None, None, :]
        scores = scores.masked_fill(~mask, -1e4)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, protein)
        pooled = (context * drug_mask[:, None, :, None]).sum(dim=2) / drug_mask.sum(dim=1).clamp_min(1)[:, None, None]
        return self.output(pooled.flatten(start_dim=1)), attention


class DrugBANForDTI(nn.Module):
    """DrugBAN model with HF-like constructors and local preprocessing."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        model = config["model"]
        self.preprocessor = DrugBANPreprocessor.from_config(config)
        self.drug_encoder = MolecularGCN(
            vocab_size=int(model["drug_vocab_size"]),
            embedding_dim=int(model["drug_embedding_dim"]),
            hidden_dims=[int(v) for v in model["drug_hidden_dims"]],
            dropout=float(model.get("dropout", 0.1)),
        )
        self.protein_encoder = ProteinCNN(
            vocab_size=int(model["protein_vocab_size"]),
            embedding_dim=int(model["protein_embedding_dim"]),
            filters=[int(v) for v in model["protein_filters"]],
            kernels=[int(v) for v in model["protein_kernels"]],
            dropout=float(model.get("dropout", 0.1)),
        )
        self.attention = BilinearAttention(
            drug_dim=self.drug_encoder.output_dim,
            protein_dim=self.protein_encoder.output_dim,
            hidden_dim=int(model["ban_hidden_dim"]),
            heads=int(model["ban_heads"]),
        )
        hidden = int(model["classifier_hidden_dim"])
        self.classifier = nn.Sequential(
            nn.Linear(int(model["ban_hidden_dim"]), hidden),
            nn.ReLU(),
            nn.Dropout(float(model.get("dropout", 0.1))),
            nn.Linear(hidden, 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], *, pretrain: bool = False, downloader=None, strict: bool = False) -> "DrugBANForDTI":
        model = cls(config)
        weight_file = resolve_weight_file(config, pretrain=pretrain, downloader=downloader) if downloader else resolve_weight_file(config, pretrain=pretrain)
        load_state_dict_if_available(model, weight_file, strict=strict)
        return model

    def preprocess(self, smiles: str, protein: str) -> dict[str, torch.Tensor]:
        return self.preprocessor(smiles, protein).to_dict()

    def __iter__(self):
        yield DrugBANProteinEncoder(self)
        yield DrugBANMoleculeEncoder(self)

    def embed_protein(self, protein_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "protein_features": self.protein_encoder(protein_ids),
            "protein_mask": protein_ids.ne(0),
        }

    def embed_drug(self, drug_node_ids: torch.Tensor, drug_adj: torch.Tensor, drug_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "drug_features": self.drug_encoder(drug_node_ids, drug_adj),
            "drug_mask": drug_mask,
        }

    def predict_from_embeddings(
        self,
        protein_embedding: dict[str, torch.Tensor],
        drug_embedding: dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        interaction, attention = self.attention(
            drug_embedding["drug_features"],
            protein_embedding["protein_features"],
            drug_embedding["drug_mask"],
            protein_embedding["protein_mask"],
        )
        logits = self.classifier(interaction).squeeze(-1)
        output = {"logits": logits, "probabilities": torch.sigmoid(logits)}
        if labels is not None:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, labels.float())
        if return_attention:
            output["attention"] = attention
        return output

    def predict_pair(self, smiles: str, protein: str) -> dict[str, torch.Tensor]:
        batch = {key: value.unsqueeze(0) for key, value in self.preprocess(smiles, protein).items()}
        self.eval()
        with torch.no_grad():
            return self.forward(**batch, return_attention=True)

    def forward(
        self,
        drug_node_ids: torch.Tensor,
        drug_adj: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        protein_embedding = self.embed_protein(protein_ids)
        drug_embedding = self.embed_drug(drug_node_ids, drug_adj, drug_mask)
        return self.predict_from_embeddings(protein_embedding, drug_embedding, labels=labels, return_attention=return_attention)


class DrugBANProteinEncoder(nn.Module):
    def __init__(self, model: DrugBANForDTI):
        super().__init__()
        self.encoder = model.protein_encoder

    def embed(self, protein_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        protein_ids = protein_data["protein_ids"]
        return {
            "protein_features": self.encoder(protein_ids),
            "protein_mask": protein_ids.ne(0),
        }


class DrugBANMoleculeEncoder(nn.Module):
    def __init__(self, model: DrugBANForDTI):
        super().__init__()
        self.encoder = model.drug_encoder

    def embed(self, drug_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "drug_features": self.encoder(drug_data["drug_node_ids"], drug_data["drug_adj"]),
            "drug_mask": drug_data["drug_mask"],
        }


class DrugBANInteractionPredictor(nn.Module):
    def __init__(self, model: DrugBANForDTI):
        super().__init__()
        self.attention = model.attention
        self.classifier = model.classifier

    @classmethod
    def from_config(cls, config: dict[str, Any], *, pretrain: bool = False, downloader=None, strict: bool = False) -> "DrugBANInteractionPredictor":
        return cls(DrugBANForDTI.from_config(config, pretrain=pretrain, downloader=downloader, strict=strict))

    def forward(
        self,
        protein_embedding: dict[str, torch.Tensor],
        drug_embedding: dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        interaction, attention = self.attention(
            drug_embedding["drug_features"],
            protein_embedding["protein_features"],
            drug_embedding["drug_mask"],
            protein_embedding["protein_mask"],
        )
        logits = self.classifier(interaction).squeeze(-1)
        output = {"logits": logits, "probabilities": torch.sigmoid(logits)}
        if labels is not None:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, labels.float())
        if return_attention:
            output["attention"] = attention
        return output
