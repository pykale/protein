"""Reusable protein-sequence embedders."""

from torch import nn
from torch.nn import functional as F

from kaleprotein.core.registry import EMBEDDER_REGISTRY


@EMBEDDER_REGISTRY.register("sequence/cnn")
class ProteinCNN(nn.Module):
    """Configurable three-layer residue convolution network."""

    def __init__(
        self,
        config=None,
        vocab_size=26,
        embedding_dim=128,
        filter_dim=128,
        kernels=(3, 6, 9),
        **kwargs,
    ):
        super().__init__()
        kernels = tuple(kernels)
        if len(kernels) != 3:
            raise ValueError("sequence/cnn requires exactly three kernel sizes.")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        channels = (embedding_dim, filter_dim, filter_dim, filter_dim)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(channels[index], channels[index + 1], kernels[index])
            for index in range(3)
        )
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(filter_dim) for _ in range(3))
        self.output_feats = filter_dim
        self.output_reduction = sum(size - 1 for size in kernels)

    def forward(self, protein):
        tokens = protein["tokens"].long()
        if tokens.shape[1] <= self.output_reduction:
            raise ValueError(
                f"Protein token length must exceed {self.output_reduction} for sequence/cnn."
            )
        hidden = self.embedding(tokens).transpose(1, 2)
        for convolution, batch_norm in zip(self.convolutions, self.batch_norms):
            hidden = batch_norm(F.relu(convolution(hidden)))
        return hidden.transpose(1, 2)

    def embed(self, protein):
        embedding = self(protein)
        return {
            "embedding": embedding,
            "mask": protein["attention_mask"].bool()[:, : embedding.shape[1]],
            "sequences": protein.get("sequences"),
        }
