"""Reusable small-molecule embedders."""

from torch import nn
from torch.nn import functional as F

from kaleprotein.core.registry import EMBEDDER_REGISTRY


class DenseGraphConv(nn.Module):
    """Symmetrically normalized message passing for dense graph batches."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, adjacency):
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        inverse_sqrt = degree.rsqrt()
        normalized = adjacency * inverse_sqrt.unsqueeze(-1) * inverse_sqrt.unsqueeze(-2)
        return self.linear(normalized.bmm(features))


@EMBEDDER_REGISTRY.register("molecule/gcn")
class MolecularGCN(nn.Module):
    """Configurable DGL-free molecular graph convolution network."""

    def __init__(
        self,
        config=None,
        in_feats=75,
        embedding_dim=128,
        hidden_dim=128,
        layers=3,
        **kwargs,
    ):
        super().__init__()
        self.init_transform = nn.Linear(in_feats, embedding_dim, bias=False)
        dimensions = [embedding_dim, *([hidden_dim] * int(layers))]
        self.layers = nn.ModuleList(
            DenseGraphConv(input_dim, output_dim)
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
        )
        self.output_feats = dimensions[-1]

    def forward(self, graph):
        features = graph["node_features"].float()
        adjacency = graph["adjacency"].float()
        mask = graph["node_mask"].bool()
        hidden = self.init_transform(features) * mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = F.relu(layer(hidden, adjacency)) * mask.unsqueeze(-1)
        return hidden

    def embed(self, graph):
        return {
            "embedding": self(graph),
            "mask": graph["node_mask"].bool(),
            "smiles": graph.get("smiles"),
            "atom_symbols": graph.get("atom_symbols"),
        }
