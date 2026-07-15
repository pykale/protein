"""MapDiff-specific graph and batch data structures."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from kaleprotein.core.data.records import AMINO_ACID_ALPHABET


AA_ALPHABET = AMINO_ACID_ALPHABET
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ALPHABET)}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - package runtime dependency
        raise ImportError(
            "MapDiff graph construction requires PyTorch. Install torch before using "
            "the MapDiff example."
        ) from exc
    return torch


@dataclass
class ProteinGraph:
    """A single residue graph with backbone atoms ordered as N, CA, C, O."""

    x: Any
    atom_pos: Any
    atom_mask: Any
    edge_index: Any
    edge_attr: Any
    identifier: str = "protein"
    sequence: str = ""

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def pos(self):
        return self.atom_pos[:, 1]

    def clone(self) -> "ProteinGraph":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.clone() if hasattr(value, "clone") else value
        return ProteinGraph(**values)

    def to(self, device) -> "ProteinGraph":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if hasattr(value, "to") else value
        return ProteinGraph(**values)


@dataclass
class GraphBatch:
    """Concatenated sparse graphs used by the EGNN diffusion network."""

    x: Any
    atom_pos: Any
    atom_mask: Any
    edge_index: Any
    edge_attr: Any
    batch: Any
    ptr: Any
    identifiers: list[str]
    sequences: list[str]

    @property
    def pos(self):
        return self.atom_pos[:, 1]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_graphs(self) -> int:
        return len(self.identifiers)

    def clone(self) -> "GraphBatch":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.clone() if hasattr(value, "clone") else list(value)
        return GraphBatch(**values)

    def to(self, device) -> "GraphBatch":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if hasattr(value, "to") else value
        return GraphBatch(**values)


@dataclass
class IPABatch:
    """Padded structure batch used by the invariant point-attention prior."""

    x: Any
    atom_pos: Any
    x_pad: Any
    x_mask: Any
    label: Any

    def __iter__(self):
        yield self.x
        yield self.atom_pos
        yield self.x_pad
        yield self.x_mask
        yield self.label

    def __len__(self) -> int:
        return 5

    def __getitem__(self, index):
        return tuple(iter(self))[index]

    def clone(self) -> "IPABatch":
        return IPABatch(*(value.clone() for value in self))

    def to(self, device) -> "IPABatch":
        return IPABatch(*(value.to(device) for value in self))


@dataclass
class DiffusionBatch:
    """Paired sparse EGNN and padded IPA views of the same proteins."""

    graph: GraphBatch
    ipa: IPABatch

    def __iter__(self):
        yield self.graph
        yield self.ipa

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        return (self.graph, self.ipa)[index]

    def clone(self) -> "DiffusionBatch":
        return DiffusionBatch(self.graph.clone(), self.ipa.clone())

    def to(self, device) -> "DiffusionBatch":
        return DiffusionBatch(self.graph.to(device), self.ipa.to(device))


def sequence_to_one_hot(sequence: str, length: int | None = None):
    torch = _torch()
    if length is None:
        length = len(sequence)
    labels = torch.tensor(
        [AA_TO_INDEX.get(aa.upper(), 0) for aa in sequence[:length]], dtype=torch.long
    )
    if labels.numel() < length:
        labels = torch.cat([labels, torch.zeros(length - labels.numel(), dtype=torch.long)])
    return torch.nn.functional.one_hot(labels, num_classes=len(AA_ALPHABET)).float()


def build_residue_graph(
    atom_pos: Any,
    sequence: str = "",
    identifier: str = "protein",
    atom_mask: Any | None = None,
    max_neighbors: int = 16,
) -> ProteinGraph:
    """Build a k-nearest-neighbor residue graph from backbone coordinates."""

    torch = _torch()
    coords = torch.as_tensor(atom_pos, dtype=torch.float32)
    if coords.ndim == 2 and coords.shape[-1] == 3:
        coords = coords[:, None, :]
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(
            "Backbone coordinates must have shape [residues, atoms, 3] or "
            f"[residues, 3], got {tuple(coords.shape)}."
        )
    if coords.shape[0] == 0:
        raise ValueError("Cannot construct a protein graph with zero residues.")

    if coords.shape[1] >= 5:
        coords = coords[:, [0, 1, 2, 4]]
    elif coords.shape[1] < 4:
        # Single-CA custom inputs remain usable; PDB preprocessing always
        # supplies all four atoms and marks missing atoms explicitly.
        coords = torch.cat([coords, coords[:, -1:].expand(-1, 4 - coords.shape[1], -1)], dim=1)
    elif coords.shape[1] > 4:
        coords = coords[:, :4]

    if atom_mask is None:
        mask = torch.isfinite(coords).all(dim=-1)
    else:
        mask = torch.as_tensor(atom_mask, dtype=torch.bool)
        if mask.shape[1] >= 5:
            mask = mask[:, [0, 1, 2, 4]]
        elif mask.shape[1] < 4:
            mask = torch.cat([mask, mask[:, -1:].expand(-1, 4 - mask.shape[1])], dim=1)
    coords = torch.nan_to_num(coords)

    n = coords.shape[0]
    ca = coords[:, 1]
    if n == 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 9), dtype=torch.float32)
    else:
        distances = torch.cdist(ca, ca)
        distances.fill_diagonal_(float("inf"))
        k = min(max_neighbors, n - 1)
        nearest = distances.topk(k, largest=False).indices
        dst = torch.arange(n).repeat_interleave(k)
        src = nearest.reshape(-1)
        edge_index = torch.stack([src, dst])
        edge_dist = torch.linalg.vector_norm(ca[src] - ca[dst], dim=-1)
        centers = torch.linspace(2.0, 20.0, 8)
        rbf = torch.exp(-((edge_dist[:, None] - centers[None]) / 2.5) ** 2)
        seq_offset = ((src - dst).float() / max(float(n), 1.0)).clamp(-1.0, 1.0)
        edge_attr = torch.cat([rbf, seq_offset[:, None]], dim=-1)

    return ProteinGraph(
        x=sequence_to_one_hot(sequence, n),
        atom_pos=coords,
        atom_mask=mask,
        edge_index=edge_index,
        edge_attr=edge_attr,
        identifier=str(identifier),
        sequence=sequence[:n],
    )


def coerce_protein_graph(record: Any, identifier: str | None = None) -> ProteinGraph:
    """Convert dictionaries and PyG-like records without importing PyG."""

    if isinstance(record, ProteinGraph):
        return record
    get = record.get if isinstance(record, dict) else lambda key, default=None: getattr(record, key, default)
    coords = get("atom_pos")
    if coords is None:
        coords = get("backbone_coords")
    if coords is None:
        coords = get("coords")
    if coords is None:
        raise ValueError("Processed graph record is missing atom_pos/backbone_coords coordinates.")

    sequence = get("sequence", "") or ""
    x = get("x")
    if not sequence and x is not None:
        torch = _torch()
        tensor_x = torch.as_tensor(x)
        if tensor_x.ndim == 2 and tensor_x.shape[-1] >= len(AA_ALPHABET):
            sequence = "".join(AA_ALPHABET[i] for i in tensor_x[:, :20].argmax(dim=-1).tolist())
    record_id = identifier or get("identifier") or get("id") or get("name") or "protein"
    return build_residue_graph(
        coords,
        sequence=sequence,
        identifier=str(record_id),
        atom_mask=get("atom_mask"),
    )


__all__ = [
    "AA_ALPHABET",
    "AA_TO_INDEX",
    "DiffusionBatch",
    "GraphBatch",
    "IPABatch",
    "ProteinGraph",
    "build_residue_graph",
    "coerce_protein_graph",
]
