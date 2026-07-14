"""Reusable structure graph data and CATH/processed-PT loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from kale_protein.registry import DATASET_REGISTRY


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ALPHABET)}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - package runtime dependency
        raise ImportError(
            "MapDiff graph loading requires PyTorch. Install torch before using "
            "inverse-folding datasets."
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


def parse_pdb_backbone(path: str | Path, chain: str | None = None) -> ProteinGraph:
    """Parse canonical residues with complete N/CA/C/O atoms from a PDB file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDB file does not exist: {path}")
    residues: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen_model = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if seen_model:
                    break
                seen_model = True
                continue
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  "):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16:17]
            chain_id = line[21:22].strip()
            if atom_name not in {"N", "CA", "C", "O"} or altloc not in {" ", "A"}:
                continue
            if chain is not None and chain_id != chain:
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name not in THREE_TO_ONE:
                continue
            key = (chain_id, line[22:26].strip(), line[26:27].strip())
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError as exc:
                raise ValueError(f"Invalid coordinate in {path} at PDB line: {line.rstrip()}") from exc
            entry = residues.setdefault(key, {"name": residue_name, "atoms": {}})
            entry["atoms"].setdefault(atom_name, xyz)

    complete = [entry for entry in residues.values() if all(a in entry["atoms"] for a in ("N", "CA", "C", "O"))]
    if not complete:
        chain_hint = f" for chain {chain!r}" if chain is not None else ""
        raise ValueError(
            f"No residues with complete N/CA/C/O backbone atoms found in {path}{chain_hint}. "
            "Check the chain identifier and PDB ATOM records."
        )
    coords = [[entry["atoms"][atom] for atom in ("N", "CA", "C", "O")] for entry in complete]
    sequence = "".join(THREE_TO_ONE[entry["name"]] for entry in complete)
    return build_residue_graph(coords, sequence, identifier=path.stem)


def _load_pt(path: Path) -> list[ProteinGraph]:
    torch = _torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not load {path} because it was serialized with optional module "
            f"{exc.name!r}. Re-save it as a plain dict containing atom_pos and x/sequence, "
            "or install the module that created the file."
        ) from exc
    if isinstance(payload, dict) and "graphs" in payload:
        payload = payload["graphs"]
    records = payload if isinstance(payload, (list, tuple)) else [payload]
    graphs = []
    for index, record in enumerate(records):
        try:
            graphs.append(coerce_protein_graph(record, f"{path.stem}:{index}"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported processed graph payload in {path} at item {index}: {exc}") from exc
    return graphs


class CATHGraphDataset(Sequence[ProteinGraph]):
    """Load CATH-style individual/collection `.pt` files or PDB files."""

    def __init__(self, source: str | Path | Iterable[str | Path], split: str | None = None):
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if split and source_path.is_dir() and (source_path / split).exists():
                source_path = source_path / split
            paths = sorted(source_path.glob("*.pt")) + sorted(source_path.glob("*.pdb")) if source_path.is_dir() else [source_path]
        else:
            paths = [Path(item) for item in source]
        if not paths:
            raise FileNotFoundError(f"No .pt or .pdb structure files found under {source!s}")
        self.graphs: list[ProteinGraph] = []
        for path in paths:
            if path.suffix.lower() == ".pt":
                self.graphs.extend(_load_pt(path))
            elif path.suffix.lower() == ".pdb":
                self.graphs.append(parse_pdb_backbone(path))
            else:
                raise ValueError(f"Expected a .pt or .pdb file, got: {path}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> ProteinGraph:
        return self.graphs[index]

    def __iter__(self) -> Iterator[ProteinGraph]:
        return iter(self.graphs)


@DATASET_REGISTRY.register("InverseFolding/CATH")
def load_cath_dataset(source: str | Path | None = None, split: str | None = None):
    if source is None:
        raise ValueError(
            "InverseFolding/CATH requires source='path/to/processed-graphs-or-pdbs'."
        )
    return CATHGraphDataset(source, split=split)
