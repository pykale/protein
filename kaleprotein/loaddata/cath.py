"""CATH dataset adapter with task-neutral structure records."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from kaleprotein.auto.registry import DATASET_REGISTRY

from .records import AMINO_ACID_ALPHABET, StructureRecord
from kaleprotein.utils import parse_mmcif, parse_pdb, require_file


@DATASET_REGISTRY.register("CATH/InverseFolding")
class CATHDataset(Sequence[StructureRecord]):
    """Read processed tensors, PDB files, or mmCIF files as structure records."""

    def __init__(self, source=None, split=None):
        if source is None:
            raise ValueError(
                "CATH/InverseFolding requires source='path/to/processed-structures'."
            )
        paths = _resolve_paths(source, split=split)
        self.source = source
        self.split = split
        self.records = []
        for path in paths:
            suffix = path.suffix.casefold()
            if suffix == ".pt":
                self.records.extend(_load_pt(path))
            elif suffix == ".pdb":
                self.records.append(parse_pdb(path))
            elif suffix in {".cif", ".mmcif"}:
                self.records.append(parse_mmcif(path))
            else:
                raise ValueError(f"Expected a .pt, .pdb, .cif, or .mmcif file, got: {path}")

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index):
        return self.records[index]


def _resolve_paths(source, *, split=None):
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"CATH data source not found: {source_path}")
        if split and source_path.is_dir() and (source_path / split).exists():
            source_path = source_path / split
        if source_path.is_dir():
            paths = []
            for suffix in ("*.pt", "*.pdb", "*.cif", "*.mmcif"):
                paths.extend(sorted(source_path.glob(suffix)))
        else:
            paths = [source_path]
    elif isinstance(source, Iterable):
        paths = [require_file(item, description="CATH data file") for item in source]
    else:
        raise TypeError("CATH source must be a path or iterable of paths.")
    if not paths:
        raise FileNotFoundError(f"No supported CATH structure files found under {source!s}")
    return paths


def _load_pt(path):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading processed CATH .pt files requires PyTorch.") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not load {path} because it references optional module {exc.name!r}. "
            "Re-save it as a plain dictionary containing atom_pos and sequence."
        ) from exc
    if isinstance(payload, dict) and "graphs" in payload:
        payload = payload["graphs"]
    items = payload if isinstance(payload, (list, tuple)) else [payload]
    records = []
    for index, item in enumerate(items):
        try:
            records.append(_coerce_structure_record(item, path=path, index=index, torch=torch))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported processed CATH payload in {path} at item {index}: {exc}"
            ) from exc
    return records


def _coerce_structure_record(item, *, path, index, torch):
    if isinstance(item, StructureRecord):
        return item
    get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(
        item, key, default
    )
    coords = get("atom_pos")
    if coords is None:
        coords = get("backbone_coords")
    if coords is None:
        coords = get("coords")
    if coords is None:
        raise ValueError("record is missing atom_pos/backbone_coords/coords")
    coords = torch.as_tensor(coords, dtype=torch.float32)
    if coords.ndim == 2 and coords.shape[-1] == 3:
        coords = coords[:, None, :]
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coordinates must have shape [residues, atoms, 3], got {coords.shape}")
    if coords.shape[1] >= 5:
        coords = coords[:, [0, 1, 2, 4]]
    elif coords.shape[1] > 4:
        coords = coords[:, :4]
    elif coords.shape[1] < 4:
        padding = torch.full(
            (coords.shape[0], 4 - coords.shape[1], 3),
            float("nan"),
            dtype=coords.dtype,
        )
        coords = torch.cat([coords, padding], dim=1)
    atom_mask = get("atom_mask")
    if atom_mask is None:
        atom_mask = torch.isfinite(coords).all(dim=-1)
    else:
        atom_mask = torch.as_tensor(atom_mask, dtype=torch.bool)
        if atom_mask.shape[1] >= 5:
            atom_mask = atom_mask[:, [0, 1, 2, 4]]
        elif atom_mask.shape[1] > 4:
            atom_mask = atom_mask[:, :4]
        elif atom_mask.shape[1] < 4:
            padding = torch.zeros(
                (atom_mask.shape[0], 4 - atom_mask.shape[1]), dtype=torch.bool
            )
            atom_mask = torch.cat([atom_mask, padding], dim=1)
    sequence = str(get("sequence", "") or "")
    x = get("x")
    if not sequence and x is not None:
        tensor_x = torch.as_tensor(x)
        if tensor_x.ndim == 2 and tensor_x.shape[-1] >= len(AMINO_ACID_ALPHABET):
            sequence = "".join(
                AMINO_ACID_ALPHABET[position]
                for position in tensor_x[:, :20].argmax(dim=-1).tolist()
            )
    identifier = (
        get("identifier") or get("id") or get("name") or f"{Path(path).stem}:{index}"
    )
    return StructureRecord(
        atom_pos=coords,
        atom_mask=atom_mask,
        sequence=sequence[: coords.shape[0]],
        identifier=str(identifier),
        metadata={"source_path": str(path), "item_index": index},
    )


__all__ = ["CATHDataset"]
