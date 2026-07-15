"""Model-independent normalization of protein backbone inputs."""

from pathlib import Path

from kaleprotein.core.data.records import StructureRecord
from kaleprotein.core.data.utils import parse_mmcif, parse_pdb
from kaleprotein.core.registry import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register(
    "structure/backbone",
    aliases=("structure/backbone_coordinate_processor", "protein/structure"),
)
class BackboneCoordinateProcessor:
    """Normalize in-memory coordinates or a PDB path to N/CA/C/O features."""

    default_input_key = "backbone_coords"

    def __init__(self, input_key="backbone_coords", chain=None, **kwargs):
        self.input_key = input_key
        self.chain = chain

    def transform(self, sample):
        pdb_path = sample.get("pdb_path") if isinstance(sample, dict) else None
        if pdb_path is None and isinstance(sample, (str, Path)):
            pdb_path = sample
        if pdb_path is not None:
            suffix = Path(pdb_path).suffix.casefold()
            record = (
                parse_mmcif(pdb_path, chain=self.chain)
                if suffix in {".cif", ".mmcif"}
                else parse_pdb(pdb_path, chain=self.chain)
            )
        elif isinstance(sample, StructureRecord):
            record = sample
        elif not isinstance(sample, dict):
            coords = getattr(sample, "atom_pos", None)
            if coords is None:
                raise TypeError(
                    "Structure preprocessing expects a mapping, StructureRecord, or structure path."
                )
            record = StructureRecord(
                atom_pos=coords,
                atom_mask=getattr(sample, "atom_mask", None),
                sequence=getattr(sample, "sequence", ""),
                identifier=getattr(sample, "identifier", "protein"),
            )
        else:
            if self.input_key not in sample:
                raise ValueError(
                    f"Expected sample[{self.input_key!r}] coordinates or a pdb_path value."
                )
            record = StructureRecord(
                atom_pos=sample[self.input_key],
                atom_mask=sample.get("atom_mask"),
                sequence=sample.get("sequence", ""),
                identifier=sample.get("id", sample.get("identifier", "protein")),
            )

        import torch

        atom_pos = torch.as_tensor(record.atom_pos, dtype=torch.float32)
        if atom_pos.ndim == 2 and atom_pos.shape[-1] == 3:
            atom_pos = atom_pos[:, None, :]
        if atom_pos.ndim != 3 or atom_pos.shape[-1] != 3:
            raise ValueError(
                "Backbone coordinates must have shape [residues, atoms, 3], got "
                f"{tuple(atom_pos.shape)}."
            )
        atom_mask = record.atom_mask
        if atom_mask is None:
            atom_mask = torch.isfinite(atom_pos).all(dim=-1)
        else:
            atom_mask = torch.as_tensor(atom_mask, dtype=torch.bool)
        return {
            "coords": atom_pos.tolist(),
            "atom_pos": atom_pos,
            "atom_mask": atom_mask,
            "coord_mask": atom_mask.all(dim=-1).tolist(),
            "identifier": record.identifier,
            "sequence": record.sequence,
        }
