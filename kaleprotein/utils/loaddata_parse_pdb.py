"""PDB coordinate parsing without model or graph dependencies."""

from pathlib import Path

from .loaddata_make_backbone_record import (
    BACKBONE_ATOMS,
    THREE_TO_ONE,
    make_backbone_record,
)
from .loaddata_require_file import require_file


def parse_pdb(path, *, chain=None):
    path = require_file(path, description="PDB file")
    residues = {}
    seen_model = False
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("MODEL"):
                if seen_model:
                    break
                seen_model = True
                continue
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16:17]
            chain_id = line[21:22].strip()
            residue_name = line[17:20].strip().upper()
            if atom_name not in BACKBONE_ATOMS or altloc not in {" ", "A"}:
                continue
            if chain is not None and chain_id != chain:
                continue
            if residue_name not in THREE_TO_ONE:
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid PDB coordinate at line {line_number} in {path}: {line.rstrip()}"
                ) from exc
            key = (chain_id, line[22:26].strip(), line[26:27].strip())
            entry = residues.setdefault(key, {"name": residue_name, "atoms": {}})
            entry["atoms"].setdefault(atom_name, xyz)
    return make_backbone_record(
        residues,
        identifier=Path(path).stem,
        source_path=path,
        chain=chain,
    )


__all__ = ["parse_pdb"]
