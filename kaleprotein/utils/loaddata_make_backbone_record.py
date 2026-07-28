"""Build normalized backbone records for text structure parsers."""

from kaleprotein.loaddata.records import StructureRecord


BACKBONE_ATOMS = ("N", "CA", "C", "O")
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def make_backbone_record(residues, *, identifier, source_path, chain=None):
    entries = list(residues.values())
    if not entries:
        chain_hint = f" for chain {chain!r}" if chain is not None else ""
        raise ValueError(f"No protein backbone atoms found in {source_path}{chain_hint}.")
    atom_pos = []
    atom_mask = []
    for entry in entries:
        atoms = entry["atoms"]
        atom_pos.append([atoms.get(atom, [float("nan")] * 3) for atom in BACKBONE_ATOMS])
        atom_mask.append([atom in atoms for atom in BACKBONE_ATOMS])
    return StructureRecord(
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        sequence="".join(THREE_TO_ONE[entry["name"]] for entry in entries),
        identifier=str(identifier),
        metadata={"source_path": str(source_path), "chain": chain},
    )


__all__ = ["BACKBONE_ATOMS", "THREE_TO_ONE", "make_backbone_record"]
