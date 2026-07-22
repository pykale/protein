"""Stable data records shared by dataset adapters and parsing utilities."""

from dataclasses import dataclass, field
from typing import Any


AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_ALPHABET = AMINO_ACID_ALPHABET
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ALPHABET)}


@dataclass(frozen=True)
class SequenceRecord:
    identifier: str
    sequence: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "sequence": self.sequence,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DTISample:
    identifier: str
    smiles: str
    sequence: str
    label: int | float
    dataset: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "smiles": self.smiles,
            "sequence": self.sequence,
            "label": self.label,
            "dataset": self.dataset,
            "metadata": dict(self.metadata),
        }


@dataclass
class StructureRecord:
    """Model-independent protein structure record in N/CA/C/O atom order."""

    atom_pos: Any
    atom_mask: Any
    sequence: str = ""
    identifier: str = "protein"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.identifier

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "identifier": self.identifier,
            "sequence": self.sequence,
            "atom_pos": self.atom_pos,
            "atom_mask": self.atom_mask,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "AMINO_ACID_ALPHABET",
    "AA_ALPHABET",
    "AA_TO_INDEX",
    "DTISample",
    "SequenceRecord",
    "StructureRecord",
]
