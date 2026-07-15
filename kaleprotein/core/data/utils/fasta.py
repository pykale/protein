"""FASTA sequence reading utilities."""

from pathlib import Path

from ..records import SequenceRecord
from .files import require_file


def read_fasta(path, *, alphabet=None, uppercase=True):
    path = require_file(path, description="FASTA file")
    records = []
    identifier = None
    description = ""
    chunks = []

    def append_record():
        if identifier is None:
            return
        sequence = "".join(chunks).replace(" ", "")
        sequence = sequence.upper() if uppercase else sequence
        if not sequence:
            raise ValueError(f"FASTA record {identifier!r} has no sequence: {path}")
        if alphabet is not None:
            invalid = sorted(set(sequence) - set(alphabet))
            if invalid:
                raise ValueError(
                    f"FASTA record {identifier!r} contains invalid symbols {invalid}: {path}"
                )
        records.append(
            SequenceRecord(
                identifier=identifier,
                sequence=sequence,
                description=description,
                metadata={"source_path": str(path)},
            )
        )

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                append_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at line {line_number}: {path}")
                identifier, _, description = header.partition(" ")
                chunks = []
            elif identifier is None:
                raise ValueError(
                    f"FASTA sequence appears before the first header at line {line_number}: {path}"
                )
            else:
                chunks.append(line)
    append_record()
    if not records:
        raise ValueError(f"FASTA file contains no records: {path}")
    return records


__all__ = ["read_fasta"]
