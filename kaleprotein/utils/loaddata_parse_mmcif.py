"""Minimal, dependency-free mmCIF atom-site parsing."""

import shlex
from pathlib import Path

from .loaddata_make_backbone_record import (
    BACKBONE_ATOMS,
    THREE_TO_ONE,
    make_backbone_record,
)
from .loaddata_require_file import require_file


def parse_mmcif(path, *, chain=None):
    path = require_file(path, description="mmCIF file")
    rows = _read_atom_site_rows(Path(path))
    residues = {}
    first_model = None
    for row_number, row in enumerate(rows, start=1):
        group = _value(row, "_atom_site.group_PDB", default="ATOM").upper()
        if group not in {"ATOM", "HETATM"}:
            continue
        model = _value(row, "_atom_site.pdbx_PDB_model_num", default="1")
        if first_model is None:
            first_model = model
        if model != first_model:
            continue
        atom_name = _value(row, "_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        altloc = _value(row, "_atom_site.label_alt_id", default=".")
        chain_id = _value(
            row,
            "_atom_site.auth_asym_id",
            "_atom_site.label_asym_id",
            default="",
        )
        residue_name = _value(
            row,
            "_atom_site.auth_comp_id",
            "_atom_site.label_comp_id",
        ).upper()
        if atom_name not in BACKBONE_ATOMS or altloc not in {".", "?", "A"}:
            continue
        if chain is not None and chain_id != chain:
            continue
        if residue_name not in THREE_TO_ONE:
            continue
        residue_id = _value(
            row,
            "_atom_site.auth_seq_id",
            "_atom_site.label_seq_id",
        )
        insertion = _value(row, "_atom_site.pdbx_PDB_ins_code", default="")
        try:
            xyz = [
                float(_value(row, "_atom_site.Cartn_x")),
                float(_value(row, "_atom_site.Cartn_y")),
                float(_value(row, "_atom_site.Cartn_z")),
            ]
        except ValueError as exc:
            raise ValueError(
                f"Invalid mmCIF coordinate in atom_site row {row_number}: {path}"
            ) from exc
        key = (chain_id, residue_id, insertion)
        entry = residues.setdefault(key, {"name": residue_name, "atoms": {}})
        entry["atoms"].setdefault(atom_name, xyz)
    return make_backbone_record(
        residues,
        identifier=Path(path).stem,
        source_path=path,
        chain=chain,
    )


def _read_atom_site_rows(path):
    lexer = shlex.shlex(path.read_text(encoding="utf-8", errors="replace"), posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    index = 0
    while index < len(tokens):
        if tokens[index].casefold() != "loop_":
            index += 1
            continue
        index += 1
        headers = []
        while index < len(tokens) and tokens[index].startswith("_"):
            headers.append(tokens[index])
            index += 1
        if not headers:
            continue
        width = len(headers)
        values = []
        while index < len(tokens):
            token = tokens[index]
            if len(values) % width == 0 and _is_control_token(token):
                break
            values.append(token)
            index += 1
        if not any(header.startswith("_atom_site.") for header in headers):
            continue
        if len(values) % width:
            raise ValueError(f"Incomplete atom_site loop in mmCIF file: {path}")
        return [
            dict(zip(headers, values[offset : offset + width]))
            for offset in range(0, len(values), width)
        ]
    raise ValueError(f"mmCIF file does not contain an atom_site loop: {path}")


def _is_control_token(token):
    lowered = token.casefold()
    return token.startswith("_") or lowered == "loop_" or lowered == "stop_" or lowered.startswith(
        ("data_", "save_", "global_")
    )


def _value(row, *keys, default=None):
    for key in keys:
        value = row.get(key)
        if value not in {None, ".", "?"}:
            return value
    if default is not None:
        return default
    raise ValueError(f"mmCIF atom_site loop is missing required columns {keys!r}.")


__all__ = ["parse_mmcif"]
