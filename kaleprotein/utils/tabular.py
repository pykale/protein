"""Dependency-free CSV and TSV readers."""

import csv
from pathlib import Path

from .files import require_file


def read_dict_rows(path, *, delimiter=",", encoding="utf-8"):
    path = require_file(path, description="table file")
    with Path(path).open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Table has no header: {path}")
        return list(reader.fieldnames), list(reader)


def read_csv(path, *, encoding="utf-8"):
    return read_dict_rows(path, delimiter=",", encoding=encoding)[1]


def read_tsv(path, *, encoding="utf-8"):
    return read_dict_rows(path, delimiter="\t", encoding=encoding)[1]


__all__ = ["read_csv", "read_dict_rows", "read_tsv"]
