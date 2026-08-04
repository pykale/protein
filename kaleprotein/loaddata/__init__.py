"""Built-in datasets, stable records, and fundamental data utilities."""

from importlib import import_module

from .base_dataset import DTICsvDataset, DTIDataset, ListDataset
from .records import DTISample, SequenceRecord, StructureRecord


_DATASET_MODULES = frozenset({"bindingdb", "biosnap", "cath", "human"})


def __getattr__(name):
    if name in _DATASET_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DTICsvDataset",
    "DTIDataset",
    "DTISample",
    "ListDataset",
    "SequenceRecord",
    "StructureRecord",
    "bindingdb",
    "biosnap",
    "cath",
    "human",
]
