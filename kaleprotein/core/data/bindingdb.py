"""BindingDB dataset adapters."""

from kaleprotein.core.registry import DATASET_REGISTRY

from .datasets import DTICsvDataset


@DATASET_REGISTRY.register("BindingDB/DTI")
class BindingDBDTIDataset(DTICsvDataset):
    dataset_id = "BindingDB/DTI"
    directory_name = "bindingdb"


__all__ = ["BindingDBDTIDataset"]
