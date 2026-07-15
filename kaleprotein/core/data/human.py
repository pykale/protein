"""Human DTI dataset adapters."""

from kaleprotein.core.registry import DATASET_REGISTRY

from .base import DTICsvDataset


@DATASET_REGISTRY.register("Human/DTI")
class HumanDTIDataset(DTICsvDataset):
    dataset_id = "Human/DTI"
    directory_name = "human"


__all__ = ["HumanDTIDataset"]
