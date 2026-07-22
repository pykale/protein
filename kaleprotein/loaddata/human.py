"""Human DTI dataset adapters."""

from kaleprotein.auto.registry import DATASET_REGISTRY

from .base_dataset import DTICsvDataset


@DATASET_REGISTRY.register("Human/DTI")
class HumanDTIDataset(DTICsvDataset):
    dataset_id = "Human/DTI"
    directory_name = "human"


__all__ = ["HumanDTIDataset"]
