"""BioSNAP dataset adapters."""

from kaleprotein.core.registry import DATASET_REGISTRY

from .datasets import DTICsvDataset


@DATASET_REGISTRY.register("BioSNAP/DTI")
class BioSNAPDTIDataset(DTICsvDataset):
    dataset_id = "BioSNAP/DTI"
    directory_name = "biosnap"


__all__ = ["BioSNAPDTIDataset"]
