"""Dataset scaffolding for task-level data logic.

Task-level pair construction, splitting, and dataset loading should live here
rather than inside modality processors.
"""

from kale_protein.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DTI/PDBBind")
def load_pdbbind_example():
    return (
        {"id": "sample_1", "smiles": "CCO", "sequence": "MKTFFVLLL"},
        1,
    )
