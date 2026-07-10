"""Dataset scaffolding for task-level data logic.

Task-level pair construction, splitting, and dataset loading should live here
rather than inside modality processors.
"""

from kale_protein.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("InverseFolding/CATH")
def load_cath_example():
    return (
        {
            "id": "protein_1",
            "backbone_coords": [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
            "sequence": "MA",
        },
        "MA",
    )
