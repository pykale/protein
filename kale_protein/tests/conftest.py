import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fake_rdkit_graph(monkeypatch):
    """Keep Auto pipeline tests independent from the optional RDKit wheel."""

    import torch

    from kale_protein.core.data.modalities.molecule.processors import RDKitGraphProcessor

    def transform(processor, sample):
        smiles = sample[processor.input_key]
        atom_count = max(1, min(len(smiles), processor.max_nodes or len(smiles)))
        features = torch.zeros(atom_count, 74)
        features[:, 0] = 1.0
        mask = torch.ones(atom_count, dtype=torch.bool)
        return {
            "smiles": smiles,
            "node_features": features,
            "adjacency": torch.eye(atom_count),
            "node_mask": mask,
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "atom_symbols": ["C"] * atom_count,
            "num_atoms": atom_count,
        }

    monkeypatch.setattr(RDKitGraphProcessor, "transform", transform)
