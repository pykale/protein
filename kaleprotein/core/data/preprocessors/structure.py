"""Protein backbone preprocessing shared by inverse-folding model cards."""

from pathlib import Path

from kaleprotein.core.registry import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register(
    "structure/backbone",
    aliases=("structure/backbone_coordinate_processor", "protein/structure"),
)
class BackboneCoordinateProcessor:
    """Normalize in-memory coordinates or a PDB path to N/CA/C/O features."""

    default_input_key = "backbone_coords"

    def __init__(self, input_key="backbone_coords", chain=None, **kwargs):
        self.input_key = input_key
        self.chain = chain

    def transform(self, sample):
        from kaleprotein.core.data.datasets.inverse_folding import (
            coerce_protein_graph,
            parse_pdb_backbone,
        )

        pdb_path = sample.get("pdb_path") if isinstance(sample, dict) else None
        if pdb_path is None and isinstance(sample, (str, Path)):
            pdb_path = sample
        if pdb_path is not None:
            graph = parse_pdb_backbone(pdb_path, chain=self.chain)
        elif not isinstance(sample, dict):
            graph = coerce_protein_graph(sample)
        else:
            if self.input_key not in sample:
                raise ValueError(
                    f"Expected sample[{self.input_key!r}] coordinates or a pdb_path value."
                )
            graph = coerce_protein_graph(
                {
                    "atom_pos": sample[self.input_key],
                    "sequence": sample.get("sequence", ""),
                    "id": sample.get("id", "protein"),
                }
            )
        return {
            # Keep the legacy generic encoder's list input while exposing the
            # tensor-native graph/atom_pos fields used by inverse-folding models.
            "coords": graph.atom_pos.tolist(),
            "atom_pos": graph.atom_pos,
            "atom_mask": graph.atom_mask,
            "coord_mask": graph.atom_mask.all(dim=-1).tolist(),
            "edge_index": graph.edge_index,
            "edge_attr": graph.edge_attr,
            "identifier": graph.identifier,
            "sequence": graph.sequence,
            "graph": graph,
        }
