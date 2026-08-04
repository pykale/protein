"""MapDiff-specific per-structure feature preparation.

Adapted from MapDiff, copyright 2024 Peizhen Bai, under the MIT License.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from kaleprotein.loaddata.records import AMINO_ACID_ALPHABET
from kaleprotein.prepdata.structure import BackboneCoordinateProcessor
from kaleprotein.utils import (
    build_residue_graph,
    compute_backbone_angle_features,
    compute_edge_orientations,
    compute_neighbor_direction_features,
    place_virtual_cb,
)


STANDARD_ALPHABET = AMINO_ACID_ALPHABET
MAPDIFF_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
STANDARD_TO_MAPDIFF = [
    MAPDIFF_ALPHABET.index(amino_acid) for amino_acid in STANDARD_ALPHABET
]
AA_TO_INDEX = {
    amino_acid: index for index, amino_acid in enumerate(STANDARD_ALPHABET)
}


class MapDiffPreprocessor:
    """Prepare one structure at a time for the published MapDiff feature schema."""

    def __init__(self, config, **kwargs):
        self.config = config
        stream = config.get_streams()["structure"]
        self.backbone_processor = BackboneCoordinateProcessor(
            input_key=stream.input_key,
            **stream.processor_kwargs,
        )
        settings = config.get("preprocessing", {})
        self.cutoff = float(settings.get("cutoff", 30.0))
        self.max_neighbors = int(settings.get("max_neighbors", 10))

    def transform_sample(self, sample):
        raw_sample = _unwrap_sample(sample)
        normalized = self.backbone_processor.transform(
            _backbone_processor_input(raw_sample)
        )
        backbone, backbone_mask = _select_backbone(
            normalized["atom_pos"], normalized["atom_mask"]
        )
        raw_features = _raw_features(raw_sample)
        sequence = str(normalized["sequence"] or "")
        (
            backbone,
            backbone_mask,
            sequence,
            raw_features,
        ) = _filter_incomplete_residues(
            backbone,
            backbone_mask,
            sequence,
            raw_features,
        )

        target_residue_one_hot = _target_one_hot(
            sequence,
            backbone.shape[0],
            raw_features.get("x"),
        )
        graph = _graph_features(
            backbone,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            raw_features=raw_features,
        )
        ipa_atom_positions = _ipa_atom_positions(
            backbone, raw_features.get("atom_pos")
        )
        if not sequence and target_residue_one_hot.numel():
            sequence = "".join(
                STANDARD_ALPHABET[index]
                for index in target_residue_one_hot.argmax(dim=-1).tolist()
            )

        return {
            "target_residue_one_hot": target_residue_one_hot,
            "target_tokens": target_residue_one_hot.argmax(dim=-1),
            "ca_coordinates": graph["ca_coordinates"],
            "backbone_mask": backbone_mask,
            "extra_residue_features": graph["extra_residue_features"],
            "edge_index": graph["edge_index"],
            "edge_features": graph["edge_features"],
            "secondary_structure": graph["secondary_structure"],
            "ipa_atom_positions": ipa_atom_positions,
            "reference_sequence": sequence[: backbone.shape[0]],
            "sample_id": str(normalized["identifier"]),
        }

    def process(self, dataset):
        return {
            "samples": [self.transform_sample(sample) for sample in dataset]
        }


def _unwrap_sample(sample):
    value = _get(sample, "structure", sample)
    return _get(value, "graph", value)


def _backbone_processor_input(sample):
    if not isinstance(sample, dict):
        return sample
    if "backbone_coords" in sample or "pdb_path" in sample:
        return sample
    for key in ("atom_pos", "coords"):
        if key in sample:
            value = dict(sample)
            value["backbone_coords"] = sample[key]
            return value
    return sample


def _select_backbone(atom_pos, atom_mask):
    atom_pos = torch.as_tensor(atom_pos, dtype=torch.float32)
    atom_mask = torch.as_tensor(atom_mask, dtype=torch.bool)
    if atom_pos.shape[1] >= 5:
        atom_pos = atom_pos[:, [0, 1, 2, 4]]
        atom_mask = atom_mask[:, [0, 1, 2, 4]]
    elif atom_pos.shape[1] > 4:
        atom_pos = atom_pos[:, :4]
        atom_mask = atom_mask[:, :4]
    elif atom_pos.shape[1] < 4:
        missing = 4 - atom_pos.shape[1]
        atom_pos = torch.cat(
            [atom_pos, atom_pos[:, -1:].expand(-1, missing, -1)], dim=1
        )
        atom_mask = torch.cat(
            [
                atom_mask,
                torch.zeros(
                    atom_mask.shape[0],
                    missing,
                    dtype=torch.bool,
                    device=atom_mask.device,
                ),
            ],
            dim=1,
        )
    if atom_pos.shape[0] == 0:
        raise ValueError("MapDiff cannot preprocess a structure with zero residues.")
    if atom_mask.shape != atom_pos.shape[:2]:
        raise ValueError(
            "MapDiff atom_mask must match the residue and atom dimensions of "
            f"atom_pos, got {tuple(atom_mask.shape)} and {tuple(atom_pos.shape)}."
        )
    return atom_pos, atom_mask


def _filter_incomplete_residues(
    backbone,
    backbone_mask,
    sequence,
    raw_features,
):
    length = backbone.shape[0]
    if sequence and len(sequence) != length:
        raise ValueError(
            "MapDiff sequence length must match the structure length before "
            f"filtering, got {len(sequence)} and {length}."
        )
    keep = backbone_mask[:, :3].all(dim=-1)
    keep &= torch.isfinite(backbone[:, :3]).all(dim=(1, 2))
    if not keep.any():
        raise ValueError(
            "MapDiff requires at least one residue with complete N, CA, and C "
            "coordinates."
        )
    if keep.all():
        return (
            torch.nan_to_num(backbone),
            backbone_mask,
            sequence,
            raw_features,
        )

    indices = keep.nonzero(as_tuple=False).flatten().tolist()
    filtered_sequence = (
        "".join(sequence[index] for index in indices) if sequence else ""
    )
    return (
        torch.nan_to_num(backbone[keep]),
        backbone_mask[keep],
        filtered_sequence,
        _filter_raw_features(raw_features, keep),
    )


def _filter_raw_features(raw_features, keep):
    filtered = dict(raw_features)
    length = keep.shape[0]
    for key in (
        "x",
        "extra_x",
        "pos",
        "atom_pos",
        "ss",
        "mu_r_norm",
    ):
        if key not in filtered:
            continue
        value = torch.as_tensor(filtered[key])
        if value.ndim == 0 or value.shape[0] != length:
            raise ValueError(
                f"MapDiff feature {key!r} must have {length} residue rows."
            )
        filtered[key] = value[keep]

    local_scalars = None
    if "extra_x" in filtered and filtered["extra_x"].ndim == 2:
        if filtered["extra_x"].shape[1] >= 2:
            local_scalars = filtered["extra_x"][:, :2]
    if local_scalars is None and (
        "x" in filtered
        and filtered["x"].ndim == 2
        and filtered["x"].shape[1] >= 22
    ):
        local_scalars = filtered["x"][:, 20:22]
    if "x" in filtered and filtered["x"].ndim == 2:
        filtered["x"] = filtered["x"][:, :20]
    if local_scalars is None:
        filtered.pop("extra_x", None)
    else:
        filtered["extra_x"] = local_scalars

    # Graph topology, dihedrals, and neighborhood statistics change when
    # residues are removed and must be recomputed from the filtered backbone.
    filtered.pop("edge_index", None)
    filtered.pop("edge_attr", None)
    filtered.pop("mu_r_norm", None)
    return filtered


def _raw_features(sample):
    metadata = _get(sample, "metadata", {}) or {}
    values = {}
    for key in (
        "x",
        "extra_x",
        "pos",
        "atom_pos",
        "edge_index",
        "edge_attr",
        "ss",
        "mu_r_norm",
    ):
        value = _get(sample, key)
        metadata_value = (
            metadata.get(key) if isinstance(metadata, dict) else None
        )
        if key == "atom_pos" and metadata_value is not None:
            metadata_atoms = torch.as_tensor(metadata_value)
            if (
                metadata_atoms.ndim == 3
                and metadata_atoms.shape[1] >= 5
                and metadata_atoms.shape[2] == 3
            ):
                value = metadata_value
        elif value is None:
            value = metadata_value
        if value is not None:
            values[key] = value
    return values


def _target_one_hot(sequence, length, raw_x=None):
    if sequence:
        one_hot = torch.zeros(length, 20, dtype=torch.float32)
        for position, amino_acid in enumerate(str(sequence)[:length]):
            index = AA_TO_INDEX.get(amino_acid.upper())
            if index is not None:
                one_hot[position, index] = 1.0
        return one_hot

    if raw_x is not None:
        raw_x = torch.as_tensor(raw_x, dtype=torch.float32)
        if raw_x.ndim == 2 and raw_x.shape[0] == length and raw_x.shape[1] >= 20:
            # Original CATH/MapDiff tensors use ARNDCQEGHILKMFPSTWYV order.
            return raw_x[:, :20][:, STANDARD_TO_MAPDIFF]

    return torch.zeros(length, 20, dtype=torch.float32)


def _graph_features(backbone, *, cutoff, max_neighbors, raw_features):
    length = backbone.shape[0]
    raw_extra = raw_features.get("extra_x")
    raw_x = raw_features.get("x")
    raw_mu = raw_features.get("mu_r_norm")
    if raw_extra is None and raw_x is not None:
        raw_x = torch.as_tensor(raw_x, dtype=torch.float32)
        if raw_x.ndim == 2 and raw_x.shape[1] > 20:
            raw_extra = raw_x[:, 20:]
            if raw_mu is not None:
                raw_extra = torch.cat(
                    [raw_extra, torch.as_tensor(raw_mu, dtype=torch.float32)],
                    dim=-1,
                )

    edge_index = raw_features.get("edge_index")
    edge_features = raw_features.get("edge_attr")
    secondary_structure = raw_features.get("ss")
    if (
        edge_index is not None
        and edge_features is not None
        and torch.as_tensor(edge_features).shape[-1] == 93
    ):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_features = torch.as_tensor(edge_features, dtype=torch.float32)
    else:
        graph = build_residue_graph(
            backbone[:, 1],
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
        edge_index = graph["edge_index"]
        edge_distances = graph["edge_distances"]
        source, target = edge_index
        sequence_distance = (source - target).abs().clamp_max(64)
        sequence_features = F.one_hot(sequence_distance, 65).float()
        scales = backbone.new_tensor([1.5**value for value in range(15)])
        distance_features = torch.exp(
            -((edge_distances[:, None] / 4).square()) / scales
        )
        contact_features = (edge_distances <= 8).float()[:, None]
        orientation_features = compute_edge_orientations(backbone, edge_index)
        edge_features = torch.cat(
            [
                sequence_features,
                distance_features,
                contact_features,
                orientation_features,
            ],
            dim=-1,
        )

    if raw_extra is not None and torch.as_tensor(raw_extra).shape == (length, 11):
        extra_residue_features = torch.as_tensor(
            raw_extra, dtype=torch.float32
        )
    else:
        if "edge_distances" not in locals():
            source, target = edge_index
            edge_distances = torch.linalg.vector_norm(
                backbone[source, 1] - backbone[target, 1], dim=-1
            )
        neighborhood = compute_neighbor_direction_features(
            backbone[:, 1], edge_index, edge_distances
        )
        local_scalars = backbone.new_zeros(length, 2)
        if raw_extra is not None:
            raw_extra = torch.as_tensor(raw_extra, dtype=torch.float32)
            if raw_extra.ndim == 2 and raw_extra.shape[0] == length:
                local_scalars[:, : min(raw_extra.shape[1], 2)] = raw_extra[
                    :, :2
                ]
        extra_residue_features = torch.cat(
            [
                local_scalars,
                compute_backbone_angle_features(backbone),
                neighborhood,
            ],
            dim=-1,
        )

    if (
        secondary_structure is not None
        and torch.as_tensor(secondary_structure).shape == (length, 8)
    ):
        secondary_structure = torch.as_tensor(
            secondary_structure, dtype=torch.float32
        )
    else:
        secondary_structure = backbone.new_zeros(length, 8)

    return {
        "ca_coordinates": backbone[:, 1],
        "extra_residue_features": extra_residue_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "secondary_structure": secondary_structure,
    }


def _ipa_atom_positions(backbone, raw_atom_pos=None):
    if raw_atom_pos is not None:
        raw_atom_pos = torch.as_tensor(raw_atom_pos, dtype=torch.float32)
        if (
            raw_atom_pos.ndim == 3
            and raw_atom_pos.shape[0] == backbone.shape[0]
            and raw_atom_pos.shape[1] >= 5
            and raw_atom_pos.shape[2] == 3
        ):
            return torch.nan_to_num(raw_atom_pos[:, :5])
    return torch.stack(
        [
            backbone[:, 0],
            backbone[:, 1],
            backbone[:, 2],
            place_virtual_cb(backbone),
            backbone[:, 3],
        ],
        dim=1,
    )


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


__all__ = ["MapDiffPreprocessor"]
