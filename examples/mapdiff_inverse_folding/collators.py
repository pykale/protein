"""Pure batching for already-prepared MapDiff structure features."""

from __future__ import annotations

import torch


REQUIRED_FIELDS = {
    "target_residue_one_hot",
    "target_tokens",
    "ca_coordinates",
    "extra_residue_features",
    "edge_index",
    "edge_features",
    "secondary_structure",
    "ipa_atom_positions",
    "reference_sequence",
    "sample_id",
}


class MapDiffCollator:
    """Concatenate sparse graphs and pad IPA tensors without featurizing data."""

    def __init__(self, config=None, stage="diffusion", **kwargs):
        if stage not in {"diffusion", "ipa"}:
            raise ValueError("MapDiff collator stage must be 'diffusion' or 'ipa'.")
        self.stage = stage

    def __call__(self, samples, **context):
        samples = list(samples)
        if not samples:
            raise ValueError("MapDiff cannot collate an empty sample list.")
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(
                    "MapDiffCollator expects preprocessed dictionaries, got "
                    f"{type(sample).__name__} at index {index}."
                )
            missing = sorted(REQUIRED_FIELDS - sample.keys())
            if missing:
                raise ValueError(
                    f"MapDiff preprocessed sample {index} is missing {missing}."
                )

        lengths = [
            int(sample["target_residue_one_hot"].shape[0]) for sample in samples
        ]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        batch_size = len(samples)
        max_length = max(lengths)
        ipa_residue_one_hot = torch.zeros(batch_size, max_length, 20)
        ipa_atom_positions = torch.zeros(batch_size, max_length, 5, 3)
        ipa_sequence_mask = torch.zeros(
            batch_size, max_length, dtype=torch.bool
        )
        ipa_target_tokens = torch.full(
            (batch_size, max_length), 20, dtype=torch.long
        )
        for index, (sample, length) in enumerate(zip(samples, lengths)):
            ipa_residue_one_hot[index, :length] = sample[
                "target_residue_one_hot"
            ]
            ipa_atom_positions[index, :length] = sample["ipa_atom_positions"]
            ipa_sequence_mask[index, :length] = True
            ipa_target_tokens[index, :length] = sample["target_tokens"]

        return {
            "training_stage": self.stage,
            "target_residue_one_hot": torch.cat(
                [sample["target_residue_one_hot"] for sample in samples]
            ),
            "target_tokens": torch.cat(
                [sample["target_tokens"] for sample in samples]
            ),
            "ca_coordinates": torch.cat(
                [sample["ca_coordinates"] for sample in samples]
            ),
            "extra_residue_features": torch.cat(
                [sample["extra_residue_features"] for sample in samples]
            ),
            "edge_index": torch.cat(
                [
                    sample["edge_index"] + offset
                    for sample, offset in zip(samples, offsets)
                ],
                dim=1,
            ),
            "edge_features": torch.cat(
                [sample["edge_features"] for sample in samples]
            ),
            "secondary_structure": torch.cat(
                [sample["secondary_structure"] for sample in samples]
            ),
            "graph_index": torch.cat(
                [
                    torch.full((length,), index, dtype=torch.long)
                    for index, length in enumerate(lengths)
                ]
            ),
            "graph_ptr": torch.tensor(offsets, dtype=torch.long),
            "ipa_residue_one_hot": ipa_residue_one_hot,
            "ipa_atom_positions": ipa_atom_positions,
            "ipa_sequence_mask": ipa_sequence_mask,
            "ipa_target_tokens": ipa_target_tokens,
            "reference_sequences": [
                sample["reference_sequence"] for sample in samples
            ],
            "sample_ids": [sample["sample_id"] for sample in samples],
        }


__all__ = ["MapDiffCollator"]
