"""Sequence diversity for generative predictions."""

from itertools import combinations

from kaleprotein.auto.registry import EVALUATOR_REGISTRY
from kaleprotein.utils import extract_sequences


class Diversity:
    """Mean normalized pairwise Hamming distance among generated sequences."""

    def __call__(self, outputs, data=None):
        sequences = extract_sequences(outputs)
        sample_ids = (
            outputs.get("sample_ids")
            if isinstance(outputs, dict)
            else None
        )
        if sample_ids is None:
            return _mean_pairwise_distance(sequences)
        sample_ids = list(sample_ids)
        if len(sample_ids) != len(sequences):
            raise ValueError(
                "Diversity requires one sample_id per generated sequence."
            )
        groups = {}
        for sample_id, sequence in zip(sample_ids, sequences):
            groups.setdefault(sample_id, []).append(sequence)
        return sum(
            _mean_pairwise_distance(group)
            for group in groups.values()
        ) / max(len(groups), 1)


def _mean_pairwise_distance(sequences):
    if len(sequences) < 2:
        return 0.0
    distances = []
    for first, second in combinations(sequences, 2):
        length = max(len(first), len(second), 1)
        matches = sum(a == b for a, b in zip(first, second))
        distances.append(1.0 - matches / length)
    return sum(distances) / len(distances)


EVALUATOR_REGISTRY.register(("inverse_folding", "diversity"), Diversity)

__all__ = ["Diversity"]
