"""Sequence diversity for generative predictions."""

from itertools import combinations

from kaleprotein.auto.registry import EVALUATOR_REGISTRY

from ._sequence import sequences_from


class Diversity:
    """Mean normalized pairwise Hamming distance among generated sequences."""

    def __call__(self, outputs, data=None):
        sequences = sequences_from(outputs)
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
