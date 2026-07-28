"""Sequence novelty for generative predictions."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY
from kaleprotein.utils import extract_sequences


class Novelty:
    def __call__(self, outputs, data):
        generated = extract_sequences(outputs)
        known = set(extract_sequences(data))
        return sum(
            sequence not in known for sequence in generated
        ) / max(len(generated), 1)


EVALUATOR_REGISTRY.register(("inverse_folding", "novelty"), Novelty)

__all__ = ["Novelty"]
