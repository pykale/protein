"""Sequence novelty for generative predictions."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY

from ._sequence import sequences_from


class Novelty:
    def __call__(self, outputs, data):
        generated = sequences_from(outputs)
        known = set(sequences_from(data))
        return sum(
            sequence not in known for sequence in generated
        ) / max(len(generated), 1)


EVALUATOR_REGISTRY.register(("inverse_folding", "novelty"), Novelty)

__all__ = ["Novelty"]
