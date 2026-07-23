"""Sequence recovery for inverse-folding predictions."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY
from kaleprotein.utils import extract_sequences


class SequenceRecovery:
    def __call__(self, outputs, data):
        predicted = extract_sequences(outputs)
        native = extract_sequences(data)
        if not predicted or not native:
            raise ValueError(
                "Sequence recovery needs generated and native sequences."
            )
        scores = []
        for index, sequence in enumerate(predicted):
            reference = native[index % len(native)]
            length = min(len(sequence), len(reference))
            recovered = sum(
                first == second
                for first, second in zip(
                    sequence[:length], reference[:length]
                )
            )
            scores.append(recovered / max(len(reference), 1))
        return sum(scores) / len(scores)


EVALUATOR_REGISTRY.register(
    ("inverse_folding", "sequence_recovery"), SequenceRecovery
)

__all__ = ["SequenceRecovery"]
