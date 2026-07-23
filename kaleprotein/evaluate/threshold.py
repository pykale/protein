"""Optimal binary classification threshold selection."""

import math

from kaleprotein.auto.registry import EVALUATOR_REGISTRY

from ._binary import labels_from, probabilities_from, validate_binary
from .f1 import f1_score


def optimal_f1_threshold(labels, probabilities):
    labels, probabilities = validate_binary(labels, probabilities)
    candidates = sorted(set(probabilities), reverse=True)
    candidates.append(math.nextafter(min(candidates), -math.inf))
    return max(
        candidates,
        key=lambda threshold: (
            f1_score(labels, probabilities, threshold),
            threshold,
        ),
    )


class Threshold:
    def __call__(self, outputs, data):
        return optimal_f1_threshold(
            labels_from(data), probabilities_from(outputs)
        )


EVALUATOR_REGISTRY.register(("dti", "threshold"), Threshold)

__all__ = ["Threshold", "optimal_f1_threshold"]
