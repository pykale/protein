"""Optimal binary classification threshold selection."""

import math

from kaleprotein.auto.registry import EVALUATOR_REGISTRY
from kaleprotein.utils import extract_binary_inputs, validate_binary_inputs
from .f1 import f1_score


def optimal_f1_threshold(labels, probabilities):
    labels, probabilities = validate_binary_inputs(labels, probabilities)
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
        labels, probabilities = extract_binary_inputs(outputs, data)
        return optimal_f1_threshold(labels, probabilities)


EVALUATOR_REGISTRY.register(("dti", "threshold"), Threshold)

__all__ = ["Threshold", "optimal_f1_threshold"]
