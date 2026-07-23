"""Area under the receiver operating characteristic curve."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY
from kaleprotein.utils import (
    MetricUndefinedError,
    extract_binary_inputs,
    validate_binary_inputs,
)


def auroc_score(labels, probabilities):
    labels, probabilities = validate_binary_inputs(labels, probabilities)
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise MetricUndefinedError(
            "AUROC is undefined when labels contain only one class"
        )

    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


class AUROC:
    def __call__(self, outputs, data):
        labels, probabilities = extract_binary_inputs(outputs, data)
        return auroc_score(labels, probabilities)


EVALUATOR_REGISTRY.register(("dti", "auroc"), AUROC)

__all__ = ["AUROC", "auroc_score"]
