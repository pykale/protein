"""Area under the receiver operating characteristic curve."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY

from ._binary import (
    MetricUndefinedError,
    labels_from,
    probabilities_from,
    validate_binary,
)


def auroc_score(labels, probabilities):
    labels, probabilities = validate_binary(labels, probabilities)
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
        return auroc_score(labels_from(data), probabilities_from(outputs))


EVALUATOR_REGISTRY.register(("dti", "auroc"), AUROC)

__all__ = ["AUROC", "auroc_score"]
