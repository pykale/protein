"""Area under the precision-recall curve."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY
from kaleprotein.utils import (
    MetricUndefinedError,
    extract_binary_inputs,
    validate_binary_inputs,
)


def auprc_score(labels, probabilities):
    labels, probabilities = validate_binary_inputs(labels, probabilities)
    positive_count = sum(labels)
    if positive_count == 0:
        raise MetricUndefinedError(
            "AUPRC is undefined when labels contain no positive examples"
        )

    ordered = sorted(
        zip(probabilities, labels), key=lambda item: item[0], reverse=True
    )
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        true_positives += sum(label for _, label in ordered[index:end])
        false_positives += sum(1 - label for _, label in ordered[index:end])
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


class AUPRC:
    def __call__(self, outputs, data):
        labels, probabilities = extract_binary_inputs(outputs, data)
        return auprc_score(labels, probabilities)


EVALUATOR_REGISTRY.register(("dti", "auprc"), AUPRC)

__all__ = ["AUPRC", "auprc_score"]
