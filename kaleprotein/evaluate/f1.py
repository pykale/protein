"""Binary classification F1 score."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY
from kaleprotein.utils import extract_binary_inputs, validate_binary_inputs


def f1_score(labels, probabilities, threshold=0.5):
    labels, probabilities = validate_binary_inputs(labels, probabilities)
    predictions = [
        int(probability >= threshold) for probability in probabilities
    ]
    true_positives = sum(
        prediction == label == 1
        for prediction, label in zip(predictions, labels)
    )
    false_positives = sum(
        prediction == 1 and label == 0
        for prediction, label in zip(predictions, labels)
    )
    false_negatives = sum(
        prediction == 0 and label == 1
        for prediction, label in zip(predictions, labels)
    )
    denominator = 2 * true_positives + false_positives + false_negatives
    return 0.0 if denominator == 0 else 2 * true_positives / denominator


class F1:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, outputs, data):
        labels, probabilities = extract_binary_inputs(outputs, data)
        return f1_score(labels, probabilities, self.threshold)


EVALUATOR_REGISTRY.register(("dti", "f1"), F1)

__all__ = ["F1", "f1_score"]
