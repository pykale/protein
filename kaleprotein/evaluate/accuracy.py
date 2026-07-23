"""Binary classification accuracy."""

from kaleprotein.auto.registry import EVALUATOR_REGISTRY

from ._binary import labels_from, probabilities_from, validate_binary


def accuracy_score(labels, probabilities, threshold=0.5):
    labels, probabilities = validate_binary(labels, probabilities)
    predictions = [
        int(probability >= threshold) for probability in probabilities
    ]
    return sum(
        prediction == label
        for prediction, label in zip(predictions, labels)
    ) / len(labels)


class Accuracy:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, outputs, data):
        return accuracy_score(
            labels_from(data), probabilities_from(outputs), self.threshold
        )


EVALUATOR_REGISTRY.register(("dti", "accuracy"), Accuracy)

__all__ = ["Accuracy", "accuracy_score"]
