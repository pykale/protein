"""Dependency-free binary classification metrics for DTI models."""

from __future__ import annotations

import math

import torch

from kale_protein.registry import EVALUATOR_REGISTRY


class MetricUndefinedError(ValueError):
    """Raised when a metric is not defined for the supplied labels."""


def _flatten_numbers(value):
    if torch.is_tensor(value):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, dict):
        for key in ("probabilities", "labels", "label"):
            if key in value:
                return _flatten_numbers(value[key])
        raise ValueError(f"Could not find probabilities or labels in keys {tuple(value)}")
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    return [float(value)]


def _labels(data):
    if isinstance(data, dict) and "label" in data:
        return _flatten_numbers(data["label"])
    if isinstance(data, (list, tuple)) and data and all(
        isinstance(item, dict) and "label" in item for item in data
    ):
        return [number for item in data for number in _flatten_numbers(item["label"])]
    return _flatten_numbers(data)


def _probabilities(outputs):
    if isinstance(outputs, dict):
        return _flatten_numbers(outputs["probabilities"])
    if isinstance(outputs, (list, tuple)) and outputs and all(isinstance(item, dict) for item in outputs):
        return [number for item in outputs for number in _flatten_numbers(item["probabilities"])]
    return _flatten_numbers(outputs)


def _validate(labels, probabilities):
    labels = [int(label) for label in labels]
    probabilities = [float(probability) for probability in probabilities]
    if not labels:
        raise ValueError("Metrics require at least one example")
    if len(labels) != len(probabilities):
        raise ValueError(
            f"Labels and probabilities have different lengths: {len(labels)} != {len(probabilities)}"
        )
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Binary DTI metrics require labels containing only 0 and 1")
    if any(not math.isfinite(probability) for probability in probabilities):
        raise ValueError("Probabilities must all be finite")
    return labels, probabilities


def accuracy_score(labels, probabilities, threshold=0.5):
    labels, probabilities = _validate(labels, probabilities)
    predictions = [int(probability >= threshold) for probability in probabilities]
    return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)


def f1_score(labels, probabilities, threshold=0.5):
    labels, probabilities = _validate(labels, probabilities)
    predictions = [int(probability >= threshold) for probability in probabilities]
    true_positives = sum(prediction == label == 1 for prediction, label in zip(predictions, labels))
    false_positives = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels))
    false_negatives = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels))
    denominator = 2 * true_positives + false_positives + false_negatives
    return 0.0 if denominator == 0 else 2 * true_positives / denominator


def auroc_score(labels, probabilities):
    labels, probabilities = _validate(labels, probabilities)
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise MetricUndefinedError("AUROC is undefined when labels contain only one class")

    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def auprc_score(labels, probabilities):
    labels, probabilities = _validate(labels, probabilities)
    positive_count = sum(labels)
    if positive_count == 0:
        raise MetricUndefinedError("AUPRC is undefined when labels contain no positive examples")

    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True)
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


def optimal_f1_threshold(labels, probabilities):
    labels, probabilities = _validate(labels, probabilities)
    candidates = sorted(set(probabilities), reverse=True)
    candidates.append(math.nextafter(min(candidates), -math.inf))
    return max(candidates, key=lambda threshold: (f1_score(labels, probabilities, threshold), threshold))


def compute_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = _validate(_flatten_numbers(labels), _flatten_numbers(probabilities))
    if threshold is None:
        threshold = optimal_f1_threshold(labels, probabilities)
    return {
        "auroc": auroc_score(labels, probabilities),
        "auprc": auprc_score(labels, probabilities),
        "f1": f1_score(labels, probabilities, threshold),
        "accuracy": accuracy_score(labels, probabilities, threshold),
        "threshold": float(threshold),
    }


class Accuracy:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, outputs, data):
        return accuracy_score(_labels(data), _probabilities(outputs), self.threshold)


class F1:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, outputs, data):
        return f1_score(_labels(data), _probabilities(outputs), self.threshold)


class AUROC:
    def __call__(self, outputs, data):
        return auroc_score(_labels(data), _probabilities(outputs))


class AUPRC:
    def __call__(self, outputs, data):
        return auprc_score(_labels(data), _probabilities(outputs))


class Threshold:
    def __call__(self, outputs, data):
        return optimal_f1_threshold(_labels(data), _probabilities(outputs))


for name, metric in (
    ("accuracy", Accuracy), ("f1", F1), ("auroc", AUROC),
    ("auprc", AUPRC), ("threshold", Threshold),
):
    EVALUATOR_REGISTRY.register(("drug_target_interaction", name), metric)
