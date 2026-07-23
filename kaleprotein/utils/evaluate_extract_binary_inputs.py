"""Extract and validate binary-classification metric inputs."""

import math

class MetricUndefinedError(ValueError):
    """Raised when a metric is not defined for the supplied labels."""


def _flatten_numbers(value):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, dict):
        for key in ("probabilities", "labels", "label"):
            if key in value:
                return _flatten_numbers(value[key])
        raise ValueError(
            f"Could not find probabilities or labels in keys {tuple(value)}"
        )
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    return [float(value)]


def _labels_from(data):
    if isinstance(data, dict) and "label" in data:
        return _flatten_numbers(data["label"])
    if isinstance(data, (list, tuple)) and data and all(
        isinstance(item, dict) and "label" in item for item in data
    ):
        return [
            number
            for item in data
            for number in _flatten_numbers(item["label"])
        ]
    return _flatten_numbers(data)


def _probabilities_from(outputs):
    if isinstance(outputs, dict):
        return _flatten_numbers(outputs["probabilities"])
    if (
        isinstance(outputs, (list, tuple))
        and outputs
        and all(isinstance(item, dict) for item in outputs)
    ):
        return [
            number
            for item in outputs
            for number in _flatten_numbers(item["probabilities"])
        ]
    return _flatten_numbers(outputs)


def validate_binary_inputs(labels, probabilities):
    labels = [int(label) for label in _flatten_numbers(labels)]
    probabilities = [
        float(probability) for probability in _flatten_numbers(probabilities)
    ]
    if not labels:
        raise ValueError("Metrics require at least one example")
    if len(labels) != len(probabilities):
        raise ValueError(
            "Labels and probabilities have different lengths: "
            f"{len(labels)} != {len(probabilities)}"
        )
    if any(label not in (0, 1) for label in labels):
        raise ValueError(
            "Binary DTI metrics require labels containing only 0 and 1"
        )
    if any(not math.isfinite(probability) for probability in probabilities):
        raise ValueError("Probabilities must all be finite")
    return labels, probabilities


def extract_binary_inputs(outputs, data):
    """Return validated labels and probabilities from metric stage mappings."""
    return validate_binary_inputs(
        _labels_from(data),
        _probabilities_from(outputs),
    )


__all__ = [
    "MetricUndefinedError",
    "extract_binary_inputs",
    "validate_binary_inputs",
]
