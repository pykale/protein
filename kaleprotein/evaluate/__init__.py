"""Reusable metrics, exposed as first-level metric modules."""

from kaleprotein.utils import MetricUndefinedError, validate_binary_inputs


def compute_metrics(labels, probabilities, threshold=0.5):
    """Compute the standard binary classification metric bundle."""
    from .accuracy import accuracy_score
    from .auprc import auprc_score
    from .auroc import auroc_score
    from .f1 import f1_score
    from .threshold import optimal_f1_threshold

    labels, probabilities = validate_binary_inputs(labels, probabilities)
    if threshold is None:
        threshold = optimal_f1_threshold(labels, probabilities)
    return {
        "auroc": auroc_score(labels, probabilities),
        "auprc": auprc_score(labels, probabilities),
        "f1": f1_score(labels, probabilities, threshold),
        "accuracy": accuracy_score(labels, probabilities, threshold),
        "threshold": float(threshold),
    }


__all__ = ["MetricUndefinedError", "compute_metrics"]
