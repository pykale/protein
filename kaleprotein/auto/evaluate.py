"""Task-metric dispatch for model-card evaluations."""

import importlib

from .registry import EVALUATOR_REGISTRY


class AutoProteinEvaluator:
    @classmethod
    def from_config(cls, config):
        return MultiMetricEvaluator(config)


class MultiMetricEvaluator:
    def __init__(self, config):
        self.config = config
        task = config["task"]
        metric_names = config.get("evaluation", {}).get("metrics", [])
        for name in metric_names:
            if not EVALUATOR_REGISTRY.has((task, name)):
                _import_metric(name)
        self.metrics = [(name, EVALUATOR_REGISTRY.get((task, name))()) for name in metric_names]

    def evaluate(self, outputs=None, data=None, labels=None, **prediction):
        """Evaluate a prediction mapping passed directly or expanded with ``**``."""

        if outputs is not None:
            if prediction:
                raise TypeError(
                    "Pass evaluator outputs either as a mapping or as keyword "
                    "fields, not both."
                )
            prediction = outputs
        if not isinstance(prediction, dict) or not prediction:
            raise ValueError("Evaluation requires a non-empty prediction mapping.")
        if data is None:
            labels = prediction.get("labels") if labels is None else labels
            references = prediction.get("reference_sequences")
            if labels is not None:
                data = {"label": labels}
            elif references is not None:
                data = references
            else:
                raise ValueError(
                    "Evaluation requires data, labels, or reference_sequences "
                    "alongside prediction fields."
                )
        return {name: metric(prediction, data) for name, metric in self.metrics}

    __call__ = evaluate


def _import_metric(metric_name):
    if not isinstance(metric_name, str) or not metric_name.isidentifier():
        raise ValueError(
            "Metric names used for Auto discovery must be identifiers; "
            f"got {metric_name!r}."
        )
    module_name = f"kaleprotein.evaluate.{metric_name}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return
        raise
