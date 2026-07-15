"""Task-metric dispatch for model-card evaluations."""

import importlib

from kale_protein.core.registry import EVALUATOR_REGISTRY


class AutoProteinEvaluator:
    @classmethod
    def from_config(cls, config):
        return MultiMetricEvaluator(config)


class MultiMetricEvaluator:
    def __init__(self, config):
        self.config = config
        task = config["task"]
        metric_names = config.get("evaluation", {}).get("metrics", [])
        if any(not EVALUATOR_REGISTRY.has((task, name)) for name in metric_names):
            _import_task_module(task, "metrics")
        self.metrics = [(name, EVALUATOR_REGISTRY.get((task, name))()) for name in metric_names]

    def evaluate(self, outputs=None, data=None, labels=None, **prediction):
        """Evaluate a prediction mapping passed directly or expanded with ``**``."""

        if outputs is not None:
            if prediction:
                raise TypeError("Pass evaluator outputs either as a mapping or as keyword fields, not both.")
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


def _import_task_module(task, module):
    if not isinstance(task, str) or not task.isidentifier():
        raise ValueError(f"Task names used for Auto discovery must be identifiers; got {task!r}.")
    try:
        importlib.import_module(f"kale_protein.core.evaluation.tasks.{task}.{module}")
    except ModuleNotFoundError as error:
        if error.name == f"kale_protein.core.evaluation.tasks.{task}":
            return
        raise
