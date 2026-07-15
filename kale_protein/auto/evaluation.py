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

    def evaluate(self, outputs, data):
        return {name: metric(outputs, data) for name, metric in self.metrics}


def _import_task_module(task, module):
    if not isinstance(task, str) or not task.isidentifier():
        raise ValueError(f"Task names used for Auto discovery must be identifiers; got {task!r}.")
    try:
        importlib.import_module(f"kale_protein.core.tasks.{task}.{module}")
    except ModuleNotFoundError as error:
        if error.name == f"kale_protein.core.tasks.{task}":
            return
        raise
