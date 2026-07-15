"""Task-interpreter dispatch with lazy task registration."""

from kale_protein.core.registry import INTERPRETER_REGISTRY

from .evaluation import _import_task_module


class AutoProteinInterpreter:
    @classmethod
    def from_config(cls, config):
        task = config["task"]
        method = config["interpretation"]["method"]
        key = (task, method)
        if not INTERPRETER_REGISTRY.has(key):
            _import_task_module(task, "interpreters")
        return INTERPRETER_REGISTRY.get(key)(config)
