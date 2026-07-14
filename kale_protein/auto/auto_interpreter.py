"""Task-interpreter dispatch with lazy task registration."""

from __future__ import annotations

from kale_protein.registry import INTERPRETER_REGISTRY

from .auto_evaluator import _import_task_module


class AutoProteinInterpreter:
    @classmethod
    def from_config(cls, config):
        task = config["task"]
        method = config["interpretation"]["method"]
        key = (task, method)
        if not INTERPRETER_REGISTRY.has(key):
            _import_task_module(task, "interpreters")
        return INTERPRETER_REGISTRY.get(key)(config)
