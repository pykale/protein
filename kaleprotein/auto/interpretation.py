"""Task-interpreter dispatch with lazy task registration."""

import importlib

from kaleprotein.core.registry import INTERPRETER_REGISTRY


class AutoProteinInterpreter:
    @classmethod
    def from_config(cls, config):
        task = config["task"]
        method = config["interpretation"]["method"]
        key = (task, method)
        if not INTERPRETER_REGISTRY.has(key):
            _import_task_interpreters(task)
        return INTERPRETER_REGISTRY.get(key)(config)


def _import_task_interpreters(task):
    if not isinstance(task, str) or not task.isidentifier():
        raise ValueError(
            "Task names used for Auto discovery must be identifiers; "
            f"got {task!r}."
        )
    try:
        importlib.import_module(
            f"kaleprotein.core.interpretation.tasks.{task}.interpreters"
        )
    except ModuleNotFoundError as error:
        if error.name == f"kaleprotein.core.interpretation.tasks.{task}":
            return
        raise
