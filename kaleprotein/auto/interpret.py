"""Task-interpreter dispatch with lazy task registration."""

import importlib

from .registry import INTERPRETER_REGISTRY


class AutoProteinInterpreter:
    @classmethod
    def from_config(cls, config):
        task = config["task"]
        method = config["interpretation"]["method"]
        key = (task, method)
        if not INTERPRETER_REGISTRY.has(key):
            _import_interpreter(method)
        return INTERPRETER_REGISTRY.get(key)(config)


def _import_interpreter(method):
    if not isinstance(method, str) or not method.isidentifier():
        raise ValueError(
            "Interpreter names used for Auto discovery must be identifiers; "
            f"got {method!r}."
        )
    module_name = f"kaleprotein.interpret.{method}"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return
        raise
