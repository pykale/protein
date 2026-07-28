"""Lazy public exports for the Auto API."""

from importlib import import_module


_IMPORT_STRUCTURE = {
    "AutoMoleculePreprocessor": (".prepdata", "AutoMoleculePreprocessor"),
    "AutoProteinCollator": (".loaddata", "AutoProteinCollator"),
    "AutoProteinConfig": (".config", "AutoProteinConfig"),
    "AutoProteinData": (".loaddata", "AutoProteinData"),
    "AutoProteinDataLoader": (".loaddata", "AutoProteinDataLoader"),
    "AutoProteinEmbedder": (".model", "AutoProteinEmbedder"),
    "AutoProteinEvaluator": (".evaluate", "AutoProteinEvaluator"),
    "AutoProteinInterpreter": (".interpret", "AutoProteinInterpreter"),
    "AutoProteinModel": (".model", "AutoProteinModel"),
    "AutoProteinPredictor": (".model", "AutoProteinPredictor"),
    "AutoProteinPreprocessor": (".prepdata", "AutoProteinPreprocessor"),
}

__all__ = list(_IMPORT_STRUCTURE)


def __getattr__(name):
    try:
        module_name, object_name = _IMPORT_STRUCTURE[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
