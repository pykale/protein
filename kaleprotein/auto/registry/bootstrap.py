"""Lazy registration of reusable components shipped with KaleProtein."""

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def register_builtin_data():
    """Import built-in dataset modules exactly once."""

    for module_name in ("bindingdb", "biosnap", "cath", "human"):
        import_module(f"kaleprotein.loaddata.{module_name}")

    return True


@lru_cache(maxsize=1)
def register_builtin_preprocessors():
    """Import built-in preprocessing modules exactly once."""

    from kaleprotein import prepdata as _prepdata  # noqa: F401

    return True


def register_builtin_components():
    """Explicitly register all reusable built-in data-side components."""

    register_builtin_data()
    register_builtin_preprocessors()
    return True


__all__ = [
    "register_builtin_components",
    "register_builtin_data",
    "register_builtin_preprocessors",
]
