"""Registry-backed data loading entry points."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Mapping


_DATASET_DISCOVERY_ATTEMPTED = False


class AutoProteinData:
    """Load a registered dataset directly by its public identifier."""

    def __new__(cls, data_id, *args, **kwargs):
        import kale_protein  # noqa: F401 - bootstrap registrations
        from kale_protein.registry import DATASET_REGISTRY

        if not DATASET_REGISTRY.has(data_id):
            _discover_task_datasets()
        loader = DATASET_REGISTRY.get(data_id)
        if not callable(loader):
            if args or kwargs:
                raise TypeError(
                    f"Registered dataset {data_id!r} is not callable and cannot accept arguments."
                )
            return loader
        return loader(*args, **kwargs)


class AutoProteinDataLoader:
    """Config adapter for code that prefers a delayed ``load()`` call.

    Direct workflows should normally use :class:`AutoProteinData`.
    """

    def __init__(self, data_id, **loader_kwargs):
        self.data_id = data_id
        self.loader_kwargs = loader_kwargs

    @classmethod
    def from_config(cls, config, **overrides):
        data = config.get("data")
        if isinstance(data, str):
            return cls(data, **overrides)
        if not isinstance(data, Mapping):
            raise ValueError(
                "Config-driven data loading requires data: <dataset-id> or a data mapping."
            )
        data = dict(data)
        data_id = data.pop("dataset_id", data.pop("id", None))
        if not data_id:
            raise ValueError("The config data mapping must define dataset_id or id.")
        data.update(overrides)
        return cls(data_id, **data)

    def load(self):
        return AutoProteinData(self.data_id, **self.loader_kwargs)


def _discover_task_datasets():
    """Import task dataset modules by package convention on first use."""

    global _DATASET_DISCOVERY_ATTEMPTED
    if _DATASET_DISCOVERY_ATTEMPTED:
        return
    _DATASET_DISCOVERY_ATTEMPTED = True

    from kale_protein import tasks

    for module_info in pkgutil.iter_modules(tasks.__path__, f"{tasks.__name__}."):
        if not module_info.ispkg:
            continue
        try:
            importlib.import_module(f"{module_info.name}.datasets")
        except ImportError:
            # A task may depend on an optional backend. Its own Auto call will
            # still surface a missing-key/dependency error without preventing
            # lightweight task modules from registering.
            continue
