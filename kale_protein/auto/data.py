"""Registry-backed data loading entry points."""

from collections.abc import Mapping


class AutoProteinData:
    """Load a reusable dataset directly by its public identifier."""

    def __new__(cls, data_id, *args, **kwargs):
        import kale_protein  # noqa: F401 - bootstrap dataset registrations
        from kale_protein.core.registry import DATASET_REGISTRY

        loader = DATASET_REGISTRY.get(data_id)
        if not callable(loader):
            if args or kwargs:
                raise TypeError(f"Registered dataset {data_id!r} is not callable and cannot accept arguments.")
            return loader
        return loader(*args, **kwargs)

    @classmethod
    def register(cls, data_id, loader=None, *, aliases=()):
        from kale_protein.core.registry import DATASET_REGISTRY

        return DATASET_REGISTRY.register(data_id, loader, aliases=aliases)


class AutoProteinDataLoader:
    def __init__(self, data_id, **loader_kwargs):
        self.data_id = data_id
        self.loader_kwargs = loader_kwargs

    @classmethod
    def from_config(cls, config, **overrides):
        data = config.get("data")
        if isinstance(data, str):
            return cls(data, **overrides)
        if not isinstance(data, Mapping):
            raise ValueError("Config-driven data loading requires data: <dataset-id> or a data mapping.")
        data = dict(data)
        data_id = data.pop("dataset_id", data.pop("id", None))
        if not data_id:
            raise ValueError("The config data mapping must define dataset_id or id.")
        data.update(overrides)
        return cls(data_id, **data)

    def load(self):
        return AutoProteinData(self.data_id, **self.loader_kwargs)


__all__ = ["AutoProteinData", "AutoProteinDataLoader"]
