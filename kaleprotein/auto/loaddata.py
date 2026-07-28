"""Registry-backed datasets and model-ready batch loading."""

from collections.abc import Mapping


class AutoProteinData:
    """Load a reusable dataset by its ``Dataset/Task`` identifier."""

    def __new__(cls, data_id, *args, **kwargs):
        from .registry import DATASET_REGISTRY, register_builtin_data

        register_builtin_data()
        _validate_data_id(data_id)
        loader = DATASET_REGISTRY.get(data_id)
        if not callable(loader):
            if args or kwargs:
                raise TypeError(
                    f"Registered dataset {data_id!r} is not callable and cannot "
                    "accept arguments."
                )
            return loader
        return loader(*args, **kwargs)

    @classmethod
    def register(cls, data_id, loader=None, *, aliases=()):
        from .registry import DATASET_REGISTRY

        _validate_data_id(data_id)
        return DATASET_REGISTRY.register(data_id, loader, aliases=aliases)


class AutoProteinCollator:
    """Build the data collator declared by a model card."""

    def __new__(cls, model_id_or_config=None, **kwargs):
        if cls is not AutoProteinCollator:
            return super().__new__(cls)
        return cls.from_config(model_id_or_config, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        config = _coerce_config(config)
        implementation = config.auto_class("AutoProteinCollator")
        return implementation(config=config, **kwargs)


class AutoProteinDataLoader:
    """Compose a dataset, preprocessor, collator, and batch iterator.

    The dataset id selects normalized records. A model config selects the
    preprocessing and collation contract without creating or retaining a model
    instance.
    """

    def __init__(
        self,
        data_id,
        *dataset_args,
        config=None,
        for_model=None,
        preprocessor=None,
        collator=None,
        dataset=None,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        loader_class=None,
        loader_kwargs=None,
        **dataset_kwargs,
    ):
        _validate_data_id(data_id)
        if config is not None and for_model is not None:
            raise ValueError("Pass either config=... or for_model=..., not both.")
        config_value = config if config is not None else for_model
        self.config = (
            _coerce_config(config_value) if config_value is not None else None
        )
        if self.config is None and (preprocessor is None or collator is None):
            raise ValueError(
                "AutoProteinDataLoader requires config=... or for_model=... to "
                "select preprocessing and collation. Alternatively, pass both a "
                "custom preprocessor and custom collator. Use AutoProteinData when "
                "only normalized records are needed."
            )
        if self.config is not None:
            _validate_task_compatibility(data_id, self.config)

        if loader_kwargs is None:
            loader_kwargs = {}
        elif not isinstance(loader_kwargs, Mapping):
            raise TypeError("loader_kwargs must be a mapping or None.")
        reserved = {
            "batch_size",
            "shuffle",
            "num_workers",
            "drop_last",
            "collate_fn",
        }
        duplicates = sorted(reserved.intersection(loader_kwargs))
        if duplicates:
            raise ValueError(
                "Pass standard DataLoader options as direct arguments, not through "
                f"loader_kwargs: {duplicates}."
            )

        self.data_id = data_id
        if dataset is not None and (dataset_args or dataset_kwargs):
            raise ValueError(
                "When dataset=... is provided, do not also pass dataset constructor "
                "arguments."
            )
        self.dataset = (
            dataset
            if dataset is not None
            else AutoProteinData(data_id, *dataset_args, **dataset_kwargs)
        )
        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else _preprocessor_from_config(self.config)
        )
        process = getattr(self.preprocessor, "process", None)
        if not callable(process):
            raise TypeError(
                "AutoProteinDataLoader preprocessors must define process(dataset)."
            )
        processed = process(self.dataset)
        if not isinstance(processed, Mapping) or "samples" not in processed:
            raise TypeError(
                "preprocessor.process(dataset) must return a mapping containing "
                "the 'samples' field."
            )
        self.processed = dict(processed)
        self.collator = (
            collator if collator is not None else _collator_from_config(self.config)
        )
        if not callable(self.collator):
            raise TypeError("AutoProteinDataLoader collator must be callable.")

        loader_class = loader_class or _torch_data_loader()
        self.loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": drop_last,
            **dict(loader_kwargs),
        }
        self.loader = loader_class(
            self.processed["samples"],
            collate_fn=_NamedCollate(
                self.collator,
                {
                    key: value
                    for key, value in self.processed.items()
                    if key != "samples"
                },
            ),
            **self.loader_kwargs,
        )

    @classmethod
    def from_config(cls, config, data_id=None, **overrides):
        config = _coerce_config(config)
        if data_id is not None:
            return cls(data_id, config=config, **overrides)
        data = config.get("data")
        if isinstance(data, str):
            return cls(data, config=config, **overrides)
        if not isinstance(data, Mapping):
            raise ValueError(
                "AutoProteinDataLoader.from_config requires data_id=... or a "
                "config data entry containing a dataset id."
            )
        data = dict(data)
        data_id = data.pop("dataset_id", data.pop("id", None))
        if not data_id:
            raise ValueError("The config data mapping must define dataset_id or id.")
        data.update(overrides)
        return cls(data_id, config=config, **data)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class _NamedCollate:
    """Adapt PyTorch's positional collate hook to the named stage contract."""

    def __init__(self, collator, context):
        self.collator = collator
        self.context = context

    def __call__(self, samples):
        batch = self.collator(samples=samples, **self.context)
        if not isinstance(batch, Mapping):
            raise TypeError("AutoProteinDataLoader collators must return a mapping.")
        batch = dict(batch)
        invalid_keys = [key for key in batch if not isinstance(key, str)]
        if invalid_keys:
            raise TypeError(
                "AutoProteinDataLoader batch keys must be strings so the batch can "
                f"be passed as **inputs; invalid keys: {invalid_keys!r}."
            )
        return batch


def _coerce_config(value):
    from .config import AutoProteinConfig

    if isinstance(value, AutoProteinConfig):
        return value
    if isinstance(value, str):
        return AutoProteinConfig.from_pretrained(value)
    raise TypeError(
        "Expected a model id string or AutoProteinConfig, got "
        f"{type(value).__name__}."
    )


def _preprocessor_from_config(config):
    from .prepdata import AutoProteinPreprocessor

    return AutoProteinPreprocessor.from_config(config)


def _collator_from_config(config):
    return AutoProteinCollator.from_config(config)


def _torch_data_loader():
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "AutoProteinDataLoader requires PyTorch for batching. Install torch "
            "or pass loader_class=... for a compatible custom loader."
        ) from exc
    return DataLoader


def _validate_task_compatibility(data_id, config):
    data_task = data_id.rsplit("/", 1)[1]
    model_task = config.get("task")
    if _normalize_task(data_task) != _normalize_task(model_task):
        raise ValueError(
            f"Dataset {data_id!r} targets task {data_task!r}, but model "
            f"{config.get('model_id')!r} targets {model_task!r}."
        )


def _normalize_task(value):
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _validate_data_id(data_id):
    if not isinstance(data_id, str):
        raise TypeError(f"Dataset id must be a string, got {type(data_id).__name__}.")
    parts = [part.strip() for part in data_id.split("/")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Dataset id must use 'Dataset/Task' format, got {data_id!r}."
        )


__all__ = ["AutoProteinCollator", "AutoProteinData", "AutoProteinDataLoader"]
