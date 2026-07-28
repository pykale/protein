"""Generic Auto dispatch for complete models and reusable components."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from kaleprotein.utils.model_verify_checksum import verify_checksum

from .config import AutoProteinConfig, ComponentSpec
from .registry import EMBEDDER_REGISTRY, PREDICTOR_REGISTRY


def _coerce_model_config(model_id_or_config):
    if isinstance(model_id_or_config, AutoProteinConfig):
        return model_id_or_config
    if isinstance(model_id_or_config, str):
        return AutoProteinConfig.from_pretrained(model_id_or_config)
    raise TypeError(
        "Expected a model id string or AutoProteinConfig, got "
        f"{type(model_id_or_config).__name__}."
    )


def _build_component(registry, spec, *, config=None, **overrides):
    spec = ComponentSpec.from_value(spec)
    if not registry.has(spec.id):
        _import_shared_component_namespace(registry, spec.id)
    implementation = registry.get(spec.id)
    kwargs = dict(spec.kwargs)
    kwargs.update(overrides)
    return implementation(config=config, **kwargs)


def _import_shared_component_namespace(registry, component_id):
    namespace, separator, component = component_id.partition("/")
    if not separator or not namespace or not component:
        return
    filename = f"{namespace}_{component}".casefold().replace("-", "_")
    if not filename.isidentifier():
        return
    if registry is EMBEDDER_REGISTRY:
        module_name = f"kaleprotein.model.embed.{filename}"
    elif registry is PREDICTOR_REGISTRY:
        module_name = f"kaleprotein.model.predict.{filename}"
    else:
        return
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return
        raise


def resolve_pretrained_weight(config, downloader=urlretrieve):
    """Resolve and, when configured, download a model card's checkpoint."""
    block = config.get("pretrained", {})
    config_dir = Path(config.get("_config_dir", "."))
    local_dir = config_dir / block.get("local_dir", "weights")
    filename = block.get("filename")
    if not filename:
        raise ValueError(_missing_weight_message(config))

    weight_path = local_dir / filename
    expected_checksum = block.get("sha256")
    if weight_path.is_file():
        verify_checksum(weight_path, expected_checksum)
        return weight_path

    url = block.get("url")
    if not _valid_url(url):
        raise ValueError(_missing_weight_message(config))

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=weight_path.parent,
        prefix=f".{weight_path.name}.",
        suffix=".part",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        downloader(url, temporary_path)
        verify_checksum(temporary_path, expected_checksum)
        os.replace(temporary_path, weight_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return weight_path


class AutoProteinModel:
    """Load a complete model implementation declared by its model card."""

    def __new__(cls, model_id=None, *, pretrain=False, checkpoint=None, **kwargs):
        if cls is not AutoProteinModel:
            return super().__new__(cls)
        return cls.from_config(
            model_id,
            pretrain=pretrain,
            checkpoint=checkpoint,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config, *, pretrain=False, checkpoint=None, **kwargs):
        config = _coerce_model_config(config)
        if pretrain and checkpoint is not None:
            raise ValueError("Pass either pretrain=True or checkpoint=..., not both.")
        weight_path = resolve_pretrained_weight(config) if pretrain else checkpoint
        implementation = config.auto_class("AutoProteinModel")
        model = implementation(config=config, **kwargs)
        if weight_path is not None:
            load_checkpoint = getattr(model, "load_checkpoint", None)
            if not callable(load_checkpoint):
                raise TypeError(
                    f"{type(model).__name__} must define load_checkpoint() to load weights "
                    "through AutoProteinModel."
                )
            load_checkpoint(weight_path)
            model.weight_path = weight_path
        return model


class AutoProteinEmbedder:
    """Resolve one reusable modality or condition encoder by component id."""

    def __new__(cls, component_id=None, *, config=None, **kwargs):
        if cls is not AutoProteinEmbedder:
            return super().__new__(cls)
        return cls.from_config(component_id, config=config, **kwargs)

    @classmethod
    def from_config(cls, spec, *, config=None, **kwargs):
        return _build_component(EMBEDDER_REGISTRY, spec, config=config, **kwargs)

    @classmethod
    def register(cls, component_id, implementation=None, *, aliases=()):
        return EMBEDDER_REGISTRY.register(component_id, implementation, aliases=aliases)


class AutoProteinPredictor:
    """Resolve a reusable task predictor or generator by component id."""

    def __new__(cls, component_id=None, *, config=None, **kwargs):
        if cls is not AutoProteinPredictor:
            return super().__new__(cls)
        return cls.from_config(component_id, config=config, **kwargs)

    @classmethod
    def from_config(cls, spec, *, config=None, **kwargs):
        return _build_component(PREDICTOR_REGISTRY, spec, config=config, **kwargs)

    @classmethod
    def register(cls, component_id, implementation=None, *, aliases=()):
        return PREDICTOR_REGISTRY.register(component_id, implementation, aliases=aliases)


def _valid_url(url):
    return isinstance(url, str) and url.startswith(("https://", "http://"))


def _missing_weight_message(config):
    model_id = config.get("model_id", config.get("name", "this model"))
    return (
        f"No pretrained weight file is available for {model_id}. "
        "Place the expected file in this model card's weights/ folder, provide "
        "a valid URL in config.yaml, or train the model yourself with "
        "pretrain=False."
    )


__all__ = [
    "AutoProteinEmbedder",
    "AutoProteinModel",
    "AutoProteinPredictor",
    "resolve_pretrained_weight",
]
