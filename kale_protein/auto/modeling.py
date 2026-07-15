"""Generic Auto dispatch for complete models and reusable components."""

from __future__ import annotations

import importlib

from kale_protein.core.config import AutoProteinConfig, ComponentSpec
from kale_protein.core.registry import EMBEDDER_REGISTRY, PREDICTOR_REGISTRY


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
    namespace = component_id.partition("/")[0].casefold()
    if registry is EMBEDDER_REGISTRY and namespace in {"sequence", "molecule", "structure"}:
        importlib.import_module(f"kale_protein.core.modalities.{namespace}.embedders")
    elif registry is PREDICTOR_REGISTRY and namespace in {"dti", "inverse_folding"}:
        importlib.import_module(f"kale_protein.core.tasks.{namespace}.predictors")


class AutoProteinModel:
    """Load a complete model implementation declared by its model card."""

    def __new__(cls, model_id=None, *, pretrain=False, **kwargs):
        if cls is not AutoProteinModel:
            return super().__new__(cls)
        return cls.from_config(model_id, pretrain=pretrain, **kwargs)

    @classmethod
    def from_config(cls, config, *, pretrain=False, **kwargs):
        config = _coerce_model_config(config)
        implementation = config.auto_class("AutoProteinModel")
        return implementation(config=config, pretrain=pretrain, **kwargs)


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


__all__ = ["AutoProteinEmbedder", "AutoProteinModel", "AutoProteinPredictor"]
