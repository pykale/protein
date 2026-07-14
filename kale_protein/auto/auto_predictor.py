"""Model-card dispatch for trainable and generative Auto classes."""

from __future__ import annotations

from .config import AutoProteinConfig


def _coerce_config(model_id_or_config):
    if isinstance(model_id_or_config, AutoProteinConfig):
        return model_id_or_config
    if isinstance(model_id_or_config, str):
        return AutoProteinConfig.from_pretrained(model_id_or_config)
    raise TypeError(
        "Expected a model id string or AutoProteinConfig, got "
        f"{type(model_id_or_config).__name__}."
    )


def _from_config(config, auto_name, *, pretrain=False, **kwargs):
    config = _coerce_config(config)
    implementation = config.auto_class(auto_name)
    return implementation(config=config, pretrain=pretrain, **kwargs)


class AutoProteinModel:
    """Resolve the model implementation declared by a model card."""

    def __new__(cls, model_id=None, *, pretrain=False, component=None, **kwargs):
        if cls is not AutoProteinModel:
            return super().__new__(cls)
        model = cls.from_config(model_id, pretrain=pretrain, **kwargs)
        return model.component(component) if component is not None else model

    @classmethod
    def from_config(cls, config, *, pretrain=False, component=None, **kwargs):
        model = _from_config(
            config, "AutoProteinModel", pretrain=pretrain, **kwargs
        )
        return model.component(component) if component is not None else model


class AutoProteinPredictor:
    """Resolve a task predictor declared by a model card."""

    def __new__(cls, model_id=None, *, pretrain=False, **kwargs):
        if cls is not AutoProteinPredictor:
            return super().__new__(cls)
        return cls.from_config(model_id, pretrain=pretrain, **kwargs)

    @classmethod
    def from_config(cls, config, *, pretrain=False, **kwargs):
        return _from_config(
            config, "AutoProteinPredictor", pretrain=pretrain, **kwargs
        )


class AutoProteinGenerator:
    """Resolve a generative pipeline declared by a model card."""

    def __new__(cls, model_id=None, *, pretrain=False, **kwargs):
        if cls is not AutoProteinGenerator:
            return super().__new__(cls)
        return cls.from_config(model_id, pretrain=pretrain, **kwargs)

    @classmethod
    def from_config(cls, config, *, pretrain=False, **kwargs):
        return _from_config(
            config, "AutoProteinGenerator", pretrain=pretrain, **kwargs
        )
