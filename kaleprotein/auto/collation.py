"""Model-card-driven data collator selection."""

from kaleprotein.core.config import AutoProteinConfig


def _coerce_config(model_id_or_config):
    if isinstance(model_id_or_config, AutoProteinConfig):
        return model_id_or_config
    if isinstance(model_id_or_config, str):
        return AutoProteinConfig.from_pretrained(model_id_or_config)
    raise TypeError(
        "Expected a model id string or AutoProteinConfig, got "
        f"{type(model_id_or_config).__name__}."
    )


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


__all__ = ["AutoProteinCollator"]
