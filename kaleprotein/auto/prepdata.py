"""Registry-backed preprocessing Auto classes."""

import importlib
from collections.abc import Mapping

from .registry import PREPROCESSOR_REGISTRY, register_builtin_preprocessors


def _load_preprocessor(preprocessor_id):
    register_builtin_preprocessors()
    if not PREPROCESSOR_REGISTRY.has(preprocessor_id):
        namespace = preprocessor_id.partition("/")[0].casefold()
        if namespace in {"sequence", "molecule", "structure"}:
            importlib.import_module(f"kaleprotein.prepdata.{namespace}")
    return PREPROCESSOR_REGISTRY.get(preprocessor_id)


class AutoProteinPreprocessor:
    def __new__(cls, preprocessor_id=None, **kwargs):
        if cls is AutoProteinPreprocessor and isinstance(preprocessor_id, str):
            return SingleModalityPreprocessor(preprocessor_id, **kwargs)
        return super().__new__(cls)

    @classmethod
    def from_config(cls, config):
        target = config.get("auto_map", {}).get("AutoProteinPreprocessor")
        if target:
            implementation = config.auto_class("AutoProteinPreprocessor")
            return implementation(config=config)
        return MultiStreamPreprocessor(config)

    @classmethod
    def register(cls, preprocessor_id, implementation=None, *, aliases=()):
        return PREPROCESSOR_REGISTRY.register(preprocessor_id, implementation, aliases=aliases)


class AutoMoleculePreprocessor:
    def __new__(cls, preprocessor_id=None, **kwargs):
        if isinstance(preprocessor_id, str):
            return SingleModalityPreprocessor(preprocessor_id, **kwargs)
        return super().__new__(cls)


class SingleModalityPreprocessor:
    def __init__(self, preprocessor_id, **kwargs):
        processor_cls = _load_preprocessor(preprocessor_id)
        input_key = getattr(processor_cls, "default_input_key", None)
        if input_key is not None:
            kwargs.setdefault("input_key", input_key)
        self.processor = processor_cls(**kwargs)

    def tokenize(self, data):
        return self.processor.transform(data)

    def featurize(self, data):
        return self.processor.transform(data)

    def transform_sample(self, sample):
        return self.processor.transform(sample)

    def process(self, dataset):
        return {"samples": [self.transform_sample(sample) for sample in dataset]}


class MultiStreamPreprocessor:
    def __init__(self, config):
        self.config = config
        self.processors = {}
        for name, stream in config.get_streams().items():
            processor_cls = _load_preprocessor(stream.processor_id)
            self.processors[name] = processor_cls(input_key=stream.input_key, **stream.processor_kwargs)

    def transform_sample(self, sample):
        output = {name: processor.transform(sample) for name, processor in self.processors.items()}
        for key in ("label", "domain", "id", "metadata"):
            if isinstance(sample, Mapping) and key in sample:
                output[key] = sample[key]
            elif not isinstance(sample, Mapping) and hasattr(sample, key):
                output[key] = getattr(sample, key)
        return output

    def process(self, dataset):
        return {"samples": [self.transform_sample(sample) for sample in dataset]}


__all__ = ["AutoMoleculePreprocessor", "AutoProteinPreprocessor"]
