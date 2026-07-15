"""The intentionally small set of reusable component registries."""

from .base import Registry


DATASET_REGISTRY = Registry("datasets", normalize_strings=True)
PREPROCESSOR_REGISTRY = Registry("preprocessors", normalize_strings=True)
EMBEDDER_REGISTRY = Registry("embedders", normalize_strings=True)
PREDICTOR_REGISTRY = Registry("predictors", normalize_strings=True)
EVALUATOR_REGISTRY = Registry("evaluators", normalize_strings=True)
INTERPRETER_REGISTRY = Registry("interpreters", normalize_strings=True)
