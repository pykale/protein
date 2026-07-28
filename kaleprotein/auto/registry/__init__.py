"""Shared registries. Concrete model definitions live in model-card examples."""

from .base import Registry
from .bootstrap import (
    register_builtin_components,
    register_builtin_data,
    register_builtin_preprocessors,
)
from .catalog import (
    DATASET_REGISTRY,
    EMBEDDER_REGISTRY,
    EVALUATOR_REGISTRY,
    INTERPRETER_REGISTRY,
    PREDICTOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
)
from .model_cards import MODEL_CARD_REGISTRY, discover_model_cards, register_model_card

__all__ = [
    "DATASET_REGISTRY",
    "EMBEDDER_REGISTRY",
    "EVALUATOR_REGISTRY",
    "INTERPRETER_REGISTRY",
    "MODEL_CARD_REGISTRY",
    "PREDICTOR_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "Registry",
    "discover_model_cards",
    "register_builtin_components",
    "register_builtin_data",
    "register_builtin_preprocessors",
    "register_model_card",
]
