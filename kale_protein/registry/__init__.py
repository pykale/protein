from .base import Registry
from .modality_processors import MODALITY_PROCESSORS_REGISTRY as MODALITY_PROCESSOR_REGISTRY
from .modality_encoders import MODALITY_ENCODERS_REGISTRY as MODALITY_ENCODER_REGISTRY
from .fusion import FUSION_REGISTRY
from .conditioners import CONDITIONERS_REGISTRY as CONDITIONER_REGISTRY
from .heads import HEADS_REGISTRY as HEAD_REGISTRY
from .runners import RUNNERS_REGISTRY as RUNNER_REGISTRY
from .evaluators import EVALUATORS_REGISTRY as EVALUATOR_REGISTRY
from .interpreters import INTERPRETERS_REGISTRY as INTERPRETER_REGISTRY
from .presets import PRESETS_REGISTRY as PRESET_REGISTRY
from .model_cards import MODEL_CARD_REGISTRY
from .datasets import DATASET_REGISTRY
