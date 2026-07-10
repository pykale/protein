from pathlib import Path

from kale_protein.registry import MODEL_CARD_REGISTRY

MODEL_CARD_REGISTRY.register("InverseFolding/MapDiff", Path(__file__).with_name("config.yaml"))
