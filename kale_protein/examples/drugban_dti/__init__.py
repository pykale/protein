from pathlib import Path

from kale_protein.registry import MODEL_CARD_REGISTRY

MODEL_CARD_REGISTRY.register("DTI/DrugBAN", Path(__file__).with_name("config.yaml"))
