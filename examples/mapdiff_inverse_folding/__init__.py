"""MapDiff model card and runnable workflows."""

from pathlib import Path


MODEL_CARD_PATH = Path(__file__).with_name("config.yaml")


def register_model_card():
    """Register this example's model id before using an Auto model API."""

    from kaleprotein.auto.registry import register_model_card as register

    return register(MODEL_CARD_PATH)


__all__ = ["MODEL_CARD_PATH", "register_model_card"]
