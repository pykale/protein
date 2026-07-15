from pathlib import Path


def register_builtin_components():
    from .core.modalities.molecule import processors as _molecule_processors  # noqa: F401
    from .core.modalities.sequence import processors as _sequence_processors  # noqa: F401
    from .core.modalities.structure import processors as _structure_processors  # noqa: F401
    from .core.registry import discover_model_cards
    from .core.tasks.dti import datasets as _dti_datasets  # noqa: F401
    from .core.tasks.inverse_folding import datasets as _inverse_folding_datasets  # noqa: F401

    discover_model_cards(Path(__file__).resolve().parent)
    return True


register_builtin_components()
