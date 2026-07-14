from pathlib import Path


def register_builtin_components():
    from .modalities.small_molecule import processors as _smp
    from .modalities.protein_sequence import processors as _psp
    from .modalities.protein_structure import processors as _pstp
    from .registry import discover_model_cards

    discover_model_cards(Path(__file__).resolve().parent)
    return True
register_builtin_components()
