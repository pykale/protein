from pathlib import Path


def register_builtin_components():
    from .core.data.datasets import dti as _dti_datasets  # noqa: F401
    from .core.data.datasets import inverse_folding as _inverse_folding_datasets  # noqa: F401
    from .core.data.preprocessors import molecule as _molecule_preprocessors  # noqa: F401
    from .core.data.preprocessors import sequence as _sequence_preprocessors  # noqa: F401
    from .core.data.preprocessors import structure as _structure_preprocessors  # noqa: F401
    from .core.registry import discover_model_cards

    package_dir = Path(__file__).resolve().parent
    repository_dir = package_dir.parent
    examples_dir = repository_dir / "examples"
    if (repository_dir / "pyproject.toml").is_file() and examples_dir.is_dir():
        discover_model_cards(examples_dir)
    return True


register_builtin_components()
