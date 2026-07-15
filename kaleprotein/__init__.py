from pathlib import Path


def register_builtin_components():
    from .core import data as _data  # noqa: F401
    from .core import preprocessing as _preprocessing  # noqa: F401
    from .core.registry import discover_model_cards

    package_dir = Path(__file__).resolve().parent
    repository_dir = package_dir.parent
    examples_dir = repository_dir / "examples"
    if (repository_dir / "pyproject.toml").is_file() and examples_dir.is_dir():
        discover_model_cards(examples_dir)
    return True


register_builtin_components()
