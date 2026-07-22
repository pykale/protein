from pathlib import Path


def register_builtin_components():
    from . import loaddata as _loaddata  # noqa: F401
    from . import prepdata as _prepdata  # noqa: F401
    from .auto.registry import discover_model_cards

    package_dir = Path(__file__).resolve().parent
    repository_dir = package_dir.parent
    examples_dir = repository_dir / "examples"
    if (repository_dir / "pyproject.toml").is_file() and examples_dir.is_dir():
        discover_model_cards(examples_dir)
    return True


register_builtin_components()
