"""Small filesystem helpers used by dataset adapters."""

from pathlib import Path


def require_file(path, *, description="data file") -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description.capitalize()} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description.capitalize()} is not a file: {path}")
    return path


__all__ = ["require_file"]
