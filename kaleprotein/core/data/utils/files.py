"""Small filesystem helpers used by dataset adapters."""

from pathlib import Path


def require_file(path, *, description="data file") -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description.capitalize()} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description.capitalize()} is not a file: {path}")
    return path


def safe_path_part(value, *, name="path component") -> str:
    text = str(value).strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"{name} must be a single safe path component; got {value!r}.")
    return text


__all__ = ["require_file", "safe_path_part"]
