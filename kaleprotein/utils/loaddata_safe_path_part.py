"""Safe path-component validation for dataset adapters."""

from pathlib import Path


def safe_path_part(value, *, name="path component") -> str:
    text = str(value).strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(
            f"{name} must be a single safe path component; got {value!r}."
        )
    return text


__all__ = ["safe_path_part"]
