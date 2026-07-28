"""Checkpoint integrity verification for model loading."""

from hashlib import sha256
from pathlib import Path


def verify_checksum(path, expected):
    """Verify an optional SHA-256 digest for a checkpoint file."""
    if not expected:
        return
    if not isinstance(expected, str):
        raise TypeError("Checkpoint sha256 must be a hexadecimal string.")

    digest = sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.casefold() != expected.casefold():
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}"
        )


__all__ = ["verify_checksum"]
