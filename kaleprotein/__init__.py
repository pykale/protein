"""KaleProtein public package metadata."""

from pathlib import Path


_VERSION_FILE = Path(__file__).with_name("_version.txt")
__version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()

__all__ = ["__version__"]
