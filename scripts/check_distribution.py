"""Validate KaleProtein wheel and source-distribution boundaries."""

from __future__ import annotations

import argparse
import tarfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

EXPECTED_EXTRAS = {"dev", "drugban-dti", "mapdiff-inverse-folding", "test"}
FORBIDDEN_PARTS = {".DS_Store", "__pycache__"}


def _artifacts(dist_dir: Path) -> tuple[Path, Path]:
    wheels = sorted(dist_dir.glob("*.whl"))
    source_archives = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(source_archives) != 1:
        raise AssertionError(
            "Expected exactly one wheel and one source archive in "
            f"{dist_dir}, found {wheels} and {source_archives}."
        )
    return wheels[0], source_archives[0]


def _assert_clean_paths(names: list[str]) -> None:
    for name in names:
        parts = set(Path(name).parts)
        if parts & FORBIDDEN_PARTS or name.endswith((".pyc", ".pyo")):
            raise AssertionError(f"Generated artifact contains {name!r}.")


def check_wheel(wheel: Path, expected_version: str) -> None:
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        _assert_clean_paths(names)

        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise AssertionError(f"Expected one METADATA file, found {metadata_names}.")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))

    if metadata["Name"] != "kaleprotein":
        raise AssertionError(f"Unexpected project name: {metadata['Name']!r}.")
    if metadata["Version"] != expected_version:
        raise AssertionError(
            f"Wheel version {metadata['Version']!r} does not match "
            f"{expected_version!r}."
        )
    if metadata["Requires-Python"] != ">=3.10":
        raise AssertionError(
            f"Unexpected Requires-Python: {metadata['Requires-Python']!r}."
        )
    if metadata["License-Expression"] != "MIT":
        raise AssertionError(
            f"Unexpected license expression: {metadata['License-Expression']!r}."
        )

    extras = set(metadata.get_all("Provides-Extra", []))
    if extras != EXPECTED_EXTRAS:
        raise AssertionError(f"Unexpected extras: {sorted(extras)}.")

    requirements = set(metadata.get_all("Requires-Dist", []))
    if "torch>=2.0" not in requirements:
        raise AssertionError("Wheel must require torch>=2.0 without an extra marker.")

    if not any(name.startswith("kaleprotein/") for name in names):
        raise AssertionError("Wheel does not contain the kaleprotein package.")
    for prefix in ("examples/", "tests/", "docs/", "scripts/"):
        if any(name.startswith(prefix) for name in names):
            raise AssertionError(f"Wheel unexpectedly contains {prefix}.")


def check_source_archive(source_archive: Path) -> None:
    with tarfile.open(source_archive, "r:gz") as archive:
        names = archive.getnames()
    _assert_clean_paths(names)

    relative_names = {
        "/".join(Path(name).parts[1:]) for name in names if len(Path(name).parts) > 1
    }
    required = {
        "README.md",
        "pyproject.toml",
        "kaleprotein/__init__.py",
        "examples/drugban_dti/train.py",
        "examples/mapdiff_inverse_folding/train.py",
        "tests/test_package_metadata_architecture.py",
        "docs/architecture.md",
        "scripts/check_distribution.py",
    }
    missing = required - relative_names
    if missing:
        raise AssertionError(
            f"Source archive is missing required files: {sorted(missing)}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    expected_version = (
        (root / "kaleprotein" / "_version.txt").read_text(encoding="utf-8").strip()
    )
    wheel, source_archive = _artifacts(args.dist_dir)
    check_wheel(wheel, expected_version)
    check_source_archive(source_archive)
    print(f"Validated {wheel.name} and {source_archive.name}.")


if __name__ == "__main__":
    main()
