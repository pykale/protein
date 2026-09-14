"""Download a standalone example from the official repository."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from kaleprotein import __version__

_REPOSITORY = "pykale/protein"
_MAX_FILE_BYTES = 16 * 1024 * 1024


def _fetch_bytes(url):
    request = Request(url, headers={"User-Agent": "kaleprotein-example-downloader"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read(_MAX_FILE_BYTES + 1)
    except HTTPError as error:
        detail = " Check the requested ref and example name."
        if error.code in (403, 429):
            detail = " GitHub may be rate limiting requests; retry later."
        raise RuntimeError(f"Download failed (HTTP {error.code}): {url}.{detail}") from error
    except URLError as error:
        raise RuntimeError(f"Download failed: {url}: {error.reason}") from error
    if len(payload) > _MAX_FILE_BYTES:
        raise ValueError(f"Example asset exceeds the 16 MiB limit: {url}")
    return payload


def download_example(name, output="examples", *, ref=None):
    """Fetch one example, excluding datasets and weights, and return its path.

    Defaults to the installed library's release tag. Downloaded Python files
    are not imported or executed. Existing destinations are never overwritten.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Use an example folder name, such as 'drugban_dti'.")
    ref = f"v{__version__}" if ref is None else ref
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("ref must be a non-empty branch, tag, or commit.")
    output = Path(output).resolve()
    destination = output / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Example directory already exists: {destination}. Choose another output directory.")

    api = f"https://api.github.com/repos/{_REPOSITORY}"
    commit = json.loads(_fetch_bytes(f"{api}/commits/{quote(ref, safe='')}"))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("GitHub returned an invalid commit SHA.")
    tree = json.loads(_fetch_bytes(f"{api}/git/trees/{commit}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned an incomplete file listing; no example was downloaded.")

    prefix = f"examples/{name}/"
    files = {}
    notices = {}
    for entry in tree["tree"]:
        source = entry["path"]
        if source.startswith(prefix):
            relative = source[len(prefix):]
        elif source in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            notices[source] = entry
            continue
        else:
            continue
        path = PurePosixPath(relative)
        if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError(f"Invalid example asset path: {source}")
        if entry["type"] == "tree":
            continue
        if path.parts[0] in ("data", "weights") and path.name != ".gitkeep":
            continue
        if entry["type"] != "blob" or entry["mode"] not in ("100644", "100755"):
            raise ValueError(f"Unsupported example asset (symlink or submodule): {source}")
        files[relative] = entry

    if not {"config.yaml", "__init__.py"} <= files.keys():
        raise ValueError(f"No complete example named {name!r} found at {_REPOSITORY}@{ref}.")
    for name_, entry in notices.items():
        files.setdefault(name_, entry)

    output.mkdir(parents=True, exist_ok=True)
    # Stage the complete bundle before exposing it; failures leave no partial example.
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output) as temporary:
        staging = Path(temporary) / name
        staging.mkdir()
        for relative, entry in files.items():
            url = f"https://raw.githubusercontent.com/{_REPOSITORY}/{commit}/{quote(entry['path'], safe='/')}"
            payload = _fetch_bytes(url)
            blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
            if hashlib.sha1(blob).hexdigest() != entry["sha"]:
                raise ValueError(f"Git blob checksum mismatch: {entry['path']}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        for folder in ("data", "weights"):
            (staging / folder).mkdir(exist_ok=True)
        provenance = {"repository": _REPOSITORY, "ref": ref, "commit": commit, "example": name}
        (staging / ".kaleprotein-example.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Example directory already exists: {destination}")
        staging.rename(destination)
    return destination
