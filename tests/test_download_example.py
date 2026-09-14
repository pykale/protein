import hashlib
import json

import pytest

from kaleprotein.__main__ import main
from kaleprotein.utils import model_download_example as downloader

COMMIT = "a" * 40


@pytest.fixture
def fake_github(monkeypatch):
    payloads = {
        "examples/sample/__init__.py": b"",
        "examples/sample/config.yaml": b"model_id: Test/Sample\n",
        "examples/sample/maps/vocab.json": b"{}",
        "examples/sample/weights/large.pt": b"not a real weight",
        "examples/sample/data/train.csv": b"not real data",
        "examples/other/model.py": b"not requested",
        "kaleprotein/__init__.py": b"not requested",
        "LICENSE": b"License text",
        "THIRD_PARTY_NOTICES.md": b"Attribution",
    }
    entries = [
        {
            "path": path,
            "type": "blob",
            "mode": "100644",
            "sha": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
        }
        for path, data in payloads.items()
    ]
    requests = []

    def fetch(url):
        requests.append(url)
        if "/commits/" in url:
            return json.dumps({"sha": COMMIT}).encode()
        if "/git/trees/" in url:
            return json.dumps({"tree": entries, "truncated": False}).encode()
        return payloads[url.split(f"/{COMMIT}/", 1)[1]]

    monkeypatch.setattr(downloader, "_fetch_bytes", fetch)
    return payloads, entries, requests


def test_downloads_only_one_example_with_pinned_revision(tmp_path, fake_github):
    _, _, requests = fake_github
    result = downloader.download_example("sample", tmp_path, ref="feature/test")
    assert result == tmp_path / "sample"
    assert (result / "config.yaml").is_file()
    assert (result / "maps/vocab.json").read_text() == "{}"
    assert (result / "LICENSE").is_file()
    assert (result / "THIRD_PARTY_NOTICES.md").is_file()
    assert list((result / "weights").iterdir()) == []
    assert list((result / "data").iterdir()) == []
    assert not (tmp_path / "kaleprotein").exists()
    assert not any("large.pt" in url or "train.csv" in url or "other/" in url for url in requests)
    assert requests[0].endswith("/commits/feature%2Ftest")
    assert all(COMMIT in url for url in requests[1:])
    assert json.loads((result / ".kaleprotein-example.json").read_text())["commit"] == COMMIT


def test_default_ref_matches_library_version(tmp_path, fake_github):
    downloader.download_example("sample", tmp_path)
    assert fake_github[2][0].endswith(f"/commits/v{downloader.__version__}")


def test_existing_directory_is_not_overwritten(tmp_path, fake_github):
    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "user.py").write_text("user code")
    with pytest.raises(FileExistsError, match="already exists"):
        downloader.download_example("sample", tmp_path)
    assert (folder / "user.py").read_text() == "user code"
    assert fake_github[2] == []


@pytest.mark.parametrize("name", ["../sample", "/sample", "sample/child", "", "sample\\child"])
def test_invalid_example_names(tmp_path, name):
    with pytest.raises(ValueError, match="folder name"):
        downloader.download_example(name, tmp_path)


def test_missing_example_leaves_no_directory(tmp_path, fake_github):
    with pytest.raises(ValueError, match="No complete example"):
        downloader.download_example("missing", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failed_download_cleans_staging(tmp_path, fake_github):
    fake_github[0]["examples/sample/maps/vocab.json"] = b"corrupted"
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.download_example("sample", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path,mode", [("../escape.py", "100644"), ("link.py", "120000")])
def test_unsafe_assets_rejected(tmp_path, fake_github, path, mode):
    fake_github[1].append({"path": f"examples/sample/{path}", "type": "blob", "mode": mode, "sha": COMMIT})
    with pytest.raises(ValueError, match="Invalid example asset|Unsupported example asset"):
        downloader.download_example("sample", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cli_download_and_existing_directory_error(tmp_path, fake_github, capsys):
    args = ["download-example", "sample", "--output", str(tmp_path), "--ref", "main"]
    assert main(args) == 0
    assert "Downloaded example" in capsys.readouterr().out
    assert main(args) == 1
    assert "already exists" in capsys.readouterr().err
