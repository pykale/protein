from hashlib import sha256
from pathlib import Path

import pytest

from kale_protein.core.weights import (
    extract_checkpoint_state_dict,
    load_checkpoint_state_dict,
    resolve_pretrained_weight,
)


def _weight_config(tmp_path, *, checksum=None):
    return {
        "model_id": "Architecture/Fake",
        "_config_dir": str(tmp_path),
        "pretrained": {
            "local_dir": "weights",
            "filename": "fake.ckpt",
            "url": "https://example.test/fake.ckpt",
            "sha256": checksum,
        },
    }


def test_local_checkpoint_wins_and_is_checksum_verified(tmp_path):
    content = b"local checkpoint"
    checkpoint = tmp_path / "weights" / "fake.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(content)

    def unexpected_download(*args):
        raise AssertionError("local checkpoint should prevent downloading")

    resolved = resolve_pretrained_weight(
        _weight_config(tmp_path, checksum=sha256(content).hexdigest()),
        downloader=unexpected_download,
    )

    assert resolved == checkpoint


def test_download_is_verified_then_atomically_published(tmp_path):
    content = b"downloaded checkpoint"
    seen_paths = []

    def fake_downloader(url, path):
        seen_paths.append(Path(path))
        assert url == "https://example.test/fake.ckpt"
        assert Path(path).name != "fake.ckpt"
        Path(path).write_bytes(content)

    resolved = resolve_pretrained_weight(
        _weight_config(tmp_path, checksum=sha256(content).hexdigest()),
        downloader=fake_downloader,
    )

    assert resolved.read_bytes() == content
    assert seen_paths[0].suffix == ".part"
    assert not seen_paths[0].exists()


@pytest.mark.parametrize("failure", ["download", "checksum"])
def test_failed_downloads_leave_no_final_or_partial_file(tmp_path, failure):
    def fake_downloader(url, path):
        Path(path).write_bytes(b"partial")
        if failure == "download":
            raise OSError("simulated download failure")

    expected = sha256(b"different").hexdigest() if failure == "checksum" else None
    with pytest.raises((OSError, ValueError)):
        resolve_pretrained_weight(
            _weight_config(tmp_path, checksum=expected), downloader=fake_downloader
        )

    weight_dir = tmp_path / "weights"
    assert not (weight_dir / "fake.ckpt").exists()
    assert list(weight_dir.glob("*.part")) == []
    assert list(weight_dir.glob(".*.part")) == []


@pytest.mark.parametrize("container_key", [None, "model", "state_dict", "model_state_dict"])
def test_checkpoint_loader_accepts_raw_and_common_containers(tmp_path, container_key):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fake")
    state_dict = {"layer.weight": object()}
    payload = state_dict if container_key is None else {container_key: state_dict, "epoch": 2}
    calls = []

    def fake_loader(path, *, map_location):
        calls.append((path, map_location))
        return payload

    loaded = load_checkpoint_state_dict(
        checkpoint_path, map_location="meta", loader=fake_loader
    )

    assert loaded is state_dict
    assert calls == [(checkpoint_path, "meta")]


def test_checkpoint_loader_reports_invalid_payloads_clearly():
    with pytest.raises(TypeError, match="Checkpoint must be a state-dict mapping"):
        extract_checkpoint_state_dict(["not", "a", "mapping"])
    with pytest.raises(TypeError, match="'state_dict'.*mapping"):
        extract_checkpoint_state_dict({"state_dict": None})
    with pytest.raises(ValueError, match="State-dict keys must be strings"):
        extract_checkpoint_state_dict({1: "parameter"})
