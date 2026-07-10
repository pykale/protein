from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kaleprotein.hub import load_model_config, resolve_weight_file, sha256_file


def test_load_packaged_model_card() -> None:
    config = load_model_config("DTI/DrugBAN")

    assert config["model_type"] == "drugban"
    assert config["architecture"] == "DrugBANForDTI"
    assert Path(config["_model_dir"]).name == "DrugBAN"


def test_load_packaged_data_cards() -> None:
    assert load_model_config("DTI/PDBBind")["model_type"] == "drugban"
    assert load_model_config("InverseFolding/CATH")["model_type"] == "mapdiff"


def test_resolve_weight_file_local_hit(tmp_path: Path) -> None:
    weight = tmp_path / "weights" / "pytorch_model.bin"
    weight.parent.mkdir()
    torch.save({"model": {}}, weight)
    config = {
        "_model_dir": str(tmp_path),
        "model_id": "Unit/Test",
        "weights": {"local_path": "weights/pytorch_model.bin", "url": ""},
    }

    assert resolve_weight_file(config, pretrain=True) == weight


def test_resolve_weight_file_downloads_and_checks_hash(tmp_path: Path) -> None:
    payload = b"downloaded checkpoint"
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)

    def fake_download(url: str, destination: Path) -> None:
        assert url == "https://example.test/model.bin"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    config = {
        "_model_dir": str(tmp_path),
        "model_id": "Unit/Test",
        "weights": {
            "local_path": "weights/pytorch_model.bin",
            "url": "https://example.test/model.bin",
            "sha256": sha256_file(source),
        },
    }

    resolved = resolve_weight_file(config, pretrain=True, downloader=fake_download)

    assert resolved == tmp_path / "weights" / "pytorch_model.bin"
    assert resolved.read_bytes() == payload


def test_resolve_weight_file_missing_url_error(tmp_path: Path) -> None:
    config = {
        "_model_dir": str(tmp_path),
        "model_id": "DTI/DrugBAN",
        "weights": {"local_path": "weights/pytorch_model.bin", "url": ""},
    }

    with pytest.raises(FileNotFoundError, match="Train the model locally"):
        resolve_weight_file(config, pretrain=True)


def test_resolve_weight_file_disabled_returns_none(tmp_path: Path) -> None:
    assert resolve_weight_file({"_model_dir": str(tmp_path)}, pretrain=False) is None
