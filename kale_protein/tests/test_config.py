from pathlib import Path

import pytest
from kale_protein.auto import AutoProteinConfig, StreamSpec

def test_load_presets_and_streams():
    d=AutoProteinConfig.from_preset('drugban')
    m=AutoProteinConfig.from_preset('mapdiff')
    assert d['task']=='drug_target_interaction'
    assert m['runner']=='diffusion_generate'
    assert isinstance(d.get_streams()['drug'], StreamSpec)

def test_validate_required_fields():
    with pytest.raises(ValueError): AutoProteinConfig.from_dict({'task':'x'})


def test_load_yaml_file(tmp_path):
    cfg_path = tmp_path / "drugban_copy.yaml"
    cfg_path.write_text(Path("kale_protein/presets/drugban.yaml").read_text(), encoding="utf-8")
    cfg = AutoProteinConfig.from_yaml(cfg_path)
    assert cfg["name"] == "drugban"
    assert cfg.get_streams()["target"].processor == "amino_acid_tokenizer"
