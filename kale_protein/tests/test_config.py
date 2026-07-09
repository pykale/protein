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
