import pytest
from kale_protein.auto import AutoProteinConfig
from kale_protein.core.config import ComponentSpec, StreamSpec

def test_load_presets_and_streams():
    d=AutoProteinConfig.from_preset('drugban')
    m=AutoProteinConfig.from_preset('mapdiff')
    assert d['task']=='dti'
    assert m['objective']=='generative'
    assert isinstance(d.get_streams()['drug'], StreamSpec)
    assert isinstance(d.get_embedders()['drug'], ComponentSpec)
    assert d.get_predictor().id == 'dti/ban'

def test_validate_required_fields():
    with pytest.raises(ValueError): AutoProteinConfig.from_dict({'task':'x'})
