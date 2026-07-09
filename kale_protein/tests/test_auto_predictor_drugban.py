from pathlib import Path
from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor

def test_drugban_predictor_predicts():
    cfg=AutoProteinConfig.from_preset('drugban')
    data=AutoProteinPreprocessor.from_config(cfg).transform_sample({'smiles':'CCO','sequence':'MKTFFVLLL','label':1})
    out=AutoProteinPredictor.from_config(cfg).predict(data)
    assert 'logits' in out and 'probabilities' in out


def test_drugban_predictor_loads_pretrained_checkpoint(tmp_path):
    cfg_dict=AutoProteinConfig.from_preset('drugban').to_dict()
    checkpoint_path=tmp_path / 'toy_checkpoint.json'
    checkpoint_path.write_text('{"state_dict": {}, "metadata": {"name": "toy"}}', encoding='utf-8')
    cfg_dict['checkpoint']['path']=str(checkpoint_path)
    cfg=AutoProteinConfig.from_dict(cfg_dict)
    predictor=AutoProteinPredictor.from_config(cfg, pretrained=True)
    assert predictor.model.loaded_checkpoint['path'] == str(checkpoint_path)
