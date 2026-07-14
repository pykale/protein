from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor

def test_drugban_predictor_predicts(fake_rdkit_graph):
    cfg=AutoProteinConfig.from_preset('drugban')
    data=AutoProteinPreprocessor.from_config(cfg).transform_sample({'smiles':'CCO','sequence':'MKTFFVLLL','label':1})
    out=AutoProteinPredictor.from_config(cfg).predict(data)
    assert 'logits' in out and 'probabilities' in out
