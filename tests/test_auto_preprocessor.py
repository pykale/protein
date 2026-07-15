from kaleprotein.auto import AutoProteinConfig, AutoProteinPreprocessor

def test_drugban_preprocess(fake_rdkit_graph):
    cfg=AutoProteinConfig.from_preset('drugban')
    p=AutoProteinPreprocessor.from_config(cfg)
    out=p.transform_sample({'smiles':'CCO','sequence':'MKTFFVLLL','label':1})
    assert 'drug' in out and 'target' in out and out['label']==1
    assert 'node_features' in out['drug'] and 'tokens' in out['target']


def test_dataset_preprocessing_returns_keyword_expandable_mapping(fake_rdkit_graph):
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    preprocessor = AutoProteinPreprocessor.from_config(config)

    processed = preprocessor.transform_dataset(
        [{"smiles": "CCO", "sequence": "MKTFFVLLLMKTFFVLLL", "label": 1}]
    )

    assert set(processed) == {"samples"}
    assert processed["samples"][0]["label"] == 1

def test_mapdiff_preprocess():
    cfg=AutoProteinConfig.from_preset('mapdiff')
    p=AutoProteinPreprocessor.from_config(cfg)
    out=p.transform_sample({'backbone_coords':[[[0,0,0]],[[1,0,0]]],'sequence':'MA'})
    assert 'structure' in out and 'noisy_sequence' in out
