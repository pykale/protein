from kaleprotein.auto import AutoProteinConfig, AutoProteinPreprocessor


def test_drugban_preprocess(fake_rdkit_graph):
    config = AutoProteinConfig.from_preset("drugban")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    output = preprocessor.transform_sample(
        {"smiles": "CCO", "sequence": "MKTFFVLLL", "label": 1}
    )
    assert "drug" in output and "target" in output and output["label"] == 1
    assert "node_features" in output["drug"] and "tokens" in output["target"]


def test_dataset_preprocessing_returns_keyword_expandable_mapping(fake_rdkit_graph):
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    preprocessor = AutoProteinPreprocessor.from_config(config)

    processed = preprocessor.process(
        [{"smiles": "CCO", "sequence": "MKTFFVLLLMKTFFVLLL", "label": 1}]
    )

    assert set(processed) == {"samples"}
    assert processed["samples"][0]["label"] == 1


def test_single_modality_dataset_preprocessing_returns_mapping():
    preprocessor = AutoProteinPreprocessor("protein/structure")

    processed = preprocessor.process(
        [
            {
                "backbone_coords": [
                    [[0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0]],
                ],
                "sequence": "MA",
            }
        ]
    )

    assert set(processed) == {"samples"}
    assert processed["samples"][0]["sequence"] == "MA"


def test_mapdiff_preprocess():
    config = AutoProteinConfig.from_preset("mapdiff")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    output = preprocessor.transform_sample(
        {"backbone_coords": [[[0, 0, 0]], [[1, 0, 0]]], "sequence": "MA"}
    )
    assert set(output) == {"structure"}
