import torch

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
        {
            "backbone_coords": [
                [
                    [-1.2, 0.1, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.4, 0.2, 0.0],
                    [2.0, 1.2, 0.0],
                ],
                [
                    [2.1, -0.8, 0.1],
                    [3.5, -0.7, 0.0],
                    [4.2, 0.6, 0.1],
                    [5.4, 0.7, 0.0],
                ],
            ],
            "sequence": "MA",
        }
    )
    assert output["edge_features"].shape[-1] == 93
    assert output["extra_residue_features"].shape[-1] == 11
    assert output["ipa_atom_positions"].shape == (2, 5, 3)


def test_mapdiff_preprocess_filters_incomplete_backbone_residues():
    config = AutoProteinConfig.from_preset("mapdiff")
    preprocessor = AutoProteinPreprocessor.from_config(config)
    coordinates = [
        [
            [-1.2, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [1.4, 0.2, 0.0],
            [2.0, 1.2, 0.0],
        ],
        [
            [2.1, -0.8, 0.1],
            [float("nan"), float("nan"), float("nan")],
            [4.2, 0.6, 0.1],
            [5.4, 0.7, 0.0],
        ],
        [
            [6.1, -0.8, 0.1],
            [7.5, -0.7, 0.0],
            [8.2, 0.6, 0.1],
            [9.4, 0.7, 0.0],
        ],
    ]

    edge_features = torch.arange(6 * 93, dtype=torch.float32).reshape(6, 93)
    node_features = torch.zeros(3, 26)
    node_features[:, 20] = torch.tensor([1.0, 2.0, 3.0])
    node_features[:, 21] = torch.tensor([10.0, 20.0, 30.0])
    output = preprocessor.transform_sample(
        {
            "backbone_coords": coordinates,
            "sequence": "MAD",
            "x": node_features,
            "mu_r_norm": torch.zeros(3, 5),
            "ss": torch.zeros(3, 8),
            "edge_index": torch.tensor(
                [
                    [0, 1, 0, 2, 1, 2],
                    [1, 0, 2, 0, 2, 1],
                ]
            ),
            "edge_attr": edge_features,
        }
    )

    assert output["reference_sequence"] == "MD"
    assert output["target_tokens"].shape == (2,)
    assert output["edge_index"].max().item() == 1
    assert torch.equal(
        output["edge_index"],
        torch.tensor([[0, 1], [1, 0]]),
    )
    assert torch.equal(
        output["extra_residue_features"][:, :2],
        torch.tensor([[1.0, 10.0], [3.0, 30.0]]),
    )
    assert not torch.equal(output["edge_features"], edge_features[2:4])
