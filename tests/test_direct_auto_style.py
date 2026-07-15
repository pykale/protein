import builtins
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from kaleprotein.auto import (
    AutoMoleculePreprocessor,
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinEvaluator,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPreprocessor,
)
from kaleprotein.core.weights import resolve_pretrained_weight


def test_model_ids_are_card_driven_not_auto_hardcoded():
    source = (Path(__file__).resolve().parents[1] / "kaleprotein" / "auto" / "modeling.py").read_text(
        encoding="utf-8"
    )
    assert "DTI/DrugBAN" not in source
    assert "InverseFolding/MapDiff" not in source

    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    assert config["auto_map"]["AutoProteinModel"] == "modeling.DrugBANModel"


def test_model_cards_do_not_require_pyyaml(monkeypatch):
    original_import = builtins.__import__

    def import_without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("PyYAML intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")

    assert config["model_id"] == "DTI/DrugBAN"
    assert config.get_predictor().id == "dti/ban"


def test_drugban_direct_pipeline_style(fake_rdkit_graph, tmp_path):
    dataset_dir = tmp_path / "bindingdb"
    dataset_dir.mkdir()
    (dataset_dir / "full.csv").write_text(
        "SMILES,Protein,Y\nCCO,MKTFFVLLLMKTFFVLLL,1\n", encoding="utf-8"
    )
    sample = AutoProteinData("BindingDB/DTI", root=tmp_path)[0]
    protein_data = AutoProteinPreprocessor("protein/sequence").tokenize(sample)
    molecule_data = AutoMoleculePreprocessor("molecule/SMILE").featurize(sample)
    model = AutoProteinModel("DTI/DrugBAN", pretrain=False)

    processed = {
        "samples": [
            {"target": protein_data, "drug": molecule_data, "label": 0},
            {"target": protein_data, "drug": molecule_data, "label": 1},
        ]
    }
    batch = model.collator(**processed)
    embeddings = model.embed(**batch)
    prediction = model.predictor(**embeddings)

    def custom_head(protein_embedding, molecule_embedding, labels=None, **metadata):
        return {
            "scores": protein_embedding.mean((1, 2)) + molecule_embedding.mean((1, 2)),
            "labels": labels,
        }

    custom_prediction = custom_head(**embeddings)
    metrics = model.evaluate(**prediction)
    attention = model.extract_attention(**prediction)
    auto_metrics = AutoProteinEvaluator.from_config(model.config).evaluate(**prediction)
    interpretation = AutoProteinInterpreter.from_config(model.config).explain(**attention)

    assert sample["label"] == 1
    assert embeddings["protein_embedding"].shape[0] == 2
    assert embeddings["molecule_embedding"].shape[0] == 2
    assert "probabilities" in prediction
    assert custom_prediction["scores"].shape == (2,)
    assert set(metrics) == {"auroc", "auprc", "f1", "accuracy", "threshold"}
    assert set(auto_metrics) == {"auroc", "auprc", "f1", "accuracy", "threshold"}
    assert attention["attention"].shape[0] == 2
    assert len(interpretation["samples"]) == 2
    assert set(model.state_dict()) == {
        *[key for key in model.state_dict() if key.startswith("protein_embedder.")],
        *[key for key in model.state_dict() if key.startswith("molecule_embedder.")],
        *[key for key in model.state_dict() if key.startswith("predictor.")],
    }


def test_pretrained_missing_url_errors_clearly():
    try:
        AutoProteinModel("DTI/DrugBAN", pretrain=True)
    except ValueError as exc:
        assert "No pretrained weight file is available for DTI/DrugBAN" in str(exc)
    else:
        raise AssertionError("Expected missing pretrained weight error")


def test_pretrained_resolver_uses_fake_downloader():
    with TemporaryDirectory() as tmpdir:
        config = AutoProteinConfig.from_dict(
            {
                "model_id": "Fake/Model",
                "_config_dir": tmpdir,
                "task": "fake",
                "objective": "generative",
                "auto_map": {"AutoProteinModel": "modeling.FakeModel"},
                "streams": {
                    "sequence": {
                        "modality": "sequence",
                        "input_key": "sequence",
                        "processor": "amino_acid",
                    }
                },
                "components": {
                    "embedders": {"sequence": {"id": "fake/embedder"}},
                    "predictor": {"id": "fake/predictor"},
                },
                "pretrained": {
                    "local_dir": "weights",
                    "filename": "fake.pt",
                    "url": "https://example.test/fake.pt",
                },
            }
        )

        def fake_downloader(url, path):
            Path(path).write_text(f"downloaded from {url}", encoding="utf-8")

        path = resolve_pretrained_weight(config, downloader=fake_downloader)
        assert path.read_text(encoding="utf-8") == "downloaded from https://example.test/fake.pt"


def test_mapdiff_direct_generative_pipeline_style(tmp_path):
    torch.save(
        {
            "atom_pos": [
                [[-1.2, 0.1, 0.0], [0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [2.0, 1.2, 0.0]],
                [[2.1, -0.8, 0.1], [3.5, -0.7, 0.0], [4.2, 0.6, 0.1], [5.4, 0.7, 0.0]],
            ],
            "sequence": "MA",
        },
        tmp_path / "protein.pt",
    )
    graph = AutoProteinData("CATH/InverseFolding", source=tmp_path)[0]
    structure_data = AutoProteinPreprocessor("protein/structure").featurize(
        {"backbone_coords": graph.atom_pos, "sequence": graph.sequence}
    )
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=False)

    processed = {"samples": [structure_data]}
    batch = model.collator(**processed)
    conditioning = model.embed(**batch)
    generated = model.predictor.generate(**conditioning, steps=1)
    metrics = model.evaluate(**generated)
    auto_metrics = AutoProteinEvaluator.from_config(model.config).evaluate(**generated)
    interpretation = AutoProteinInterpreter.from_config(model.config).explain(**generated)

    assert graph.sequence == "MA"
    assert "structure_embedding" in conditioning
    assert generated["sequences"]
    assert set(metrics) == {"sequence_recovery", "perplexity", "diversity"}
    assert set(auto_metrics) == {"sequence_recovery", "perplexity", "diversity"}
    assert interpretation["final_sequences"] == generated["sequences"]
    assert all(key.startswith("predictor.network.") for key in model.state_dict())
