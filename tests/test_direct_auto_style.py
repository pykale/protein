import ast
import builtins
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinEvaluator,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from kaleprotein.auto.model import resolve_pretrained_weight


def test_model_ids_are_card_driven_not_auto_hardcoded():
    source = (Path(__file__).resolve().parents[1] / "kaleprotein" / "auto" / "model.py").read_text(
        encoding="utf-8"
    )
    assert "DTI/DrugBAN" not in source
    assert "InverseFolding/MapDiff" not in source

    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    assert config["auto_map"]["AutoProteinModel"] == "model_drugban.DrugBANModel"


def test_example_evaluations_expose_the_named_auto_pipeline():
    root = Path(__file__).resolve().parents[1]
    scripts = {
        "examples/drugban_dti/evaluate.py": "prediction = model.predict(**embeddings)",
        "examples/mapdiff_inverse_folding/evaluate.py": (
            "generation = model.generate("
        ),
    }

    for relative_path, prediction_stage in scripts.items():
        source = (root / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AutoProteinDataLoader"
            for node in ast.walk(tree)
        )
        for stage in (
            "loader = AutoProteinDataLoader(",
            "for inputs in loader:",
            "embeddings = model.embed(**inputs)",
            prediction_stage,
            "metrics = model.evaluate(",
        ):
            assert stage in source, f"{relative_path} is missing the visible stage: {stage}"
        assert "AutoProteinInterpreter" not in source
        assert ".explain(" not in source
        assert '"interpretation"' not in source


def test_example_interpreters_expose_independent_named_pipelines():
    root = Path(__file__).resolve().parents[1]
    scripts = {
        "examples/drugban_dti/interpret.py": "prediction = model.predict(**embeddings)",
        "examples/mapdiff_inverse_folding/interpret.py": "generation = model.generate(",
    }

    for relative_path, output_stage in scripts.items():
        source = (root / relative_path).read_text(encoding="utf-8")
        for stage in (
            "loader = AutoProteinDataLoader(",
            "interpreter = AutoProteinInterpreter.from_config(config)",
            "for inputs in loader:",
            "embeddings = model.embed(**inputs)",
            output_stage,
            "interpreter.explain(**",
        ):
            assert stage in source, f"{relative_path} is missing the visible stage: {stage}"
        assert "model.evaluate(" not in source


def test_all_workflows_keep_embedder_and_predictor_stages_explicit():
    root = Path(__file__).resolve().parents[1]
    scripts = {
        "examples/drugban_dti/train.py": "model.predict(**embeddings)",
        "examples/drugban_dti/evaluate.py": "model.predict(**embeddings)",
        "examples/drugban_dti/predict.py": "model.predict(**embeddings)",
        "examples/drugban_dti/interpret.py": "model.predict(**embeddings)",
        "examples/mapdiff_inverse_folding/train.py": (
            "model.predict(**embeddings)"
        ),
        "examples/mapdiff_inverse_folding/evaluate.py": (
            "model.generate("
        ),
        "examples/mapdiff_inverse_folding/interpret.py": (
            "model.generate("
        ),
        "examples/mapdiff_inverse_folding/generate.py": (
            "model.generate("
        ),
    }

    for relative_path, predictor_stage in scripts.items():
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "AutoProteinDataLoader(" in source
        assert "for inputs in loader:" in source
        assert "model.embed(**inputs)" in source
        assert predictor_stage in source
        assert "model(**inputs)" not in source
        assert "model.predictor" not in source


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
        "SMILES,Protein,Y\n"
        "CCO,MKTFFVLLLMKTFFVLLL,0\n"
        "CCN,MKTFFVLLLMKTFFVLLL,1\n",
        encoding="utf-8",
    )
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    loader = AutoProteinDataLoader(
        "BindingDB/DTI",
        config=config,
        root=tmp_path,
        batch_size=2,
    )
    batch = next(iter(loader))
    model = AutoProteinModel.from_config(config)
    embeddings = model.embed(**batch)
    prediction = model.predict(**embeddings)

    def custom_head(protein_embedding, molecule_embedding, labels=None, **metadata):
        return {
            "scores": protein_embedding.mean((1, 2)) + molecule_embedding.mean((1, 2)),
            "labels": labels,
        }

    custom_prediction = custom_head(**embeddings)
    metrics = model.evaluate(**prediction)
    auto_metrics = AutoProteinEvaluator.from_config(model.config).evaluate(**prediction)
    interpretation = AutoProteinInterpreter.from_config(model.config).explain(
        **prediction
    )

    assert loader.dataset[1]["label"] == 1
    assert embeddings["protein_embedding"].shape[0] == 2
    assert embeddings["molecule_embedding"].shape[0] == 2
    assert "probabilities" in prediction
    assert custom_prediction["scores"].shape == (2,)
    assert set(metrics) == {"auroc", "auprc", "f1", "accuracy", "threshold"}
    assert set(auto_metrics) == {"auroc", "auprc", "f1", "accuracy", "threshold"}
    assert prediction["attention"].shape[0] == 2
    assert len(interpretation["samples"]) == 2
    assert not hasattr(model, "extract_attention")
    assert not hasattr(model, "collator")
    assert not hasattr(model, "make_dataloader")
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
    config_data = AutoProteinConfig.from_pretrained(
        "InverseFolding/MapDiff"
    ).to_dict()
    config_data["model"].update(
        {
            "hidden_dim": 16,
            "egnn_depth": 1,
            "ipa_depth": 1,
            "ipa_heads": 2,
            "qk_points": 2,
            "v_points": 2,
            "timesteps": 3,
            "egnn_dropout": 0.0,
            "ipa_dropout": 0.0,
        }
    )
    config_data["model"].pop("marginal_map", None)
    config = AutoProteinConfig.from_dict(config_data)
    loader = AutoProteinDataLoader(
        "CATH/InverseFolding",
        config=config,
        source=tmp_path,
    )
    batch = next(iter(loader))
    model = AutoProteinModel.from_config(config)
    conditioning = model.embed(**batch)
    generated = model.generate(**conditioning, steps=1)
    training_output = model.predict(**conditioning)
    metrics = model.evaluate(**generated)
    auto_metrics = AutoProteinEvaluator.from_config(model.config).evaluate(**generated)
    interpretation = AutoProteinInterpreter.from_config(model.config).explain(**generated)

    assert loader.dataset[0].sequence == "MA"
    assert "structure_embedding" in conditioning
    assert "loss" in training_output
    assert generated["sequences"]
    assert set(metrics) == {"sequence_recovery", "perplexity", "diversity"}
    assert set(auto_metrics) == {"sequence_recovery", "perplexity", "diversity"}
    assert interpretation["final_sequences"] == generated["sequences"]
    assert not hasattr(model, "collator")
    assert all(key.startswith("predictor.network.") for key in model.state_dict())
