import builtins
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from kale_protein.auto import (
    AutoMoleculePreprocessor,
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinGenerator,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)
from kale_protein.auto.weights import resolve_pretrained_weight


def test_model_ids_are_model_card_driven_not_auto_hardcoded():
    auto_source = Path(__file__).resolve().parents[1] / "auto" / "auto_predictor.py"
    source = auto_source.read_text(encoding="utf-8")

    assert "DTI/DrugBAN" not in source
    assert "InverseFolding/MapDiff" not in source

    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    assert config["model_type"] == "drugban"
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
    assert config["streams"]["target"]["processor_kwargs"]["max_length"] == 1000


def test_drugban_direct_pipeline_style(fake_rdkit_graph, tmp_path):
    dataset_dir = tmp_path / "bindingdb"
    dataset_dir.mkdir()
    (dataset_dir / "full.csv").write_text(
        "SMILES,Protein,Y\nCCO,MKTFFVLLLMKTFFVLLL,1\n",
        encoding="utf-8",
    )
    data = AutoProteinData("DTI/BindingDB", root=tmp_path)[0]
    label = data["label"]
    preprocessor_protein = AutoProteinPreprocessor("protein/sequence")
    preprocessor_drug = AutoMoleculePreprocessor("molecule/SMILE")
    protein_model, molecule_model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
    interaction_predictor = AutoProteinPredictor("DTI/DrugBAN", pretrain=False)

    protein_data = preprocessor_protein.tokenize(data)
    drug_data = preprocessor_drug.tokenize(data)

    protein_embedding = protein_model.embed(protein_data)
    drug_embedding = molecule_model.embed(drug_data)

    interaction_prediction = interaction_predictor(protein_embedding, drug_embedding)

    assert label == 1
    assert "embedding" in protein_embedding
    assert "embedding" in drug_embedding
    assert "probabilities" in interaction_prediction


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
                "runner": "diffusion_generate",
                "streams": {
                    "sequence": {
                        "modality": "protein_sequence",
                        "input_key": "sequence",
                        "processor": "amino_acid_tokenizer",
                        "encoder": "residue_token_embedding",
                    }
                },
                "head": {"type": "diffusion_sequence_decoder"},
                "sampling": {},
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
        assert path.name == "fake.pt"
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
    graph = AutoProteinData("InverseFolding/CATH", source=tmp_path)[0]
    native_sequence = graph.sequence
    data = {"backbone_coords": graph.atom_pos, "sequence": native_sequence}
    structure_preprocessor = AutoProteinPreprocessor("protein/structure")
    sequence_preprocessor = AutoProteinPreprocessor("protein/masked_sequence")
    structure_encoder = AutoProteinModel("InverseFolding/MapDiff", pretrain=False)
    sequence_generator = AutoProteinGenerator("InverseFolding/MapDiff", pretrain=False)

    structure_data = {
        "structure": structure_preprocessor.featurize(data),
        "noisy_sequence": sequence_preprocessor.tokenize(data),
    }
    structure_embedding = structure_encoder.embed(structure_data)
    generated_sequence = sequence_generator.generate(structure_embedding)

    assert native_sequence == "MA"
    assert "hidden" in structure_embedding
    assert "sequences" in generated_sequence
