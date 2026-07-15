import importlib
from pathlib import Path

import pytest
import torch

from kaleprotein.auto import AutoProteinConfig
from examples.drugban_dti.modeling import DrugBANModel
from kaleprotein.core.data.modalities.molecule import processors
from kaleprotein.core.evaluation.tasks.dti.metrics import (
    MetricUndefinedError,
    compute_metrics,
)


class _FakeHybridization:
    SP = "SP"
    SP2 = "SP2"
    SP3 = "SP3"
    SP3D = "SP3D"
    SP3D2 = "SP3D2"


class _FakeAtom:
    def __init__(self, symbol, degree):
        self.symbol = symbol
        self.degree = degree

    def GetSymbol(self):
        return self.symbol

    def GetDegree(self):
        return self.degree

    def GetImplicitValence(self):
        return 4 - self.degree

    def GetFormalCharge(self):
        return 0

    def GetNumRadicalElectrons(self):
        return 0

    def GetHybridization(self):
        return _FakeHybridization.SP3

    def GetIsAromatic(self):
        return False

    def GetTotalNumHs(self):
        return max(0, 4 - self.degree)


class _FakeBond:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def GetBeginAtomIdx(self):
        return self.source

    def GetEndAtomIdx(self):
        return self.target


class _FakeMol:
    def __init__(self, smiles):
        self.atoms = [_FakeAtom(symbol, 1) for symbol in smiles]
        if len(self.atoms) == 1:
            self.atoms[0].degree = 0
        self.bonds = [_FakeBond(index, index + 1) for index in range(len(self.atoms) - 1)]

    def GetAtoms(self):
        return self.atoms

    def GetBonds(self):
        return self.bonds


class _FakeChem:
    class rdchem:
        HybridizationType = _FakeHybridization

    @staticmethod
    def MolFromSmiles(smiles):
        return None if smiles == "invalid" else _FakeMol(smiles)


@pytest.fixture
def fake_rdkit(monkeypatch):
    monkeypatch.setattr(processors, "_load_rdkit", lambda: _FakeChem)


@pytest.fixture
def tiny_config():
    data = AutoProteinConfig.from_pretrained("DTI/DrugBAN").to_dict()
    data["streams"]["drug"]["processor_kwargs"]["max_nodes"] = 4
    data["streams"]["target"]["processor_kwargs"]["max_length"] = 20
    data["components"]["embedders"]["drug"]["kwargs"].update(
        {
            "embedding_dim": 8,
            "hidden_dim": 8,
            "layers": 2,
        }
    )
    data["components"]["embedders"]["target"]["kwargs"].update(
        {"embedding_dim": 8, "filter_dim": 8}
    )
    data["components"]["predictor"]["kwargs"].update(
        {
            "molecule_dim": 8,
            "protein_dim": 8,
            "hidden_dim": 12,
            "heads": 2,
            "decoder_hidden_dim": 16,
            "decoder_output_dim": 8,
        }
    )
    data["training"].update({"batch_size": 2, "epochs": 1, "learning_rate": 1e-3})
    return AutoProteinConfig.from_dict(data)


def _tensor_sample(label):
    features = torch.zeros(4, 75)
    features[:2, 0] = 1.0
    features[2:, 74] = 1.0
    return {
        "drug": {
            "smiles": "CC",
            "node_features": features,
            "adjacency": torch.eye(4),
            "node_mask": torch.tensor([True, True, False, False]),
            "atom_symbols": ["C", "C"],
        },
        "target": {"sequence": "MKTFFVLLLMKTFFVLLL"},
        "label": label,
    }


def test_rdkit_graph_has_reusable_canonical_atom_features(fake_rdkit, tiny_config):
    graph = processors.RDKitGraphProcessor(max_nodes=4).transform({"smiles": "CC"})

    assert graph["node_features"].shape == (2, 74)
    assert graph["node_features"][0, 0] == 1
    assert torch.equal(graph["node_mask"], torch.tensor([True, True]))
    assert graph["adjacency"][0, 1] == graph["adjacency"][1, 0] == 1

    batch = DrugBANModel(tiny_config).collator.collate_drugs([graph])
    assert batch["node_features"].shape == (1, 2, 75)
    assert torch.equal(batch["node_features"][0, :, 74], torch.tensor([0.0, 0.0]))


def test_missing_rdkit_error_is_actionable(monkeypatch):
    def missing():
        raise ImportError("RDKit is required for the 'rdkit_graph' molecule processor")

    monkeypatch.setattr(processors, "_load_rdkit", missing)
    with pytest.raises(ImportError, match="RDKit is required"):
        processors.RDKitGraphProcessor().transform({"smiles": "CC"})


def test_optimizer_step_changes_parameter(tiny_config):
    predictor = DrugBANModel(tiny_config)
    predictor.train()
    batch = predictor.collator([_tensor_sample(0), _tensor_sample(1)])
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    before = predictor.predictor.decoder.fc4.weight.detach().clone()

    output = predictor(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output["logits"], batch["label"])
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, predictor.predictor.decoder.fc4.weight.detach())


def test_component_embeddings_equal_full_batch_and_raw_preprocessed(tiny_config, fake_rdkit):
    predictor = DrugBANModel(tiny_config)
    predictor.eval()
    batch = predictor.collator([_tensor_sample(0), _tensor_sample(1)])

    full = predictor(**batch)
    embeddings = predictor.embed_components(**batch)
    component = predictor.predictor(**embeddings)
    assert torch.allclose(full["logits"], component["logits"], atol=1e-6)

    raw = [
        {"smiles": "CC", "sequence": "MKTFFVLLLMKTFFVLLL", "label": 0},
        {"smiles": "CC", "sequence": "MKTFFVLLLMKTFFVLLL", "label": 1},
    ]
    raw_output = predictor.predict(raw, batch_size=2)
    processed_output = predictor.predict(predictor.collator(raw))
    assert torch.allclose(raw_output["probabilities"], processed_output["probabilities"], atol=1e-6)


def test_checkpoint_round_trip(tiny_config, tmp_path):
    predictor = DrugBANModel(tiny_config)
    checkpoint = predictor.save_checkpoint(tmp_path / "drugban.pt")
    expected = predictor.predictor.decoder.fc4.weight.detach().clone()
    with torch.no_grad():
        predictor.predictor.decoder.fc4.weight.add_(1)

    predictor.load_checkpoint(checkpoint)
    assert torch.equal(expected, predictor.predictor.decoder.fc4.weight.detach())


def test_complete_model_resolves_pretrained_checkpoint_once(tiny_config, tmp_path, monkeypatch):
    import examples.drugban_dti.modeling as modeling

    source = DrugBANModel(tiny_config)
    source.save_checkpoint(tmp_path / "drugban.pt")
    data = tiny_config.to_dict()
    data["_config_dir"] = str(tmp_path)
    data["pretrained"] = {
        "local_dir": ".",
        "filename": "drugban.pt",
        "url": "",
    }
    config = AutoProteinConfig.from_dict(data)
    original = modeling.resolve_pretrained_weight
    calls = []

    def counted_resolver(value):
        calls.append(value["model_id"])
        return original(value)

    monkeypatch.setattr(modeling, "resolve_pretrained_weight", counted_resolver)
    loaded = DrugBANModel(config, pretrain=True)

    assert calls == ["DTI/DrugBAN"]
    assert loaded.weight_path == tmp_path / "drugban.pt"
    assert not hasattr(loaded.protein_embedder, "weight_path")
    assert not hasattr(loaded.molecule_embedder, "weight_path")
    assert not hasattr(loaded.predictor, "weight_path")


def test_binary_metrics_and_degenerate_classes():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert metrics == pytest.approx(
        {"auroc": 0.75, "auprc": 5 / 6, "f1": 2 / 3, "accuracy": 0.75, "threshold": 0.5}
    )
    with pytest.raises(MetricUndefinedError, match="one class"):
        compute_metrics([1, 1], [0.2, 0.8])


@pytest.mark.parametrize("name", ["train", "evaluate", "predict", "interpret"])
def test_scripts_are_import_safe(name):
    module = importlib.import_module(f"examples.drugban_dti.{name}")
    assert callable(module.main)


def test_predict_script_accepts_a_single_pair_without_dataset_files(fake_rdkit):
    module = importlib.import_module("examples.drugban_dti.predict")

    predictions = module.main(
        [
            "--smiles", "CC",
            "--sequence", "MKTFFVLLLMKTFFVLLL",
            "--batch-size", "1",
            "--device", "cpu",
        ]
    )

    assert len(predictions) == 1
    assert predictions[0]["id"] == "input_0"
    assert 0.0 <= predictions[0]["probability"] <= 1.0


def test_all_workflow_mains_run_with_fake_csv_and_tiny_model(
    fake_rdkit, monkeypatch, tiny_config, tmp_path
):
    csv_path = tmp_path / "dti.csv"
    csv_path.write_text(
        "SMILES,Protein,Y\n"
        "CC,MKTFFVLLLMKTFFVLLL,0\n"
        "CN,MKTFFVLLLMKTFFVLLL,1\n",
        encoding="utf-8",
    )
    modules = {
        name: importlib.import_module(f"examples.drugban_dti.{name}")
        for name in ("train", "evaluate", "predict", "interpret")
    }

    class TinyConfigFactory:
        @staticmethod
        def from_pretrained(model_id):
            assert model_id == "DTI/DrugBAN"
            return tiny_config

    def tiny_model(*args, **kwargs):
        return DrugBANModel(tiny_config, pretrain=False)

    for module in modules.values():
        monkeypatch.setattr(module, "AutoProteinConfig", TinyConfigFactory)
        monkeypatch.setattr(module, "AutoProteinModel", tiny_model)

    checkpoint = tmp_path / "drugban.pt"
    training = modules["train"].main(
        [
            "--path", str(csv_path),
            "--epochs", "1",
            "--batch-size", "2",
            "--checkpoint", str(checkpoint),
        ]
    )
    metrics = modules["evaluate"].main(
        ["--path", str(csv_path), "--batch-size", "2", "--checkpoint", str(checkpoint)]
    )
    predictions = modules["predict"].main(
        ["--path", str(csv_path), "--batch-size", "2", "--checkpoint", str(checkpoint)]
    )
    interpretation = modules["interpret"].main(
        ["--path", str(csv_path), "--batch-size", "2", "--checkpoint", str(checkpoint)]
    )

    assert training["epochs"] == 1 and checkpoint.is_file()
    assert {"auroc", "auprc", "f1", "accuracy", "threshold"} <= metrics.keys()
    assert len(predictions) == 2
    assert len(interpretation["samples"]) == 2


def test_drugban_card_has_no_upstream_or_dgl_runtime_imports():
    card = Path(__file__).resolve().parents[1] / "examples" / "drugban_dti"
    source = "\n".join(path.read_text(encoding="utf-8") for path in card.glob("*.py"))
    assert "kaleprotein-drugban-reference" not in source
    assert "import dgl" not in source
    assert "dgllife" not in source
