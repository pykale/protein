import importlib
import math

import pytest
import torch

from kale_protein.auto import AutoProteinConfig, AutoProteinModel
from kale_protein.examples.mapdiff_inverse_folding.modeling import MapDiffModel
from kale_protein.examples.mapdiff_inverse_folding.upstream_compat import UpstreamMapDiff
from kale_protein.core.data.modalities.structure.processors import BackboneCoordinateProcessor
from kale_protein.core.data.tasks.inverse_folding.collators import CollatorDiff, CollatorIPAPretrain
from kale_protein.core.data.tasks.inverse_folding.datasets import CATHGraphDataset, build_residue_graph
from kale_protein.core.evaluation.tasks.inverse_folding.interpreters import DenoisingTrajectoryInterpreter
from kale_protein.core.evaluation.tasks.inverse_folding.metrics import Diversity, Perplexity, SequenceRecovery


def _coords(length=4, shift=0.0):
    residues = []
    for index in range(length):
        center = shift + index * 3.8
        residues.append(
            [
                [center - 1.2, 0.2, 0.0],
                [center, 0.0, 0.0],
                [center + 1.4, 0.3, 0.1],
                [center + 2.0, 1.2, 0.0],
            ]
        )
    return residues


def _tiny_config(tmp_path=None, filename="weights.pt"):
    config = {
        "model_id": "InverseFolding/MapDiffTiny",
        "task": "inverse_folding",
        "objective": "generative",
        "auto_map": {"AutoProteinModel": "modeling.MapDiffModel"},
        "streams": {
            "structure": {
                "modality": "structure",
                "input_key": "backbone_coords",
                "processor": "backbone",
            }
        },
        "components": {
            "embedders": {
                "structure": {"id": "structure/mapdiff_condition", "kwargs": {}}
            },
            "predictor": {"id": "inverse_folding/mapdiff_generator", "kwargs": {}},
        },
        "sampling": {"steps": 3, "method": "ddim", "num_samples": 1},
        "model": {
            "hidden_dim": 24,
            "egnn_layers": 1,
            "ipa_layers": 1,
            "ipa_heads": 2,
            "ipa_head_dim": 6,
            "ipa_points": 2,
            "dropout": 0.0,
        },
        "diffusion": {"timesteps": 4, "min_mask_ratio": 0.4, "mask_ratio_deviation": 0.1},
    }
    if tmp_path is not None:
        config.update(
            {
                "_config_dir": str(tmp_path),
                "pretrained": {"local_dir": ".", "filename": filename},
            }
        )
    return AutoProteinConfig.from_dict(config)


def _graphs():
    return [
        build_residue_graph(_coords(4), "ACDE", "first", max_neighbors=3),
        build_residue_graph(_coords(3, 0.5), "FGH", "second", max_neighbors=2),
    ]


def _tiny_upstream_config(tmp_path, filename="upstream.pt"):
    config = _tiny_config().to_dict()
    config["_config_dir"] = str(tmp_path)
    config["pretrained"] = {
        "local_dir": ".",
        "filename": filename,
        "url": "https://example.test/mapdiff_weight.pt",
        "format": "upstream-mapdiff-v1",
        "architecture": "upstream-mapdiff-v1",
    }
    config["upstream"] = {
        "hidden_dim": 128,
        "depth": 1,
        "ipa_depth": 1,
        "ipa_heads": 4,
        "qk_points": 4,
        "v_points": 8,
        "drop_out": 0.1,
        "ipa_drop_out": 0.2,
        "timesteps": 4,
        "min_mask_ratio": 0.4,
        "dev_mask_ratio": 0.2,
    }
    return AutoProteinConfig.from_dict(config)


def test_processed_pt_and_pdb_backbone_loading(tmp_path):
    pt_path = tmp_path / "fake_cath.pt"
    torch.save({"graphs": [{"atom_pos": _coords(3), "sequence": "ACD", "id": "fake"}]}, pt_path)
    dataset = CATHGraphDataset(pt_path)
    assert len(dataset) == 1
    assert dataset[0].atom_pos.shape == (3, 4, 3)
    assert dataset[0].edge_attr.shape[-1] == 9

    pdb_path = tmp_path / "tiny.pdb"
    pdb_path.write_text(
        """ATOM      1  N   ALA A   1      -1.200   0.200   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.400   0.300   0.100  1.00 20.00           C
ATOM      4  O   ALA A   1       2.000   1.200   0.000  1.00 20.00           O
ATOM      5  N   CYS A   2       2.600  -0.700   0.000  1.00 20.00           N
ATOM      6  CA  CYS A   2       3.800   0.000   0.000  1.00 20.00           C
ATOM      7  C   CYS A   2       5.200   0.300   0.100  1.00 20.00           C
ATOM      8  O   CYS A   2       5.800   1.200   0.000  1.00 20.00           O
END
""",
        encoding="utf-8",
    )
    processed = BackboneCoordinateProcessor().transform({"pdb_path": pdb_path})
    assert processed["sequence"] == "AC"
    assert processed["atom_pos"].shape == (2, 4, 3)


def test_both_mapdiff_collators_are_functional():
    graphs = _graphs()
    ipa = CollatorIPAPretrain(seed=7)(graphs)
    assert ipa.x.shape == (2, 4, 20)
    assert ipa.x_pad.sum().item() == 7
    assert (ipa.x_mask[ipa.x_pad] > 0).any()
    assert (ipa.label[~ipa.x_pad] == 20).all()

    diffusion = CollatorDiff()(graphs)
    graph_batch, ipa_batch = diffusion
    assert graph_batch.num_graphs == 2
    assert graph_batch.x.shape == (7, 20)
    assert graph_batch.edge_index.max().item() < 7
    assert ipa_batch.atom_pos.shape == (2, 4, 4, 3)


def test_tiny_optimizer_step_and_iterative_sampling():
    torch.manual_seed(11)
    model = MapDiffModel(_tiny_config(), pretrain=False)
    assert len(model.state_dict()) == len(model.network.state_dict())
    assert all(key.startswith("predictor.network.") for key in model.state_dict())
    batch = CollatorDiff()(_graphs())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.network.denoiser.output[-1].weight.detach().clone()
    output = model(batch)
    assert torch.isfinite(output["loss"])
    optimizer.zero_grad()
    output["loss"].backward()
    optimizer.step()
    assert not torch.equal(before, model.network.denoiser.output[-1].weight)

    sampled = model.sample(batch, {"steps": 3, "method": "ddpm", "num_samples": 2})
    assert len(sampled["sequences"]) == 4
    assert len(sampled["trajectory"]) == 4
    assert sampled["trajectory"][0]["timestep"] == 4
    assert sampled["trajectory"][-1]["timestep"] == 0
    interpreted = DenoisingTrajectoryInterpreter().explain(sampled)
    assert interpreted["steps"] == 3
    assert interpreted["final_sequences"] == sampled["trajectory"][-1]["sequences"]


def test_generator_consumes_precomputed_structure_condition(monkeypatch):
    model = MapDiffModel(_tiny_config(), pretrain=False)
    batch = CollatorDiff()([_graphs()[0]])
    encoded = model.embed(batch=batch)
    original_forward = model.network.forward
    seen = []

    def recording_forward(value, conditioning=None):
        seen.append(conditioning)
        return original_forward(value, conditioning=conditioning)

    monkeypatch.setattr(model.network, "forward", recording_forward)
    model.predictor(**encoded)

    assert len(seen) == 1
    assert seen[0] is encoded["conditioning"]


def test_inverse_folding_metrics_use_sequences_and_logits():
    assert SequenceRecovery()({"sequences": ["ACD"]}, [{"sequence": "ACF"}]) == pytest.approx(2 / 3)
    assert Diversity()({"sequences": ["AAAA", "AAAC", "CCCC"]}) > 0.0
    logits = torch.full((3, 20), -5.0)
    logits[torch.arange(3), torch.tensor([0, 1, 2])] = 5.0
    perplexity = Perplexity()({"logits": logits}, [{"sequence": "ACD"}])
    assert 1.0 <= perplexity < 1.01
    assert math.isfinite(perplexity)


def test_compatible_checkpoint_loads_and_incompatible_one_fails(tmp_path):
    config = _tiny_config(tmp_path)
    source = MapDiffModel(config, pretrain=False)
    torch.save({"format": "kale-mapdiff-v1", "state_dict": source.state_dict()}, tmp_path / "weights.pt")
    loaded = MapDiffModel(config, pretrain=True)
    assert loaded.weight_path == tmp_path / "weights.pt"
    key = next(iter(source.state_dict()))
    assert torch.equal(source.state_dict()[key], loaded.state_dict()[key])

    torch.save({"state_dict": {"upstream.egnn.weight": torch.ones(2, 2)}}, tmp_path / "bad.pt")
    bad_config = _tiny_config(tmp_path, "bad.pt")
    with pytest.raises(RuntimeError, match="checkpoint compatibility error.*exact state-key"):
        MapDiffModel(bad_config, pretrain=True)


def test_upstream_container_selects_release_architecture_and_strictly_loads(tmp_path):
    source = UpstreamMapDiff(egnn_depth=1, ipa_depth=1, timesteps=4)
    state = source.state_dict()
    representative = "model.mpnn_layes.0.edge_mlp.0.weight"
    assert state[representative].shape == (768, 384)
    assert "prior_model.ipa.ipa_layers.0.0.linear_out.weight" in state
    assert "noise_schedule.betas" in state
    torch.save(
        {"config": {"model": {"depth": 1}}, "step": 3, "model": state, "opt": {}},
        tmp_path / "upstream.pt",
    )

    loaded = MapDiffModel(_tiny_upstream_config(tmp_path), pretrain=True)
    assert loaded.architecture == "upstream-mapdiff-v1"
    assert len(loaded.network.state_dict()) == len(state)
    assert torch.equal(loaded.network.state_dict()[representative], state[representative])


def test_configured_release_url_selects_exact_embedded_profile():
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    assert config["pretrained"]["url"].endswith("/v1.0.1/mapdiff_weight.pt")
    assert config["pretrained"]["format"] == "upstream-mapdiff-v1"
    assert config["pretrained"]["architecture"] == "upstream-mapdiff-v1"
    assert config["upstream"]["input_feat_dim"] == 31
    assert config["upstream"]["edge_attr_dim"] == 93
    assert config["upstream"]["depth"] == 6
    assert config["upstream"]["timesteps"] == 500


def test_auto_direct_style_and_workflow_modules_are_import_safe():
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=False)
    batch = CollatorDiff()([_graphs()[0]])
    embeddings = model.embed(batch=batch)
    output = model.predictor.generate(**embeddings, steps=2, num_samples=1)
    assert output["sequences"] and output["trajectory"]

    for module_name in ("pretrain_ipa", "train_diffusion", "evaluate", "generate"):
        module = importlib.import_module(f"kale_protein.examples.mapdiff_inverse_folding.{module_name}")
        assert callable(module.main)
        assert callable(module.build_parser)


def test_all_workflow_mains_run_with_fake_graph_and_tiny_model(monkeypatch, tmp_path):
    graph_path = tmp_path / "graph.pt"
    torch.save({"atom_pos": _coords(4), "sequence": "ACDE", "id": "tiny"}, graph_path)
    modules = {
        name: importlib.import_module(f"kale_protein.examples.mapdiff_inverse_folding.{name}")
        for name in ("pretrain_ipa", "train_diffusion", "evaluate", "generate")
    }

    def tiny_model(*args, **kwargs):
        return MapDiffModel(_tiny_config(), pretrain=False)

    for module in modules.values():
        monkeypatch.setattr(module, "AutoProteinModel", tiny_model)

    ipa_path = tmp_path / "ipa.pt"
    diffusion_path = tmp_path / "diffusion.pt"
    generated_path = tmp_path / "generated.json"
    assert modules["pretrain_ipa"].main(
        [str(graph_path), "--epochs", "1", "--batch-size", "1", "--output", str(ipa_path)]
    ) == ipa_path
    assert modules["train_diffusion"].main(
        [str(graph_path), "--epochs", "1", "--batch-size", "1", "--output", str(diffusion_path)]
    ) == diffusion_path
    metrics = modules["evaluate"].main(
        [str(graph_path), "--steps", "1", "--num-samples", "1"]
    )
    generated = modules["generate"].main(
        [str(graph_path), "--steps", "1", "--output", str(generated_path)]
    )

    assert ipa_path.is_file() and diffusion_path.is_file() and generated_path.is_file()
    assert {"sequence_recovery", "perplexity", "diversity"} <= metrics.keys()
    assert generated["sequences"] and generated["trajectory"]
