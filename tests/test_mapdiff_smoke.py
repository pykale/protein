"""End-to-end smoke coverage for the public MapDiff example."""

import importlib
import json
from hashlib import sha256
from pathlib import Path

import torch

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from examples.mapdiff_inverse_folding.model_mapdiff import (
    MapDiffModel,
    to_mapdiff_order,
    to_standard_order,
)


def _coordinates(length=4):
    return [
        [
            [index * 3.8 - 1.2, 0.2, 0.0],
            [index * 3.8, 0.0, 0.0],
            [index * 3.8 + 1.4, 0.3, 0.1],
            [index * 3.8 + 2.0, 1.2, 0.0],
        ]
        for index in range(length)
    ]


def _five_atom_coordinates(length=4):
    return [
        [
            row[0],
            row[1],
            row[2],
            [index * 3.8, -1.5, 2.25],
            row[3],
        ]
        for index, row in enumerate(_coordinates(length))
    ]


def _tiny_config():
    data = AutoProteinConfig.from_pretrained(
        "InverseFolding/MapDiff"
    ).to_dict()
    data["model"].update(
        {
            "hidden_dim": 16,
            "egnn_depth": 1,
            "ipa_depth": 1,
            "ipa_heads": 2,
            "qk_points": 2,
            "v_points": 2,
            "egnn_dropout": 0.0,
            "ipa_dropout": 0.0,
            "timesteps": 3,
        }
    )
    data["model"].pop("marginal_map", None)
    data["sampling"].update({"steps": 1, "num_samples": 1})
    return AutoProteinConfig.from_dict(data)


def test_mapdiff_amino_acid_order_round_trip():
    values = torch.eye(20)
    assert torch.equal(
        to_standard_order(to_mapdiff_order(values)),
        values,
    )


def test_mapdiff_release_parameter_tree_manifest():
    config = AutoProteinConfig.from_pretrained(
        "InverseFolding/MapDiff"
    )
    model = AutoProteinModel.from_config(config)
    state = model.network.state_dict()
    canonical = "\n".join(
        f"{key}:{','.join(str(size) for size in state[key].shape)}"
        for key in sorted(state)
    )
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "mapdiff_inverse_folding"
        / "maps"
        / "release_state_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["architecture"] == model.architecture
    assert manifest["tensor_count"] == len(state)
    assert manifest["key_shape_sha256"] == sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def test_mapdiff_auto_pipeline_smoke(tmp_path):
    graph_path = tmp_path / "structure.pt"
    atom_positions = _five_atom_coordinates()
    torch.save(
        {
            "atom_pos": atom_positions,
            "sequence": "ACDE",
            "id": "tiny",
        },
        graph_path,
    )
    config = _tiny_config()
    loader = AutoProteinDataLoader(
        "CATH/InverseFolding",
        config=config,
        source=graph_path,
    )
    inputs = next(iter(loader))
    model = AutoProteinModel.from_config(config)

    assert torch.equal(
        inputs["ipa_atom_positions"][0, :, 3],
        torch.tensor(atom_positions)[:, 3],
    )
    node_t = torch.tensor([3])
    node_s = torch.tensor([1])
    alpha = model.network.noise_schedule.alphas_bar
    q_s = model.network._matrix(alpha[node_s])
    q_t = model.network._matrix(alpha[node_t])
    q_step = model.network._ddpm_step_matrix(node_t, node_s)
    assert torch.allclose(torch.bmm(q_s, q_step), q_t, atol=1e-5)

    embeddings = model.embed(**inputs)
    prediction = model.predict(**embeddings)
    generation = model.generate(**embeddings, steps=1)
    metrics = model.evaluate(**generation)
    interpretation = AutoProteinInterpreter.from_config(config).explain(
        **generation
    )

    assert "batch" not in inputs
    assert inputs["edge_features"].shape[-1] == 93
    assert embeddings["structure_embedding"].shape == (4, 16)
    assert torch.isfinite(prediction["loss"])
    assert generation["sequences"]
    assert set(metrics) == {
        "sequence_recovery",
        "perplexity",
        "diversity",
    }
    grouped_metrics = model.evaluate(
        sequences=["AAAA", "CCCC", "AAAT", "CCCG"],
        reference_sequences=["AAAA", "CCCC", "AAAA", "CCCC"],
        sample_ids=["first", "second", "first", "second"],
    )
    assert grouped_metrics["diversity"] == 0.25
    assert interpretation["final_sequences"] == generation["sequences"]

    checkpoint = model.save_checkpoint(tmp_path / "mapdiff.pt")
    restored = AutoProteinModel(config, checkpoint=checkpoint)
    assert restored.architecture == "mapdiff-v1.0.1"


def test_mapdiff_public_workflows_smoke(monkeypatch, tmp_path):
    graph_path = tmp_path / "structure.pt"
    torch.save(
        {
            "atom_pos": _coordinates(),
            "sequence": "ACDE",
            "id": "tiny",
        },
        graph_path,
    )
    config = _tiny_config()
    modules = {
        name: importlib.import_module(
            f"examples.mapdiff_inverse_folding.{name}"
        )
        for name in ("train", "evaluate", "generate", "interpret")
    }

    class TinyConfigFactory:
        @staticmethod
        def from_pretrained(model_id):
            assert model_id == "InverseFolding/MapDiff"
            return config

    def tiny_model(*args, **kwargs):
        model = MapDiffModel(config)
        checkpoint = kwargs.get("checkpoint")
        if checkpoint is not None:
            model.load_checkpoint(checkpoint)
        return model

    for module in modules.values():
        monkeypatch.setattr(
            module, "AutoProteinConfig", TinyConfigFactory
        )
        monkeypatch.setattr(module, "AutoProteinModel", tiny_model)

    ipa_checkpoint = tmp_path / "ipa.pt"
    diffusion_checkpoint = tmp_path / "diffusion.pt"
    generated_path = tmp_path / "generated.json"
    assert modules["train"].main(
        [
            str(graph_path),
            "--stage",
            "ipa",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--output",
            str(ipa_checkpoint),
        ]
    ) == ipa_checkpoint
    assert modules["train"].main(
        [
            str(graph_path),
            "--stage",
            "diffusion",
            "--checkpoint",
            str(ipa_checkpoint),
            "--validation-data",
            str(graph_path),
            "--validation-steps",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--output",
            str(diffusion_checkpoint),
        ]
    ) == diffusion_checkpoint
    metrics = modules["evaluate"].main(
        [
            str(graph_path),
            "--checkpoint",
            str(diffusion_checkpoint),
            "--steps",
            "1",
        ]
    )
    generation = modules["generate"].main(
        [
            str(graph_path),
            "--checkpoint",
            str(diffusion_checkpoint),
            "--steps",
            "1",
            "--num-samples",
            "2",
            "--output",
            str(generated_path),
        ]
    )
    interpretation = modules["interpret"].main(
        [
            str(graph_path),
            "--checkpoint",
            str(diffusion_checkpoint),
            "--steps",
            "1",
        ]
    )

    assert ipa_checkpoint.is_file()
    assert diffusion_checkpoint.is_file()
    assert generated_path.is_file()
    assert {"sequence_recovery", "perplexity", "diversity"} <= metrics.keys()
    assert generation["sequences"]
    serialized = json.loads(generated_path.read_text(encoding="utf-8"))
    assert len(serialized["sequences"]) == 2
    assert len(serialized["trajectories"]) == 2
    assert interpretation["final_sequences"]
