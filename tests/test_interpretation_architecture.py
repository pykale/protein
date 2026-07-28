from pathlib import Path

import torch

from kaleprotein.auto import AutoProteinConfig, AutoProteinInterpreter


def test_auto_interpreter_discovers_dti_implementation_independently():
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    interpreter = AutoProteinInterpreter.from_config(config)
    output = interpreter.explain(
        attention=torch.ones(1, 2, 2, 3),
        molecule_mask=torch.tensor([[True, True]]),
        protein_mask=torch.tensor([[True, True, False]]),
        molecule_atom_symbols=[["C", "O"]],
        protein_sequences=["AC"],
    )

    assert len(output["samples"]) == 1
    assert [item["symbol"] for item in output["samples"][0]["atoms"]] == [
        "C",
        "O",
    ]
    assert [
        item["residue"] for item in output["samples"][0]["residues"]
    ] == ["A", "C"]


def test_auto_interpreter_discovers_inverse_folding_implementation_independently():
    config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")
    interpreter = AutoProteinInterpreter.from_config(config)
    output = interpreter.explain(
        trajectory=[
            {"timestep": 2, "sequences": ["AA"]},
            {"timestep": 0, "sequences": ["AC"]},
        ]
    )

    assert output["steps"] == 1
    assert output["trajectory"][1]["changed_residues"] == [1]
    assert output["final_sequences"] == ["AC"]


def test_auto_interpretation_does_not_import_evaluation_dispatch():
    root = Path(__file__).resolve().parents[1]
    interpretation_source = (
        root / "kaleprotein" / "auto" / "interpret.py"
    ).read_text(encoding="utf-8")
    evaluation_source = (
        root / "kaleprotein" / "auto" / "evaluate.py"
    ).read_text(encoding="utf-8")

    assert "from .evaluation" not in interpretation_source
    assert "kaleprotein.interpret.{method}" in interpretation_source
    assert "kaleprotein.interpret.tasks" not in interpretation_source
    assert "kaleprotein.evaluate.{metric_name}" in evaluation_source
    assert "kaleprotein.evaluate.tasks" not in evaluation_source
    assert "INTERPRETER_REGISTRY" not in evaluation_source
