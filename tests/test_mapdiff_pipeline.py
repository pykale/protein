from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from kaleprotein import AutoProteinData, AutoProteinGenerator, AutoProteinModel, AutoProteinPreprocessor


def write_fake_graph(path: Path, length: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": F.one_hot(torch.arange(length) % 20, num_classes=20).float(),
            "pos": torch.zeros(length, 3),
        },
        path,
    )


def test_auto_protein_model_builds_mapdiff() -> None:
    model = AutoProteinModel("InverseFolding/MapDiff")
    structure_data = {
        "x": F.one_hot(torch.arange(4).unsqueeze(0) % 20, num_classes=20).float(),
        "pos": torch.zeros(1, 4, 3),
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "labels": torch.arange(4).unsqueeze(0),
    }

    structure_embedding = model.embed(structure_data)
    generation = AutoProteinGenerator("InverseFolding/MapDiff").generate(structure_embedding)
    output = model(**structure_data)

    assert generation["logits"].shape == (1, 4, 20)
    assert generation["sequences"].shape == (1, 1, 4)
    assert output["logits"].shape == (1, 4, 20)
    assert output["predictions"].shape == (1, 4)
    assert torch.isfinite(output["loss"])


def test_auto_protein_data_loads_mapdiff_graphs(tmp_path: Path) -> None:
    write_fake_graph(tmp_path / "train" / "sample.pt", length=5)
    data = AutoProteinData("InverseFolding/CATH", data_dir=tmp_path)

    batch = next(iter(data.dataloader("train")))
    loaded_data, _ = data.load("train")
    structure_data = AutoProteinPreprocessor("protein/structure").featurize(loaded_data)

    assert batch["x"].shape == (1, 5, 20)
    assert batch["pos"].shape == (1, 5, 3)
    assert batch["mask"].tolist() == [[True, True, True, True, True]]
    assert batch["labels"].shape == (1, 5)
    assert structure_data["x"].shape == (1, 5, 20)
