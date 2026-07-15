from kale_protein.auto import AutoProteinModel
from kale_protein.core.data.tasks.inverse_folding.collators import CollatorDiff
from kale_protein.core.data.tasks.inverse_folding.datasets import build_residue_graph


def test_mapdiff_complete_model_generates():
    coordinates = [
        [[-1.2, 0.2, 0.0], [0.0, 0.0, 0.0], [1.4, 0.3, 0.1], [2.0, 1.2, 0.0]],
        [[2.6, -0.7, 0.0], [3.8, 0.0, 0.0], [5.2, 0.3, 0.1], [5.8, 1.2, 0.0]],
    ]
    batch = CollatorDiff()([build_residue_graph(coordinates, "MA")])
    model = AutoProteinModel("InverseFolding/MapDiff", pretrain=False)
    batch_fields = {"batch": batch}
    conditioning = model.embed(**batch_fields)
    output = model.predictor.generate(**conditioning, steps=1)

    assert "structure_embedding" in conditioning
    assert output["sequences"]
