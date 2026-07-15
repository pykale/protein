from kaleprotein.auto import AutoProteinConfig, AutoProteinModel, AutoProteinPreprocessor


def test_drugban_complete_model_predicts(fake_rdkit_graph):
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    sample = AutoProteinPreprocessor.from_config(config).transform_sample(
        {"smiles": "CCO", "sequence": "MKTFFVLLLMKTFFVLLL", "label": 1}
    )
    model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
    output = model.predict(sample)

    assert "logits" in output and "probabilities" in output
    assert model.protein_embedder is not model.molecule_embedder
    assert model.predictor is not model.protein_embedder
