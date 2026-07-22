from kaleprotein.auto import (
    AutoProteinCollator,
    AutoProteinConfig,
    AutoProteinModel,
    AutoProteinPreprocessor,
)


def test_drugban_complete_model_predicts(fake_rdkit_graph):
    config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
    processed = AutoProteinPreprocessor.from_config(config).process(
        [{"smiles": "CCO", "sequence": "MKTFFVLLLMKTFFVLLL", "label": 1}]
    )
    collator = AutoProteinCollator.from_config(config)
    batch = collator(**processed)
    model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
    embeddings = model.embed(**batch)
    output = model.predict(**embeddings)

    assert "logits" in output and "probabilities" in output
    assert model.protein_embedder is not model.molecule_embedder
    assert type(model.molecule_embedder).__module__ == (
        "kaleprotein.model.embed.molecule_gcn"
    )
    assert type(model.protein_embedder).__module__ == (
        "kaleprotein.model.embed.sequence_cnn"
    )
    assert type(model.predictor).__module__ == "kaleprotein.model.predict.dti_ban"
    assert AutoProteinCollator.__module__ == "kaleprotein.auto.loaddata"
    assert callable(model.predict)
