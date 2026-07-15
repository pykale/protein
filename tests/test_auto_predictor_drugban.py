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
    output = model.predictor(**embeddings)

    assert "logits" in output and "probabilities" in output
    assert model.protein_embedder is not model.molecule_embedder
    assert model.predictor is not model.protein_embedder
