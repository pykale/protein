import pytest

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinModel,
)


def _write_bindingdb(root):
    dataset_dir = root / "bindingdb"
    dataset_dir.mkdir()
    (dataset_dir / "full.csv").write_text(
        "SMILES,Protein,Y\nCCO,MKTFFVLLLMKTFFVLLL,1\nCCN,MKTFFVLLLMKTFFVLLL,0\n",
        encoding="utf-8",
    )


def test_loader_composes_dataset_preprocessor_and_collator(fake_rdkit_graph, tmp_path):
    _write_bindingdb(tmp_path)

    loader = AutoProteinDataLoader(
        "BindingDB/DTI",
        for_model="DTI/DrugBAN",
        root=tmp_path,
        batch_size=2,
    )
    batch = next(iter(loader))
    model = AutoProteinModel.from_config(loader.config)
    embeddings = model.embed(**batch)
    prediction = model.predictor(**embeddings)

    assert loader.config["model_id"] == "DTI/DrugBAN"
    assert len(loader.dataset) == 2
    assert set(loader.processed) == {"samples"}
    assert loader.preprocessor is not None
    assert loader.collator is not None
    assert loader.loader is not None
    assert len(loader) == 1
    assert type(batch) is dict
    assert set(batch) == {"drug", "target", "ids", "label"}
    assert batch["label"].tolist() == [1.0, 0.0]
    assert prediction["probabilities"].shape == (2,)
    assert not hasattr(loader, "model")


def test_loader_accepts_in_memory_data_and_custom_components():
    class Preprocessor:
        def process(self, dataset):
            return {
                "samples": [{"value": item * 2} for item in dataset],
                "offset": 1,
            }

    class Collator:
        def __call__(self, *, samples, offset):
            return {
                "values": [sample["value"] + offset for sample in samples]
            }

    loader = AutoProteinDataLoader(
        "Memory/Regression",
        dataset=[1, 2, 3],
        preprocessor=Preprocessor(),
        collator=Collator(),
        batch_size=2,
    )

    assert list(loader) == [{"values": [3, 5]}, {"values": [7]}]


def test_loader_from_config_reads_data_card_entry(fake_rdkit_graph, tmp_path):
    _write_bindingdb(tmp_path)
    data = AutoProteinConfig.from_pretrained("DTI/DrugBAN").to_dict()
    data["data"] = {
        "id": "BindingDB/DTI",
        "root": str(tmp_path),
        "batch_size": 2,
    }
    config = AutoProteinConfig.from_dict(data)

    loader = AutoProteinDataLoader.from_config(config)

    assert loader.data_id == "BindingDB/DTI"
    assert next(iter(loader))["label"].shape == (2,)


def test_loader_rejects_missing_pipeline_selection():
    with pytest.raises(ValueError, match="requires config=.*or for_model"):
        AutoProteinDataLoader("Memory/DTI", dataset=[])


def test_loader_rejects_dataset_model_task_mismatch():
    with pytest.raises(ValueError, match="targets task 'InverseFolding'.*targets 'dti'"):
        AutoProteinDataLoader(
            "Memory/InverseFolding",
            for_model="DTI/DrugBAN",
            dataset=[],
        )


def test_loader_requires_mapping_stage_outputs():
    class Preprocessor:
        def process(self, dataset):
            return {"samples": list(dataset)}

    def bad_collator(*, samples):
        return list(samples)

    loader = AutoProteinDataLoader(
        "Memory/Regression",
        dataset=[1],
        preprocessor=Preprocessor(),
        collator=bad_collator,
    )
    with pytest.raises(TypeError, match="collators must return a mapping"):
        next(iter(loader))


def test_loader_requires_string_batch_keys_for_keyword_expansion():
    class Preprocessor:
        def process(self, dataset):
            return {"samples": list(dataset)}

    def bad_collator(*, samples):
        return {0: list(samples)}

    loader = AutoProteinDataLoader(
        "Memory/Regression",
        dataset=[1],
        preprocessor=Preprocessor(),
        collator=bad_collator,
    )
    with pytest.raises(TypeError, match="batch keys must be strings"):
        next(iter(loader))
