import pytest

from kale_protein.auto import AutoProteinData
from kale_protein.tasks.drug_target_interaction.datasets import DrugTargetInteractionDataset


def test_bindingdb_full_loader_normalizes_drugban_columns(tmp_path):
    dataset_dir = tmp_path / "bindingdb"
    dataset_dir.mkdir()
    (dataset_dir / "full.csv").write_text(
        "SMILES,Target Sequence,Label\nCCO,MKTFFVLLL,1\nCCC,GGGG,0\n",
        encoding="utf-8",
    )

    dataset = AutoProteinData("DTI/BindingDB", root=tmp_path)

    assert isinstance(dataset, DrugTargetInteractionDataset)
    assert len(dataset) == 2
    assert dataset[0]["smiles"] == "CCO"
    assert dataset[0]["sequence"] == "MKTFFVLLL"
    assert dataset[0]["label"] == 1
    assert dataset.labels == [1, 0]

    direct_dataset = AutoProteinData("DTI/bindingdb", root=dataset_dir, limit=1)
    assert len(direct_dataset) == 1
    assert direct_dataset[0]["smiles"] == "CCO"


def test_human_random_split_loader_supports_common_aliases(tmp_path):
    split_dir = tmp_path / "human" / "random"
    split_dir.mkdir(parents=True)
    (split_dir / "train.csv").write_text(
        "compound_iso_smiles,target_sequence,Y\nNCC,MAAA,1.0\n",
        encoding="utf-8",
    )

    dataset = AutoProteinData("DTI/Human", root=tmp_path, split="random", subset="train")

    assert len(dataset) == 1
    assert dataset.split == "random"
    assert dataset.subset == "train"
    assert dataset[0]["dataset"] == "DTI/Human"
    assert dataset[0]["label"] == 1


def test_biosnap_cluster_split_can_load_from_direct_path(tmp_path):
    csv_path = tmp_path / "biosnap_test.csv"
    csv_path.write_text("drug,protein,interaction\nCCN,MMMM,0\n", encoding="utf-8")

    dataset = AutoProteinData("DTI/BioSNAP", path=csv_path)

    assert len(dataset) == 1
    assert dataset.path == csv_path
    assert dataset[0]["smiles"] == "CCN"
    assert dataset[0]["sequence"] == "MMMM"
    assert dataset[0]["label"] == 0


def test_split_loader_requires_subset(tmp_path):
    (tmp_path / "bindingdb" / "random").mkdir(parents=True)

    with pytest.raises(ValueError, match="requires subset"):
        AutoProteinData("DTI/BindingDB", root=tmp_path, split="random")
