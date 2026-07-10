"""Dataset scaffolding for task-level data logic.

Task-level pair construction, splitting, and dataset loading should live here
rather than inside modality processors.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from kale_protein.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("DTI/PDBBind")
def load_pdbbind_example():
    return (
        {"id": "sample_1", "smiles": "CCO", "sequence": "MKTFFVLLL"},
        1,
    )


@dataclass(frozen=True)
class DrugTargetInteractionSample:
    id: str
    smiles: str
    sequence: str
    label: int | float
    dataset: str
    metadata: dict

    def to_dict(self):
        return {
            "id": self.id,
            "smiles": self.smiles,
            "sequence": self.sequence,
            "label": self.label,
            "dataset": self.dataset,
            "metadata": dict(self.metadata),
        }


class DrugTargetInteractionDataset:
    """List-like DTI dataset normalized to smiles/sequence/label samples."""

    def __init__(self, name, path, samples, split="full", subset=None):
        self.name = name
        self.path = Path(path)
        self.samples = list(samples)
        self.split = split
        self.subset = subset

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @property
    def labels(self):
        return [sample["label"] for sample in self.samples]

    def to_list(self):
        return list(self.samples)


class DrugTargetInteractionCSVLoader:
    """Load local DrugBAN-style DTI CSV datasets for any DTI model."""

    def __init__(self, dataset_key, directory_name, default_split="full"):
        self.dataset_key = dataset_key
        self.directory_name = directory_name
        self.default_split = default_split

    def __call__(self, root=None, split=None, subset=None, limit=None, path=None):
        csv_path = self.resolve_path(root=root, split=split, subset=subset, path=path)
        samples = _read_dti_csv(csv_path, self.dataset_key, limit=limit)
        return DrugTargetInteractionDataset(
            name=self.dataset_key,
            path=csv_path,
            samples=samples,
            split=split or self.default_split,
            subset=subset,
        )

    def resolve_path(self, root=None, split=None, subset=None, path=None):
        if path is not None:
            return _require_file(Path(path))
        if root is None:
            raise ValueError(
                f"{self.dataset_key} requires a local dataset root. "
                "Pass root='path/to/DrugBAN/datasets' or path='path/to/file.csv'."
            )
        root = Path(root)
        dataset_root = root if root.name == self.directory_name else root / self.directory_name
        split = split or self.default_split
        if split in (None, "full"):
            return _require_file(dataset_root / "full.csv")
        if subset is None:
            raise ValueError(
                f"Loading split {split!r} for {self.dataset_key} requires subset='train', 'val', or 'test'."
            )
        return _require_file(dataset_root / split / f"{subset}.csv")


def _register_dti_dataset(*ids, directory_name, default_split="full"):
    loader = DrugTargetInteractionCSVLoader(ids[0], directory_name, default_split=default_split)
    for dataset_id in ids:
        DATASET_REGISTRY.register(dataset_id, loader)


_register_dti_dataset("DTI/BindingDB", "DTI/bindingdb", directory_name="bindingdb")
_register_dti_dataset("DTI/Human", "DTI/human", directory_name="human")
_register_dti_dataset("DTI/BioSNAP", "DTI/biosnap", directory_name="biosnap")


def _require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DTI dataset file not found: {path}")
    if not path.is_file():
        raise ValueError(f"DTI dataset path is not a file: {path}")
    return path


def _read_dti_csv(path, dataset_name, limit=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"DTI dataset CSV has no header: {path}")
        fields = {field.lower().strip(): field for field in reader.fieldnames}
        smiles_field = _find_column(fields, _SMILES_COLUMNS, path)
        sequence_field = _find_column(fields, _SEQUENCE_COLUMNS, path)
        label_field = _find_column(fields, _LABEL_COLUMNS, path)
        id_field = _find_optional_column(fields, _ID_COLUMNS)
        samples = []
        for index, row in enumerate(reader):
            if limit is not None and len(samples) >= limit:
                break
            sample_id = row.get(id_field) if id_field else f"{dataset_name}:{index}"
            samples.append(
                DrugTargetInteractionSample(
                    id=sample_id,
                    smiles=row[smiles_field],
                    sequence=row[sequence_field],
                    label=_parse_label(row[label_field]),
                    dataset=dataset_name,
                    metadata={
                        "row_index": index,
                        "source_path": str(path),
                        "source_columns": {
                            "smiles": smiles_field,
                            "sequence": sequence_field,
                            "label": label_field,
                        },
                    },
                ).to_dict()
            )
    return samples


def _find_column(fields, candidates, path):
    column = _find_optional_column(fields, candidates)
    if column is None:
        raise ValueError(f"Could not find any of columns {candidates!r} in DTI dataset CSV: {path}")
    return column


def _find_optional_column(fields, candidates):
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in fields:
            return fields[key]
    return None


def _parse_label(value):
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text
    return int(as_float) if as_float.is_integer() else as_float


_SMILES_COLUMNS = (
    "SMILES",
    "smiles",
    "compound_iso_smiles",
    "compound_smiles",
    "drug_smiles",
    "Drug SMILES",
    "drug",
    "compound",
)
_SEQUENCE_COLUMNS = (
    "Target Sequence",
    "target_sequence",
    "protein_sequence",
    "sequence",
    "Sequence",
    "Protein",
    "protein",
    "target",
    "Target",
)
_LABEL_COLUMNS = (
    "Label",
    "label",
    "Y",
    "y",
    "interaction",
    "Interaction",
    "bind",
)
_ID_COLUMNS = (
    "id",
    "ID",
    "sample_id",
    "Sample ID",
)
