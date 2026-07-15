"""Reusable dataset classes and task-level sample normalization."""

from collections.abc import Sequence
from pathlib import Path

from .schemas import DTISample
from .utils import read_dict_rows, require_file, safe_path_part


class ListDataset(Sequence):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def to_list(self):
        return list(self.samples)


class DTIDataset(ListDataset):
    """Common list-like contract for drug-target interaction datasets."""

    def __init__(self, *, name, path, samples, split="full", subset=None):
        super().__init__(samples)
        self.name = name
        self.path = Path(path)
        self.split = split
        self.subset = subset

    @property
    def labels(self):
        return [sample["label"] for sample in self.samples]


class DTICsvDataset(DTIDataset):
    """Base for common DTI datasets distributed as CSV files."""

    dataset_id = None
    directory_name = None
    default_split = "full"

    def __init__(self, root=None, split=None, subset=None, limit=None, path=None):
        if self.dataset_id is None or self.directory_name is None:
            raise TypeError("DTICsvDataset subclasses must define dataset_id and directory_name.")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
        ):
            raise ValueError("limit must be a non-negative integer or None.")
        split = split or self.default_split
        csv_path = self.resolve_path(root=root, split=split, subset=subset, path=path)
        fieldnames, rows = read_dict_rows(csv_path)
        samples = _normalize_dti_rows(
            rows,
            fieldnames=fieldnames,
            dataset_id=self.dataset_id,
            path=csv_path,
            limit=limit,
        )
        super().__init__(
            name=self.dataset_id,
            path=csv_path,
            samples=samples,
            split=split,
            subset=subset,
        )

    def resolve_path(self, *, root=None, split="full", subset=None, path=None):
        if path is not None:
            return require_file(path, description=f"{self.dataset_id} dataset file")
        if root is None:
            raise ValueError(
                f"{self.dataset_id} requires a local dataset root. "
                "Pass root='path/to/datasets' or path='path/to/file.csv'."
            )
        root = Path(root)
        dataset_root = (
            root
            if root.name.casefold() == self.directory_name.casefold()
            else root / self.directory_name
        )
        if split == "full":
            return require_file(
                dataset_root / "full.csv",
                description=f"{self.dataset_id} dataset file",
            )
        if subset is None:
            raise ValueError(
                f"Loading split {split!r} for {self.dataset_id} requires subset "
                "such as 'train', 'test', or 'target_test'."
            )
        split = safe_path_part(split, name="split")
        subset = safe_path_part(subset, name="subset")
        filename = subset if subset.casefold().endswith(".csv") else f"{subset}.csv"
        return require_file(
            dataset_root / split / filename,
            description=f"{self.dataset_id} dataset file",
        )


def _normalize_dti_rows(rows, *, fieldnames, dataset_id, path, limit=None):
    fields = {field.casefold().strip(): field for field in fieldnames}
    smiles_field = _find_column(fields, _SMILES_COLUMNS, path)
    sequence_field = _find_column(fields, _SEQUENCE_COLUMNS, path)
    label_field = _find_column(fields, _LABEL_COLUMNS, path)
    id_field = _find_optional_column(fields, _ID_COLUMNS)
    samples = []
    for index, row in enumerate(rows):
        if limit is not None and len(samples) >= limit:
            break
        sample_id = row.get(id_field, "").strip() if id_field else ""
        sample_id = sample_id or f"{dataset_id}:{index}"
        smiles = row[smiles_field].strip()
        sequence = row[sequence_field].strip()
        if not smiles:
            raise ValueError(f"Empty SMILES at row {index + 2} in DTI dataset CSV: {path}")
        if not sequence:
            raise ValueError(
                f"Empty protein sequence at row {index + 2} in DTI dataset CSV: {path}"
            )
        mapped_fields = {smiles_field, sequence_field, label_field}
        if id_field:
            mapped_fields.add(id_field)
        samples.append(
            DTISample(
                identifier=sample_id,
                smiles=smiles,
                sequence=sequence,
                label=_parse_label(row[label_field], path=path, row_number=index + 2),
                dataset=dataset_id,
                metadata={
                    "row_index": index,
                    "source_path": str(path),
                    "source_columns": {
                        "smiles": smiles_field,
                        "sequence": sequence_field,
                        "label": label_field,
                    },
                    "extra_fields": {
                        key: value for key, value in row.items() if key not in mapped_fields
                    },
                },
            ).to_dict()
        )
    return samples


def _find_column(fields, candidates, path):
    column = _find_optional_column(fields, candidates)
    if column is None:
        raise ValueError(f"Could not find any of columns {candidates!r} in DTI CSV: {path}")
    return column


def _find_optional_column(fields, candidates):
    for candidate in candidates:
        key = candidate.casefold().strip()
        if key in fields:
            return fields[key]
    return None


def _parse_label(value, *, path, row_number):
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError as exc:
        raise ValueError(
            f"DTI labels must be numeric; got {value!r} at row {row_number} in {path}."
        ) from exc
    return int(as_float) if as_float.is_integer() else as_float


_SMILES_COLUMNS = (
    "SMILES",
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
    "Protein",
    "protein",
    "target",
    "Target",
)
_LABEL_COLUMNS = ("Label", "Y", "interaction", "Interaction", "bind")
_ID_COLUMNS = ("id", "ID", "sample_id", "Sample ID")


__all__ = ["DTICsvDataset", "DTIDataset", "ListDataset"]
