import pytest

from kaleprotein.utils import (
    move_to_device,
    parse_mmcif,
    parse_pdb,
    read_csv,
    read_fasta,
    read_tsv,
)


class _Movable:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


def test_fasta_and_tabular_readers_are_dataset_independent(tmp_path):
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(
        ">protein-1 first chain\nMKT\nFF\n>protein-2\nacde\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("id,sequence\nfirst,MKT\n", encoding="utf-8")
    tsv_path = tmp_path / "records.tsv"
    tsv_path.write_text("id\tsequence\nsecond\tACD\n", encoding="utf-8")

    records = read_fasta(fasta_path)

    assert [record.identifier for record in records] == ["protein-1", "protein-2"]
    assert [record.sequence for record in records] == ["MKTFF", "ACDE"]
    assert read_csv(csv_path) == [{"id": "first", "sequence": "MKT"}]
    assert read_tsv(tsv_path) == [{"id": "second", "sequence": "ACD"}]


def test_pdb_and_mmcif_parsers_return_the_same_structure_contract(tmp_path):
    pdb_path = tmp_path / "protein.pdb"
    pdb_path.write_text(
        """ATOM      1  N   ALA A   1      -1.200   0.200   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       1.400   0.300   0.100  1.00 20.00           C
ATOM      4  O   ALA A   1       2.000   1.200   0.000  1.00 20.00           O
END
""",
        encoding="utf-8",
    )
    cif_path = tmp_path / "protein.cif"
    cif_path.write_text(
        """data_protein
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM N  . ALA A 1 -1.200 0.200 0.000 1
ATOM CA . ALA A 1  0.000 0.000 0.000 1
ATOM C  . ALA A 1  1.400 0.300 0.100 1
ATOM O  . ALA A 1  2.000 1.200 0.000 1
#
""",
        encoding="utf-8",
    )

    pdb_record = parse_pdb(pdb_path, chain="A")
    cif_record = parse_mmcif(cif_path, chain="A")

    assert pdb_record.sequence == cif_record.sequence == "A"
    assert pdb_record.atom_mask == cif_record.atom_mask == [[True, True, True, True]]
    assert pdb_record.atom_pos == cif_record.atom_pos


def test_fasta_reader_rejects_sequence_before_header(tmp_path):
    path = tmp_path / "invalid.fasta"
    path.write_text("MKT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="before the first header"):
        read_fasta(path)


def test_move_to_device_preserves_nested_container_shapes():
    first = _Movable()
    second = _Movable()

    moved = move_to_device(
        {"list": [first, "metadata"], "tuple": (second, 3)},
        "test-device",
    )

    assert moved == {"list": [first, "metadata"], "tuple": (second, 3)}
    assert first.device == second.device == "test-device"
