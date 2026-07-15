import subprocess
import sys
from pathlib import Path


def test_pyproject_packages_only_kaleprotein():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '[project.optional-dependencies]' in text
    assert "drugban = [" in text
    assert "mapdiff = [" in text
    assert "dev = [" in text
    assert '"**/*.yaml"' in text
    assert '"**/*.map"' in text
    assert 'include = ["kaleprotein*"]' in text
    assert 'name = "kaleprotein"' in text
    assert "kaleprotein.tests" not in text
    assert "examples*" not in text
    assert "tests*" not in text


def test_base_import_and_dti_data_do_not_require_model_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dataset_dir = tmp_path / "bindingdb"
    dataset_dir.mkdir()
    (dataset_dir / "full.csv").write_text(
        "SMILES,Protein,Y\nCCO,MKT,1\n", encoding="utf-8"
    )
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        "from kaleprotein.auto import AutoProteinData; "
        f"assert AutoProteinData('BindingDB/DTI', root={str(tmp_path)!r})[0]['label'] == 1"
    )

    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_repository_uses_auto_core_examples_layers_only():
    root = Path(__file__).resolve().parents[1]
    package = root / "kaleprotein"
    assert (package / "auto" / "modeling.py").is_file()
    assert (package / "core" / "registry").is_dir()
    assert (package / "core" / "data" / "utils" / "fasta.py").is_file()
    assert (package / "core" / "data" / "utils" / "pdb.py").is_file()
    assert (package / "core" / "data" / "utils" / "mmcif.py").is_file()
    assert (package / "core" / "data" / "bindingdb.py").is_file()
    assert (package / "core" / "data" / "cath.py").is_file()
    assert (package / "core" / "preprocessing" / "sequence.py").is_file()
    assert (package / "core" / "modeling" / "modalities" / "sequence").is_dir()
    assert (package / "core" / "modeling" / "tasks" / "dti").is_dir()
    assert (package / "core" / "evaluation" / "tasks" / "dti").is_dir()
    assert (package / "core" / "modeling" / "modalities" / "sequence" / "embedders.py").is_file()
    assert (package / "core" / "modeling" / "tasks" / "dti" / "predictors.py").is_file()
    assert (package / "core" / "evaluation" / "tasks" / "dti" / "metrics.py").is_file()
    assert (
        package / "core" / "evaluation" / "tasks" / "inverse_folding" / "interpreters.py"
    ).is_file()
    assert (root / "examples" / "drugban_dti" / "modeling.py").is_file()
    assert (root / "examples" / "mapdiff_inverse_folding" / "modeling.py").is_file()
    assert (root / "tests" / "test_registry.py").is_file()
    assert (root / "docs" / "architecture.md").is_file()
    assert not (package / "examples").exists()
    assert not (package / "tests").exists()

    for legacy_data_dir in ("collators", "datasets", "modalities", "preprocessors", "tasks"):
        assert not any((package / "core" / "data" / legacy_data_dir).rglob("*.py"))
    assert not any((package / "core" / "modalities").rglob("*.py"))
    assert not any((package / "core" / "tasks").rglob("*.py"))

    for legacy in ("registry", "modalities", "tasks", "fusion", "heads", "runners", "conditioners"):
        assert not any((package / legacy).glob("*.py"))
