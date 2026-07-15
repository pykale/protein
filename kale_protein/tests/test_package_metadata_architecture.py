import subprocess
import sys
from pathlib import Path


def test_pyproject_declares_extras_and_package_data_exclusions():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '[project.optional-dependencies]' in text
    assert "drugban = [" in text
    assert "mapdiff = [" in text
    assert "dev = [" in text
    assert '"**/*.yaml"' in text
    assert '"**/*.map"' in text
    assert '"**/data/**/*"' in text
    assert '"**/weights/**/*"' in text


def test_base_import_and_dti_data_do_not_require_model_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[2]
    dataset_dir = tmp_path / "bindingdb"
    dataset_dir.mkdir()
    (dataset_dir / "full.csv").write_text(
        "SMILES,Protein,Y\nCCO,MKT,1\n", encoding="utf-8"
    )
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        "from kale_protein.auto import AutoProteinData; "
        f"assert AutoProteinData('DTI/BindingDB', root={str(tmp_path)!r})[0]['label'] == 1"
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
    package = Path(__file__).resolve().parents[1]
    assert (package / "auto" / "modeling.py").is_file()
    assert (package / "core" / "registry").is_dir()
    assert (package / "core" / "modalities" / "sequence").is_dir()
    assert (package / "core" / "tasks" / "dti").is_dir()
    assert (package / "examples" / "drugban_dti" / "modeling.py").is_file()
    assert (package / "examples" / "mapdiff_inverse_folding" / "modeling.py").is_file()

    for legacy in ("registry", "modalities", "tasks", "fusion", "heads", "runners", "conditioners"):
        assert not any((package / legacy).glob("*.py"))
