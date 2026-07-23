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


def test_repository_uses_flat_verb_oriented_package_layers():
    root = Path(__file__).resolve().parents[1]
    package = root / "kaleprotein"
    assert (package / "auto" / "model.py").is_file()
    assert (package / "auto" / "loaddata.py").is_file()
    assert (package / "auto" / "prepdata.py").is_file()
    assert (package / "auto" / "evaluate.py").is_file()
    assert (package / "auto" / "interpret.py").is_file()
    assert (package / "auto" / "config" / "model_config.py").is_file()
    assert (package / "auto" / "registry" / "base.py").is_file()
    for legacy_auto_module in (
        "configuration.py",
        "data.py",
        "preprocessing.py",
        "collation.py",
        "collate.py",
        "modeling.py",
        "evaluation.py",
        "interpretation.py",
    ):
        assert not (package / "auto" / legacy_auto_module).exists()
    expected_utils = {
        "evaluate_build_sequence_targets.py",
        "evaluate_extract_binary_inputs.py",
        "evaluate_extract_sequences.py",
        "loaddata_make_backbone_record.py",
        "loaddata_parse_mmcif.py",
        "loaddata_parse_pdb.py",
        "loaddata_read_dict_rows.py",
        "loaddata_read_fasta.py",
        "loaddata_require_file.py",
        "loaddata_safe_path_part.py",
        "model_load_checkpoint_state_dict.py",
        "model_move_to_device.py",
        "model_verify_checksum.py",
    }
    utils_dir = package / "utils"
    assert {
        path.name for path in utils_dir.glob("*.py") if path.name != "__init__.py"
    } == expected_utils
    operation_steps = {"loaddata", "prepdata", "model", "evaluate", "interpret"}
    for helper_path in utils_dir.glob("*.py"):
        if helper_path.name == "__init__.py":
            continue
        step, separator, helper_name = helper_path.stem.partition("_")
        assert separator and helper_name
        assert step in operation_steps
    assert (package / "loaddata" / "base_dataset.py").is_file()
    assert (package / "loaddata" / "bindingdb.py").is_file()
    assert (package / "loaddata" / "cath.py").is_file()
    assert (package / "prepdata" / "sequence.py").is_file()
    assert (package / "model" / "embed" / "sequence_cnn.py").is_file()
    assert (package / "model" / "embed" / "molecule_gcn.py").is_file()
    assert (package / "model" / "predict" / "dti_ban.py").is_file()
    for metric in (
        "accuracy",
        "auprc",
        "auroc",
        "diversity",
        "f1",
        "novelty",
        "perplexity",
        "sequence_recovery",
        "threshold",
    ):
        assert (package / "evaluate" / f"{metric}.py").is_file()
    for interpreter in ("bilinear_attention_map", "denoising_trajectory"):
        assert (package / "interpret" / f"{interpreter}.py").is_file()
    assert not (package / "evaluate" / "tasks").exists()
    assert not (package / "interpret" / "tasks").exists()
    assert not (package / "evaluate" / "_binary.py").exists()
    assert not (package / "evaluate" / "_sequence.py").exists()
    assert not any((package / "evaluate").rglob("interpreters.py"))
    assert (root / "examples" / "drugban_dti" / "model_drugban.py").is_file()
    assert (root / "examples" / "drugban_dti" / "collators.py").is_file()
    assert (root / "examples" / "mapdiff_inverse_folding" / "model_mapdiff.py").is_file()
    assert (root / "examples" / "mapdiff_inverse_folding" / "collators.py").is_file()
    assert not (root / "examples" / "_utils.py").exists()
    assert (root / "tests" / "test_registry.py").is_file()
    assert (root / "docs" / "architecture.md").is_file()
    assert not (package / "core").exists()
    assert not (package / "weights").exists()
    assert not (package / "examples").exists()
    assert not (package / "tests").exists()

    for legacy in ("data", "evaluation", "interpretation", "preprocessing", "modeling"):
        assert not (package / legacy).exists()
    for deep_model_dir in ("modalities", "tasks"):
        assert not (package / "model" / deep_model_dir).exists()

    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.rglob("*.py")
    )
    assert "kaleprotein.core" not in source


def test_example_models_do_not_own_data_workflow_objects():
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "examples/drugban_dti/model_drugban.py",
        "examples/mapdiff_inverse_folding/model_mapdiff.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        for forbidden in (
            "DataLoader",
            "self.collator",
            "make_dataloader",
            "LazyPreprocessedDataset",
            "load_requested_checkpoint",
            "extract_attention",
        ):
            assert forbidden not in source, f"{relative_path} contains {forbidden}"
