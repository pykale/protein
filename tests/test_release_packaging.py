from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _project_config():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_public_install_extras_have_expected_relationships():
    optional = _project_config()["optional-dependencies"]

    assert set(optional) == {"drugban_dti", "mapdiff_inverse_folding", "test", "dev"}
    assert optional["drugban_dti"] == ["rdkit>=2022.9"]
    assert optional["mapdiff_inverse_folding"] == ["torch-geometric>=2.4"]
    example_dependencies = set(optional["drugban_dti"]) | set(optional["mapdiff_inverse_folding"])
    test_dependencies = set(optional["test"])
    dev_dependencies = set(optional["dev"])

    assert dev_dependencies >= test_dependencies
    assert dev_dependencies.isdisjoint(example_dependencies)
    assert {"build>=1.2", "ruff>=0.9", "twine>=6.0"} <= dev_dependencies
    for name in ("drugban_dti", "mapdiff_inverse_folding"):
        assert (ROOT / "examples" / name / "config.yaml").is_file()


def test_pypi_metadata_is_release_ready():
    project = _project_config()

    assert "torch>=2.0" in project["dependencies"]
    assert "PyYAML>=6.0" in project["dependencies"]
    assert project["name"] == "kaleprotein"
    assert project["license"] == "MIT"
    assert set(project["license-files"]) == {"LICENSE", "THIRD_PARTY_NOTICES.md"}
    assert project["requires-python"] == ">=3.10"
    assert project["urls"]["Source"] == "https://github.com/pykale/protein"
    assert "Development Status :: 3 - Alpha" in project["classifiers"]


def test_publish_workflow_uses_trusted_publishing():
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "types: [published]" in workflow
    assert "environment:\n      name: testpypi" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert workflow.count("id-token: write") == 2
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "PYPI_API_TOKEN" not in workflow
