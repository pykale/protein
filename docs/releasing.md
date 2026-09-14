# Releasing KaleProtein

KaleProtein publishes one pure-Python distribution named `kaleprotein`. The
wheel contains only the `kaleprotein` package. Examples, tests, documentation,
and release scripts remain available in the source archive and Git repository.

## Installation Contracts

The public installation contracts are:

```bash
python -m pip install kaleprotein
python -m pip install "kaleprotein[example-name]"
python -m pip install "kaleprotein[dev]"
```

Replace `example-name` with the selected example's folder name. The base
package requires PyTorch and PyYAML. `drugban_dti` adds RDKit, and
`mapdiff_inverse_folding` adds PyTorch Geometric. There is no aggregate
`examples` extra. `dev` contains test, build, lint, and publishing tools only;
example dependencies are selected separately. All extras inherit the base
dependencies. Packaging extras add dependencies, not example source files.

`python -m kaleprotein download-example <name>` fetches only that example's
source and small assets from the installed version's release tag. Keep the
tagged examples compatible with the wheel published for that tag. The command
does not install extras or download datasets and pretrained weights.

## One-Time Trusted Publishing Setup

KaleProtein uses PyPI Trusted Publishing. Do not add a long-lived PyPI token to
GitHub secrets.

Create pending publishers with these values on both PyPI and TestPyPI:

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project | `kaleprotein` | `kaleprotein` |
| GitHub owner | `pykale` | `pykale` |
| Repository | `protein` | `protein` |
| Workflow | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

Create matching GitHub environments named `pypi` and `testpypi`. Configure the
`pypi` environment with required reviewers so every production upload requires
manual approval.

## Validate A Candidate

Start from a clean checkout and install all development dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[drugban_dti]"
python -m pip install -e ".[mapdiff_inverse_folding]"
python -m pytest -q
python -m compileall -q kaleprotein
python -m build
python -m twine check --strict dist/*
python scripts/check_distribution.py dist
```

The distribution check verifies metadata, extras, package boundaries, source
archive contents, and the version shared by the wheel and
`kaleprotein/_version.txt`.

Run the `Publish distributions` workflow manually to upload the candidate to
TestPyPI. Test the uploaded package from a fresh virtual environment:

```bash
VERSION=$(cat kaleprotein/_version.txt)
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "kaleprotein==${VERSION}"
python -c "import kaleprotein; print(kaleprotein.__version__)"
```

The extra index is required because scientific dependencies are normally
resolved from production PyPI rather than duplicated on TestPyPI. Repeat with
each example extra separately to verify its dependency set.

## Publish To PyPI

1. Update `kaleprotein/_version.txt` to a valid PEP 440 version.
2. Merge the tested release commit into `main`.
3. Create the matching tag, such as `v0.1.0a1`.
4. Publish a GitHub Release for that tag.
5. Approve the `pypi` environment deployment after the build checks pass.
6. Verify `python -m pip install kaleprotein` in a fresh environment.

The workflow refuses a GitHub Release whose tag does not exactly match
`v<contents of kaleprotein/_version.txt>`. PyPI does not allow replacing an
uploaded version, so every retry after a successful upload requires a new
version.
