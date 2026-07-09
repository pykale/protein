# KaleProtein

KaleProtein is a stream-based Auto workflow framework for protein and small-molecule machine learning.

## Quick start

Run the unit tests:

```bash
pytest -q kale_protein/tests
```

Run the built-in toy DrugBAN workflow:

```bash
python kale_protein/examples/drugban_dti/evaluate.py
```

Run the built-in toy MapDiff workflow:

```bash
python kale_protein/examples/mapdiff_inverse_folding/generate.py
```

## Basic usage

```python
from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor

samples = [
    {
        "id": "sample_1",
        "smiles": "CCO",
        "sequence": "MKTFFVLLL",
        "label": 1,
    }
]

config = AutoProteinConfig.from_preset("drugban")
processed = AutoProteinPreprocessor.from_config(config).transform_dataset(samples)
predictor = AutoProteinPredictor.from_config(config)
outputs = predictor.predict(processed)
print(outputs)
```

## Customization

To configure your own data, model components, metrics, interpreters, or AutoX-style presets, see [CUSTOMIZE.md](CUSTOMIZE.md).

The most common workflow is:

1. Copy a preset from `kale_protein/presets/`.
2. Update `streams` so each stream points to the correct input field with `input_key`.
3. Select registered processors, encoders, fusion/conditioner modules, heads, runners, evaluators, and interpreters.
4. Load the config with `AutoProteinConfig.from_yaml("path/to/config.yaml")`.

## Continuous integration

The GitHub Actions workflow at `.github/workflows/tests.yml` runs tests on Python 3.10, 3.11, and 3.12, byte-compiles the package, and executes the toy examples.
