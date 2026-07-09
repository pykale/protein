# KaleProtein

KaleProtein is a stream-based, registry-driven Auto workflow framework for protein and small-molecule machine learning. It follows the PyKale six-step pattern:

```text
Load Data → Preprocess Data → Embed → Predict/Generate → Evaluate → Interpret
```

The first built-in presets are:

- **DrugBAN**: drug-target interaction prediction with a small-molecule stream, a protein-sequence stream, bilinear attention fusion, and a binary classification head.
- **MapDiff**: structure-conditioned inverse folding with a protein-structure stream, a noisy/masked protein-sequence stream, a denoising conditioner, and a diffusion-style sequence decoder.

These are intentionally lightweight, toy-runnable implementations that prioritize clean extension points over full reproduction of the original papers.

## Repository layout

```text
kale_protein/
  auto/          # AutoProteinConfig, preprocessing, prediction, evaluation, interpretation
  registry/      # Component registries
  modalities/    # Stream processors and encoders
  fusion/        # Cross-stream fusion modules
  conditioners/  # Conditional generation modules
  heads/         # Prediction and generation heads
  runners/       # Predict/generate runners
  tasks/         # Metrics and interpreters
  presets/       # Built-in DrugBAN and MapDiff configs
  examples/      # Toy runnable examples
  tests/         # Unit tests
```

## Quick start

### 1. Run tests

```bash
pytest -q kale_protein/tests
```

### 2. Run the DrugBAN toy example

```bash
python kale_protein/examples/drugban_dti/evaluate.py
```

### 3. Run the MapDiff toy example

```bash
python kale_protein/examples/mapdiff_inverse_folding/generate.py
```

## Use the AutoProtein API

### DrugBAN-style prediction

```python
from kale_protein.auto import (
    AutoProteinConfig,
    AutoProteinPreprocessor,
    AutoProteinPredictor,
    AutoProteinEvaluator,
)

samples = [
    {
        "id": "sample_1",
        "smiles": "CCO",
        "sequence": "MKTFFVLLL",
        "label": 1,
    },
    {
        "id": "sample_2",
        "smiles": "CCN",
        "sequence": "GAVLIPFWY",
        "label": 0,
    },
]

config = AutoProteinConfig.from_preset("drugban")
processed = AutoProteinPreprocessor.from_config(config).transform_dataset(samples)
predictor = AutoProteinPredictor.from_config(config)
outputs = predictor.predict(processed)
metrics = AutoProteinEvaluator.from_config(config).evaluate(outputs, processed)

print(metrics)
```

### MapDiff-style generation

```python
from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor

samples = [
    {
        "id": "protein_1",
        "backbone_coords": [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        "sequence": "MA",
    }
]

config = AutoProteinConfig.from_preset("mapdiff")
processed = AutoProteinPreprocessor.from_config(config).transform_dataset(samples)
predictor = AutoProteinPredictor.from_config(config)
generated = predictor.generate(processed[0])

print(generated)
```

## Use your own config

Copy one of the built-in presets and edit the stream, model, and evaluation sections:

```bash
cp kale_protein/presets/drugban.yaml my_drugban.yaml
```

Then load it:

```python
from kale_protein.auto import AutoProteinConfig

config = AutoProteinConfig.from_yaml("my_drugban.yaml")
```

A config describes:

- `task`: the task family, such as `drug_target_interaction` or `inverse_folding`.
- `objective`: `discriminative` or `generative`.
- `runner`: the execution mode, such as `predict` or `diffusion_generate`.
- `streams`: independent modality inputs with processors and encoders.
- `fusion` or `conditioner`: cross-stream computation.
- `head`: task output layer.
- `evaluation`: metric names.
- `interpretation`: explanation method.

See [`CUSTOMIZE.md`](CUSTOMIZE.md) for detailed instructions on configuring your own data, model components, metrics, interpreters, and AutoX-style presets.

## Built-in presets

### DrugBAN

```python
from kale_protein.auto import AutoProteinConfig

config = AutoProteinConfig.from_preset("drugban")
```

The DrugBAN preset is defined in `kale_protein/presets/drugban.yaml` and composes:

- small-molecule `rdkit_graph` processor
- protein-sequence `amino_acid_tokenizer`
- `drugban_molecule_gnn` and `drugban_protein_cnn` encoders
- `bilinear_attention` fusion
- `binary_classifier` head
- DTI metrics such as AUROC, AUPRC, accuracy, and F1

### MapDiff

```python
from kale_protein.auto import AutoProteinConfig

config = AutoProteinConfig.from_preset("mapdiff")
```

The MapDiff preset is defined in `kale_protein/presets/mapdiff.yaml` and composes:

- protein-structure `backbone_coordinate_processor`
- protein-sequence `masked_sequence_tokenizer`
- `mapdiff_structure_encoder` and `residue_token_embedding` encoders
- `structure_conditioned_denoising` conditioner
- `diffusion_sequence_decoder` head
- inverse-folding metrics such as sequence recovery, diversity, and novelty

## Add your own component

Register new components with the relevant registry and refer to them from config:

```python
from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY

@MODALITY_PROCESSOR_REGISTRY.register(("protein_sequence", "my_tokenizer"))
class MyTokenizer:
    def __init__(self, input_key="sequence", **kwargs):
        self.input_key = input_key

    def transform(self, sample):
        sequence = sample[self.input_key]
        return {"sequence": sequence, "tokens": [ord(char) for char in sequence]}
```

Config:

```yaml
streams:
  target:
    modality: protein_sequence
    input_key: sequence
    processor: my_tokenizer
    encoder: drugban_protein_cnn
```

## Continuous integration

This repo includes a GitHub Actions workflow in `.github/workflows/tests.yml` that runs tests on Python 3.10, 3.11, and 3.12, byte-compiles the package, and executes both toy examples.
