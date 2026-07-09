# Customizing KaleProtein

KaleProtein is designed around the PyKale-style six-step workflow:

```text
Load Data → Preprocess Data → Embed → Predict/Generate → Evaluate → Interpret
```

Instead of hard-coding a model such as DrugBAN or MapDiff, KaleProtein composes a workflow from a config file and registries. This document explains how to bring your own data, model components, metrics, interpreters, and AutoX-style presets.

## 1. Start from a preset

Use the built-in presets as templates:

- `kale_protein/presets/drugban.yaml` for drug-target interaction prediction.
- `kale_protein/presets/mapdiff.yaml` for structure-conditioned inverse folding.

```python
from kale_protein.auto import AutoProteinConfig

config = AutoProteinConfig.from_preset("drugban")
print(config.to_dict())
```

To customize, copy a preset to your own config path and update the sections described below.

## 2. Configure your own data

A KaleProtein sample is a Python dictionary. Each stream reads one field from that dictionary using `input_key`.

For a DrugBAN-like task:

```python
samples = [
    {
        "id": "pair_001",
        "smiles": "CCO",
        "sequence": "MKTFFVLLL",
        "label": 1,
    }
]
```

For a MapDiff-like task:

```python
samples = [
    {
        "id": "protein_001",
        "backbone_coords": [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        "sequence": "MA",
    }
]
```

Then point each stream at the correct input field:

```yaml
streams:
  drug:
    modality: small_molecule
    input_key: smiles
    processor: rdkit_graph
    encoder: drugban_molecule_gnn

  target:
    modality: protein_sequence
    input_key: sequence
    processor: amino_acid_tokenizer
    encoder: drugban_protein_cnn
```

Task-level fields such as `label`, `domain`, `id`, and `metadata` are preserved by the preprocessor.

## 3. Configure processors and encoders

Processors convert raw sample fields into one stream-specific representation. Encoders consume exactly one processed stream.

Example processor customization:

```yaml
streams:
  target:
    modality: protein_sequence
    input_key: protein_seq
    processor: amino_acid_tokenizer
    encoder: drugban_protein_cnn
    processor_kwargs:
      max_length: 512
    encoder_kwargs:
      hidden_dim: 128
      output_dim: 128
```

Important boundaries:

- A small-molecule processor should not read protein sequence fields.
- A protein sequence processor should not build drug-target pairs.
- Cross-stream logic belongs in a fusion module or conditioner.
- Pair construction, negative sampling, and cold splits belong in task-level data logic.

## 4. Configure the model composition

### Discriminative workflows

A discriminative workflow usually has streams, a fusion or conditioner, a head, metrics, and an interpreter:

```yaml
name: my_dti_model
task: drug_target_interaction
objective: discriminative
runner: predict

streams:
  drug:
    modality: small_molecule
    input_key: smiles
    processor: rdkit_graph
    encoder: drugban_molecule_gnn
  target:
    modality: protein_sequence
    input_key: sequence
    processor: amino_acid_tokenizer
    encoder: drugban_protein_cnn

fusion:
  type: bilinear_attention
  kwargs:
    hidden_dim: 256

head:
  type: binary_classifier
  kwargs:
    input_dim: 256
    hidden_dim: 128
    output_dim: 1

evaluation:
  metrics: [auroc, auprc, accuracy, f1]

interpretation:
  method: bilinear_attention_map
```

### Generative workflows

A generative workflow usually has streams, a conditioner, a decoder head, a sampling section, metrics, and an interpreter:

```yaml
name: my_inverse_folding_model
task: inverse_folding
objective: generative
runner: diffusion_generate

streams:
  structure:
    modality: protein_structure
    input_key: backbone_coords
    processor: backbone_coordinate_processor
    encoder: mapdiff_structure_encoder
  noisy_sequence:
    modality: protein_sequence
    input_key: sequence
    processor: masked_sequence_tokenizer
    encoder: residue_token_embedding

conditioner:
  type: structure_conditioned_denoising
  kwargs:
    hidden_dim: 128

head:
  type: diffusion_sequence_decoder
  kwargs:
    hidden_dim: 128
    vocab_size: 25

sampling:
  steps: 100
  num_samples: 8
  temperature: 1.0

evaluation:
  metrics: [sequence_recovery, diversity, novelty]

interpretation:
  method: denoising_trajectory
```

## 5. Register your own component

All components are looked up from registries. Register a new component at import time, then refer to it in config.

### Custom processor

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

### Custom encoder

```python
from kale_protein.registry import MODALITY_ENCODER_REGISTRY

@MODALITY_ENCODER_REGISTRY.register(("protein_sequence", "my_encoder"))
class MyEncoder:
    def __init__(self, output_dim=128, **kwargs):
        self.output_dim = output_dim

    def __call__(self, batch):
        return {
            "embedding": [0.0] * self.output_dim,
            "token_embeddings": [],
            "mask": None,
        }
```

Config:

```yaml
streams:
  target:
    modality: protein_sequence
    input_key: sequence
    processor: amino_acid_tokenizer
    encoder: my_encoder
    encoder_kwargs:
      output_dim: 256
```

### Custom metric

```python
from kale_protein.registry import EVALUATOR_REGISTRY

@EVALUATOR_REGISTRY.register(("drug_target_interaction", "my_metric"))
class MyMetric:
    def __call__(self, outputs, data):
        return 0.0
```

Config:

```yaml
evaluation:
  metrics: [my_metric]
```

## 6. Add a new AutoX-style preset

An AutoX preset is just a named config plus registered components. To create one:

1. Add or import all processors, encoders, fusion/conditioner modules, heads, runners, evaluators, and interpreters required by the config.
2. Create a config file under `kale_protein/presets/`, for example `my_autox.yaml`.
3. Load it with `AutoProteinConfig.from_yaml("kale_protein/presets/my_autox.yaml")`.
4. Build the workflow through `AutoProteinPreprocessor`, `AutoProteinPredictor`, `AutoProteinEvaluator`, and `AutoProteinInterpreter`.

Example:

```python
from kale_protein.auto import (
    AutoProteinConfig,
    AutoProteinPreprocessor,
    AutoProteinPredictor,
    AutoProteinEvaluator,
)

config = AutoProteinConfig.from_yaml("kale_protein/presets/my_autox.yaml")
processed = AutoProteinPreprocessor.from_config(config).transform_dataset(samples)
predictor = AutoProteinPredictor.from_config(config)
outputs = predictor.predict(processed)
metrics = AutoProteinEvaluator.from_config(config).evaluate(outputs, processed)
```

## 7. Debug registry errors

If a config references an unregistered component, KaleProtein raises an error containing the missing key and available keys. Check that:

- The registration module is imported before building the workflow.
- The registry key in code exactly matches the key in config.
- The key uses the expected shape, for example `(modality, encoder_name)` for modality encoders.

## 8. Recommended development workflow

After adding a custom config or component, run:

```bash
pytest -q kale_protein/tests
python -m compileall -q kale_protein
python kale_protein/examples/drugban_dti/evaluate.py
python kale_protein/examples/mapdiff_inverse_folding/generate.py
```

## 9. Load pretrained checkpoints

`AutoProteinPredictor.from_config()` accepts `pretrained=True` and an optional `checkpoint_path`. If `checkpoint_path` is omitted, KaleProtein reads `checkpoint.path` from the config.

```yaml
checkpoint:
  path: /path/to/model_checkpoint.json
  strict: false
```

```python
from kale_protein.auto import AutoProteinConfig, AutoProteinPredictor

config = AutoProteinConfig.from_yaml("my_drugban.yaml")
predictor = AutoProteinPredictor.from_config(config, pretrained=True)
```

You can also override the path at call time:

```python
predictor = AutoProteinPredictor.from_config(
    config,
    pretrained=True,
    checkpoint_path="/path/to/model_checkpoint.json",
)
```

The v0.1 checkpoint hook loads JSON checkpoints. A checkpoint may contain a `state_dict` object and optional metadata. The lightweight toy components do not have real learned weights, but this hook is the stable place for future torch-backed models to load pretrained parameters.
