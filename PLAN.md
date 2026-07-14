# KaleProtein Architecture Plan

## Objective

KaleProtein provides Hugging Face-style Auto entry points for protein and
molecule workflows while keeping every named model implementation in its own
model-card directory.

The public workflow is explicit:

```text
load data -> preprocess -> collate -> embed/model -> train or predict -> evaluate -> interpret (optional)
```

DrugBAN and MapDiff are production examples refactored from their MIT-licensed
upstream repositories. They must not import or require an upstream checkout at
runtime, and they must not be replaced with numerical stubs for CI.

## Ownership Boundaries

### Auto layer

`kale_protein.auto` performs lookup and dispatch only. It may contain generic
configuration, checkpoint, data, processor, and model-card discovery logic. It
must not contain model IDs, model-name conditionals, architecture definitions,
or embedded DrugBAN/MapDiff configurations.

### Shared layers

Shared modules may define reusable modality operations and task behavior:

- generic protein sequence, structure, and molecule preprocessing
- task-level datasets, collators, losses, metrics, and split handling
- model-card and processor registries
- checkpoint resolution, download, checksum, and deserialization helpers

A class whose behavior or parameterization is specific to a named model belongs
to that model card, even if it encodes only one modality.

### Model cards

Each model card owns its configuration and implementation:

```text
kale_protein/examples/<model>/
  config.yaml
  configuration.py
  modeling.py
  data/
  maps/
  weights/
  train/evaluate/predict/generate/interpret scripts
  README.md
```

`config.yaml` declares `auto_map` targets. The generic Auto classes dynamically
load those targets. Adding a new model card must not require editing an Auto
class.

## Required Auto APIs

```python
AutoProteinData("DTI/BindingDB", root="...")
AutoProteinPreprocessor("protein/sequence")
AutoMoleculePreprocessor("molecule/SMILES")
AutoProteinModel("DTI/DrugBAN", pretrain=True)
AutoProteinPredictor("DTI/DrugBAN", pretrain=True)
AutoProteinGenerator("InverseFolding/MapDiff", pretrain=True)
```

Model cards may expose task-appropriate component APIs, such as separate
protein and molecule embedders for DrugBAN or a structure encoder and sequence
generator for MapDiff. These objects must be backed by real learnable models.

## Pretrained Assets

When `pretrain=True`, the loader must:

1. Resolve the configured filename inside the model card's local `weights/`
   directory.
2. Verify the optional SHA-256 checksum.
3. If absent, atomically download a valid configured URL and verify it.
4. Deserialize the checkpoint and load it into the matching architecture.
5. Raise a clear error when no local file or valid URL exists.
6. Raise a compatibility error for missing/unexpected state keys instead of
   silently using random weights.

Large datasets and weights are never committed. Small maps and configuration
assets required to construct a model may be packaged.

## Example Acceptance Criteria

### DrugBAN

- BindingDB, Human, and BioSNAP use a shared normalized DTI dataset interface.
- SMILES become chemically meaningful molecular graphs.
- Proteins use DrugBAN-compatible residue encoding and convolution.
- BAN fusion returns logits and atom-residue attention.
- Training updates parameters and saves a reloadable checkpoint.
- Evaluation computes AUROC, AUPRC, F1, accuracy, and selected threshold.
- Prediction performs inference only; interpretation consumes real attention.

### MapDiff

- CATH processed `.pt` graphs and PDB backbone coordinates are supported.
- `CollatorIPAPretrain` and `CollatorDiff` produce real padded/graph batches.
- IPA pretraining and diffusion training update model parameters.
- Generation runs an iterative discrete diffusion schedule and returns a
  non-empty trajectory.
- Evaluation computes recovery, perplexity, and diversity from model outputs.

## Test Strategy

Tests use tiny local CSVs, graph tensors, temporary model cards, fake download
URLs, and small model configurations. They do not download real datasets or
large checkpoints. The suite verifies:

- model-card discovery and dynamic `auto_map` loading
- absence of model IDs and named architectures in the Auto layer
- local/download/missing/checksum checkpoint behavior
- state-dict loading and incompatibility errors
- reusable DTI and inverse-folding datasets/collators
- one optimizer step changes parameters for each example
- distinct scripts are import-safe and expose callable `main` workflows
- metrics are numerically correct on known small cases
- wheel installation retains model-card YAML and required maps

## Verification

Before submission:

```bash
python -m pytest -q
python -m compileall -q kale_protein
python -m build
```

CI installs the package before testing. Heavy example workflows are exercised
with fake/tiny data; full datasets and released checkpoints remain user-run
integration tests.
