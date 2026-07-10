# KaleProtein

KaleProtein is a stream-based Auto framework for protein, small-molecule, and
protein-design workflows. Its Auto APIs follow the same architectural idea as
Hugging Face Transformers: generic Auto classes resolve a model card, read its
`auto_map`, and dispatch to model-specific configuration and modeling files.

Model-specific code should not live inside `kale_protein.auto`. Built-in or
custom models should live in self-contained model-card folders with their own
`config.yaml`, `configuration_*.py`, `modeling_*.py`, `data/`, and `weights/`
layout.

## Quick Start

Install test dependencies and run the suite:

```bash
python -m pip install pytest
python -m pytest -q kale_protein/tests
```

Run the DrugBAN DTI example:

```bash
python kale_protein/examples/drugban_dti/evaluate.py
python kale_protein/examples/drugban_dti/train.py
python kale_protein/examples/drugban_dti/interpret.py
```

Run the MapDiff inverse-folding example:

```bash
python kale_protein/examples/mapdiff_inverse_folding/generate.py
python kale_protein/examples/mapdiff_inverse_folding/evaluate.py
```

## Auto Pipeline Style

Drug-target interaction:

```python
from kale_protein.auto import (
    AutoMoleculePreprocessor,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)

data, label = AutoProteinData("DTI/PDBBind")
preprocessor_protein = AutoProteinPreprocessor("protein/sequence")
preprocessor_drug = AutoMoleculePreprocessor("molecule/SMILE")

protein_model, molecule_model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
interaction_predictor = AutoProteinPredictor("DTI/DrugBAN", pretrain=False)

protein_data = preprocessor_protein.tokenize(data)
drug_data = preprocessor_drug.tokenize(data)

protein_embedding = protein_model.embed(protein_data)
drug_embedding = molecule_model.embed(drug_data)

interaction_prediction = interaction_predictor(protein_embedding, drug_embedding)
```

Generative inverse folding:

```python
from kale_protein.auto import (
    AutoProteinData,
    AutoProteinGenerator,
    AutoProteinModel,
    AutoProteinPreprocessor,
)

data, native_sequence = AutoProteinData("InverseFolding/CATH")
structure_preprocessor = AutoProteinPreprocessor("protein/structure")
sequence_preprocessor = AutoProteinPreprocessor("protein/masked_sequence")

structure_encoder = AutoProteinModel("InverseFolding/MapDiff", pretrain=False)
sequence_generator = AutoProteinGenerator("InverseFolding/MapDiff", pretrain=False)

structure_data = {
    "structure": structure_preprocessor.featurize(data),
    "noisy_sequence": sequence_preprocessor.tokenize(data),
}
structure_embedding = structure_encoder.embed(structure_data)
generated_sequence = sequence_generator.generate(structure_embedding)
```

## Model-Card Layout

Each model-card folder owns the code for that model:

```text
kale_protein/examples/<model>/
  config.yaml
  configuration_<model>.py
  modeling_<model>.py
  data/
  weights/
  README.md
```

`config.yaml` declares the task, streams, processors, encoders, heads, pretrained
weight metadata, and `auto_map` entries. For example:

```yaml
model_id: DTI/DrugBAN
model_type: drugban
auto_map:
  AutoProteinConfig: configuration_drugban.DrugBANConfig
  AutoProteinModel: modeling_drugban.DrugBANModel
  AutoProteinPredictor: modeling_drugban.DrugBANInteractionPredictor
```

`AutoProteinModel("DTI/DrugBAN")` resolves the registered model card, loads
`config.yaml`, imports the classes declared in `auto_map`, and instantiates the
model. The Auto layer performs lookup and dispatch only; DrugBAN and MapDiff
model-specific code stays in their example folders.

## Pretrained Weights

Each model card may define a pretrained block:

```yaml
pretrained:
  local_dir: weights
  filename: model.pt
  url: ""
```

When `pretrain=True`:

1. KaleProtein first looks for the local file under the model card's `weights/`
   folder.
2. If the file is missing and `url` is valid, it downloads the file into
   `weights/`.
3. If neither a local file nor a valid URL is available, it raises a clear error
   telling the user to provide weights or train the model.

DrugBAN intentionally ships with an empty pretrained URL. MapDiff points to the
upstream release weight URL declared in its model card.

## Registries

Common processors, encoders, heads, tasks, evaluators, and datasets are
registered in shared registries. This is appropriate for broadly reusable
modalities and tasks. Specific model definitions should be added through a
model-card folder and `auto_map`, not by adding model-name branches to
`kale_protein.auto`.

## Customization

To add a new model:

1. Create a folder under `kale_protein/examples/` or in your own package.
2. Add `config.yaml`, `configuration_*.py`, and `modeling_*.py`.
3. Register the model id with `MODEL_CARD_REGISTRY`.
4. Put large datasets and weights under `data/` and `weights/`, but keep them
   out of git.
5. Load it with `AutoProteinModel("<task>/<model>")` or the matching Auto class.

For lower-level component customization, see [CUSTOMIZE.md](CUSTOMIZE.md).

## Continuous Integration

The GitHub Actions workflow at `.github/workflows/tests.yml` runs tests,
byte-compiles the package, and executes the example scripts.
