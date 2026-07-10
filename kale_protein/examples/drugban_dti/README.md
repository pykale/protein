# DrugBAN DTI Example

This folder is the DrugBAN model card. It is intentionally structured like a
small model repository: configuration, model-specific code, optional local data,
and optional local weights live beside each other.

The generic Auto classes do not contain DrugBAN branches. They resolve
`DTI/DrugBAN`, load this folder's `config.yaml`, read `auto_map`, and import the
DrugBAN classes from this folder.

## Folder Layout

```text
config.yaml
configuration.py
modeling.py
data/
weights/
evaluate.py
train.py
interpret.py
```

## Auto Pipeline

The scripts use the explicit workflow:

```text
load data -> preprocess -> embed -> predict/train/evaluate -> interpret optional
```

Minimal evaluation pipeline:

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

## Scripts

```bash
python kale_protein/examples/drugban_dti/evaluate.py
python kale_protein/examples/drugban_dti/train.py
python kale_protein/examples/drugban_dti/interpret.py
```

`evaluate.py` loads data, preprocesses protein and molecule inputs, embeds both
streams, predicts interaction probability, and computes metrics.

`train.py` follows the same data/preprocess/embed/predict shape and calls the
predictor training entry point.

`interpret.py` extracts the attention-like interpretation returned by the
DrugBAN predictor.

## Pretrained Weights

`config.yaml` declares:

```yaml
pretrained:
  local_dir: weights
  filename: drugban.pt
  url: ""
```

Use `pretrain=False` for scripts that should run without large assets. Set
`pretrain=True` only after placing `weights/drugban.pt` in this folder or adding
a valid URL to `config.yaml`. If neither exists, KaleProtein raises a clear
missing-weight error instead of pretending pretrained weights are available.

## Extending

DrugBAN-specific model code belongs in `modeling.py`; configuration
logic belongs in `configuration.py`; model-card wiring belongs in
`config.yaml`. Avoid adding DrugBAN-specific branches to `kale_protein.auto`.
