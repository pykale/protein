# DrugBAN DTI Example

This implementation is adapted from
[peizhenbai/DrugBAN](https://github.com/peizhenbai/DrugBAN), commit `9923f8c`,
which is distributed under the MIT License (copyright 2022 peizhenbai). Its
PyTorch architecture and featurization conventions were reimplemented here;
the upstream repository is never imported at runtime.

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
_cli.py
data/
weights/
train.py
evaluate.py
predict.py
interpret.py
```

## Auto Pipeline

The scripts use the explicit workflow:

```text
load data -> preprocess -> collate -> embed -> predict/train/evaluate -> interpret optional
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

dataset = AutoProteinData(
    "DTI/BindingDB",
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
)
sample = dataset[0]
label = sample["label"]
preprocessor_protein = AutoProteinPreprocessor("protein/sequence")
preprocessor_drug = AutoMoleculePreprocessor("molecule/SMILE")

protein_model, molecule_model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
interaction_predictor = AutoProteinPredictor("DTI/DrugBAN", pretrain=False)

protein_data = preprocessor_protein.tokenize(sample)
drug_data = preprocessor_drug.featurize(sample)

batch = interaction_predictor.collator(
    [{"target": protein_data, "drug": drug_data, "label": label}]
)

protein_embedding = protein_model.embed(batch["target"])
drug_embedding = molecule_model.embed(batch["drug"])

interaction_prediction = interaction_predictor(protein_embedding, drug_embedding)
```

## Scripts

Install the model extra first:

```bash
python -m pip install -e ".[drugban]"
```

```bash
python kale_protein/examples/drugban_dti/train.py --dataset BindingDB --root /data/drugban --split random --subset train --validation-subset val --checkpoint drugban.pt
python kale_protein/examples/drugban_dti/evaluate.py --dataset BindingDB --root /data/drugban --split random --subset test --checkpoint drugban.pt
python kale_protein/examples/drugban_dti/predict.py --dataset Human --path /data/human/full.csv --checkpoint drugban.pt
python kale_protein/examples/drugban_dti/predict.py --smiles "CCO" --sequence "MKT..." --checkpoint drugban.pt
python kale_protein/examples/drugban_dti/interpret.py --dataset BioSNAP --path /data/biosnap/full.csv --checkpoint drugban.pt
```

`evaluate.py` loads data, preprocesses protein and molecule inputs, embeds both
streams, predicts interaction probability, and computes metrics.

`train.py` builds shuffled batches, performs optimizer steps, optionally scores
a validation subset, and saves a reloadable checkpoint.

`predict.py` performs inference only, either for every row in a normalized DTI
CSV or for one direct SMILES/protein pair.

`interpret.py` maps real BAN attention to valid molecule atoms and protein
residues.

RDKit is optional for KaleProtein as a whole but required when converting raw
SMILES for DrugBAN. Already-preprocessed tensor graph batches do not require
RDKit. DGL and DGL-LifeSci are not required.

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
