# DrugBAN DTI

This model card is a self-contained PyTorch refactor of DrugBAN. It keeps the
real data flow and learnable architecture while removing the runtime dependency
on the upstream checkout, DGL, and DGL-LifeSci.

## Card Layout

```text
drugban_dti/
  config.yaml          component ids, dimensions, training and weight metadata
  configuration.py     DrugBANConfig
  modeling.py          complete model, collation, training, checkpoint adapter
  train.py
  evaluate.py
  predict.py
  interpret.py
  data/
  weights/
```

The complete model is assembled from reusable core components:

```text
DrugBANModel
  AutoProteinEmbedder("molecule/gcn")
  AutoProteinEmbedder("sequence/cnn")
  AutoProteinPredictor("dti/ban")
```

`DrugBANModel` alone owns full checkpoints. The component factories do not
create or load another DrugBAN network.

## Evaluation Pipeline

```python
from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPreprocessor,
)

# 1. Load normalized DTI records.
data = AutoProteinData(
    "BindingDB/DTI",
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
)

# 2. Preprocess SMILES and protein sequences.
config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
preprocessor = AutoProteinPreprocessor.from_config(config)
processed = preprocessor.transform_dataset(list(data)[:64])

# 3. Build the complete model and create one named batch mapping.
model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
batch = model.collator(**processed)

# 4. Embed both modalities.
embeddings = model.embed(**batch)

# 5. Predict interactions.
prediction = model.predictor(**embeddings)

# 6. Evaluate or expose attention from the prediction mapping.
metrics = model.evaluate(**prediction)
attention = model.extract_attention(**prediction)
interpretation = AutoProteinInterpreter.from_config(config).explain(**attention)
```

The stage contracts are ordinary dictionaries. In particular, `embed()`
returns flat keys such as `protein_embedding`, `protein_mask`,
`molecule_embedding`, and `molecule_mask`; the BAN head declares those names
in its Python signature. Replacing the head or inserting a user-defined stage
only requires accepting and returning the desired named fields.

The shared preprocessing produces canonical 74-feature RDKit atoms. DrugBAN's
collator adds the model-specific virtual-node bit before dense normalized graph convolution,
DrugBAN-compatible residue encoding and CNN layers, bilinear attention, and the
MLP classifier.

## Reusable Datasets

BindingDB, Human, and BioSNAP use the same model-independent loader:

```python
bindingdb = AutoProteinData("BindingDB/DTI", root="path/to/datasets")
human = AutoProteinData("Human/DTI", root="path/to/datasets")
biosnap = AutoProteinData("BioSNAP/DTI", root="path/to/datasets")
```

Use `split` and `subset` for layouts such as `random/train.csv` or
`cluster/target_test.csv`. All records expose `smiles`, `sequence`, `label`,
`id`, and provenance metadata for reuse by other DTI models.

## Workflows

```bash
python -m pip install -e ".[drugban]"

python -m examples.drugban_dti.train \
  --dataset BindingDB --root /data/drugban --split random --subset train \
  --validation-subset val --checkpoint drugban.pt

python -m examples.drugban_dti.evaluate \
  --dataset BindingDB --root /data/drugban --split random --subset test \
  --checkpoint drugban.pt

python -m examples.drugban_dti.predict \
  --smiles "CCO" --sequence "MKT..." --checkpoint drugban.pt

python -m examples.drugban_dti.interpret \
  --dataset BioSNAP --path /data/biosnap/full.csv --checkpoint drugban.pt
```

- `train.py` performs real optimizer updates, optional validation, and full
  checkpoint saving.
- `evaluate.py` computes AUROC, AUPRC, F1, accuracy, and threshold metrics.
- `predict.py` performs inference only for a CSV or one SMILES/sequence pair.
- `interpret.py` maps BAN attention to valid atoms and protein residues.

## Pretrained Weights

The card intentionally has no default DrugBAN weight URL:

```yaml
pretrained:
  local_dir: weights
  filename: drugban.pt
  url: ""
```

With `pretrain=True`, KaleProtein first checks `weights/drugban.pt`. If it is
absent and no valid URL is configured, it raises a clear error telling the user
to provide a checkpoint or train with `pretrain=False`.

## Attribution

The architecture and adapted implementation derive from
[peizhenbai/DrugBAN](https://github.com/peizhenbai/DrugBAN) under the MIT
License. See the repository-level `THIRD_PARTY_NOTICES.md`.
