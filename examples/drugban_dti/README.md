# DrugBAN DTI

This model card is a self-contained PyTorch refactor of DrugBAN. It keeps the
real data flow and learnable architecture while removing the runtime dependency
on the upstream checkout, DGL, and DGL-LifeSci.

## Card Layout

```text
drugban_dti/
  config.yaml          component ids, dimensions, training and weight metadata
  configuration.py     DrugBANConfig
  collators.py         DrugBAN-specific tensor batching
  model_drugban.py     pure model computation and checkpoint state adapter
  train.py
  evaluate.py
  predict.py
  interpret.py
  data/
  weights/
```

The complete model is assembled from reusable first-level model components:

```text
DrugBANModel
  AutoProteinEmbedder("molecule/gcn")
  AutoProteinEmbedder("sequence/cnn")
  AutoProteinPredictor("dti/ban")
```

`AutoProteinModel` resolves and loads a requested full checkpoint once.
`DrugBANModel` implements only model-state serialization and compatibility;
component factories do not create or load another DrugBAN network.

## Evaluation Pipeline

```python
from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)

# 1. Load the model card shared by the data and model sides.
config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")

# 2. Load, preprocess, collate, and batch normalized DTI records.
loader = AutoProteinDataLoader(
    "BindingDB/DTI",
    config=config,
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
    batch_size=64,
)
inputs = next(iter(loader))

# 3. Build the complete model and load weights when requested.
model = AutoProteinModel.from_config(config, checkpoint="drugban.pt")

# 4. Pass loader outputs directly into DrugBAN's embedders.
embeddings = model.embed(**inputs)

# 5. Predict interactions from the named embeddings.
prediction = model.predict(**embeddings)

# 6a. Evaluate the prediction mapping.
metrics = model.evaluate(**prediction)

# 6b. Independently interpret its attention fields when needed.
interpretation = AutoProteinInterpreter.from_config(config).explain(**prediction)
```

The stage contracts are ordinary dictionaries. In particular, `embed()`
returns flat keys such as `protein_embedding`, `protein_mask`,
`molecule_embedding`, and `molecule_mask`; the BAN head declares those names
in its Python signature. Replacing the head or inserting a user-defined stage
only requires accepting and returning the desired named fields.

The high-level loader exposes its composed `.dataset`, `.preprocessor`,
`.processed`, `.collator`, and underlying `.loader`. Users can still construct
or replace each low-level component directly for research workflows.

The loader never invokes either model stage. It only guarantees that its output
mapping satisfies the selected model card's `embed(**inputs)` contract.

The shared preprocessing produces canonical 74-feature RDKit atoms. The independent
DrugBAN collator adds the model-specific virtual-node bit before dense normalized graph convolution,
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
- `interpret.py` independently maps BAN attention to valid atoms and protein
  residues; evaluation does not invoke it.

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
