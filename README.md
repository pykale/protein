# KaleProtein

KaleProtein provides Hugging Face-style Auto APIs for protein and multimodal
models. Users load one complete model by id; the model card decides which
reusable embedders and task predictor or generator it contains.

```python
from kaleprotein.auto import AutoProteinModel

model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
```

Named example models are discovered automatically when working from this source
repository. An installed wheel contains the reusable library only; model authors
can register their own local or downloaded model cards without changing Auto.

The examples expose a real pipeline rather than a workflow wrapper:

```text
load data -> preprocess -> collate -> embed -> predict/generate -> evaluate -> interpret (optional)
```

DrugBAN and MapDiff are self-contained PyTorch refactors. They do not import an
upstream checkout at runtime.

## Structure

```text
kaleprotein/
  auto/                         # generic selection and construction only
    configuration.py
    data.py
    preprocessing.py
    modeling.py
    evaluation.py
    interpretation.py
  core/                         # reusable building blocks
    data/
      modalities/               # sequence, molecule, structure preprocessing
      tasks/                    # task datasets and collators
    modeling/
      modalities/               # reusable encoders and neural layers
      tasks/                    # fusion, heads, predictors, generators
    evaluation/
      tasks/                    # metrics and interpretation
    registry/
    config/
    weights/
examples/                       # complete named model implementations
  drugban_dti/
  mapdiff_inverse_folding/
tests/                          # repository tests, not installed
docs/
```

`auto/` contains no DrugBAN or MapDiff branch. `core/` contains only reusable
data, layers, heads, metrics, and infrastructure. Concrete full-model assembly
and checkpoint compatibility stay in each example. See the
[architecture diagram](docs/architecture.md).

## Installation

```bash
python -m pip install -e ".[drugban]"
python -m pip install -e ".[mapdiff]"
python -m pip install -e ".[drugban,mapdiff,dev]"
```

Setuptools packages only `kaleprotein*`. Root-level `examples/`, `tests/`, and
`docs/` are repository resources and are not installed into site-packages. The
base package can load externally registered model cards and DTI CSV data without
importing PyTorch. DrugBAN raw-SMILES processing requires RDKit; MapDiff requires
PyTorch.

When running from a source checkout, `import kaleprotein` discovers model cards
under the adjacent `examples/` directory. After installing a wheel, register a
model card directory explicitly before using a named example model:

```python
from kaleprotein.core.registry import discover_model_cards

discover_model_cards("path/to/model_cards")
```

## DrugBAN

The normal user entry point is one complete model:

```python
from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPreprocessor,
)

# 1. Load reusable DTI data.
data = AutoProteinData(
    "DTI/BindingDB",
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
)

# 2. Preprocess molecule and protein streams from the model card.
config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")
preprocessor = AutoProteinPreprocessor.from_config(config)
processed = preprocessor.transform_dataset(list(data)[:8])

# 3. Build the complete model and collate a batch.
model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
batch = model.collator(**processed)

# 4. Embed with the model's registered modality encoders.
embeddings = model.embed(**batch)

# 5. Predict with the model's registered DTI task head.
prediction = model.predictor(**embeddings)

# 6. Evaluate or interpret when needed.
metrics = model.evaluate(**prediction)
attention = model.extract_attention(**prediction)
interpretation = AutoProteinInterpreter.from_config(config).explain(**attention)
```

Every public stage returns a dictionary, and the next stage consumes named
fields with `**`. DrugBAN's embedding contract includes
`protein_embedding`, `protein_mask`, `molecule_embedding`, and
`molecule_mask`; metadata such as labels, sample ids, sequences, and atom names
flows forward under explicit keys. A custom predictor can therefore implement
the same named signature without depending on a tuple position or opaque
workflow object.

```python
def my_head(protein_embedding, molecule_embedding, **metadata):
    scores = (protein_embedding.mean(1) * molecule_embedding.mean(1)).sum(-1)
    return {"scores": scores, **metadata}

custom_prediction = my_head(**embeddings)
```

Internally, `DrugBANModel` composes reusable components declared in its card:

```text
AutoProteinEmbedder("sequence/cnn")
AutoProteinEmbedder("molecule/gcn")
AutoProteinPredictor("dti/ban")
```

These component Auto APIs are primarily for model authors. They do not load a
second full DrugBAN model and do not own full-model checkpoints.

Run the complete workflows:

```bash
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

See the [DrugBAN example README](examples/drugban_dti/README.md).

## MapDiff

Generative models use the same complete-model entry point. Their predictor
component is a generator:

```python
from kaleprotein.auto import (
    AutoProteinData,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPreprocessor,
)

# 1. Load a PDB or processed CATH graph.
record = AutoProteinData("InverseFolding/CATH", source="structure.pdb")[0]

# 2. Preprocess the structure condition.
preprocessor = AutoProteinPreprocessor("protein/structure")
structure = preprocessor.featurize(
    {
        "backbone_coords": record.atom_pos,
        "sequence": record.sequence,
        "id": record.identifier,
    }
)
processed = {"samples": [structure]}

# 3. Load one complete model and collate named inputs.
model = AutoProteinModel("InverseFolding/MapDiff", pretrain=True)
batch = model.collator(**processed)

# 4. Encode the condition, then generate a sequence.
conditioning = model.embed(**batch)
generation = model.predictor.generate(
    **conditioning,
    steps=100,
    method="ddim",
)
metrics = model.evaluate(**generation)
trajectory = AutoProteinInterpreter.from_config(model.config).explain(**generation)
```

`pretrain=True` selects the release-compatible architecture, checks the local
card `weights/` directory, and downloads the configured MapDiff v1.0.1 weight
only when absent. The complete model loads that checkpoint exactly once.

```bash
python -m examples.mapdiff_inverse_folding.pretrain_ipa \
  /data/cath/train --output ipa.pt
python -m examples.mapdiff_inverse_folding.train_diffusion \
  /data/cath/train --ipa-checkpoint ipa.pt --output mapdiff.pt
python -m examples.mapdiff_inverse_folding.evaluate \
  /data/cath/test --pretrained
python -m examples.mapdiff_inverse_folding.generate \
  structure.pdb --pretrained --steps 100
```

See the [MapDiff example README](examples/mapdiff_inverse_folding/README.md).

## Reusable DTI Data

BindingDB, Human, and BioSNAP share one model-independent CSV loader:

```python
bindingdb = AutoProteinData("DTI/BindingDB", root="path/to/datasets")
human_train = AutoProteinData(
    "DTI/Human", root="path/to/datasets", split="random", subset="train"
)
biosnap_test = AutoProteinData(
    "DTI/BioSNAP", root="path/to/datasets", split="cluster", subset="target_test"
)
```

Records are normalized to `smiles`, `sequence`, `label`, `id`, and provenance
metadata, so another DTI model can reuse the same datasets unchanged.

## Model Cards

Every named model owns its implementation and assets:

```text
examples/<model>/
  config.yaml
  configuration.py
  modeling.py
  train/evaluate/predict/generate/interpret scripts
  data/
  maps/
  weights/
  README.md
```

`config.yaml` declares `auto_map.AutoProteinModel` and component ids. Adding a
new model card does not require editing `auto/`. External cards can be
registered directly:

```python
from kaleprotein.core.registry import register_model_card

register_model_card("path/to/my_model/config.yaml")
model = AutoProteinModel("MyTask/MyModel")
```

See [CUSTOMIZE.md](CUSTOMIZE.md) for a complete extension example.

## Pretrained Assets

For `pretrain=True`, the full model:

1. checks its card-local `weights/` path;
2. verifies an optional SHA-256 checksum;
3. atomically downloads a valid configured URL only when needed;
4. extracts common checkpoint containers;
5. applies card-local compatibility mapping and loads strictly;
6. raises a clear train-it-yourself error if neither a file nor URL exists.

DrugBAN intentionally has no default URL. MapDiff uses the published v1.0.1
release URL. Large weights and datasets are excluded from git and packages.

## Development

Tests use fake URLs, fake RDKit objects, temporary CSVs and graphs, and tiny
trainable model dimensions. They never download real weights.

```bash
python -m pytest -q
python -m compileall -q kaleprotein
python -m build
```

Refactored-source attribution is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
