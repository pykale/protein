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
load data -> preprocess -> collate -> embed -> predict/generate
                                                   |-> evaluate
                                                   `-> interpret (optional)
```

DrugBAN and MapDiff are self-contained PyTorch refactors. They do not import an
upstream checkout at runtime.

## Structure

```text
kaleprotein/
  auto/                         # generic selection, config, and registries
    loaddata.py                  # dataset, collator, and batch-loader dispatch
    prepdata.py
    model.py
    evaluate.py
    interpret.py
    config/
    registry/
  loaddata/                     # datasets, records, and base_dataset.py
  prepdata/                     # reusable input transformations
  model/
    embed/                      # <modality>_<model>.py encoders
    predict/                    # <task>_<model>.py heads and generators
  evaluate/                     # one quantitative metric per file
    accuracy.py
    sequence_recovery.py
  interpret/                    # one interpretation method per file
    bilinear_attention_map.py
    denoising_trajectory.py
  utils/                        # <step>_<helper_function>.py helpers
    loaddata_read_fasta.py
    model_load_checkpoint_state_dict.py
    evaluate_extract_binary_inputs.py
examples/                       # complete named model implementations
  drugban_dti/
  mapdiff_inverse_folding/
tests/                          # repository tests, not installed
docs/
```

`auto/` contains no DrugBAN or MapDiff branch. Reusable operations are
first-level verb-oriented packages beside Auto. Concrete full-model assembly
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

`import kaleprotein` only exposes package metadata and never scans the
filesystem. Register a model card explicitly before using its named model id:

```python
from kaleprotein.auto.registry import discover_model_cards

discover_model_cards("path/to/model_cards")
```

The runnable repository examples register their own card before executing.

## DrugBAN

The normal user entry point is one complete model:

```python
from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from examples.drugban_dti import register_model_card

register_model_card()
# 1. Select the model card shared by data and model composition.
config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")

# 2. Load, preprocess, collate, and batch reusable DTI data.
loader = AutoProteinDataLoader(
    "BindingDB/DTI",
    config=config,
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
    batch_size=64,
)
inputs = next(iter(loader))

# 3. Build the complete model and load requested weights.
model = AutoProteinModel.from_config(config, checkpoint="drugban.pt")

# 4. Pass loader outputs directly into the registered embedders.
embeddings = model.embed(**inputs)

# 5. Predict from the named embeddings.
prediction = model.predict(**embeddings)

# 6a. Evaluate the prediction mapping.
metrics = model.evaluate(**prediction)

# 6b. Independently interpret the same prediction mapping when needed.
interpretation = AutoProteinInterpreter.from_config(config).explain(**prediction)
```

`AutoProteinDataLoader` exposes `.dataset`, `.preprocessor`, `.processed`,
`.collator`, and `.loader`. It composes the data side from configuration but
never creates or retains a model instance. Every yielded mapping is a valid
`model.embed(**inputs)` keyword input.

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

Shared component ids map directly to flat files: `molecule/gcn` resolves
`model/embed/molecule_gcn.py`, `sequence/cnn` resolves
`model/embed/sequence_cnn.py`, and `dti/ban` resolves
`model/predict/dti_ban.py`.

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
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from examples.mapdiff_inverse_folding import register_model_card

register_model_card()
# 1. Select one shared model configuration.
config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")

# 2. Load and prepare a PDB or processed CATH graph.
loader = AutoProteinDataLoader(
    "CATH/InverseFolding",
    config=config,
    source="structure.pdb",
    batch_size=1,
)
inputs = next(iter(loader))

# 3. Load one complete model.
model = AutoProteinModel.from_config(config, pretrain=True)

# 4. Encode structure conditions from the loader mapping.
embeddings = model.embed(**inputs)

# 5. Generate from the named embedding mapping.
generation = model.generate(
    **embeddings,
    steps=100,
    method="ddim",
)
metrics = model.evaluate(**generation)
interpretation = AutoProteinInterpreter.from_config(config).explain(**generation)
```

`pretrain=True` makes `AutoProteinModel` select the release checkpoint, check
the card's local `weights/` directory, download the configured MapDiff v1.0.1
weight only when absent, and invoke the model's checkpoint state adapter once.

```bash
python -m examples.mapdiff_inverse_folding.pretrain_ipa \
  /data/cath/train --output ipa.pt
python -m examples.mapdiff_inverse_folding.train_diffusion \
  /data/cath/train --checkpoint ipa.pt --output mapdiff.pt
python -m examples.mapdiff_inverse_folding.evaluate \
  /data/cath/test --pretrained
python -m examples.mapdiff_inverse_folding.interpret \
  structure.pdb --pretrained --steps 100
python -m examples.mapdiff_inverse_folding.generate \
  structure.pdb --pretrained --steps 100
```

See the [MapDiff example README](examples/mapdiff_inverse_folding/README.md).

## Reusable DTI Data

BindingDB, Human, and BioSNAP provide dataset-specific classes over one
model-independent DTI CSV base class:

```python
from kaleprotein.auto import AutoProteinData

bindingdb = AutoProteinData("BindingDB/DTI", root="path/to/datasets")
human_train = AutoProteinData(
    "Human/DTI", root="path/to/datasets", split="random", subset="train"
)
biosnap_test = AutoProteinData(
    "BioSNAP/DTI", root="path/to/datasets", split="cluster", subset="target_test"
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
  model_<model_name>.py
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
from kaleprotein.auto.registry import register_model_card

register_model_card("path/to/my_model/config.yaml")
model = AutoProteinModel("MyTask/MyModel")
```

See [CUSTOMIZE.md](CUSTOMIZE.md) for a complete extension example.

## Pretrained Assets

The `weights/` directories shown in model-card layouts are example-local asset
folders, not a `kaleprotein.weights` Python package. Auto owns pretrained
resolution; generic checkpoint parsing lives in
`kaleprotein.utils.model_load_checkpoint_state_dict`.

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
