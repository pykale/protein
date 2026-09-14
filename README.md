<p align="center">
  <img src="docs/assets/kaleprotein-logo.png" alt="KaleProtein logo" width="200">
</p>

<h1 align="center">KaleProtein</h1>

<p align="center">
  Reusable components and consistent interfaces for protein AI.
</p>

<p align="center">
  <a href="#installation">Installation</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#models-and-datasets">Models and Datasets</a> &middot;
  <a href="#model-and-data-cards">Model and Data Cards</a>
</p>

KaleProtein is a modular toolkit for developing and using protein AI models
across multiple modalities. It brings reusable neural network components,
dataset adapters, and model cards together through consistent Auto APIs.

- **For protein AI model developers:** Compose existing protein and multimodal
  components, develop and test models against explicit interfaces, and share
  implementations through a common model-card layout.
- **For protein scientists:** Load datasets and contributed models through Auto
  APIs, run predictions or generate protein sequences, and evaluate models with
  reusable metrics.

Data preparation, embedding, prediction or generation, evaluation, and
interpretation remain explicit steps. Named dictionaries connect them, so
researchers can inspect intermediate results and replace individual components.

![KaleProtein pipeline: AutoProteinDataLoader loads, preprocesses, and collates data; AutoProteinModel embeds and predicts or generates; outputs branch independently into evaluation and optional interpretation. Auto, registry, config, and utils provide shared support.](docs/assets/kaleprotein-pipeline.png)

The data loader owns data preparation and batching. The model owns its
embedders, predictor or generator, and model weights. Auto APIs select and
construct registered implementations. See the
[architecture guide](docs/architecture.md) for component boundaries and
input/output contracts.

## Installation

Requires **Python 3.10 or later**.

```bash
python -m pip install kaleprotein
```

Choose optional dependencies for your work:

| Install | Includes |
| --- | --- |
| `kaleprotein` | PyTorch, PyYAML, Auto APIs, reusable model components, and data utilities |
| `kaleprotein[drugban]` | Base package plus RDKit |
| `kaleprotein[mapdiff]` | Base package plus PyTorch Geometric |
| `kaleprotein[examples]` | Dependencies for all repository examples |
| `kaleprotein[dev]` | Example dependencies, testing, lint, build, and release tools |

The wheel installs the `kaleprotein` library. Extras add dependencies;
they do not install the repository's `examples/`, `tests/`, or `docs/`.

Download only the example you need, using the installed library:

```bash
python -m pip install "kaleprotein[drugban]"
python -m kaleprotein download-example drugban_dti

# Or get MapDiff and its dependencies.
python -m pip install "kaleprotein[mapdiff]"
python -m kaleprotein download-example mapdiff_inverse_folding
```

Examples are saved under `./examples/<name>/`. Run the snippets below from
the directory containing `examples/`; no library checkout or editable install
is needed. The downloader fetches code, configuration, maps, and license
notices, without importing the downloaded code. It leaves existing example
directories untouched.

By default, examples come from the installed library's `v<version>` release
tag. Use `--ref <tag-or-commit>` to select a revision, or `--ref main` when
working with development code. `--output <directory>` changes the parent
download directory. The selected commit is recorded in
`.kaleprotein-example.json` inside each downloaded example.

Datasets and large checkpoints are obtained separately. The
[model and dataset index](#models-and-datasets) links to each workflow's
requirements.

## Quick Start

### Use a Model

This DrugBAN example evaluates one batch of drug-target pairs and exposes
interaction attention. Prepare the DrugBAN-formatted BindingDB CSV files and a
trained DrugBAN checkpoint first, following the
[DrugBAN guide](examples/drugban_dti/README.md). Replace the two local paths
below with your dataset root and checkpoint.

```python
import torch

from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from examples.drugban_dti import register_model_card

register_model_card()
config = AutoProteinConfig.from_pretrained("DTI/DrugBAN")

# 1. Load, preprocess, and batch the dataset.
loader = AutoProteinDataLoader(
    "BindingDB/DTI",
    config=config,
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
    batch_size=64,
)
inputs = next(iter(loader))

# 2. Build the model and load its checkpoint.
model = AutoProteinModel.from_config(config, checkpoint="path/to/drugban.pt")
model.eval()

with torch.inference_mode():
    # 3. Embed the protein and molecule.
    embeddings = model.embed(**inputs)

    # 4. Predict interactions.
    prediction = model.predict(**embeddings)

    # 5. Evaluate the predictions.
    metrics = model.evaluate(**prediction)

    # 6. Optionally interpret interaction attention.
    interpreter = AutoProteinInterpreter.from_config(config)
    interpretation = interpreter.explain(**prediction)

print(metrics)
```

Stage outputs are dictionaries with named fields. For example, DrugBAN
embeddings include `protein_embedding`, `molecule_embedding`, and their masks.
Labels and sample metadata travel alongside them.

This snippet reports **one-batch metrics**. Use the
[full evaluation workflow](examples/drugban_dti/README.md#workflows) to evaluate
a complete split. Model comparisons should use the same held-out records and
metric definitions, with each model's required preprocessing.

Generative models follow the same pattern. After constructing the MapDiff
loader and model, encode the structure and generate sequences:

```python
with torch.inference_mode():
    embeddings = model.embed(**inputs)
    generation = model.generate(**embeddings, steps=100, method="ddim")
    metrics = model.evaluate(**generation)
```

Here, `model` and `inputs` refer to MapDiff, not the DrugBAN objects above.
See the [complete MapDiff pipeline](examples/mapdiff_inverse_folding/README.md#pipeline)
for setup, pretrained weights, and structure-input requirements.

### Build a Model

Model developers can select reusable encoders and heads independently. For
example, DrugBAN combines a protein CNN, a molecular GCN, and a bilinear
attention prediction head:

```python
from kaleprotein.auto import AutoProteinEmbedder, AutoProteinPredictor

protein_encoder = AutoProteinEmbedder("sequence/cnn")
molecule_encoder = AutoProteinEmbedder("molecule/gcn")
interaction_head = AutoProteinPredictor("dti/ban")
```

These are trainable components, not pretrained complete models. A full model
implements `embed(**inputs)` and `predict(**embeddings)` or
`generate(**embeddings)`, connecting the components through named mappings.
Its card declares the component IDs, configuration, and checkpoint metadata.

Register a complete model card to make your implementation available through
the same Auto API:

```python
from kaleprotein.auto import AutoProteinModel
from kaleprotein.auto.registry import register_model_card

register_model_card("path/to/my_model/config.yaml")
model = AutoProteinModel("MyTask/MyModel")
```

Follow the [extension guide](CUSTOMIZE.md) to implement components, define
a model card, register it, and test the resulting workflow.

## Models and Datasets

### Model Index

Complete model implementations live in the repository examples. Both are
self-contained PyTorch refactors and do not import an upstream checkout at
runtime.

| Model ID | Task | Inputs | Pretrained weights | Card and workflows |
| --- | --- | --- | --- | --- |
| `DTI/DrugBAN` | Drug-target interaction prediction | Protein sequence and molecular SMILES | Supply a checkpoint or train locally; no default download URL | [DrugBAN](examples/drugban_dti/README.md) / [Config](examples/drugban_dti/config.yaml) |
| `InverseFolding/MapDiff` | Protein inverse folding | Protein structure or processed residue graph | Configured upstream v1.0.1 release download | [MapDiff](examples/mapdiff_inverse_folding/README.md) / [Config](examples/mapdiff_inverse_folding/config.yaml) |

`AutoProteinModel(..., pretrain=True)` checks the card's local weight path,
downloads from its configured URL when needed, and verifies a checksum when
provided. If no local checkpoint or valid URL exists, it raises an error
explaining that training or a supplied checkpoint is required.
Use `checkpoint="path/to/model.pt"` to load a specific local checkpoint.

### Dataset Index

Dataset IDs follow `Dataset/Task`. Adapters load local data and normalize
records for reuse across compatible models.

| Dataset ID | Supported input | Adapter and usage |
| --- | --- | --- |
| `BindingDB/DTI` | DrugBAN-formatted DTI CSVs with SMILES, sequences, and labels | [Adapter](kaleprotein/loaddata/bindingdb.py) / [Usage](examples/drugban_dti/README.md#reusable-datasets) |
| `Human/DTI` | DrugBAN-formatted human DTI CSVs | [Adapter](kaleprotein/loaddata/human.py) / [Usage](examples/drugban_dti/README.md#reusable-datasets) |
| `BioSNAP/DTI` | DrugBAN-formatted BioSNAP DTI CSVs | [Adapter](kaleprotein/loaddata/biosnap.py) / [Usage](examples/drugban_dti/README.md#reusable-datasets) |
| `CATH/InverseFolding` | Processed CATH `.pt` graphs; also accepts local PDB/mmCIF structures | [Adapter](kaleprotein/loaddata/cath.py) / [Preparation](examples/mapdiff_inverse_folding/README.md#data-responsibilities) |

Use `AutoProteinData` for dataset records alone, or `AutoProteinDataLoader`
to combine dataset loading with a configured preprocessor and collator:

```python
from kaleprotein.auto import AutoProteinData

data = AutoProteinData(
    "BindingDB/DTI",
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
)
```

DTI adapters target the processed benchmark CSVs, not arbitrary raw exports
from the original databases. For MapDiff, processed CATH graphs preserve
training features that the raw PDB/mmCIF path cannot fully reconstruct; see the
[preparation notes](examples/mapdiff_inverse_folding/README.md#data-responsibilities)
before comparing results.

## Model and Data Cards

Cards make implementations discoverable and document the inputs, configuration,
and assets needed to use them.

**Model cards:** The current layout includes `config.yaml`, a configuration
class, the complete model implementation, and local asset folders. The config
declares Auto class mappings, preprocessing streams, reusable components, and
pretrained-weight metadata. See the
[model-card format and implementation guide](CUSTOMIZE.md#add-a-complete-model-card)
and [external registration instructions](CUSTOMIZE.md#register-an-external-card).

**Data cards:** A standalone data-card file format is not implemented yet.
Datasets currently use registered Python adapters with `Dataset/Task` IDs.
They normalize task fields and preserve provenance independently of any model.
The [dataset extension guide](CUSTOMIZE.md#add-reusable-data) documents this
current interface.

Models can be shared through repository contributions or external card
directories that users register locally. A hosted upload service is not part
of the current library.

## Documentation

| I want to... | Start here |
| --- | --- |
| Add a dataset, preprocessor, encoder, head, or full model | [Extension guide](CUSTOMIZE.md) |
| Understand Auto APIs and the named stage contracts | [Architecture guide](docs/architecture.md) |
| Build and publish a package release | [Release guide](docs/releasing.md) |

## Contributing

Contributions can add reusable components, dataset adapters, complete model
cards, tests, or documentation. Use the [extension guide](CUSTOMIZE.md) for
implementation conventions and open a pull request against this repository.

For local development:

```bash
git clone https://github.com/pykale/protein.git
cd protein
python -m pip install -e ".[dev]"
python -m pytest -q
```

Tests use temporary data, mocked download clients, and small model inputs;
they do not download real pretrained checkpoints.

Report bugs and discuss proposed additions through
[GitHub Issues](https://github.com/pykale/protein/issues).

## Citation and License

When using a contributed model or dataset in research, cite its original
work as described in the corresponding example and source documentation.
Upstream attribution and license details are collected in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

KaleProtein is released under the [MIT License](LICENSE).
