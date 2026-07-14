# KaleProtein

KaleProtein provides Hugging Face-style Auto APIs for protein, molecule, and
multimodal biological models. Generic Auto classes resolve a model card and its
`auto_map`; named architectures remain inside their model-card directories.

The examples expose their real stages directly:

```text
load data -> preprocess -> collate -> embed/model -> train or predict -> evaluate -> interpret (optional)
```

DrugBAN and MapDiff are self-contained PyTorch refactors of their upstream
implementations. KaleProtein never imports either upstream checkout at runtime.

See the [architecture diagram](docs/architecture.md) for the package boundaries,
model-card discovery path, and end-to-end runtime pipeline.

## Installation

Install the library with the model extras you need:

```bash
python -m pip install -e ".[drugban]"
python -m pip install -e ".[mapdiff]"
python -m pip install -e ".[drugban,mapdiff,dev]"
```

The base package can load configurations and lightweight datasets without
PyTorch. DrugBAN raw-SMILES processing requires RDKit; MapDiff requires PyTorch.

## DrugBAN

This is the direct Auto pipeline for a real DrugBAN-format BindingDB split:

```python
from kale_protein.auto import (
    AutoMoleculePreprocessor,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)

# 1. Load a normalized DTI dataset.
dataset = AutoProteinData(
    "DTI/BindingDB",
    root="path/to/DrugBAN/datasets",
    split="random",
    subset="test",
)
sample = dataset[0]

# 2. Build reusable modality preprocessors.
protein_preprocessor = AutoProteinPreprocessor("protein/sequence")
molecule_preprocessor = AutoMoleculePreprocessor("molecule/SMILES")

# 3. Resolve the card-owned DrugBAN components.
protein_model, molecule_model = AutoProteinModel(
    "DTI/DrugBAN",
    pretrain=False,
)
interaction_predictor = AutoProteinPredictor(
    "DTI/DrugBAN",
    pretrain=False,
)

# 4. Preprocess and collate.
protein_data = protein_preprocessor.tokenize(sample)
molecule_data = molecule_preprocessor.featurize(sample)
batch = interaction_predictor.collator(
    [{"target": protein_data, "drug": molecule_data, "label": sample["label"]}]
)

# 5. Embed each modality and predict the interaction.
protein_embedding = protein_model.embed(batch["target"])
molecule_embedding = molecule_model.embed(batch["drug"])
interaction_prediction = interaction_predictor(
    protein_embedding,
    molecule_embedding,
)
```

The card implements the original model family as learnable PyTorch modules:
canonical 74-feature RDKit atoms plus the virtual-node bit, molecular graph
convolutions, DrugBAN protein encoding and CNNs, bilinear attention, and the MLP
decoder. It does not require DGL or DGL-LifeSci.

Run distinct workflows:

```bash
python -m kale_protein.examples.drugban_dti.train \
  --dataset BindingDB --root /data/drugban --split random --subset train \
  --validation-subset val --checkpoint drugban.pt

python -m kale_protein.examples.drugban_dti.evaluate \
  --dataset BindingDB --root /data/drugban --split random --subset test \
  --checkpoint drugban.pt

python -m kale_protein.examples.drugban_dti.predict \
  --smiles "CCO" --sequence "MKT..." --checkpoint drugban.pt

python -m kale_protein.examples.drugban_dti.interpret \
  --dataset BioSNAP --path /data/biosnap/full.csv --checkpoint drugban.pt
```

Training performs optimizer updates and checkpoint saving. Evaluation computes
AUROC, AUPRC, F1, accuracy, and the selected threshold. Prediction performs
inference only. Interpretation maps actual BAN attention to valid atoms and
protein residues.

See [the DrugBAN model-card README](kale_protein/examples/drugban_dti/README.md).

## MapDiff

MapDiff applies the same direct style to a generative pipeline:

```python
from kale_protein.auto import (
    AutoProteinData,
    AutoProteinGenerator,
    AutoProteinModel,
    AutoProteinPreprocessor,
)
from kale_protein.tasks.inverse_folding import CollatorDiff

# 1. Load processed CATH .pt graphs, a directory of PDBs, or one PDB.
dataset = AutoProteinData(
    "InverseFolding/CATH",
    source="path/to/cath/test",
)
record = dataset[0]

# 2. Preprocess the protein backbone.
structure_preprocessor = AutoProteinPreprocessor("protein/structure")
structure_data = structure_preprocessor.featurize(
    {
        "backbone_coords": record.atom_pos,
        "sequence": record.sequence,
        "id": record.identifier,
    }
)

# 3. Collate sparse EGNN and padded IPA views.
batch = CollatorDiff()([structure_data["graph"]])

# 4. Load the released architecture and weights.
structure_model = AutoProteinModel(
    "InverseFolding/MapDiff",
    pretrain=True,
)
sequence_generator = AutoProteinGenerator(
    "InverseFolding/MapDiff",
    pretrain=True,
)

# 5. Embed the structure and run iterative sequence diffusion.
structure_embedding = structure_model.embed(batch)
generation = sequence_generator.generate(
    structure_embedding,
    steps=100,
    method="ddim",
)
```

`pretrain=True` selects the release-compatible architecture and strictly loads
the published v1.0.1 checkpoint: 464 state entries and 14,821,937 tensor
elements. The runtime is plain PyTorch and does not require PyG, `torch_scatter`,
OpenFold, `einops`, Hydra, Biopython, or DSSP.

The card also contains a smaller `kale-mapdiff-v1` architecture for local
training and fast tests:

```bash
python -m kale_protein.examples.mapdiff_inverse_folding.pretrain_ipa \
  /data/cath/train --output ipa.pt

python -m kale_protein.examples.mapdiff_inverse_folding.train_diffusion \
  /data/cath/train --ipa-checkpoint ipa.pt --output mapdiff.pt

python -m kale_protein.examples.mapdiff_inverse_folding.evaluate \
  /data/cath/test --pretrained

python -m kale_protein.examples.mapdiff_inverse_folding.generate \
  structure.pdb --pretrained --steps 100
```

Generation returns sequences, logits, and a non-empty denoising trajectory.
Evaluation reports sequence recovery, perplexity, and diversity. Raw PDB inputs
cannot reproduce CATH-normalized SASA, B-factor, and DSSP channels, so those
channels are zero-filled; processed CATH graphs remain the best input for
published-checkpoint quality.

See [the MapDiff model-card README](kale_protein/examples/mapdiff_inverse_folding/README.md).

## Reusable DTI Data

BindingDB, Human, and BioSNAP share one task-level CSV implementation, so any DTI
model can reuse them:

```python
bindingdb = AutoProteinData("DTI/BindingDB", root="path/to/datasets")
human_train = AutoProteinData(
    "DTI/Human",
    root="path/to/datasets",
    split="random",
    subset="train",
)
biosnap_target_test = AutoProteinData(
    "DTI/BioSNAP",
    root="path/to/datasets",
    split="cluster",
    subset="target_test",
)
```

`root` may be the upstream `datasets/` directory or one dataset directory.
`full.csv` loads the complete set; `<split>/<subset>.csv` loads named splits,
including upstream domain-adaptation files such as `source_train` and
`target_test`. Column aliases are normalized to `smiles`, `sequence`, and a
numeric `label`, while source fields and provenance remain in `metadata`.

## Model Cards

Every named model owns its implementation and assets:

```text
kale_protein/examples/<model>/
  config.yaml
  configuration.py
  modeling.py
  data/
  maps/
  weights/
  README.md
```

`config.yaml` declares `model_id`, architecture metadata, pretrained assets, and
`auto_map` targets. Built-in cards are discovered from their configuration
metadata. External cards can be registered without editing an Auto class:

```python
from kale_protein.registry import register_model_card

register_model_card("path/to/my_model/config.yaml")
model = AutoProteinModel("MyTask/MyModel")
```

The Auto layer contains no DrugBAN or MapDiff definitions, branches, or embedded
presets. Relative imports between external card files are isolated in a private
per-card module namespace.

## Pretrained Assets

For `pretrain=True`, KaleProtein:

1. checks the card's local `weights/` path;
2. verifies an optional SHA-256 checksum;
3. atomically downloads a valid configured URL when the file is absent;
4. extracts raw, `model`, `state_dict`, or `model_state_dict` checkpoints;
5. strictly loads the architecture-specific state dictionary;
6. removes partial files when download or verification fails.

DrugBAN intentionally has no pretrained URL and raises a train-it-yourself
error when no local checkpoint exists. MapDiff uses the published v1.0.1 release
URL. Large weights and datasets are excluded from packages and version control;
small construction maps are packaged with their cards.

## Development

Tests use temporary CSVs and `.pt` graphs, fake RDKit objects, fake URLs and
downloaders, and small trainable configurations. They never download real large
weights.

```bash
python -m pytest -q
python -m compileall -q kale_protein
python -m build
```

See [CUSTOMIZE.md](CUSTOMIZE.md) for adding datasets, preprocessors, and model
cards. Refactored-source attribution and licenses are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
