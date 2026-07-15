# MapDiff Inverse Folding

This model card contains a self-contained PyTorch MapDiff refactor. It supports
both the published v1.0.1 checkpoint layout and a smaller architecture for
local training and tests, without importing the upstream checkout at runtime.

## Card Layout

```text
mapdiff_inverse_folding/
  config.yaml
  configuration.py
  modeling.py
  egnn.py
  ipa.py
  diffusion.py
  upstream_compat.py
  pretrain_ipa.py
  train_diffusion.py
  evaluate.py
  generate.py
  maps/
  data/
  weights/
```

The complete model contains two registered roles:

```text
MapDiffModel
  AutoProteinEmbedder("structure/mapdiff_condition")
  AutoProteinPredictor("inverse_folding/mapdiff_generator")
```

The predictor is the diffusion generator and owns the learned denoising
network. The embedder is a non-owning condition-encoding view, so parameters
and checkpoint keys are not duplicated. `MapDiffModel` is the only full
checkpoint owner.

## Generation Pipeline

```python
from kale_protein.auto import (
    AutoProteinData,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPreprocessor,
)

# 1. Load a processed CATH graph or PDB.
record = AutoProteinData("InverseFolding/CATH", source="structure.pdb")[0]

# 2. Preprocess the backbone.
preprocessor = AutoProteinPreprocessor("protein/structure")
structure = preprocessor.featurize(
    {
        "backbone_coords": record.atom_pos,
        "sequence": record.sequence,
        "id": record.identifier,
    }
)

# 3. Load one complete model and build its named batch mapping.
model = AutoProteinModel("InverseFolding/MapDiff", pretrain=True)
processed = {"samples": [structure]}
batch = model.collator(**processed)

# 4. Encode the structural condition.
conditioning = model.embed(**batch)

# 5. Run iterative sequence generation and evaluation.
generation = model.predictor.generate(
    **conditioning,
    steps=100,
    method="ddim",
    num_samples=1,
)
metrics = model.evaluate(**generation)
trajectory = AutoProteinInterpreter.from_config(model.config).explain(**generation)
```

Generation returns sequences, logits, token ids, and a non-empty denoising
trajectory. It also carries `reference_sequences` from the condition mapping,
so generation metrics can consume the dictionary directly.

## Architectures

- `upstream-mapdiff-v1` matches the published v1.0.1 parameter tree under
  `model`, `prior_model`, and `noise_schedule`.
- `kale-mapdiff-v1` is a smaller real EGNN, IPA, and categorical-diffusion
  implementation used for local training and tests.

The release profile uses 31 node inputs, 93 edge inputs, 128 hidden channels,
six EGNN layers, six IPA layers, 500 diffusion steps, and the packaged CATH
marginal in `maps/train_marginal_x.json`.

## Data Compatibility

`CATHGraphDataset` reads plain `.pt` graph dictionaries, directories, and PDB
files. PDB preprocessing retains `N, CA, C, O` backbones; the release adapter
constructs the virtual C-beta and expected geometric channels.

Raw PDBs do not provide the normalized solvent-accessibility, B-factor, or DSSP
channels used during upstream CATH training, so those channels are zero-filled.
Processed CATH records are preferred for published-checkpoint quality.

## Workflows

```bash
python -m pip install -e ".[mapdiff]"

python -m kale_protein.examples.mapdiff_inverse_folding.pretrain_ipa \
  /data/cath/train --output ipa.pt

python -m kale_protein.examples.mapdiff_inverse_folding.train_diffusion \
  /data/cath/train --ipa-checkpoint ipa.pt --output mapdiff.pt

python -m kale_protein.examples.mapdiff_inverse_folding.evaluate \
  /data/cath/test --pretrained

python -m kale_protein.examples.mapdiff_inverse_folding.generate \
  structure.pdb --pretrained --steps 100
```

- `pretrain_ipa.py` trains the masking prior.
- `train_diffusion.py` trains the full lightweight diffusion model.
- `evaluate.py` reports sequence recovery, perplexity, and diversity.
- `generate.py` preprocesses a structure and samples sequences.

## Pretrained Weights

`pretrain=True` checks `weights/mapdiff_weight.pt` and otherwise atomically
downloads the configured release:

```text
https://github.com/peizhenbai/MapDiff/releases/download/v1.0.1/mapdiff_weight.pt
```

The card detects lightweight versus release parameter layouts and requires an
exact state-key and tensor-shape match. It never falls back to `strict=False`.

## Dependencies And Attribution

Plain PDB and dictionary `.pt` inputs require PyTorch but not PyG,
`torch_scatter`, OpenFold, Hydra, Biopython, DSSP, or `einops`.

The architecture and adapted code derive from
[peizhenbai/MapDiff](https://github.com/peizhenbai/MapDiff) under the MIT
License. IPA-related attribution is listed in `THIRD_PARTY_NOTICES.md`.
