# MapDiff Inverse Folding

This model card contains a self-contained PyTorch MapDiff refactor. It supports
both the published v1.0.1 checkpoint layout and a smaller architecture for
local training and tests, without importing the upstream checkout at runtime.

## Card Layout

```text
mapdiff_inverse_folding/
  config.yaml
  configuration.py
  collators.py
  data.py
  model_mapdiff.py
  egnn.py
  ipa.py
  diffusion.py
  upstream_compat.py
  pretrain_ipa.py
  train_diffusion.py
  evaluate.py
  interpret.py
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
and checkpoint keys are not duplicated. `AutoProteinModel` resolves and loads
the requested full checkpoint once; `MapDiffModel` only adapts checkpoint state
to the selected parameter tree.

## Evaluation, Generation, And Interpretation Pipeline

```python
from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)

# 1. Load the model card shared by the data and model sides.
config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")

# 2. Load, preprocess, collate, and batch inverse-folding records.
loader = AutoProteinDataLoader(
    "CATH/InverseFolding",
    config=config,
    source="structure.pdb",
    batch_size=1,
)
inputs = next(iter(loader))

# 3. Build the complete model and resolve its pretrained checkpoint.
model = AutoProteinModel.from_config(config, pretrain=True)

# 4. Encode structure conditions from the loader mapping.
embeddings = model.embed(**inputs)

# 5. Generate from the named embedding mapping.
generation = model.generate(
    **embeddings,
    steps=100,
    method="ddim",
    num_samples=1,
)

# 6a. Evaluate the generation mapping.
metrics = model.evaluate(**generation)

# 6b. Independently interpret its denoising trajectory when needed.
interpretation = AutoProteinInterpreter.from_config(config).explain(**generation)
```

Generation returns sequences, logits, token ids, and a non-empty denoising
trajectory. It also carries `reference_sequences` from the condition mapping,
so generation metrics can consume the dictionary directly.

The loader does not invoke MapDiff. It only guarantees that each yielded
mapping can enter `model.embed(**inputs)` directly.

## Architectures

- `upstream-mapdiff-v1` matches the published v1.0.1 parameter tree under
  `model`, `prior_model`, and `noise_schedule`.
- `kale-mapdiff-v1` is a smaller real EGNN, IPA, and categorical-diffusion
  implementation used for local training and tests.

The release profile uses 31 node inputs, 93 edge inputs, 128 hidden channels,
six EGNN layers, six IPA layers, 500 diffusion steps, and the packaged CATH
marginal in `maps/train_marginal_x.json`.

## Data Compatibility

`CATHDataset` reads plain `.pt` structure dictionaries, directories, PDB files,
and mmCIF files into model-independent structure records. MapDiff's example-local
data layer then constructs residue graphs, paired IPA batches, the virtual C-beta,
and the expected geometric channels.

Raw PDBs do not provide the normalized solvent-accessibility, B-factor, or DSSP
channels used during upstream CATH training, so those channels are zero-filled.
Processed CATH records are preferred for published-checkpoint quality.

## Workflows

```bash
python -m pip install -e ".[mapdiff]"

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

- `pretrain_ipa.py` trains the masking prior and saves a complete model checkpoint.
- `train_diffusion.py` trains the full lightweight diffusion model.
- `evaluate.py` reports sequence recovery, perplexity, and diversity.
- `interpret.py` independently reports changes along the denoising trajectory.
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
