# MapDiff Inverse Folding

This card is a self-contained refactor of the complete MapDiff v1.0.1
architecture and workflow. It does not import a MapDiff checkout at runtime and
does not contain a toy or fallback network.

## Layout

```text
mapdiff_inverse_folding/
  config.yaml
  configuration.py
  prepdata_mapdiff.py
  collators.py
  model_mapdiff.py
  train.py
  evaluate.py
  generate.py
  interpret.py
  maps/
  data/
  weights/
```

The example composes two registered model roles:

```text
MapDiffModel
  AutoProteinEmbedder("structure/mapdiff_condition")
  AutoProteinPredictor("inverse_folding/mapdiff_generator")
```

The generator owns the full IPA prior, EGNN denoiser, categorical noise
schedule, training objectives, and iterative sampler. The embedder is a
non-owning condition-encoder view, so parameters and checkpoint keys are not
duplicated. Reusable sparse EGNN and invariant point attention layers live in
`kaleprotein.model.layers`; MapDiff-specific assembly remains here.

## Pipeline

Install the dependencies and download this example without cloning the library:

```bash
python -m pip install "kaleprotein[mapdiff_inverse_folding]"
python -m kaleprotein download-example mapdiff_inverse_folding
```

Run the following snippets and workflow commands from the directory containing
the downloaded `examples/` folder. Data and large pretrained weights are
obtained separately; `pretrain=True` resolves the configured checkpoint.

```python
from kaleprotein.auto import (
    AutoProteinConfig,
    AutoProteinDataLoader,
    AutoProteinInterpreter,
    AutoProteinModel,
)
from examples.mapdiff_inverse_folding import register_model_card

register_model_card()

# 1. Load one model card for the data and model sides.
config = AutoProteinConfig.from_pretrained("InverseFolding/MapDiff")

# 2. Load, preprocess, and batch a PDB, mmCIF, or processed CATH file.
loader = AutoProteinDataLoader(
    "CATH/InverseFolding",
    config=config,
    source="structure.pdb",
    batch_size=1,
)
inputs = next(iter(loader))

# 3. Build the complete model and resolve the release checkpoint.
model = AutoProteinModel.from_config(config, pretrain=True)

# 4. Encode the named structural conditions.
embeddings = model.embed(**inputs)

# 5. Generate inverse-folded sequences.
generation = model.generate(
    **embeddings,
    steps=100,
    method="ddim",
    num_samples=1,
)

# 6a. Evaluate the generated mapping.
metrics = model.evaluate(**generation)

# 6b. Interpret the same mapping independently when needed.
interpretation = AutoProteinInterpreter.from_config(config).explain(
    **generation
)
```

Every boundary is a normal dictionary. `model.embed(**inputs)` returns named
fields including `structure_embedding`, `edge_embedding`, and
`ipa_pair_embedding`; `model.generate(**embeddings)` returns sequences, logits,
token ids, references, and denoising trajectories.

## Data Responsibilities

`AutoProteinDataLoader` composes two independent card classes:

- `MapDiffPreprocessor` handles one structure at a time. It constructs the
  release-compatible 31-channel residue input, 93-channel edge input, virtual
  C-beta, secondary-structure input, and five-atom IPA view.
- `MapDiffCollator` only batchizes prepared samples. It concatenates sparse
  graphs, offsets edge indices, and pads IPA tensors.

Reusable residue-neighbor and backbone-geometry functions live in
`kaleprotein.utils` under `prepdata_*` filenames. Random IPA masking and
categorical diffusion noise are training objectives, so the model applies them
at each forward pass rather than freezing them during preprocessing.

Processed CATH `.pt` records retain existing `x`, `extra_x`, `edge_attr`, `ss`,
`mu_r_norm`, and five-atom coordinates and use them directly when available.
Residues missing N, CA, or C are removed during per-sample preprocessing, with
sequence, node features, and graph edges filtered and remapped together. For
raw PDB/mmCIF inputs, geometry is computed from coordinates. The generic parser
does not provide the normalized SASA, B-factor, or DSSP channels used to train
the release model, so those unavailable channels are zero-filled.

## Full Model

The single configured architecture matches the MapDiff v1.0.1 parameter tree:

```text
model.*          # six-layer EGNN denoiser
prior_model.*    # six-layer IPA masking prior
noise_schedule.* # 500-step categorical schedule
```

The release profile uses 128 hidden channels, 31 node inputs, 93 edge inputs,
four IPA heads, and the CATH marginal stored in
`maps/train_marginal_x.json`. The compact
`maps/release_state_manifest.json` records the official 464-tensor key/shape
contract, so smoke tests can detect parameter-tree drift without downloading
the large checkpoint. Tests may reduce dimensions and depth for workflow
execution, but they instantiate this same implementation rather than another
model.

## Training

MapDiff retains its original two-stage objectives and release optimization
settings through one entry point:

```bash
python -m examples.mapdiff_inverse_folding.train \
  /data/cath/train \
  --stage ipa \
  --validation-data /data/cath/validation \
  --output ipa.pt

python -m examples.mapdiff_inverse_folding.train \
  /data/cath/train \
  --stage diffusion \
  --validation-data /data/cath/validation \
  --checkpoint ipa.pt \
  --output mapdiff.pt
```

The `mapdiff` extra includes PyTorch Geometric because official processed CATH
`.pt` files serialize `torch_geometric.data.Data` objects. PyTorch Geometric is
only needed to deserialize those source files; the refactored model and
collator operate on ordinary tensors and dictionaries.

Without command-line overrides, IPA pretraining runs for 200 epochs and
diffusion training for 100 epochs. Both use Adam with `lr=5e-4`,
`betas=(0.95, 0.999)`, gradient clipping, and OneCycleLR, matching the release
training configuration. `--epochs`, `--learning-rate`, `--no-scheduler`, and
the other flags remain available for controlled runs.

Both stages execute the visible pipeline:

```python
inputs = next(iter(loader))
embeddings = model.embed(**inputs)
prediction = model.predict(**embeddings)
loss = prediction["loss"]
```

The IPA stage optimizes `prior_model`; the diffusion stage optimizes the full
model. With `--validation-data`, IPA selects the lowest masking loss while
diffusion performs iterative generation and selects the highest sequence
recovery, using perplexity as the tie-breaker. Both save complete checkpoints
that `AutoProteinModel` can restore.

## Evaluation And Generation

```bash
python -m examples.mapdiff_inverse_folding.evaluate \
  /data/cath/test --pretrained --steps 100

python -m examples.mapdiff_inverse_folding.generate \
  structure.pdb --pretrained --steps 100 --output generated.json

python -m examples.mapdiff_inverse_folding.interpret \
  structure.pdb --pretrained --steps 100
```

Evaluation reports sequence recovery, perplexity, and diversity. Interpretation
is separate and summarizes residue changes along the denoising trajectory.

## Pretrained Weights

`pretrain=True` first checks `weights/mapdiff_weight.pt`. If absent, Auto
atomically downloads the configured v1.0.1 release:

```text
https://github.com/peizhenbai/MapDiff/releases/download/v1.0.1/mapdiff_weight.pt
```

Checkpoint loading requires an exact key and tensor-shape match. It never uses
`strict=False` and never switches to another architecture.

## Attribution

The architecture, data features, training objectives, and sampling procedure
are refactored from
[wenruifan/MapDiff](https://github.com/wenruifan/MapDiff) and the original
[peizhenbai/MapDiff](https://github.com/peizhenbai/MapDiff), distributed under
the MIT License. IPA-related attribution is listed in
`THIRD_PARTY_NOTICES.md`.
