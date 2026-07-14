# MapDiff Inverse Folding

This card provides two self-contained PyTorch architectures:

- `upstream-mapdiff-v1` exactly matches the published MapDiff v1.0.1 release
  checkpoint: 464 state entries and 14,821,937 tensor elements under `model`,
  `prior_model`, and `noise_schedule`.
- `kale-mapdiff-v1` is the smaller refactor used for quick training, tests, and
  local architectural experiments.

Neither path imports the upstream checkout at runtime. The release-compatible
EGNN replaces PyG message aggregation and graph normalization with equivalent
plain-PyTorch operations; IPA reshapes are implemented without `einops`.

## Attribution

The MapDiff architecture and adapted code are copyright 2024 Peizhen Bai and
used under the MIT License from
[peizhenbai/MapDiff](https://github.com/peizhenbai/MapDiff). The IPA design in
MapDiff derives from OpenFold code copyright 2021 AlQuraishi Laboratory and
DeepMind Technologies Limited under Apache License 2.0. See the upstream paper,
*Mask-prior-guided denoising diffusion improves inverse protein folding*, for
the published method and training results.

## Pretrained Release

Install the model extra first:

```bash
python -m pip install -e ".[mapdiff]"
```

`AutoProteinModel("InverseFolding/MapDiff", pretrain=True)` and
`AutoProteinGenerator(..., pretrain=True)` select `upstream-mapdiff-v1`, resolve
the configured v1.0.1 URL through `kale_protein.auto.weights`, extract the common
`{"model": state_dict}` container, and load all keys and shapes strictly.

The card profile reproduces the checkpoint's embedded construction config:
31 node inputs, 93 edge inputs, 128 hidden channels, six EGNN layers, six IPA
layers, 500 diffusion steps, marginal noise, and mask ratio 0.4 +/- 0.2. The
CATH 4.2 marginal is packaged as `maps/train_marginal_x.json` in the upstream
amino-acid order `ARNDCQEGHILKMFPSTWYV`.

```python
from kale_protein.auto import AutoProteinGenerator, AutoProteinModel
from kale_protein.tasks.inverse_folding import CATHGraphDataset, CollatorDiff

batch = CollatorDiff()([CATHGraphDataset("structure.pdb")[0]])
model = AutoProteinModel("InverseFolding/MapDiff", pretrain=True)
generator = AutoProteinGenerator("InverseFolding/MapDiff", pretrain=True)
result = generator.generate(model.embed(batch), steps=100, method="ddim")
```

## Data Compatibility

`CATHGraphDataset` reads plain `.pt` graph dictionaries/directories and PDB
files. PDB preprocessing retains complete `N, CA, C, O` backbones. The release
adapter adds virtual C-beta atoms and reconstructs the trained 31/93-dimensional
CATH geometric channels, including the historical release channel ordering.

Raw PDBs do not provide the normalized solvent-accessibility, B-factor, or DSSP
channels used in CATH training; those channels are set to zero. This does not
affect checkpoint compatibility or execution, but can affect prediction quality
relative to the authors' fully processed CATH graphs. Loading a `.pt` object
pickled as a PyG class still requires PyG solely to unpickle that object; plain
dictionary records do not.

## Workflows

```bash
python -m kale_protein.examples.mapdiff_inverse_folding.generate structure.pdb --pretrained --steps 100
python -m kale_protein.examples.mapdiff_inverse_folding.evaluate CATH_DIR --pretrained
python -m kale_protein.examples.mapdiff_inverse_folding.pretrain_ipa CATH_DIR --output ipa.pt
python -m kale_protein.examples.mapdiff_inverse_folding.train_diffusion CATH_DIR --ipa-checkpoint ipa.pt --output mapdiff.pt
```

`pretrain_ipa.py` and `train_diffusion.py` train the lightweight
`kale-mapdiff-v1` architecture. `generate.py` and `evaluate.py` accept either
`--pretrained` for the release or `--checkpoint` for automatic architecture
detection and strict local loading.

## Dependencies

Both architectures require PyTorch 2.0 or newer. PyG, `torch_scatter`, OpenFold,
Hydra, Biopython, DSSP, SciPy, and `einops` are not runtime requirements for
plain PDB or dictionary `.pt` inputs.
