# MapDiff Inverse Folding Example

This folder is the MapDiff model card. It keeps MapDiff-specific configuration
and modeling code beside the example scripts, data folder, and weights folder.

The generic Auto classes do not contain MapDiff branches. They resolve
`InverseFolding/MapDiff`, load this folder's `config.yaml`, read `auto_map`, and
import the MapDiff classes from this folder.

## Folder Layout

```text
config.yaml
configuration.py
modeling.py
data/
weights/
generate.py
evaluate.py
```

## Auto Pipeline

The generative workflow keeps the same explicit shape:

```text
load data -> preprocess -> embed condition -> generate -> evaluate optional
```

Minimal generation pipeline:

```python
from kale_protein.auto import (
    AutoProteinData,
    AutoProteinGenerator,
    AutoProteinModel,
    AutoProteinPreprocessor,
)

data, native_sequence = AutoProteinData("InverseFolding/CATH")
structure_preprocessor = AutoProteinPreprocessor("protein/structure")
sequence_preprocessor = AutoProteinPreprocessor("protein/masked_sequence")

structure_encoder = AutoProteinModel("InverseFolding/MapDiff", pretrain=False)
sequence_generator = AutoProteinGenerator("InverseFolding/MapDiff", pretrain=False)

structure_data = {
    "structure": structure_preprocessor.featurize(data),
    "noisy_sequence": sequence_preprocessor.tokenize(data),
}

structure_embedding = structure_encoder.embed(structure_data)
generated_sequence = sequence_generator.generate(structure_embedding)
```

## Scripts

```bash
python kale_protein/examples/mapdiff_inverse_folding/generate.py
python kale_protein/examples/mapdiff_inverse_folding/evaluate.py
```

`generate.py` loads a CATH-style structure sample, preprocesses structure and
masked sequence streams, embeds the conditioning signal, and samples a sequence.

`evaluate.py` runs the same generation path and computes inverse-folding metrics
against the native sequence.

## Pretrained Weights

`config.yaml` declares:

```yaml
pretrained:
  local_dir: weights
  filename: mapdiff_weight.pt
  url: https://github.com/peizhenbai/MapDiff/releases/download/v1.0.1/mapdiff_weight.pt
```

Use `pretrain=False` for scripts that should run without downloading large
assets. Set `pretrain=True` to resolve `weights/mapdiff_weight.pt`; if it is
missing, KaleProtein downloads the upstream release weight into this folder.

## Extending

MapDiff-specific generation code belongs in `modeling.py`;
configuration logic belongs in `configuration.py`; model-card wiring
belongs in `config.yaml`. Avoid adding MapDiff-specific branches to
`kale_protein.auto`.
