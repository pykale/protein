# MapDiff Example

This example is a runnable script layout for the self-contained KaleProtein MapDiff-style refactor:

`load CATH/PDB data -> graph preprocessing/collation -> IPA or diffusion model build -> train/evaluate/predict`

The scripts do not import the original MapDiff repository. They use:

```python
from kaleprotein import AutoProteinData, AutoProteinGenerator, AutoProteinModel, AutoProteinPreprocessor

data, _ = AutoProteinData("InverseFolding/CATH", data_dir="examples/mapdiff/data/processed").load("test")
structure_preprocessor = AutoProteinPreprocessor("protein/structure")
structure_encoder = AutoProteinModel("InverseFolding/MapDiff", pretrain=False)
sequence_generator = AutoProteinGenerator("InverseFolding/MapDiff", pretrain=False)

structure_data = structure_preprocessor.featurize(data)
structure_embedding = structure_encoder.embed(structure_data)
generated_sequence = sequence_generator.generate(structure_embedding)
```

## Commands

```bash
python examples/mapdiff/train_diffusion.py --config examples/mapdiff/config.yaml
python examples/mapdiff/eval.py --config examples/mapdiff/config.yaml
python examples/mapdiff/predict.py --config examples/mapdiff/config.yaml
```

`pretrain=True` is handled by `AutoProteinModel("InverseFolding/MapDiff", pretrain=True)`. The model card contains the upstream release URL, but tests use fake data and fake download functions, so pytest never downloads the large checkpoint.
