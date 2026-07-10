# MapDiff Inverse Folding Example

This example is a self-contained model card folder. `config.yaml` declares the
MapDiff architecture and `auto_map`, while `configuration_mapdiff.py` and
`modeling_mapdiff.py` own the model-specific code. The generic Auto classes
only resolve the card and dispatch to those files.

Folder layout:

```text
config.yaml
configuration_mapdiff.py
modeling_mapdiff.py
data/
weights/
```

This applies the same explicit pipeline style to a generative model:

```python
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

Run:

```bash
python kale_protein/examples/mapdiff_inverse_folding/generate.py
python kale_protein/examples/mapdiff_inverse_folding/evaluate.py
```

Set `pretrain=True` to resolve `weights/mapdiff_weight.pt`; if it is absent,
the URL in `config.yaml` is used to download the upstream release weight.
