# DrugBAN DTI Example

This example is a self-contained model card folder. `config.yaml` declares the
model architecture and `auto_map`, while `configuration_drugban.py` and
`modeling_drugban.py` own the DrugBAN-specific code. The generic Auto classes
only resolve the card and dispatch to those files.

Folder layout:

```text
config.yaml
configuration_drugban.py
modeling_drugban.py
data/
weights/
```

The runnable scripts keep the explicit pipeline shape:

```python
data, label = AutoProteinData("DTI/PDBBind")
preprocessor_protein = AutoProteinPreprocessor("protein/sequence")
preprocessor_drug = AutoMoleculePreprocessor("molecule/SMILE")
protein_model, molecule_model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
interaction_predictor = AutoProteinPredictor("DTI/DrugBAN", pretrain=False)

protein_data = preprocessor_protein.tokenize(data)
drug_data = preprocessor_drug.tokenize(data)

protein_embedding = protein_model.embed(protein_data)
drug_embedding = molecule_model.embed(drug_data)

interaction_prediction = interaction_predictor(protein_embedding, drug_embedding)
```

Run:

```bash
python kale_protein/examples/drugban_dti/evaluate.py
python kale_protein/examples/drugban_dti/train.py
python kale_protein/examples/drugban_dti/interpret.py
```

Set `pretrain=True` only after placing `weights/drugban.pt` in this folder or
adding a valid pretrained URL to `config.yaml`; otherwise the Auto loader raises
a clear missing-weight error.
