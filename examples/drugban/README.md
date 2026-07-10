# DrugBAN Example

This example is a runnable script layout for the self-contained KaleProtein DrugBAN refactor:

`load CSV data -> SMILES/protein preprocessing -> graph/sequence embedding -> train/evaluate/predict -> optional BAN attention interpretation`

The scripts do not import the original DrugBAN repository. They use:

```python
from kaleprotein import (
    AutoMoleculePreprocessor,
    AutoProteinData,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
)

data, label = AutoProteinData("DTI/PDBBind", data_dir="examples/drugban/data").load("test")
preprocessor_protein = AutoProteinPreprocessor("protein/sequence")
preprocessor_drug = AutoMoleculePreprocessor("molecule/smiles")
protein_model, molecule_model = AutoProteinModel("DTI/DrugBAN", pretrain=False)
interaction_predictor = AutoProteinPredictor("DTI/DrugBAN", pretrain=False)

protein_data = preprocessor_protein.tokenize(data)
drug_data = preprocessor_drug.tokenize(data)

protein_embedding = protein_model.embed(protein_data)
drug_embedding = molecule_model.embed(drug_data)

interaction_prediction = interaction_predictor(protein_embedding, drug_embedding)
```

## Commands

```bash
python examples/drugban/train.py --config examples/drugban/config.yaml
python examples/drugban/eval.py --config examples/drugban/config.yaml
python examples/drugban/predict.py --config examples/drugban/config.yaml
python examples/drugban/interpret.py --config examples/drugban/config.yaml
```

`pretrain=True` is handled by `AutoProteinModel("DTI/DrugBAN", pretrain=True)`. It first checks the model-card `weights/` folder, then downloads from `weights.url` if configured, and otherwise raises a clear training-required error.
