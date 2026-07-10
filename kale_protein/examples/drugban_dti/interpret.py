import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kale_protein.auto import (
    AutoMoleculePreprocessor,
    AutoProteinData,
    AutoProteinInterpreter,
    AutoProteinModel,
    AutoProteinPredictor,
    AutoProteinPreprocessor,
    AutoProteinConfig,
)


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
interpreter = AutoProteinInterpreter.from_config(AutoProteinConfig.from_preset("drugban"))
interpretation = interpreter.explain(
    interaction_predictor,
    [{"target": protein_data, "drug": drug_data, "label": label}],
)

print(interaction_prediction)
print(interpretation)
