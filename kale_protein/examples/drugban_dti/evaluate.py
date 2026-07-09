import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor, AutoProteinEvaluator
samples=[{'id':'sample_1','smiles':'CCO','sequence':'MKTFFVLLL','label':1},{'id':'sample_2','smiles':'CCN','sequence':'GAVLIPFWY','label':0}]
config_dict=AutoProteinConfig.from_preset('drugban').to_dict()
config_dict['checkpoint']['path']=str(Path(__file__).with_name('toy_checkpoint.json'))
config=AutoProteinConfig.from_dict(config_dict)
data=AutoProteinPreprocessor.from_config(config).transform_dataset(samples)
outputs=AutoProteinPredictor.from_config(config, pretrained=True).predict(data)
print(AutoProteinEvaluator.from_config(config).evaluate(outputs, data))
