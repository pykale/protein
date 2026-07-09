import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor, AutoProteinInterpreter
sample={'id':'sample_1','smiles':'CCO','sequence':'MKTFFVLLL','label':1}
config_dict=AutoProteinConfig.from_preset('drugban').to_dict()
config_dict['checkpoint']['path']=str(Path(__file__).with_name('toy_checkpoint.json'))
config=AutoProteinConfig.from_dict(config_dict)
data=AutoProteinPreprocessor.from_config(config).transform_sample(sample)
predictor=AutoProteinPredictor.from_config(config, pretrained=True)
print(AutoProteinInterpreter.from_config(config).explain(predictor, data))
