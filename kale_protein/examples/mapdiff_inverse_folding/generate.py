import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kale_protein.auto import AutoProteinConfig, AutoProteinPreprocessor, AutoProteinPredictor
samples=[{'id':'protein_1','backbone_coords':[[[0.0,0.0,0.0]],[[1.0,0.0,0.0]]],'sequence':'MA'}]
config=AutoProteinConfig.from_preset('mapdiff')
data=AutoProteinPreprocessor.from_config(config).transform_dataset(samples)
print(AutoProteinPredictor.from_config(config).generate(data[0]))
