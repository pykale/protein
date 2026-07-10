import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kale_protein.auto import (
    AutoProteinConfig,
    AutoProteinData,
    AutoProteinEvaluator,
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

metrics = AutoProteinEvaluator.from_config(AutoProteinConfig.from_preset("mapdiff")).evaluate(
    generated_sequence,
    [{"sequence": native_sequence}],
)

print(generated_sequence)
print(metrics)
