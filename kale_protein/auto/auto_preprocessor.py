import kale_protein  # noqa: F401 bootstrap registrations
from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY

class AutoProteinPreprocessor:
    def __new__(cls, preprocessor_id=None, **kwargs):
        if cls is AutoProteinPreprocessor and isinstance(preprocessor_id, str):
            return SingleModalityPreprocessor(preprocessor_id, **kwargs)
        return super().__new__(cls)

    @classmethod
    def from_config(cls, config): return MultiStreamPreprocessor(config)

class AutoMoleculePreprocessor:
    def __new__(cls, preprocessor_id=None, **kwargs):
        if isinstance(preprocessor_id, str):
            return SingleModalityPreprocessor(preprocessor_id, **kwargs)
        return super().__new__(cls)

class SingleModalityPreprocessor:
    ALIASES = {
        "protein/sequence": ("protein_sequence", "amino_acid_tokenizer", "sequence"),
        "protein/masked_sequence": ("protein_sequence", "masked_sequence_tokenizer", "sequence"),
        "protein/structure": ("protein_structure", "backbone_coordinate_processor", "backbone_coords"),
        "molecule/SMILE": ("small_molecule", "rdkit_graph", "smiles"),
        "molecule/SMILES": ("small_molecule", "rdkit_graph", "smiles"),
        "molecule/smiles": ("small_molecule", "rdkit_graph", "smiles"),
    }
    def __init__(self, preprocessor_id, **kwargs):
        if preprocessor_id not in self.ALIASES:
            raise ValueError(f"Unknown preprocessor id: {preprocessor_id}")
        modality, processor, input_key = self.ALIASES[preprocessor_id]
        self.processor = MODALITY_PROCESSOR_REGISTRY.get((modality, processor))(input_key=input_key, **kwargs)
    def tokenize(self, data): return self.processor.transform(data)
    def featurize(self, data): return self.processor.transform(data)

class MultiStreamPreprocessor:
    def __init__(self, config):
        self.config=config; self.processors={}
        for name, stream in config.get_streams().items():
            cls=MODALITY_PROCESSOR_REGISTRY.get((stream.modality, stream.processor))
            self.processors[name]=cls(input_key=stream.input_key, **stream.processor_kwargs)
    def transform_sample(self, sample):
        out={name:proc.transform(sample) for name,proc in self.processors.items()}
        for key in ['label','domain','id','metadata']:
            if key in sample: out[key]=sample[key]
        return out
    def transform_dataset(self, dataset): return [self.transform_sample(s) for s in dataset]
