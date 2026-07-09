import kale_protein  # noqa: F401 bootstrap registrations
from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY

class AutoProteinPreprocessor:
    @classmethod
    def from_config(cls, config): return MultiStreamPreprocessor(config)

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
