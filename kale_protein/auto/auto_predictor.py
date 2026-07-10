import kale_protein  # noqa
from kale_protein.registry import MODALITY_ENCODER_REGISTRY,FUSION_REGISTRY,CONDITIONER_REGISTRY,HEAD_REGISTRY,RUNNER_REGISTRY
from kale_protein.auto.config import AutoProteinConfig

class AutoProteinModel:
    def __new__(cls, model_id=None, *, pretrain=False, component=None):
        if cls is AutoProteinModel and isinstance(model_id, str):
            config = AutoProteinConfig.from_pretrained(model_id)
            model = config.auto_class("AutoProteinModel")(config=config, pretrain=pretrain)
            if component:
                return model.component(component)
            return model
        return super().__new__(cls)

class AutoProteinGenerator:
    def __new__(cls, model_id=None, *, pretrain=False):
        if isinstance(model_id, str):
            config = AutoProteinConfig.from_pretrained(model_id)
            return config.auto_class("AutoProteinGenerator")(config=config, pretrain=pretrain)
        return super().__new__(cls)

class MultiStreamProteinModel:
    def __init__(self, config):
        self.config=config; self.encoders={}
        for name, stream in config.get_streams().items():
            self.encoders[name]=MODALITY_ENCODER_REGISTRY.get((stream.modality, stream.encoder))(**stream.encoder_kwargs)
        self.fusion=None
        if config.get('fusion'):
            f=config['fusion']; self.fusion=FUSION_REGISTRY.get(f['type'])(**f.get('kwargs',{}))
        self.conditioner=None
        if config.get('conditioner'):
            c=config['conditioner']; self.conditioner=CONDITIONER_REGISTRY.get(c['type'])(**c.get('kwargs',{}))
        self.head=HEAD_REGISTRY.get((config['task'], config['head']['type']))(**config['head'].get('kwargs',{}))
    def encode_streams(self,batch): return {n:e(batch[n]) for n,e in self.encoders.items()}
    def component(self, stream_name): return StreamEncoder(self.encoders[stream_name], stream_name)
    def embed(self, stream_data):
        if self.config.get('conditioner') and 'structure' in stream_data:
            batch = dict(stream_data)
            if 'noisy_sequence' not in batch:
                sequence = batch.get('sequence', {'tokens':[0], 'attention_mask':[1]})
                batch['noisy_sequence'] = sequence
            return self.conditioner(self.encode_streams(batch), timestep=batch.get('timestep'))
        if len(self.encoders) == 1:
            name = next(iter(self.encoders))
            return self.encoders[name](stream_data)
        raise ValueError('embed() for multi-stream models requires a named component from AutoProteinModel(..., component=...).')
    def __call__(self,batch):
        so=self.encode_streams(batch)
        if self.fusion is not None: return self.head(self.fusion(so))
        if self.conditioner is not None: return self.head(self.conditioner(so, timestep=batch.get('timestep')))
        return self.head(so)
    def sample(self,batch,sampling_config):
        so=self.encode_streams(batch); features=self.conditioner(so, timestep=batch.get('timestep')) if self.conditioner else so
        return self.head.sample(features, sampling_config)
class AutoProteinPredictor:
    def __new__(cls, model_id=None, *, pretrain=False):
        if cls is AutoProteinPredictor and isinstance(model_id, str):
            config = AutoProteinConfig.from_pretrained(model_id)
            return config.auto_class("AutoProteinPredictor")(config=config, pretrain=pretrain)
        return super().__new__(cls)

    @classmethod
    def from_config(cls, config): return RUNNER_REGISTRY.get(config['runner'])(model=MultiStreamProteinModel(config), config=config)

class StreamEncoder:
    def __init__(self, encoder, stream_name): self.encoder=encoder; self.stream_name=stream_name
    def embed(self, stream_data): return self.encoder(stream_data)
