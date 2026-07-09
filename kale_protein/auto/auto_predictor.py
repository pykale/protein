import kale_protein  # noqa
from kale_protein.registry import MODALITY_ENCODER_REGISTRY,FUSION_REGISTRY,CONDITIONER_REGISTRY,HEAD_REGISTRY,RUNNER_REGISTRY

try:
    from torch import nn
except ImportError:
    class _BaseModule:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
    class _ModuleDict(dict):
        pass
else:
    _BaseModule = nn.Module
    _ModuleDict = dict


class MultiStreamProteinModel(_BaseModule):
    def __init__(self, config):
        super().__init__(); self.config=config; self.encoders=_ModuleDict()
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
    def forward(self,batch):
        so=self.encode_streams(batch)
        if self.fusion is not None: return self.head(self.fusion(so))
        if self.conditioner is not None: return self.head(self.conditioner(so, timestep=batch.get('timestep')))
        return self.head(so)
    def sample(self,batch,sampling_config):
        so=self.encode_streams(batch); features=self.conditioner(so, timestep=batch.get('timestep')) if self.conditioner else so
        return self.head.sample(features, sampling_config)
class AutoProteinPredictor:
    @classmethod
    def from_config(cls, config): return RUNNER_REGISTRY.get(config['runner'])(model=MultiStreamProteinModel(config), config=config)
