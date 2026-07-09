import json
from pathlib import Path

import kale_protein  # noqa
from kale_protein.registry import MODALITY_ENCODER_REGISTRY,FUSION_REGISTRY,CONDITIONER_REGISTRY,HEAD_REGISTRY,RUNNER_REGISTRY


class MultiStreamProteinModel:
    def __init__(self, config, pretrained=False, checkpoint_path=None):
        self.config=config; self.encoders={}; self.loaded_checkpoint=None
        for name, stream in config.get_streams().items():
            self.encoders[name]=MODALITY_ENCODER_REGISTRY.get((stream.modality, stream.encoder))(**stream.encoder_kwargs)
        self.fusion=None
        if config.get('fusion'):
            f=config['fusion']; self.fusion=FUSION_REGISTRY.get(f['type'])(**f.get('kwargs',{}))
        self.conditioner=None
        if config.get('conditioner'):
            c=config['conditioner']; self.conditioner=CONDITIONER_REGISTRY.get(c['type'])(**c.get('kwargs',{}))
        self.head=HEAD_REGISTRY.get((config['task'], config['head']['type']))(**config['head'].get('kwargs',{}))
        if pretrained:
            self.load_checkpoint(checkpoint_path or self._checkpoint_path_from_config())
    def _checkpoint_path_from_config(self):
        checkpoint = self.config.get('checkpoint', {}) or {}
        path = checkpoint.get('path')
        if not path:
            raise ValueError('pretrained=True requires checkpoint.path in config or checkpoint_path argument.')
        return path
    def load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint file not found: {path}')
        with path.open('r', encoding='utf-8') as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
        self.loaded_checkpoint = {'path': str(path), 'data': checkpoint}
        state = checkpoint.get('state_dict', checkpoint)
        if hasattr(self, 'load_state_dict'):
            self.load_state_dict(state)
        return self.loaded_checkpoint
    def encode_streams(self,batch): return {n:e(batch[n]) for n,e in self.encoders.items()}
    def __call__(self,batch):
        so=self.encode_streams(batch)
        if self.fusion is not None: return self.head(self.fusion(so))
        if self.conditioner is not None: return self.head(self.conditioner(so, timestep=batch.get('timestep')))
        return self.head(so)
    def sample(self,batch,sampling_config):
        so=self.encode_streams(batch); features=self.conditioner(so, timestep=batch.get('timestep')) if self.conditioner else so
        return self.head.sample(features, sampling_config)
class AutoProteinPredictor:
    @classmethod
    def from_config(cls, config, pretrained=False, checkpoint_path=None):
        model=MultiStreamProteinModel(config, pretrained=pretrained, checkpoint_path=checkpoint_path)
        return RUNNER_REGISTRY.get(config['runner'])(model=model, config=config)
