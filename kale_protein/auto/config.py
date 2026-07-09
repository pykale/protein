from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import copy

from .yaml_utils import load_yaml

@dataclass
class StreamSpec:
    name: str
    modality: str
    input_key: str
    processor: str
    encoder: str
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False



class AutoProteinConfig:
    def __init__(self, config_dict):
        self._config = copy.deepcopy(config_dict)
        self.validate()

    @classmethod
    def from_yaml(cls, path):
        return cls(load_yaml(path))

    @classmethod
    def from_dict(cls, config_dict): return cls(config_dict)

    @classmethod
    def from_preset(cls, preset_name):
        path = Path(__file__).resolve().parents[1] / 'presets' / f'{preset_name}.yaml'
        if path.exists():
            return cls.from_yaml(path)
        raise FileNotFoundError(f"Unknown preset {preset_name!r}: {path}")

    def to_dict(self): return copy.deepcopy(self._config)
    def get(self, key, default=None): return self._config.get(key, default)
    def __getitem__(self, key): return self._config[key]
    def __contains__(self, key): return key in self._config
    def get_streams(self): return {name: StreamSpec(name=name, **spec) for name, spec in self._config.get('streams', {}).items()}
    def validate(self):
        missing=[k for k in ['task','objective','runner','streams'] if k not in self._config]
        if missing: raise ValueError(f"Config missing required fields: {missing}")
        for name, stream in self._config['streams'].items():
            miss=[k for k in ['modality','input_key','processor','encoder'] if k not in stream]
            if miss: raise ValueError(f"Stream {name!r} missing required fields: {miss}")
        if self._config['objective']=='discriminative' and not (self._config.get('fusion') or self._config.get('conditioner')): raise ValueError('Discriminative tasks must define fusion or conditioner.')
        if self._config['objective']=='discriminative' and 'head' not in self._config: raise ValueError('Discriminative tasks must define head.')
        if self._config['objective']=='generative' and 'head' not in self._config: raise ValueError('Generative tasks must define head.')
        if self._config['runner']=='diffusion_generate' and 'sampling' not in self._config: raise ValueError('diffusion_generate runner requires sampling config.')
        return True
