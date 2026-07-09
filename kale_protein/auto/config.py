from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import copy
import json

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

_DRUGBAN = {
    'name':'drugban','task':'drug_target_interaction','objective':'discriminative','runner':'predict',
    'streams':{
        'drug':{'modality':'small_molecule','input_key':'smiles','processor':'rdkit_graph','encoder':'drugban_molecule_gnn','processor_kwargs':{},'encoder_kwargs':{'hidden_dim':128,'output_dim':128}},
        'target':{'modality':'protein_sequence','input_key':'sequence','processor':'amino_acid_tokenizer','encoder':'drugban_protein_cnn','processor_kwargs':{'max_length':1000},'encoder_kwargs':{'hidden_dim':128,'output_dim':128}},
    },
    'fusion':{'type':'bilinear_attention','kwargs':{'hidden_dim':256}},
    'head':{'type':'binary_classifier','kwargs':{'input_dim':256,'hidden_dim':128,'output_dim':1}},
    'loss':{'type':'binary_cross_entropy'},
    'checkpoint':{'path':None, 'strict':False},
    'evaluation':{'metrics':['auroc','auprc','accuracy','f1']},
    'interpretation':{'method':'bilinear_attention_map'},
}
_MAPDIFF = {
    'name':'mapdiff','task':'inverse_folding','objective':'generative','runner':'diffusion_generate',
    'streams':{
        'structure':{'modality':'protein_structure','input_key':'backbone_coords','processor':'backbone_coordinate_processor','encoder':'mapdiff_structure_encoder','processor_kwargs':{},'encoder_kwargs':{'hidden_dim':128}},
        'noisy_sequence':{'modality':'protein_sequence','input_key':'sequence','processor':'masked_sequence_tokenizer','encoder':'residue_token_embedding','processor_kwargs':{'max_length':512,'mask_token':'<mask>'},'encoder_kwargs':{'hidden_dim':128}},
    },
    'conditioner':{'type':'structure_conditioned_denoising','kwargs':{'hidden_dim':128}},
    'head':{'type':'diffusion_sequence_decoder','kwargs':{'hidden_dim':128,'vocab_size':25}},
    'sampling':{'steps':100,'num_samples':8,'temperature':1.0},
    'checkpoint':{'path':None, 'strict':False},
    'evaluation':{'metrics':['sequence_recovery','diversity','novelty']},
    'interpretation':{'method':'denoising_trajectory'},
}

class AutoProteinConfig:
    def __init__(self, config_dict):
        self._config = copy.deepcopy(config_dict)
        self.validate()

    @classmethod
    def from_yaml(cls, path):
        text = Path(path).read_text(encoding='utf-8')
        try:
            return cls(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError('YAML parsing requires PyYAML in this minimal environment; use from_preset or JSON-formatted config.') from exc

    @classmethod
    def from_dict(cls, config_dict): return cls(config_dict)

    @classmethod
    def from_preset(cls, preset_name):
        if preset_name == 'drugban': return cls(_DRUGBAN)
        if preset_name == 'mapdiff': return cls(_MAPDIFF)
        path = Path(__file__).resolve().parents[1] / 'presets' / f'{preset_name}.yaml'
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
