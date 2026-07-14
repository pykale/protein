import copy
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

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
        path = Path(path).resolve()
        text = path.read_text(encoding='utf-8')
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError:
                data = _parse_simple_yaml(text, path)
            else:
                data = yaml.safe_load(text)
            if data is None:
                raise ValueError(f"Config file is empty: {path}")
        data['_config_path'] = str(path)
        data['_config_dir'] = str(path.parent)
        config_cls = cls
        target = data.get('auto_map', {}).get('AutoProteinConfig')
        if target:
            loaded_cls = _load_auto_object(target, str(path.parent))
            if issubclass(loaded_cls, cls):
                config_cls = loaded_cls
        return config_cls(data)

    @classmethod
    def from_pretrained(cls, model_id):
        import kale_protein  # noqa: F401 bootstrap registrations
        from kale_protein.registry import MODEL_CARD_REGISTRY
        return cls.from_yaml(MODEL_CARD_REGISTRY.get(model_id))

    @classmethod
    def from_dict(cls, config_dict): return cls(config_dict)

    @classmethod
    def from_preset(cls, preset_name):
        import kale_protein  # noqa: F401 bootstrap registrations
        from kale_protein.registry import PRESET_REGISTRY

        registered = PRESET_REGISTRY.get(preset_name)
        if isinstance(registered, Mapping):
            return cls.from_dict(registered)
        if isinstance(registered, (str, os.PathLike)):
            path = Path(registered)
            if path.is_file():
                return cls.from_yaml(path)
            if isinstance(registered, str):
                return cls.from_pretrained(registered)
        raise TypeError(
            f"Preset {preset_name!r} must resolve to a config mapping, model id, or config path; "
            f"got {type(registered).__name__}."
        )

    def to_dict(self): return copy.deepcopy(self._config)
    def get(self, key, default=None): return self._config.get(key, default)
    def __getitem__(self, key): return self._config[key]
    def __contains__(self, key): return key in self._config
    def get_streams(self): return {name: StreamSpec(name=name, **spec) for name, spec in self._config.get('streams', {}).items()}
    def auto_class(self, auto_name):
        target = self._config.get('auto_map', {}).get(auto_name)
        if not target:
            model_id = self._config.get('model_id', self._config.get('name', '<unknown>'))
            raise ValueError(f"Config for {model_id!r} does not define auto_map entry for {auto_name}.")
        return _load_auto_object(target, self._config.get('_config_dir'))
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

def _load_auto_object(target, config_dir=None):
    module_name, object_name = target.rsplit('.', 1)
    module = None
    if config_dir:
        module_path = Path(config_dir) / f"{module_name.replace('.', '/')}.py"
        if module_path.exists():
            module = _load_card_module(module_name, module_path, Path(config_dir))
    if module is None:
        module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _load_card_module(module_name, module_path, config_dir):
    """Load one card module in a stable private package.

    Giving each card its own package prevents two cards named ``modeling.py``
    from colliding and permits ordinary relative imports between card files.
    """

    digest = hashlib.sha256(str(config_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    namespace = "kale_protein.dynamic"
    card_package = f"{namespace}.card_{digest}"
    _ensure_namespace_package(namespace, [])
    _ensure_namespace_package(card_package, [str(config_dir.resolve())])

    parts = module_name.split(".")
    for index in range(1, len(parts)):
        package_name = f"{card_package}.{'/'.join(parts[:index])}".replace("/", ".")
        package_path = config_dir.joinpath(*parts[:index])
        _ensure_namespace_package(package_name, [str(package_path.resolve())])

    qualified_name = f"{card_package}.{module_name}"
    module = sys.modules.get(qualified_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model-card module {qualified_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def _ensure_namespace_package(name, paths):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = list(paths)
        sys.modules[name] = module
    return module

def _parse_simple_yaml(text, path):
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    if not lines:
        raise ValueError(f"Config file is empty: {path}")
    data, index = _parse_yaml_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError(f"Cannot parse YAML config near line: {lines[index][1]!r}")
    return data

def _parse_yaml_block(lines, index, indent):
    if index >= len(lines):
        return {}, index
    if lines[index][0] < indent:
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)

def _parse_yaml_mapping(lines, index, indent):
    out = {}
    while index < len(lines):
        current_indent, stripped = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation near line: {stripped!r}")
        if stripped.startswith("- "):
            break
        if ":" not in stripped:
            raise ValueError(f"Expected key/value YAML line, got: {stripped!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        index += 1
        if value:
            out[key] = _parse_yaml_scalar(value)
        elif index < len(lines) and lines[index][0] > indent:
            out[key], index = _parse_yaml_block(lines, index, lines[index][0])
        else:
            out[key] = {}
    return out, index

def _parse_yaml_list(lines, index, indent):
    out = []
    while index < len(lines):
        current_indent, stripped = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or not stripped.startswith("- "):
            break
        value = stripped[2:].strip()
        index += 1
        if value:
            out.append(_parse_yaml_scalar(value))
        elif index < len(lines) and lines[index][0] > indent:
            item, index = _parse_yaml_block(lines, index, lines[index][0])
            out.append(item)
        else:
            out.append(None)
    return out, index

def _parse_yaml_scalar(value):
    if value in ("{}", "[]"):
        return {} if value == "{}" else []
    if value in ("null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
