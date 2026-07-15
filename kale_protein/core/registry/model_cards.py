import json
from collections.abc import Iterable
from pathlib import Path

from .base import Registry

MODEL_CARD_REGISTRY = Registry("model cards", normalize_strings=True)

_CONFIG_FILENAMES = ("config.yaml", "config.yml", "config.json")


def discover_model_cards(search_roots, *, model_card_registry=None, preset_registry=None):
    """Discover and register model cards below one or more filesystem roots."""
    if isinstance(search_roots, (str, Path)):
        search_roots = (search_roots,)
    model_card_registry = model_card_registry or MODEL_CARD_REGISTRY

    config_paths = set()
    for root in search_roots:
        root = Path(root)
        if root.is_file():
            config_paths.add(root.resolve())
            continue
        for filename in _CONFIG_FILENAMES:
            config_paths.update(path.resolve() for path in root.rglob(filename))

    discovered = []
    for config_path in sorted(config_paths):
        metadata = read_model_card_metadata(config_path)
        if metadata.get("model_id"):
            discovered.append(
                register_model_card(
                    config_path,
                    metadata=metadata,
                    model_card_registry=model_card_registry,
                    preset_registry=preset_registry,
                )
            )
    return discovered


def register_model_card(
    config_path, *, metadata=None, model_card_registry=None, preset_registry=None
):
    """Register a config path under its declared model id and preset aliases."""
    config_path = Path(config_path).resolve()
    metadata = metadata or read_model_card_metadata(config_path)
    model_id = metadata.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"Model card does not declare a non-empty model_id: {config_path}")

    model_card_registry = model_card_registry or MODEL_CARD_REGISTRY
    _register_path(model_card_registry, model_id, config_path)
    for alias in _preset_aliases(metadata):
        model_card_registry.register_alias(alias, model_id)
        if preset_registry is not None:
            _register_path(preset_registry, alias, config_path)
    return model_id


def read_model_card_metadata(config_path):
    """Read model-card registration fields without importing card-local code."""
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:
            data = _read_top_level_yaml_scalars(text)
        else:
            data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Model card config must contain a mapping: {config_path}")
    return data


def _register_path(registry, key, path):
    if registry.has(key):
        existing = Path(registry.get(key)).resolve()
        if existing != path:
            raise ValueError(
                f"Duplicate {registry.name} key {key!r}: {existing} and {path}"
            )
        return
    registry.register(key, path)


def _preset_aliases(metadata):
    aliases = []
    for field in ("name", "model_type", "preset", "presets", "aliases"):
        value = metadata.get(field)
        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
            aliases.extend(alias for alias in value if isinstance(alias, str))
    return tuple(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))


def _read_top_level_yaml_scalars(text):
    data = {}
    for raw_line in text.splitlines():
        if not raw_line or raw_line[0].isspace() or raw_line.lstrip().startswith("#"):
            continue
        key, separator, value = raw_line.partition(":")
        if not separator or not value.strip():
            continue
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        data[key.strip()] = value
    return data
