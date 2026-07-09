"""Small YAML loading helpers for KaleProtein config files.

PyYAML is used when available. The fallback parser intentionally supports only
KaleProtein's simple config shape: nested mappings, scalar values, inline
lists/dicts, and block lists of scalars.
"""

import json
from pathlib import Path


def _parse_scalar(value):
    value = value.strip()
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        output = {}
        for part in inner.split(","):
            key, item = part.split(":", 1)
            output[key.strip().strip('"\'')] = _parse_scalar(item)
        return output
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _simple_yaml_load(text):
    root = {}
    stack = [(-1, root)]
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            parent.append(_parse_scalar(stripped[2:]))
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"Cannot parse config line: {line}")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue
        next_container = {}
        if index + 1 < len(lines) and lines[index + 1].strip().startswith("- "):
            next_container = []
        parent[key] = next_container
        stack.append((indent, next_container))
    return root


def load_yaml(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _simple_yaml_load(text)
    return yaml.safe_load(text)
