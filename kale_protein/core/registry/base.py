"""Small registries used by the public Auto factories."""

import inspect


class Registry:
    """Map stable public ids to implementations without knowing model names."""

    def __init__(self, name, *, normalize_strings=False):
        self.name = name
        self._mapping = {}
        self._aliases = {}
        self._normalize_strings = normalize_strings

    def _normalize(self, key):
        if self._normalize_strings and isinstance(key, str):
            return key.strip().casefold()
        if self._normalize_strings and isinstance(key, tuple):
            return tuple(self._normalize(part) for part in key)
        return key

    def register(self, key, value=None, *, aliases=()):
        canonical = self._normalize(key)

        def decorator(obj):
            existing = self._mapping.get(canonical)
            if (
                existing is not None
                and existing is not obj
                and existing != obj
                and not _same_source_object(existing, obj)
            ):
                raise ValueError(f"Duplicate {self.name} id: {key!r}")
            if existing is None:
                self._mapping[canonical] = obj
            for alias in aliases:
                self.register_alias(alias, canonical)
            return obj

        if value is None:
            return decorator
        return decorator(value)

    def register_alias(self, alias, target):
        alias = self._normalize(alias)
        target = self.resolve_key(target)
        if alias in self._mapping and alias != target:
            raise ValueError(f"{alias!r} is already a canonical {self.name} id")
        existing = self._aliases.get(alias)
        if existing is not None and existing != target:
            raise ValueError(f"Duplicate {self.name} alias: {alias!r}")
        self._aliases[alias] = target

    def resolve_key(self, key):
        key = self._normalize(key)
        seen = set()
        while key in self._aliases:
            if key in seen:
                raise RuntimeError(f"Cyclic alias in {self.name}: {key!r}")
            seen.add(key)
            key = self._aliases[key]
        return key

    def get(self, key):
        resolved = self.resolve_key(key)
        if resolved not in self._mapping:
            available = self.available_keys()
            message = f"No {self.name} registered for id: {key!r}."
            if available:
                message += "\nAvailable ids: " + ", ".join(str(item) for item in available)
            raise KeyError(message)
        return self._mapping[resolved]

    def has(self, key):
        return self.resolve_key(key) in self._mapping

    def available_keys(self, *, include_aliases=False):
        keys = list(self._mapping)
        if include_aliases:
            keys.extend(self._aliases)
        return sorted(set(keys), key=str)


def _same_source_object(left, right):
    try:
        return (
            left.__qualname__ == right.__qualname__
            and inspect.getsourcefile(left) == inspect.getsourcefile(right)
        )
    except (AttributeError, TypeError):
        return False
