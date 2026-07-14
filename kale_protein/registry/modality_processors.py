from collections.abc import Mapping

from .base import Registry


class ModalityProcessorRegistry(Registry):
    def __init__(self):
        super().__init__("modality_processors")
        self._aliases = {}

    def register(self, key, value=None, *, aliases=()):
        def decorator(obj):
            Registry.register(self, key, obj)
            for alias, input_key in self._iter_aliases(aliases):
                self.register_alias(alias, key, input_key=input_key)
            return obj

        if value is None:
            return decorator
        return decorator(value)

    def register_alias(self, alias, processor_key, *, input_key=None):
        normalized = self._normalize_alias(alias)
        resolution = (processor_key, input_key)
        existing = self._aliases.get(normalized)
        if existing is not None and existing != resolution:
            raise ValueError(f"Processor alias {alias!r} is already registered for {existing[0]!r}.")
        self._aliases[normalized] = resolution

    def resolve(self, processor_id):
        if isinstance(processor_id, tuple):
            return self.get(processor_id), None
        if not isinstance(processor_id, str):
            raise TypeError(f"Processor id must be a string or registry key, got {type(processor_id).__name__}.")

        normalized = self._normalize_alias(processor_id)
        if normalized in self._aliases:
            processor_key, input_key = self._aliases[normalized]
            return self.get(processor_key), input_key

        modality, separator, processor = processor_id.partition("/")
        canonical_key = (modality, processor) if separator else None
        if canonical_key is not None and self.has(canonical_key):
            return self.get(canonical_key), None

        aliases = ", ".join(sorted(self._aliases))
        suffix = f" Registered aliases: {aliases}." if aliases else ""
        raise ValueError(f"Unknown preprocessor id: {processor_id!r}.{suffix}")

    def available_aliases(self):
        return sorted(self._aliases)

    @staticmethod
    def _normalize_alias(alias):
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("Processor aliases must be non-empty strings.")
        return alias.strip().casefold()

    @staticmethod
    def _iter_aliases(aliases):
        if isinstance(aliases, Mapping):
            return aliases.items()

        normalized = []
        for alias in aliases:
            if isinstance(alias, str):
                normalized.append((alias, None))
                continue
            try:
                alias_name, input_key = alias
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Processor aliases must be strings or (alias, input_key) pairs."
                ) from exc
            normalized.append((alias_name, input_key))
        return normalized


MODALITY_PROCESSORS_REGISTRY = ModalityProcessorRegistry()
