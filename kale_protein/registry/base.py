class Registry:
    def __init__(self, name):
        self.name = name
        self._mapping = {}

    def register(self, key, value=None):
        def decorator(obj):
            self._mapping[key] = obj
            return obj
        if value is None:
            return decorator
        self._mapping[key] = value
        return value

    def get(self, key):
        if key not in self._mapping:
            available = self.available_keys()
            msg = f"No {self.name} registered for key: {key!r}."
            if isinstance(key, tuple) and key:
                filtered = [k for k in available if isinstance(k, tuple) and k[0] == key[0]]
                if filtered:
                    msg += f"\nAvailable {self.name} for {key[0]!r}: " + ", ".join(str(k) for k in filtered)
            if available:
                msg += "\nAvailable keys: " + ", ".join(str(k) for k in available)
            raise KeyError(msg)
        return self._mapping[key]

    def has(self, key):
        return key in self._mapping

    def available_keys(self):
        return sorted(self._mapping.keys(), key=str)
