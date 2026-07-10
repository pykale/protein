class AutoProteinData:
    """Registry-backed data entrypoint for readable demos and tests."""

    def __new__(cls, data_id):
        import kale_protein  # noqa: F401 bootstrap registrations
        from kale_protein.registry import DATASET_REGISTRY
        loader = DATASET_REGISTRY.get(data_id)
        return loader() if callable(loader) else loader


class AutoProteinDataLoader:
    @classmethod
    def from_config(cls, config):
        return cls(config)
    def __init__(self, config): self.config = config
    def load(self): return self.config.get('data', [])
