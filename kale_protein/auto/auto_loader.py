class AutoProteinDataLoader:
    @classmethod
    def from_config(cls, config):
        return cls(config)
    def __init__(self, config): self.config = config
    def load(self): return self.config.get('data', [])
