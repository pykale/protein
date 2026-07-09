from kale_protein.registry import RUNNER_REGISTRY
@RUNNER_REGISTRY.register('diffusion_generate')
class DiffusionGenerateRunner:
    def __init__(self, model, config): self.model=model; self.config=config; self.last_trajectory=None
    def generate(self, batch_or_dataset):
        if isinstance(batch_or_dataset, list): return [self.generate(x) for x in batch_or_dataset]
        out=self.model.sample(batch_or_dataset, self.config.get('sampling', {})); self.last_trajectory=out.get('trajectory'); return out
    def predict(self,*a,**k): raise NotImplementedError('DiffusionGenerateRunner does not support predict(). Available methods: generate.')
    def fit(self,*a,**k): raise NotImplementedError('DiffusionGenerateRunner does not support fit(). Available methods: generate.')
    def score(self,*a,**k): return self.generate(*a,**k)
