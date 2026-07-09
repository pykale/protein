from kale_protein.registry import RUNNER_REGISTRY
@RUNNER_REGISTRY.register('predict')
class PredictRunner:
    def __init__(self, model, config): self.model=model; self.config=config
    def predict(self, batch_or_dataset):
        if isinstance(batch_or_dataset, list): return [self.model(x) for x in batch_or_dataset]
        return self.model(batch_or_dataset)
    def fit(self, train_data, valid_data=None): return {'status':'not_trained','message':'Training loop placeholder.'}
    def generate(self,*a,**k): raise NotImplementedError('PredictRunner does not support generate(). Available methods: predict, fit.')
    def score(self, batch_or_dataset): return self.predict(batch_or_dataset)
