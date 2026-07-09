from kale_protein.registry import HEAD_REGISTRY
@HEAD_REGISTRY.register(('drug_target_interaction','regression'))
class RegressionHead:
    def __init__(self, input_dim=256, output_dim=1, **kwargs): pass
    def __call__(self, features): return {'prediction':[sum(features['embedding'])/max(1,len(features['embedding']))]}
