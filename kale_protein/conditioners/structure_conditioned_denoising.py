from kale_protein.registry import CONDITIONER_REGISTRY
@CONDITIONER_REGISTRY.register('structure_conditioned_denoising')
class StructureConditionedDenoising:
    def __init__(self, hidden_dim=128, **kwargs): self.hidden_dim=hidden_dim
    def __call__(self, stream_outputs, timestep=None):
        s=stream_outputs['structure']['embedding']; q=stream_outputs['noisy_sequence']['embedding']
        return {'hidden':[(a+b)/2 for a,b in zip(s,q)], 'conditioning':s}
