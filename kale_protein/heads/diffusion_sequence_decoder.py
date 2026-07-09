from kale_protein.registry import HEAD_REGISTRY
AA='ACDEFGHIKLMNPQRSTVWY'
@HEAD_REGISTRY.register(('inverse_folding','diffusion_sequence_decoder'))
class DiffusionSequenceDecoder:
    def __init__(self, hidden_dim=128, vocab_size=25, **kwargs): self.vocab_size=vocab_size
    def __call__(self, hidden, timestep=None):
        h=hidden['hidden'] if isinstance(hidden,dict) else hidden
        return {'logits':h[:self.vocab_size]}
    def sample(self, hidden, sampling_config):
        logits=self(hidden)['logits']; seq=''.join(AA[int(abs(v)*100)%len(AA)] for v in logits[:max(1, min(10, len(logits)))])
        return {'token_ids':[int(abs(v)*100)%self.vocab_size for v in logits], 'sequences':[seq], 'trajectory':[]}
