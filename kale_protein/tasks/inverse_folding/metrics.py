from kale_protein.registry import EVALUATOR_REGISTRY
class SequenceRecovery:
    def __call__(self, outputs, data): return 0.0
class Diversity:
    def __call__(self, outputs, data):
        seqs=[]
        for o in (outputs if isinstance(outputs,list) else [outputs]): seqs += o.get('sequences',[])
        return len(set(seqs))/max(1,len(seqs))
class Novelty:
    def __call__(self, outputs, data): return 1.0
class Perplexity:
    def __call__(self, outputs, data): return None
for n,c in [('sequence_recovery',SequenceRecovery),('diversity',Diversity),('novelty',Novelty),('perplexity',Perplexity)]: EVALUATOR_REGISTRY.register(('inverse_folding',n), c)
