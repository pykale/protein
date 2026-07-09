import math
from kale_protein.registry import HEAD_REGISTRY
@HEAD_REGISTRY.register(('drug_target_interaction','binary_classifier'))
class BinaryClassificationHead:
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=1, **kwargs): pass
    def __call__(self, features):
        x=features['embedding']; logit=sum(x)/max(1,len(x)); prob=1/(1+math.exp(-logit))
        out={'logits':[logit],'probabilities':[prob]}
        if 'attention' in features: out['attention']=features['attention']
        return out
