from kale_protein.registry import FUSION_REGISTRY
@FUSION_REGISTRY.register('bilinear_attention')
class BilinearAttentionFusion:
    def __init__(self, hidden_dim=256, **kwargs): pass
    def __call__(self, stream_outputs):
        vals=list(stream_outputs.values()); a,b=vals[0],vals[1]
        att=[[sum(x*y for x,y in zip(ta,tb)) for tb in b.get('token_embeddings') or [b['embedding']]] for ta in a.get('token_embeddings') or [a['embedding']]]
        return {'embedding':a['embedding']+b['embedding'],'attention':att}
@FUSION_REGISTRY.register('cross_attention')
class CrossAttentionFusion(BilinearAttentionFusion): pass
