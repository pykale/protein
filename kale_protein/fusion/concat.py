from kale_protein.registry import FUSION_REGISTRY
@FUSION_REGISTRY.register('concat')
class ConcatFusion:
    def __call__(self, stream_outputs): return {'embedding':sum([v['embedding'] for v in stream_outputs.values()], [])}
