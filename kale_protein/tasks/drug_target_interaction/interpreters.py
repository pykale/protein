from kale_protein.registry import INTERPRETER_REGISTRY
@INTERPRETER_REGISTRY.register(('drug_target_interaction','bilinear_attention_map'))
class BilinearAttentionMapInterpreter:
    def __init__(self, config): self.config=config
    def explain(self, predictor, data):
        out=predictor.predict(data[0] if isinstance(data,list) else data)
        return {'attention': out.get('attention'), 'message': 'Attention map returned when available.'}
