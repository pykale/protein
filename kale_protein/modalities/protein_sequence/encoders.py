from kale_protein.registry import MODALITY_ENCODER_REGISTRY
@MODALITY_ENCODER_REGISTRY.register(('protein_sequence','drugban_protein_cnn'))
class DrugBANProteinCNNEncoder:
    def __init__(self, hidden_dim=128, output_dim=128, vocab_size=32, **kwargs): self.output_dim=output_dim
    def __call__(self,batch):
        toks=batch['tokens']; mean=sum(toks)/max(1,len(toks))/32.0
        return {'embedding':[mean]*self.output_dim,'token_embeddings':[[t/32.0]*self.output_dim for t in toks],'mask':batch.get('attention_mask')}
@MODALITY_ENCODER_REGISTRY.register(('protein_sequence','residue_token_embedding'))
class ResidueTokenEmbedding(DrugBANProteinCNNEncoder): pass
