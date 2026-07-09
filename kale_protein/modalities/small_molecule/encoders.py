from kale_protein.registry import MODALITY_ENCODER_REGISTRY
@MODALITY_ENCODER_REGISTRY.register(('small_molecule','drugban_molecule_gnn'))
class DrugBANMoleculeGNNEncoder:
    def __init__(self, hidden_dim=128, output_dim=128, **kwargs): self.output_dim=output_dim
    def __call__(self, batch):
        vals=[row[0] for row in batch.get('node_features', [[0.0]])]; mean=sum(vals)/max(1,len(vals))/32.0
        emb=[mean]*self.output_dim
        return {'embedding':emb,'token_embeddings':[[v/32.0]*self.output_dim for v in vals],'mask':None}
