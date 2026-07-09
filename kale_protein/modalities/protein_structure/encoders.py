from kale_protein.registry import MODALITY_ENCODER_REGISTRY
@MODALITY_ENCODER_REGISTRY.register(('protein_structure','mapdiff_structure_encoder'))
class MapDiffStructureEncoder:
    def __init__(self, hidden_dim=128, **kwargs): self.hidden_dim=hidden_dim
    def __call__(self,batch):
        coords=batch['coords']; flat=[]
        for r in coords:
            p=r[0] if r and isinstance(r[0], list) else r; flat.append(sum(p)/3.0)
        mean=sum(flat)/max(1,len(flat))
        return {'embedding':[mean]*self.hidden_dim,'token_embeddings':[[v]*self.hidden_dim for v in flat],'mask':batch.get('coord_mask')}
