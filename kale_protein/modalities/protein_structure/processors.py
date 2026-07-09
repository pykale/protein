from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY
@MODALITY_PROCESSOR_REGISTRY.register(('protein_structure','backbone_coordinate_processor'))
class BackboneCoordinateProcessor:
    def __init__(self, input_key='backbone_coords', **kwargs): self.input_key=input_key
    def transform(self, sample):
        coords=sample[self.input_key]
        return {'coords':coords,'coord_mask':[True]*len(coords)}
