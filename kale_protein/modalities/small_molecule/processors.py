from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY
@MODALITY_PROCESSOR_REGISTRY.register(('small_molecule','smiles_tokenizer'))
class SMILESTokenizer:
    def __init__(self, input_key='smiles', **kwargs): self.input_key=input_key
    def transform(self, sample):
        s=sample[self.input_key]
        return {'smiles':s,'tokens':[ord(c)%128 for c in s],'attention_mask':[1]*len(s)}
@MODALITY_PROCESSOR_REGISTRY.register(('small_molecule','dummy_graph'))
class DummyMoleculeGraphProcessor:
    def __init__(self, input_key='smiles', **kwargs): self.input_key=input_key
    def transform(self, sample):
        s=sample[self.input_key]
        return {'smiles':s,'node_features':[[float(ord(c)%32)] for c in s] or [[0.0]],'edge_index':[[],[]],'edge_features':[]}
@MODALITY_PROCESSOR_REGISTRY.register(('small_molecule','rdkit_graph'))
class RDKitGraphProcessor(DummyMoleculeGraphProcessor): pass
