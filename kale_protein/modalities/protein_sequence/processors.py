from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY
AA=list('ACDEFGHIKLMNPQRSTVWY')
VOCAB={tok:i for i,tok in enumerate(['<pad>','<unk>','<mask>','<bos>','<eos>']+AA)}
@MODALITY_PROCESSOR_REGISTRY.register(('protein_sequence','amino_acid_tokenizer'))
class AminoAcidTokenizer:
    def __init__(self, input_key='sequence', max_length=None, **kwargs): self.input_key=input_key; self.max_length=max_length
    def transform(self, sample):
        seq=sample[self.input_key]; seq=seq[:self.max_length] if self.max_length else seq
        ids=[VOCAB.get(a,VOCAB['<unk>']) for a in seq]
        return {'sequence':seq,'tokens':ids,'attention_mask':[1]*len(ids)}
@MODALITY_PROCESSOR_REGISTRY.register(('protein_sequence','masked_sequence_tokenizer'))
class MaskedSequenceTokenizer(AminoAcidTokenizer):
    def __init__(self, input_key='sequence', mask_token='<mask>', **kwargs): super().__init__(input_key=input_key, **kwargs); self.mask_token=mask_token
