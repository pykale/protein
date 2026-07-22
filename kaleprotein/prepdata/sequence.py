"""Reusable protein-sequence preprocessing."""

from kaleprotein.auto.registry import PREPROCESSOR_REGISTRY


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ("<pad>", "<unk>", "<mask>", "<bos>", "<eos>")
DEFAULT_VOCAB = {
    token: index for index, token in enumerate((*SPECIAL_TOKENS, *AMINO_ACIDS))
}


@PREPROCESSOR_REGISTRY.register(
    "sequence/amino_acid",
    aliases=("sequence/amino_acid_tokenizer", "protein/sequence"),
)
class AminoAcidTokenizer:
    default_input_key = "sequence"

    def __init__(self, input_key="sequence", max_length=None, vocab=None, **kwargs):
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be a positive integer or None.")
        self.input_key = input_key
        self.max_length = max_length
        self.vocab = dict(vocab or DEFAULT_VOCAB)

    def transform(self, sample):
        sequence = str(sample[self.input_key]).upper()
        if self.max_length is not None:
            sequence = sequence[: self.max_length]
        unknown_id = self.vocab["<unk>"]
        tokens = [self.vocab.get(residue, unknown_id) for residue in sequence]
        return {
            "sequence": sequence,
            "tokens": tokens,
            "attention_mask": [1] * len(tokens),
        }


@PREPROCESSOR_REGISTRY.register(
    "sequence/masked",
    aliases=("sequence/masked_sequence_tokenizer", "protein/masked_sequence"),
)
class MaskedSequenceTokenizer(AminoAcidTokenizer):
    """Tokenize a sequence and apply optional explicit mask positions."""

    def __init__(self, input_key="sequence", mask_positions_key="mask_positions", **kwargs):
        super().__init__(input_key=input_key, **kwargs)
        self.mask_positions_key = mask_positions_key

    def transform(self, sample):
        output = super().transform(sample)
        positions = [int(index) for index in sample.get(self.mask_positions_key, ())]
        for index in positions:
            if index < 0 or index >= len(output["tokens"]):
                raise ValueError(
                    f"Mask position {index} is outside sequence length {len(output['tokens'])}."
                )
            output["tokens"][index] = self.vocab["<mask>"]
        output["mask_positions"] = positions
        return output


__all__ = ["AminoAcidTokenizer", "MaskedSequenceTokenizer"]
