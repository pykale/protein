"""Mask-aware DrugBAN attention interpretation."""

from __future__ import annotations

import torch

from kale_protein.registry import INTERPRETER_REGISTRY


@INTERPRETER_REGISTRY.register(("drug_target_interaction", "bilinear_attention_map"))
class BilinearAttentionMapInterpreter:
    def __init__(self, config):
        self.config = config

    def explain(self, predictor, data):
        output = predictor.extract_attention(data)
        attention = output.get("attention")
        if not torch.is_tensor(attention):
            raise ValueError("Attention interpretation requires a single, consistently padded tensor batch")
        drug_mask = torch.as_tensor(output["drug_mask"], dtype=torch.bool)
        protein_mask = torch.as_tensor(output["protein_mask"], dtype=torch.bool)
        atom_symbols = output.get("atom_symbols") or [[] for _ in range(attention.shape[0])]
        sequences = output.get("sequences") or [None for _ in range(attention.shape[0])]

        samples = []
        normalized_maps = []
        for batch_index in range(attention.shape[0]):
            atom_indices = torch.nonzero(drug_mask[batch_index], as_tuple=False).flatten()
            residue_indices = torch.nonzero(protein_mask[batch_index], as_tuple=False).flatten()
            raw = attention[batch_index].float().index_select(1, atom_indices).index_select(2, residue_indices)
            raw = raw.mean(dim=0)
            normalized = torch.softmax(raw.flatten(), dim=0).view_as(raw) if raw.numel() else raw
            normalized_maps.append(normalized)
            atom_scores = normalized.sum(dim=1) if normalized.numel() else torch.empty(0)
            residue_scores = normalized.sum(dim=0) if normalized.numel() else torch.empty(0)
            sequence = sequences[batch_index] or ""
            symbols = atom_symbols[batch_index] if batch_index < len(atom_symbols) else []
            samples.append(
                {
                    "atoms": [
                        {
                            "index": atom_index,
                            "symbol": symbols[atom_index] if atom_index < len(symbols) else None,
                            "attention": float(atom_scores[score_index]),
                        }
                        for score_index, atom_index in enumerate(atom_indices.tolist())
                    ],
                    "residues": [
                        {
                            "index": residue_index,
                            "residue": sequence[residue_index] if residue_index < len(sequence) else None,
                            "attention": float(residue_scores[score_index]),
                        }
                        for score_index, residue_index in enumerate(residue_indices.tolist())
                    ],
                    "attention": normalized.tolist(),
                }
            )
        return {"samples": samples, "attention": normalized_maps}
