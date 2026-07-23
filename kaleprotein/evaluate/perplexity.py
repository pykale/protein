"""Perplexity for inverse-folding residue logits."""

import math

from kaleprotein.auto.registry import EVALUATOR_REGISTRY

from ._sequence import targets_from


class Perplexity:
    def __call__(self, outputs, data):
        import torch

        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs
        if logits is None:
            raise ValueError(
                "Perplexity requires logits in outputs['logits']."
            )
        logits = torch.as_tensor(logits).float()
        targets = targets_from(data, logits).reshape(-1)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        usable = min(flat_logits.shape[0], targets.shape[0])
        targets = targets[:usable]
        flat_logits = flat_logits[:usable]
        valid = targets != -100
        if not valid.any():
            raise ValueError("Perplexity received no valid target residues.")
        loss = torch.nn.functional.cross_entropy(
            flat_logits[valid], targets[valid]
        )
        return math.exp(float(loss.detach()))


EVALUATOR_REGISTRY.register(("inverse_folding", "perplexity"), Perplexity)

__all__ = ["Perplexity"]
