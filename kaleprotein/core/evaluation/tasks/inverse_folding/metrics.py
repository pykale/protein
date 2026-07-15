"""Metrics for inverse-folding predictions and diffusion logits."""

from __future__ import annotations

import math
from itertools import combinations

from kaleprotein.core.registry import EVALUATOR_REGISTRY
from kaleprotein.core.data.datasets.inverse_folding import AA_ALPHABET, AA_TO_INDEX


def _sequences(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if "sequences" in value:
            return list(value["sequences"])
        if "sequence" in value:
            return [value["sequence"]]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_sequences(item))
        return result
    return []


def _targets(data, logits):
    import torch

    if isinstance(data, dict):
        for key in ("labels", "label", "targets", "target"):
            if key in data:
                return torch.as_tensor(data[key], dtype=torch.long, device=logits.device)
        if "x" in data:
            x = torch.as_tensor(data["x"], device=logits.device)
            return x.argmax(dim=-1) if x.ndim == logits.ndim else x.long()
    if hasattr(data, "label"):
        return data.label.to(logits.device)
    native = _sequences(data)
    if native:
        encoded = [[AA_TO_INDEX.get(aa, 0) for aa in sequence] for sequence in native]
        if logits.ndim == 2:
            flattened = [token for sequence in encoded for token in sequence]
            return torch.tensor(flattened, dtype=torch.long, device=logits.device)
        max_length = logits.shape[-2]
        target = torch.full((len(encoded), max_length), -100, dtype=torch.long, device=logits.device)
        for row, tokens in enumerate(encoded[: logits.shape[0]]):
            target[row, : min(len(tokens), max_length)] = torch.tensor(tokens[:max_length], device=logits.device)
        return target
    raise ValueError("Perplexity needs residue labels or native sequences in data.")


class SequenceRecovery:
    def __call__(self, outputs, data):
        predicted = _sequences(outputs)
        native = _sequences(data)
        if not predicted or not native:
            raise ValueError("Sequence recovery needs generated and native sequences.")
        scores = []
        for index, sequence in enumerate(predicted):
            reference = native[index % len(native)]
            length = min(len(sequence), len(reference))
            scores.append(sum(a == b for a, b in zip(sequence[:length], reference[:length])) / max(len(reference), 1))
        return sum(scores) / len(scores)


class Diversity:
    """Mean normalized pairwise Hamming distance among generated sequences."""

    def __call__(self, outputs, data=None):
        sequences = _sequences(outputs)
        if len(sequences) < 2:
            return 0.0
        distances = []
        for first, second in combinations(sequences, 2):
            length = max(len(first), len(second), 1)
            matches = sum(a == b for a, b in zip(first, second))
            distances.append(1.0 - matches / length)
        return sum(distances) / len(distances)


class Novelty:
    def __call__(self, outputs, data):
        generated = _sequences(outputs)
        known = set(_sequences(data))
        return sum(sequence not in known for sequence in generated) / max(len(generated), 1)


class Perplexity:
    def __call__(self, outputs, data):
        import torch

        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs
        if logits is None:
            raise ValueError("Perplexity requires logits in outputs['logits'].")
        logits = torch.as_tensor(logits).float()
        targets = _targets(data, logits).reshape(-1)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        usable = min(flat_logits.shape[0], targets.shape[0])
        targets = targets[:usable]
        flat_logits = flat_logits[:usable]
        valid = targets != -100
        if not valid.any():
            raise ValueError("Perplexity received no valid target residues.")
        loss = torch.nn.functional.cross_entropy(flat_logits[valid], targets[valid])
        return math.exp(float(loss.detach()))


for name, metric in (
    ("sequence_recovery", SequenceRecovery),
    ("diversity", Diversity),
    ("novelty", Novelty),
    ("perplexity", Perplexity),
):
    EVALUATOR_REGISTRY.register(("inverse_folding", name), metric)


__all__ = ["Diversity", "Novelty", "Perplexity", "SequenceRecovery"]
