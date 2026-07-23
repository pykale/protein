"""Shared sequence extraction for inverse-folding metrics."""

from kaleprotein.loaddata.records import AA_TO_INDEX


def sequences_from(value):
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
            result.extend(sequences_from(item))
        return result
    return []


def targets_from(data, logits):
    import torch

    if isinstance(data, dict):
        for key in ("labels", "label", "targets", "target"):
            if key in data:
                return torch.as_tensor(
                    data[key], dtype=torch.long, device=logits.device
                )
        if "x" in data:
            x = torch.as_tensor(data["x"], device=logits.device)
            return x.argmax(dim=-1) if x.ndim == logits.ndim else x.long()
    if hasattr(data, "label"):
        return data.label.to(logits.device)
    native = sequences_from(data)
    if native:
        encoded = [
            [AA_TO_INDEX.get(aa, 0) for aa in sequence]
            for sequence in native
        ]
        if logits.ndim == 2:
            flattened = [token for sequence in encoded for token in sequence]
            return torch.tensor(
                flattened, dtype=torch.long, device=logits.device
            )
        max_length = logits.shape[-2]
        target = torch.full(
            (len(encoded), max_length),
            -100,
            dtype=torch.long,
            device=logits.device,
        )
        for row, tokens in enumerate(encoded[: logits.shape[0]]):
            target[row, : min(len(tokens), max_length)] = torch.tensor(
                tokens[:max_length], device=logits.device
            )
        return target
    raise ValueError(
        "Perplexity needs residue labels or native sequences in data."
    )


__all__ = ["sequences_from", "targets_from"]
