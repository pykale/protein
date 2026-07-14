"""MapDiff-compatible collators without a torch-geometric dependency."""

from __future__ import annotations

from typing import Iterable

from .datasets import DiffusionBatch, GraphBatch, IPABatch, ProteinGraph, coerce_protein_graph


def _torch():
    import torch
    return torch


def batch_graphs(graphs: Iterable[ProteinGraph]) -> GraphBatch:
    torch = _torch()
    graphs = [coerce_protein_graph(graph) for graph in graphs]
    if not graphs:
        raise ValueError("Cannot collate an empty graph list.")
    offsets = []
    total = 0
    for graph in graphs:
        offsets.append(total)
        total += graph.num_nodes
    edges = [graph.edge_index + offset for graph, offset in zip(graphs, offsets)]
    return GraphBatch(
        x=torch.cat([graph.x for graph in graphs], dim=0),
        atom_pos=torch.cat([graph.atom_pos for graph in graphs], dim=0),
        atom_mask=torch.cat([graph.atom_mask for graph in graphs], dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_attr=torch.cat([graph.edge_attr for graph in graphs], dim=0),
        batch=torch.cat([torch.full((graph.num_nodes,), i, dtype=torch.long) for i, graph in enumerate(graphs)]),
        ptr=torch.tensor(offsets + [total], dtype=torch.long),
        identifiers=[graph.identifier for graph in graphs],
        sequences=[graph.sequence for graph in graphs],
    )


def _padded_batch(graphs: list[ProteinGraph]) -> IPABatch:
    torch = _torch()
    batch_size = len(graphs)
    max_length = max(graph.num_nodes for graph in graphs)
    x = torch.zeros(batch_size, max_length, 20)
    atom_pos = torch.zeros(batch_size, max_length, 4, 3)
    x_pad = torch.zeros(batch_size, max_length, dtype=torch.bool)
    x_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    label = torch.full((batch_size, max_length), 20, dtype=torch.long)
    for index, graph in enumerate(graphs):
        length = graph.num_nodes
        x[index, :length] = graph.x
        atom_pos[index, :length] = graph.atom_pos
        x_pad[index, :length] = True
        label[index, :length] = graph.x.argmax(dim=-1)
    return IPABatch(x, atom_pos, x_pad, x_mask, label)


class CollatorIPAPretrain:
    """Apply BERT-style candidate masking to padded residue batches."""

    def __init__(self, candi_rate=0.7, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1, seed=None):
        if candi_rate < 0 or candi_rate > 1:
            raise ValueError("candi_rate must be between 0 and 1.")
        if abs(mask_rate + replace_rate + keep_rate - 1.0) > 1e-6:
            raise ValueError("mask_rate, replace_rate, and keep_rate must sum to 1.")
        self.candi_rate = candi_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.keep_rate = keep_rate
        self.generator = None
        if seed is not None:
            self.generator = _torch().Generator().manual_seed(seed)

    def __call__(self, records) -> IPABatch:
        torch = _torch()
        graphs = [coerce_protein_graph(record) for record in records]
        batch = _padded_batch(graphs)
        for row, graph in enumerate(graphs):
            length = graph.num_nodes
            count = min(length, max(1, int(round(length * self.candi_rate))))
            candidate = torch.randperm(length, generator=self.generator)[:count]
            random_order = candidate[torch.randperm(count, generator=self.generator)]
            mask_count = max(1, int(round(count * self.mask_rate))) if count else 0
            replace_count = min(count - mask_count, int(round(count * self.replace_rate)))
            mask_ids = random_order[:mask_count]
            replace_ids = random_order[mask_count:mask_count + replace_count]
            keep_ids = random_order[mask_count + replace_count:]
            batch.x_mask[row, mask_ids] = 1
            batch.x_mask[row, replace_ids] = 2
            batch.x_mask[row, keep_ids] = 3
            if replace_ids.numel():
                old = batch.label[row, replace_ids]
                replacement = torch.randint(0, 19, old.shape, generator=self.generator)
                replacement = replacement + (replacement >= old).long()
                batch.x[row, replace_ids] = torch.nn.functional.one_hot(replacement, 20).float()
            batch.x[row, mask_ids] = 0.0
        return batch


class CollatorDiff:
    """Return sparse EGNN and padded IPA views for diffusion training."""

    def __call__(self, records) -> DiffusionBatch:
        graphs = [coerce_protein_graph(record) for record in records]
        if not graphs:
            raise ValueError("Cannot collate an empty graph list.")
        return DiffusionBatch(batch_graphs(graphs), _padded_batch(graphs))


__all__ = ["CollatorDiff", "CollatorIPAPretrain", "batch_graphs"]
