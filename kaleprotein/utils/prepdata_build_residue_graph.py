"""Build directed residue-neighborhood graphs from representative coordinates."""

from __future__ import annotations


def build_residue_graph(coordinates, *, cutoff=30.0, max_neighbors=10):
    """Return directed neighbor indices and distances for residue coordinates."""

    import torch

    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            "Residue coordinates must have shape [residues, 3], got "
            f"{tuple(coordinates.shape)}."
        )
    if coordinates.shape[0] == 0:
        raise ValueError("Cannot build a residue graph with zero residues.")
    if max_neighbors is not None and max_neighbors < 1:
        raise ValueError("max_neighbors must be positive or None.")

    distances = torch.cdist(coordinates, coordinates)
    residue_ids = torch.arange(coordinates.shape[0], device=coordinates.device)
    source_parts = []
    target_parts = []
    for residue in range(coordinates.shape[0]):
        neighbors = torch.where(
            (distances[residue] < float(cutoff)) & (residue_ids != residue)
        )[0]
        if neighbors.numel() == 0 and coordinates.shape[0] > 1:
            neighbors = distances[residue].topk(2, largest=False).indices[1:]
        if max_neighbors is not None and neighbors.numel() > max_neighbors:
            order = distances[residue, neighbors].argsort()[:max_neighbors]
            neighbors = neighbors[order]
        if neighbors.numel():
            source_parts.append(
                torch.full(
                    (neighbors.numel(),),
                    residue,
                    dtype=torch.long,
                    device=coordinates.device,
                )
            )
            target_parts.append(neighbors)

    if not source_parts:
        return {
            "edge_index": torch.zeros(
                (2, 0), dtype=torch.long, device=coordinates.device
            ),
            "edge_distances": coordinates.new_zeros((0,)),
        }

    source = torch.cat(source_parts)
    target = torch.cat(target_parts)
    return {
        "edge_index": torch.stack([source, target]),
        "edge_distances": torch.linalg.vector_norm(
            coordinates[source] - coordinates[target], dim=-1
        ),
    }


__all__ = ["build_residue_graph"]
