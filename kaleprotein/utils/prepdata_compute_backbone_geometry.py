"""Reusable geometric features derived from protein backbone coordinates."""

from __future__ import annotations

import math


def compute_dihedral(first, second, third, fourth):
    """Compute signed dihedral angles for broadcast-compatible point tensors."""

    import torch
    from torch.nn import functional as functional

    first_vector = first - second
    axis = functional.normalize(third - second, dim=-1)
    last_vector = fourth - third
    first_plane = first_vector - (first_vector * axis).sum(
        -1, keepdim=True
    ) * axis
    last_plane = last_vector - (last_vector * axis).sum(
        -1, keepdim=True
    ) * axis
    return torch.atan2(
        (torch.cross(axis, first_plane, dim=-1) * last_plane).sum(-1),
        (first_plane * last_plane).sum(-1),
    )


def compute_backbone_angle_features(backbone):
    """Return per-residue sin/cos phi and psi features."""

    import torch

    backbone = _validate_backbone(backbone)
    features = backbone.new_zeros(backbone.shape[0], 4)
    if backbone.shape[0] <= 1:
        return features
    nitrogen, alpha_carbon, carbon = (
        backbone[:, 0],
        backbone[:, 1],
        backbone[:, 2],
    )
    phi = compute_dihedral(
        carbon[:-1], nitrogen[:-1], alpha_carbon[:-1], nitrogen[1:]
    )
    psi = compute_dihedral(
        nitrogen[:-1], alpha_carbon[:-1], carbon[:-1], nitrogen[1:]
    )
    features[:-1, 0] = torch.sin(phi)
    features[:-1, 1] = torch.cos(phi)
    features[:-1, 2] = torch.sin(psi)
    features[:-1, 3] = torch.cos(psi)
    return features


def compute_backbone_frames(backbone):
    """Return normal, N-CA, and in-plane axes for every residue."""

    import torch
    from torch.nn import functional as functional

    backbone = _validate_backbone(backbone)
    nitrogen, alpha_carbon, carbon = (
        backbone[:, 0],
        backbone[:, 1],
        backbone[:, 2],
    )
    u_axis = functional.normalize(nitrogen - alpha_carbon, dim=-1)
    tangent = functional.normalize(carbon - alpha_carbon, dim=-1)
    normal = functional.normalize(
        torch.cross(u_axis, tangent, dim=-1), dim=-1
    )
    v_axis = torch.cross(normal, u_axis, dim=-1)
    return {
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
    }


def compute_edge_orientations(backbone, edge_index):
    """Return the 12 local-frame orientation values used by residue GNNs."""

    import torch

    backbone = _validate_backbone(backbone)
    edge_index = torch.as_tensor(
        edge_index, dtype=torch.long, device=backbone.device
    )
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges].")
    if edge_index.shape[1] == 0:
        return backbone.new_zeros((0, 12))

    source, target = edge_index
    frames = compute_backbone_frames(backbone)
    basis = torch.stack(
        [
            frames["normal"][target],
            frames["u_axis"][target],
            frames["v_axis"][target],
        ],
        dim=1,
    )
    alpha_carbon = backbone[:, 1]
    values = [
        torch.bmm(
            basis,
            (alpha_carbon[source] - alpha_carbon[target]).unsqueeze(-1),
        ).squeeze(-1),
        torch.bmm(
            basis, frames["normal"][source].unsqueeze(-1)
        ).squeeze(-1),
        torch.bmm(
            basis, frames["u_axis"][source].unsqueeze(-1)
        ).squeeze(-1),
        torch.bmm(
            basis, frames["v_axis"][source].unsqueeze(-1)
        ).squeeze(-1),
    ]
    return torch.cat(values, dim=-1)


def compute_neighbor_direction_features(
    coordinates,
    edge_index,
    edge_distances,
    *,
    scales=(1.0, 2.0, 5.0, 10.0, 30.0),
):
    """Summarize neighbor directions at several distance scales."""

    import torch

    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    edge_index = torch.as_tensor(
        edge_index, dtype=torch.long, device=coordinates.device
    )
    edge_distances = torch.as_tensor(
        edge_distances, dtype=coordinates.dtype, device=coordinates.device
    )
    features = coordinates.new_zeros(coordinates.shape[0], len(scales))
    if edge_index.shape[1] == 0:
        return features

    source, target = edge_index
    for residue in range(coordinates.shape[0]):
        selected = source == residue
        if not selected.any():
            continue
        distances = edge_distances[selected]
        differences = coordinates[source[selected]] - coordinates[target[selected]]
        for scale_index, scale in enumerate(scales):
            weights = torch.softmax(-distances.square() / float(scale), dim=0)
            mean = (weights[:, None] * differences).sum(0)
            denominator = (weights * distances).sum().clamp_min(1e-8)
            features[residue, scale_index] = mean.norm() / denominator
    return features


def place_virtual_cb(backbone):
    """Place a virtual C-beta from N, CA, and C coordinates."""

    import torch
    from torch.nn import functional as functional

    backbone = _validate_backbone(backbone)
    carbon, nitrogen, alpha_carbon = (
        backbone[:, 2],
        backbone[:, 0],
        backbone[:, 1],
    )
    bc = functional.normalize(nitrogen - alpha_carbon, dim=-1)
    normal = functional.normalize(
        torch.cross(nitrogen - carbon, bc, dim=-1), dim=-1
    )
    middle = torch.cross(normal, bc, dim=-1)
    length, planar, dihedral = 1.522, 1.927, -2.143
    return alpha_carbon + (
        bc * (length * math.cos(planar))
        + middle * (length * math.sin(planar) * math.cos(dihedral))
        + normal * (-length * math.sin(planar) * math.sin(dihedral))
    )


def _validate_backbone(backbone):
    import torch

    backbone = torch.as_tensor(backbone, dtype=torch.float32)
    if (
        backbone.ndim != 3
        or backbone.shape[1] < 3
        or backbone.shape[-1] != 3
    ):
        raise ValueError(
            "Backbone coordinates must have shape [residues, atoms>=3, 3], "
            f"got {tuple(backbone.shape)}."
        )
    return backbone


__all__ = [
    "compute_backbone_angle_features",
    "compute_backbone_frames",
    "compute_dihedral",
    "compute_edge_orientations",
    "compute_neighbor_direction_features",
    "place_virtual_cb",
]
