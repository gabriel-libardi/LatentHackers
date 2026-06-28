from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_flow_batch(
    room_geometry: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = room_geometry.shape[0]
    t = torch.rand(batch, device=room_geometry.device, generator=generator)
    noise = torch.randn(room_geometry.shape, device=room_geometry.device, generator=generator)
    t_view = t.view(batch, *([1] * (room_geometry.ndim - 1)))
    noisy = (1.0 - t_view) * noise + t_view * room_geometry
    target_flow = room_geometry - noise
    return noisy, t, target_flow


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def _final_geometry_estimate(
    outputs: dict[str, torch.Tensor],
    noisy_geometry: torch.Tensor | None,
    t: torch.Tensor | None,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if noisy_geometry is None or t is None:
        return fallback
    view = t.view(t.shape[0], *([1] * (noisy_geometry.ndim - 1)))
    return noisy_geometry + (1.0 - view) * outputs["flow"]


def _as_vertices(geometry: torch.Tensor) -> torch.Tensor | None:
    if geometry.shape[-1] < 6 or geometry.shape[-1] % 2 != 0:
        return None
    return geometry.reshape(*geometry.shape[:-1], geometry.shape[-1] // 2, 2)


def _polygon_signed_area(vertices: torch.Tensor) -> torch.Tensor:
    x = vertices[..., 0]
    y = vertices[..., 1]
    return 0.5 * (x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1)).sum(dim=-1)


def _edge_lengths(vertices: torch.Tensor) -> torch.Tensor:
    return (vertices - torch.roll(vertices, shifts=-1, dims=-2)).norm(dim=-1)


def _polygon_regularizers(
    geometry: torch.Tensor,
    room_mask: torch.Tensor,
    min_edge: float,
    min_area: float,
    outside_margin: float,
) -> dict[str, torch.Tensor]:
    vertices = _as_vertices(geometry)
    zero = geometry.sum() * 0.0
    if vertices is None:
        return {"edge_loss": zero, "short_edge_loss": zero, "signed_area_loss": zero, "outside_loss": zero}

    lengths = _edge_lengths(vertices)
    mean_length = masked_mean(lengths, room_mask)
    edge_loss = masked_mean((lengths - mean_length.detach()).abs(), room_mask)
    short_edge_loss = masked_mean(F.relu(float(min_edge) - lengths).pow(2), room_mask)
    signed_area = _polygon_signed_area(vertices)
    signed_area_loss = masked_mean(F.relu(float(min_area) - signed_area.abs()).pow(2), room_mask)
    outside_loss = masked_mean(F.relu(vertices.abs() - float(outside_margin)).pow(2).sum(dim=-1), room_mask)
    return {
        "edge_loss": edge_loss,
        "short_edge_loss": short_edge_loss,
        "signed_area_loss": signed_area_loss,
        "outside_loss": outside_loss,
    }


def flow_matching_losses(
    outputs: dict[str, torch.Tensor],
    target_flow: torch.Tensor,
    room_geometry: torch.Tensor,
    room_mask: torch.Tensor,
    room_presence: torch.Tensor | None = None,
    room_type_ids: torch.Tensor | None = None,
    room_count: torch.Tensor | None = None,
    type_weights: torch.Tensor | None = None,
    noisy_geometry: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    velocity_loss: str = "mse",
    flow_weight: float = 1.0,
    presence_weight: float = 0.2,
    type_weight: float = 0.2,
    count_weight: float = 0.2,
    area_weight: float = 0.1,
    lambda_edge: float = 0.0,
    lambda_short_edge: float = 0.0,
    lambda_area: float = 0.0,
    lambda_outside: float = 0.0,
    min_edge_length: float = 0.02,
    min_signed_area: float = 1e-3,
    outside_margin: float = 1.1,
) -> dict[str, torch.Tensor]:
    """Compute masked FM loss plus separate presence/type/count/area losses."""

    room_mask = room_mask.bool()
    if velocity_loss == "mse":
        velocity_values = (outputs["flow"] - target_flow).pow(2)
    elif velocity_loss == "huber":
        velocity_values = F.smooth_l1_loss(outputs["flow"], target_flow, reduction="none")
    else:
        raise ValueError("velocity_loss must be 'mse' or 'huber'.")
    flow_loss = masked_mean(velocity_values, room_mask)

    if room_presence is None:
        room_presence = room_mask.to(outputs["presence_logits"].dtype)
    presence_targets = room_presence.to(outputs["presence_logits"].dtype).clamp(0.0, 1.0)
    presence_loss = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"],
        presence_targets,
    )

    if room_type_ids is None:
        room_type_ids = torch.zeros_like(room_mask, dtype=torch.long)
    type_targets = room_type_ids.long().clamp_min(0)
    if room_mask.any():
        weights = type_weights.to(outputs["type_logits"].device) if type_weights is not None else None
        type_loss = F.cross_entropy(outputs["type_logits"][room_mask], type_targets[room_mask], weight=weights)
    else:
        type_loss = outputs["type_logits"].sum() * 0.0

    if room_count is None:
        room_count = room_mask.sum(dim=1)
    if "count_logits" in outputs:
        count_loss = F.cross_entropy(
            outputs["count_logits"],
            room_count.long().clamp(0, outputs["count_logits"].shape[-1] - 1),
        )
    else:
        count_loss = flow_loss * 0.0

    if "area_pred" in outputs and room_geometry.shape[-1] >= 3:
        area_targets = room_geometry[..., 2].clamp_min(0.0)
        area_loss = masked_mean(F.smooth_l1_loss(outputs["area_pred"], area_targets, reduction="none"), room_mask)
    else:
        area_loss = flow_loss * 0.0

    pred_geometry = _final_geometry_estimate(outputs, noisy_geometry, t, room_geometry)
    regularizers = _polygon_regularizers(
        pred_geometry,
        room_mask,
        min_edge=min_edge_length,
        min_area=min_signed_area,
        outside_margin=outside_margin,
    )

    total = (
        flow_weight * flow_loss
        + presence_weight * presence_loss
        + type_weight * type_loss
        + count_weight * count_loss
        + area_weight * area_loss
        + lambda_edge * regularizers["edge_loss"]
        + lambda_short_edge * regularizers["short_edge_loss"]
        + lambda_area * regularizers["signed_area_loss"]
        + lambda_outside * regularizers["outside_loss"]
    )
    return {
        "loss": total,
        "flow_loss": flow_loss.detach(),
        "presence_loss": presence_loss.detach(),
        "type_loss": type_loss.detach(),
        "count_loss": count_loss.detach(),
        "area_loss": area_loss.detach(),
        "edge_loss": regularizers["edge_loss"].detach(),
        "short_edge_loss": regularizers["short_edge_loss"].detach(),
        "signed_area_loss": regularizers["signed_area_loss"].detach(),
        "outside_loss": regularizers["outside_loss"].detach(),
    }
