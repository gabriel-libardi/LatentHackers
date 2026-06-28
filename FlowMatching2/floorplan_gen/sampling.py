from __future__ import annotations

import torch


@torch.no_grad()
def sample_room_geometry(
    model,
    boundary_points: torch.Tensor,
    max_rooms: int,
    geometry_dim: int | None = None,
    steps: int = 32,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Euler-integrate the learned flow from noise to room geometry."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    generator = None
    if device.type != "mps":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    torch.manual_seed(seed)
    if boundary_points.ndim == 2:
        boundary_points = boundary_points.unsqueeze(0)
    boundary_points = boundary_points.to(device=device, dtype=torch.float32)
    if geometry_dim is None:
        geometry_dim = int(getattr(model, "geometry_dim", getattr(model, "token_dim", 32)))
    x = torch.randn(
        boundary_points.shape[0],
        max_rooms,
        geometry_dim,
        device=device,
        generator=generator,
    )
    dt = 1.0 / float(steps)
    last_outputs: dict[str, torch.Tensor] = {}
    for index in range(steps):
        t = torch.full((boundary_points.shape[0],), index / float(steps), device=device)
        last_outputs = model(x, t, boundary_points)
        x = x + dt * last_outputs["flow"]
    t = torch.ones((boundary_points.shape[0],), device=device)
    last_outputs = model(x, t, boundary_points)
    return x, last_outputs


sample_room_tokens = sample_room_geometry


@torch.no_grad()
def sample_wall_graph(
    model,
    boundary_points: torch.Tensor,
    max_junctions: int,
    steps: int = 32,
    seed: int = 42,
    device: str | torch.device | None = None,
    junction_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    generator = None
    if device.type != "mps":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    torch.manual_seed(seed)
    if boundary_points.ndim == 2:
        boundary_points = boundary_points.unsqueeze(0)
    boundary_points = boundary_points.to(device=device, dtype=torch.float32)
    if junction_mask is not None and junction_mask.ndim == 1:
        junction_mask = junction_mask.unsqueeze(0)
    if junction_mask is not None:
        junction_mask = junction_mask.to(device=device, dtype=torch.bool)
    x = torch.randn(boundary_points.shape[0], max_junctions, 2, device=device, generator=generator)
    dt = 1.0 / float(steps)
    outputs: dict[str, torch.Tensor] = {}
    for index in range(steps):
        t = torch.full((boundary_points.shape[0],), index / float(steps), device=device)
        outputs = model(x, t, boundary_points, junction_mask=junction_mask)
        x = x + dt * outputs["flow"]
    t = torch.ones((boundary_points.shape[0],), device=device)
    outputs = model(x, t, boundary_points, junction_mask=junction_mask)
    return x, outputs
