from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    """Small PointNet-style encoder for sampled outline boundary points."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points with shape ``[batch, points, 2]``."""

        if points.ndim != 3:
            raise ValueError(f"Expected [batch, points, dims], got shape {tuple(points.shape)}.")
        features = self.point_mlp(points)
        pooled = features.max(dim=1).values
        return self.output_norm(pooled)

