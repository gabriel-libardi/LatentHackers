from __future__ import annotations

import math

import torch
from torch import nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ConditionalRoomFlow(nn.Module):
    """Boundary-token conditioned Transformer decoder over room geometry slots."""

    def __init__(
        self,
        token_dim: int | None = None,
        geometry_dim: int | None = None,
        max_rooms: int = 400,
        num_room_types: int = 10,
        boundary_dim: int = 2,
        point_hidden_dim: int = 128,
        cond_dim: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        encoder_layers: int | None = None,
        decoder_layers: int | None = None,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if geometry_dim is None:
            geometry_dim = token_dim if token_dim is not None else 32
        self.geometry_dim = int(geometry_dim)
        self.token_dim = self.geometry_dim
        self.max_rooms = max_rooms
        self.num_room_types = num_room_types
        self.d_model = d_model
        encoder_layers = num_layers if encoder_layers is None else encoder_layers
        decoder_layers = num_layers if decoder_layers is None else decoder_layers

        self.boundary_proj = nn.Sequential(
            nn.Linear(boundary_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.boundary_pos_proj = nn.Linear(2, d_model)
        boundary_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.boundary_encoder = nn.TransformerEncoder(boundary_layer, num_layers=encoder_layers)
        self.token_proj = nn.Linear(self.geometry_dim, d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.global_proj = nn.Linear(d_model, d_model)
        self.query_embed = nn.Parameter(torch.zeros(1, max_rooms, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.flow_head = nn.Linear(d_model, self.geometry_dim)
        self.presence_head = nn.Linear(d_model, 1)
        self.type_head = nn.Linear(d_model, num_room_types + 1)
        self.area_head = nn.Linear(d_model, 1)
        self.count_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max_rooms + 1),
        )
        nn.init.normal_(self.query_embed, std=0.02)

    def encode_boundary(self, boundary_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if boundary_points.ndim != 3:
            raise ValueError(f"Expected boundary points [B, P, 2], got {tuple(boundary_points.shape)}.")
        positions = torch.linspace(0.0, 1.0, boundary_points.shape[1], device=boundary_points.device)
        pos = torch.stack([torch.sin(2.0 * math.pi * positions), torch.cos(2.0 * math.pi * positions)], dim=-1)
        hidden = self.boundary_proj(boundary_points) + self.boundary_pos_proj(pos)[None, :, :]
        tokens = self.boundary_encoder(hidden)
        pooled = tokens.mean(dim=1)
        return tokens, self.global_proj(pooled)

    def forward(
        self,
        noisy_geometry: torch.Tensor,
        t: torch.Tensor,
        boundary_points: torch.Tensor,
        room_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if noisy_geometry.ndim != 3:
            raise ValueError(f"Expected geometry tensor [B, R, D], got {tuple(noisy_geometry.shape)}.")
        if noisy_geometry.shape[1] > self.max_rooms:
            raise ValueError(f"Got {noisy_geometry.shape[1]} rooms, model max is {self.max_rooms}.")

        boundary_tokens, outline = self.encode_boundary(boundary_points)
        time = self.time_embed(t)
        cond = outline + time
        hidden = self.query_embed[:, : noisy_geometry.shape[1], :] + self.token_proj(noisy_geometry)
        hidden = hidden + cond[:, None, :]
        tgt_padding_mask = None
        if room_mask is not None:
            tgt_padding_mask = ~room_mask.bool()
        hidden = self.decoder(hidden, boundary_tokens, tgt_key_padding_mask=tgt_padding_mask)
        hidden = self.norm(hidden)
        return {
            "flow": self.flow_head(hidden),
            "presence_logits": self.presence_head(hidden).squeeze(-1),
            "type_logits": self.type_head(hidden),
            "area_pred": self.area_head(hidden).squeeze(-1),
            "count_logits": self.count_head(outline),
            "outline_embedding": outline,
            "boundary_tokens": boundary_tokens,
        }

    def config(self) -> dict[str, int | float]:
        return {
            "geometry_dim": self.geometry_dim,
            "max_rooms": self.max_rooms,
            "num_room_types": self.num_room_types,
            "d_model": self.d_model,
        }
