from __future__ import annotations

import math

import torch
from torch import nn

from .flow import SinusoidalTimeEmbedding


class WallGraphFlow(nn.Module):
    """Small outline-conditioned model for wall junction flow and dense edge logits."""

    def __init__(
        self,
        max_junctions: int,
        num_room_types: int = 10,
        boundary_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_junctions = int(max_junctions)
        self.num_room_types = int(num_room_types)
        self.d_model = int(d_model)
        self.boundary_proj = nn.Sequential(nn.Linear(boundary_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.boundary_pos_proj = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.boundary_encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.global_proj = nn.Linear(d_model, d_model)
        self.junction_proj = nn.Linear(2, d_model)
        self.query_embed = nn.Parameter(torch.zeros(1, max_junctions, d_model))
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.flow_head = nn.Linear(d_model, 2)
        self.junction_presence_head = nn.Linear(d_model, 1)
        self.edge_left = nn.Linear(d_model, d_model)
        self.edge_right = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.query_embed, std=0.02)

    def encode_boundary(self, boundary_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.linspace(0.0, 1.0, boundary_points.shape[1], device=boundary_points.device)
        pos = torch.stack([torch.sin(2.0 * math.pi * positions), torch.cos(2.0 * math.pi * positions)], dim=-1)
        hidden = self.boundary_proj(boundary_points) + self.boundary_pos_proj(pos)[None, :, :]
        tokens = self.boundary_encoder(hidden)
        return tokens, self.global_proj(tokens.mean(dim=1))

    def forward(
        self,
        noisy_junction_xy: torch.Tensor,
        t: torch.Tensor,
        boundary_points: torch.Tensor,
        junction_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        boundary_tokens, outline = self.encode_boundary(boundary_points)
        hidden = self.query_embed[:, : noisy_junction_xy.shape[1], :] + self.junction_proj(noisy_junction_xy)
        hidden = hidden + (outline + self.time_embed(t))[:, None, :]
        padding_mask = ~junction_mask.bool() if junction_mask is not None else None
        hidden = self.decoder(hidden, boundary_tokens, tgt_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)
        left = self.edge_left(hidden)
        right = self.edge_right(hidden)
        edge_logits = torch.einsum("bid,bjd->bij", left, right) / math.sqrt(float(self.d_model)) + self.edge_bias
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        diagonal = torch.eye(edge_logits.shape[1], dtype=torch.bool, device=edge_logits.device)[None, :, :]
        edge_logits = edge_logits.masked_fill(diagonal, -30.0)
        if junction_mask is not None:
            valid = junction_mask.bool()
            pair_mask = valid[:, :, None] & valid[:, None, :]
            edge_logits = edge_logits.masked_fill(~pair_mask, -30.0)
        return {
            "flow": self.flow_head(hidden),
            "junction_presence_logits": self.junction_presence_head(hidden).squeeze(-1),
            "edge_logits": edge_logits,
            "outline_embedding": outline,
            "boundary_tokens": boundary_tokens,
        }
