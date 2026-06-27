import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        if mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x, memory, memory_mask=None):
        B, N, D = x.shape
        M = memory.shape[1]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if memory_mask is not None:
            scores = scores.masked_fill(memory_mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        if memory_mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dim = d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model))
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -(math.log(10000.0) / (half - 1)))
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)

class CornerEmbedding(nn.Module):
    def __init__(self, d_model, num_room_types=32, max_corners_per_room=512, max_rooms=512):
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.room_type_emb = nn.Embedding(num_room_types, d_model)
        self.corner_idx_emb = nn.Embedding(max_corners_per_room, d_model)
        self.room_idx_emb = nn.Embedding(max_rooms, d_model)
    def forward(self, x_t, entity_type, entity_idx, corner_idx, time_emb):
        entity_type = torch.clamp(entity_type, 0, self.room_type_emb.num_embeddings - 1)
        entity_idx = torch.clamp(entity_idx, 0, self.room_idx_emb.num_embeddings - 1)
        corner_idx = torch.clamp(corner_idx, 0, self.corner_idx_emb.num_embeddings - 1)
        emb = self.coord_proj(x_t)
        emb = emb + self.room_type_emb(entity_type)
        emb = emb + self.corner_idx_emb(corner_idx)
        emb = emb + self.room_idx_emb(entity_idx)
        emb = emb + time_emb.unsqueeze(1)
        return emb

class OutlineEncoder(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))
    def forward(self, outline):
        B, M, _ = outline.shape
        coord_emb = self.coord_proj(outline)
        pos = torch.arange(M, device=outline.device).unsqueeze(0).repeat(B, 1)
        pos = torch.clamp(pos, 0, self.pos_emb.num_embeddings - 1)
        pos_emb = self.pos_emb(pos)
        return self.mlp(coord_emb + pos_emb)

class FlowBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.csa = MaskedSelfAttention(d_model, num_heads)
        self.gsa = MaskedSelfAttention(d_model, num_heads)
        self.cross = CrossAttention(d_model, num_heads)
        self.csa_norm = nn.LayerNorm(d_model)
        self.gsa_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.csa_dropout = nn.Dropout(dropout)
        self.gsa_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
    def forward(self, x, outline_emb, csa_mask=None, cross_mask=None):
        x = x + self.csa_dropout(self.csa(self.csa_norm(x), mask=csa_mask))
        x = x + self.gsa_dropout(self.gsa(self.gsa_norm(x), mask=None))
        x = x + self.cross_dropout(self.cross(self.cross_norm(x), outline_emb, memory_mask=cross_mask))
        x = x + self.mlp(self.ffn_norm(x))
        return x

class HouseFlowModel(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, num_layers=6, dropout=0.1, num_room_types=32, max_corners_per_room=512, max_rooms=512, max_outline_len=128, time_scale=1000.0):
        super().__init__()
        self.time_scale = time_scale
        self.time_embed = TimeEmbedding(d_model)
        self.corner_embed = CornerEmbedding(d_model, num_room_types, max_corners_per_room, max_rooms)
        self.outline_encoder = OutlineEncoder(d_model, max_outline_len)
        self.blocks = nn.ModuleList([FlowBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2)
    def forward(self, x_t, timesteps, entity_type, entity_idx, corner_idx, outline, outline_mask=None):
        B, N, _ = x_t.shape
        time_emb = self.time_embed(timesteps * self.time_scale)
        x = self.corner_embed(x_t, entity_type, entity_idx, corner_idx, time_emb)
        outline_emb = self.outline_encoder(outline)
        csa_mask = (entity_idx.unsqueeze(2) == entity_idx.unsqueeze(1)).float()
        cross_mask = None
        if outline_mask is not None:
            cross_mask = outline_mask.unsqueeze(1).repeat(1, N, 1)
        for block in self.blocks:
            x = block(x, outline_emb, csa_mask=csa_mask, cross_mask=cross_mask)
        return self.out_proj(self.out_norm(x))

class FlowMatching:
    def __init__(self, steps=100, device="cpu"):
        self.steps = steps
        self.device = device
    def add_noise(self, data, noise, t):
        tt = t.view(-1, 1, 1)
        return (1.0 - tt) * noise + tt * data
    def velocity(self, data, noise):
        return data - noise
    @torch.no_grad()
    def sample_loop(self, model, shape, cond):
        x = torch.randn(shape, device=self.device)
        n = self.steps
        for i in range(n):
            t = torch.full((shape[0],), i / n, dtype=torch.float32, device=self.device)
            v = model(x_t=x, timesteps=t, entity_type=cond["entity_type"], entity_idx=cond["entity_idx"], corner_idx=cond["corner_idx"], outline=cond["outline"], outline_mask=cond.get("outline_mask", None))
            x = x + v / n
        return x
