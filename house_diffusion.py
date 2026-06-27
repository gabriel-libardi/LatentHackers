import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Attention Mechanisms
# =====================================================================

class MaskedSelfAttention(nn.Module):
    """
    Multi-head self-attention supporting custom masks (used for CSA).
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        # x: (B, N, D)
        # mask: (B, N, N) containing 1s for attended elements and 0s for masked elements
        B, N, D = x.shape
        
        # Project and split into heads
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d_h)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d_h)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d_h)
        
        # Calculate scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)
        
        if mask is not None:
            # Expand mask from (B, N, N) to (B, 1, N, N)
            expanded_mask = mask.unsqueeze(1)
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        
        if mask is not None:
            # Replace NaNs that might occur if a row is entirely masked out
            attn = torch.nan_to_num(attn, nan=0.0)
            
        out = torch.matmul(attn, v)  # (B, H, N, d_h)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for conditioning on the outer apartment outline.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, memory, memory_mask=None):
        # x: Query sequence (room corners), shape (B, N, D)
        # memory: Key/Value sequence (outline points), shape (B, M, D)
        # memory_mask: Mask for memory, shape (B, N, M)
        B, N, D = x.shape
        _, M, _ = memory.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d_h)
        k = self.k_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d_h)
        v = self.v_proj(memory).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d_h)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, M)
        
        if memory_mask is not None:
            expanded_mask = memory_mask.unsqueeze(1)  # (B, 1, N, M)
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        
        if memory_mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
            
        out = torch.matmul(attn, v)  # (B, H, N, d_h)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# =====================================================================
# 2. Embedding Layers
# =====================================================================

class TimestepEmbedding(nn.Module):
    """
    Embeds diffusion timesteps into continuous vectors using a sinusoidal schedule and MLP.
    """
    def __init__(self, d_model):
        super().__init__()
        self.time_emb_dim = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, timesteps):
        # timesteps: (B,)
        half_dim = self.time_emb_dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.time_emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class CornerEmbedding(nn.Module):
    """
    Combines the spatial coordinates of room corners with entity category,
    corner indices, entity instance IDs, and timestep embeddings.
    """
    def __init__(self, d_model, num_room_types=32, max_corners_per_room=16, max_rooms=64):
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.room_type_emb = nn.Embedding(num_room_types, d_model)
        self.corner_idx_emb = nn.Embedding(max_corners_per_room, d_model)
        self.room_idx_emb = nn.Embedding(max_rooms, d_model)
        
    def forward(self, x_t, entity_type, entity_idx, corner_idx, time_emb):
        # x_t: (B, N, 2)
        # entity_type: (B, N) -> room category (living room, bedroom, corridor, etc.)
        # entity_idx: (B, N) -> instance ID (separates room instances)
        # corner_idx: (B, N) -> index of the corner within its polygon
        # time_emb: (B, d_model)
        
        emb = self.coord_proj(x_t)
        emb = emb + self.room_type_emb(entity_type)
        emb = emb + self.corner_idx_emb(corner_idx)
        emb = emb + self.room_idx_emb(entity_idx)
        emb = emb + time_emb.unsqueeze(1)  # Broadcast time embedding to all tokens
        
        return emb


class OutlineEncoder(nn.Module):
    """
    Encodes the boundary coordinates of the outer apartment shell.
    """
    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.coord_proj = nn.Linear(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, outline):
        # outline: (B, M, 2)
        B, M, _ = outline.shape
        coord_emb = self.coord_proj(outline)
        
        # Sequential position embeddings for the outline vertices
        pos = torch.arange(M, device=outline.device).unsqueeze(0).repeat(B, 1)
        pos_emb = self.pos_emb(pos)
        
        return self.mlp(coord_emb + pos_emb)


# =====================================================================
# 3. Model Architecture
# =====================================================================

class HouseDiffusionBlock(nn.Module):
    """
    A single block of the HouseDiffusion Transformer.
    Contains CSA, GSA, Outline Cross-Attention, and a Feed-Forward Network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.csa_layer = MaskedSelfAttention(d_model, num_heads)
        self.gsa_layer = MaskedSelfAttention(d_model, num_heads)
        self.cross_attn = CrossAttention(d_model, num_heads)
        
        self.csa_norm = nn.LayerNorm(d_model)
        self.gsa_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.csa_dropout = nn.Dropout(dropout)
        self.gsa_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
        
    def forward(self, x, outline_emb, csa_mask=None, outline_mask=None):
        # 1. Component-wise Self-Attention (CSA): limit interactions to corners of the same room
        csa_out = self.csa_layer(self.csa_norm(x), mask=csa_mask)
        x = x + self.csa_dropout(csa_out)
        
        # 2. Global Self-Attention (GSA): global interaction between rooms
        gsa_out = self.gsa_layer(self.gsa_norm(x), mask=None)
        x = x + self.gsa_dropout(gsa_out)
        
        # 3. Outline Cross-Attention: align room geometries with the apartment boundaries
        cross_out = self.cross_attn(self.cross_norm(x), outline_emb, memory_mask=outline_mask)
        x = x + self.cross_dropout(cross_out)
        
        # 4. Feed-Forward Network
        ffn_out = self.mlp(self.ffn_norm(x))
        x = x + ffn_out
        
        return x


class HouseDiffusionModel(nn.Module):
    """
    Reimplemented HouseDiffusion Transformer architecture conditioned on
    outer apartment outlines instead of relational graphs.
    """
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, num_layers=6, dropout=0.1,
                 num_room_types=32, max_corners_per_room=16, max_rooms=64, max_outline_len=128):
        super().__init__()
        self.time_embed = TimestepEmbedding(d_model)
        self.corner_embed = CornerEmbedding(d_model, num_room_types, max_corners_per_room, max_rooms)
        self.outline_encoder = OutlineEncoder(d_model, max_outline_len)
        
        self.blocks = nn.ModuleList([
            HouseDiffusionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2)  # Outputs 2D noise / coordinates
        
    def forward(self, x_t, timesteps, entity_type, entity_idx, corner_idx, outline, outline_mask=None):
        """
        Args:
            x_t: (B, N, 2) Noisy room/door corner coordinates
            timesteps: (B,) Diffusion time steps
            entity_type: (B, N) Room/door category indices
            entity_idx: (B, N) Unique entity instance IDs
            corner_idx: (B, N) Indices of corners within room polygons
            outline: (B, M, 2) outer outline coordinates
            outline_mask: (B, M) Binary mask representing valid outline vertices (1 for active, 0 for padded)
        """
        B, N, _ = x_t.shape
        
        # 1. Embed time steps
        time_emb = self.time_embed(timesteps)  # (B, d_model)
        
        # 2. Embed room/door corners
        x = self.corner_embed(x_t, entity_type, entity_idx, corner_idx, time_emb)  # (B, N, d_model)
        
        # 3. Encode outer apartment outline
        outline_emb = self.outline_encoder(outline)  # (B, M, d_model)
        
        # 4. Build Component-wise Self-Attention (CSA) Mask dynamically
        # csa_mask[b, i, j] = 1 if entity_idx[b, i] == entity_idx[b, j] else 0
        csa_mask = (entity_idx.unsqueeze(2) == entity_idx.unsqueeze(1)).float()  # (B, N, N)
        
        # 5. Build cross-attention boundary mask
        cross_attn_mask = None
        if outline_mask is not None:
            # Shape: (B, N, M)
            cross_attn_mask = outline_mask.unsqueeze(1).repeat(1, N, 1)
            
        # 6. Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, outline_emb, csa_mask=csa_mask, outline_mask=cross_attn_mask)
            
        # 7. Project features back to 2D space
        x = self.out_norm(x)
        out = self.out_proj(x)  # (B, N, 2)
        
        return out


# =====================================================================
# 4. Gaussian Diffusion Helper
# =====================================================================
class GaussianDiffusion:
    """
    DDPM pipeline tailored for vector floorplan denoising.
    """
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.steps = steps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: Add Gaussian noise to room corners.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Extract variables at t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
    def p_losses(self, model, x_0, t, cond, noise=None):
        """
        Compute mean squared error loss between model predictions and target noise.
        cond is a dictionary containing:
        - entity_type
        - entity_idx
        - corner_idx
        - outline
        - outline_mask
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Forward pass: create noisy sample
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict the noise using the outline-conditioned model
        predicted_noise = model(
            x_t=x_t,
            timesteps=t,
            entity_type=cond["entity_type"],
            entity_idx=cond["entity_idx"],
            corner_idx=cond["corner_idx"],
            outline=cond["outline"],
            outline_mask=cond.get("outline_mask", None)
        )
        
        return F.mse_loss(predicted_noise, noise)
        
    @torch.no_grad()
    def p_sample(self, model, x, t, cond):
        """
        Perform a single reverse step to denoise x_t into x_{t-1}.
        """
        # Obtain alphas and betas for current timestep
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        
        # Predict noise
        predicted_noise = model(
            x_t=x,
            timesteps=t,
            entity_type=cond["entity_type"],
            entity_idx=cond["entity_idx"],
            corner_idx=cond["corner_idx"],
            outline=cond["outline"],
            outline_mask=cond.get("outline_mask", None)
        )
        
        # Compute mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)
        
        
        if t[0] == 0:
            return mean
        else:
            variance = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise
            
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond):
        """
        Complete reverse process loop to generate vector floorplan corners from pure noise.
        shape: (B, N, 2)
        """
        B = shape[0]
        # Start from isotropic Gaussian noise
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.steps)):
            t = torch.full((B,), i, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, t, cond)
            
        return x
