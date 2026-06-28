import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
    Integrates Corner Augmentation (AU) by sampling points along wall segments.
    """
    def __init__(self, d_model, num_room_types=10, max_corners_per_room=512, max_rooms=512):
        super().__init__()
        self.coord_proj = nn.Linear(18, d_model)  # Projected from augmented coordinate of size 18 (2 + 8 * 2)
        self.room_type_emb = nn.Embedding(num_room_types, d_model)
        self.corner_idx_emb = nn.Embedding(max_corners_per_room, d_model)
        self.room_idx_emb = nn.Embedding(max_rooms, d_model)
        
    def forward(self, x_t, entity_type, entity_idx, corner_idx, time_emb):
        # x_t: (B, N, 2)
        # entity_type: (B, N) -> room category (living room, bedroom, corridor, etc.)
        # entity_idx: (B, N) -> instance ID (separates room instances)
        # corner_idx: (B, N) -> index of the corner within its polygon
        # time_emb: (B, d_model)
        B, N, _ = x_t.shape
        
        # 1. Corner Augmentation (AU): sample 8 points along the segment connecting x_t to the next corner
        same_room = (entity_idx.unsqueeze(2) == entity_idx.unsqueeze(1))  # (B, N, N)
        is_next = (corner_idx.unsqueeze(2) == (corner_idx.unsqueeze(1) + 1))  # (B, N, N)
        is_wrap = (corner_idx.unsqueeze(2) == 0)  # (B, N, N)
        
        match_next = same_room & is_next
        match_wrap = same_room & is_wrap
        
        has_next = match_next.any(dim=-1, keepdim=True)  # (B, N, 1)
        transition = torch.where(has_next, match_next.float(), match_wrap.float())
        
        x_next = torch.matmul(transition, x_t)  # (B, N, 2)
        
        # Sample 8 points along the segment connecting x_t and x_next
        lambdas = torch.linspace(0.0, 1.0, 8, device=x_t.device).view(1, 1, 8, 1)
        sampled_pts = (1.0 - lambdas) * x_t.unsqueeze(2) + lambdas * x_next.unsqueeze(2)  # (B, N, 8, 2)
        sampled_pts_flat = sampled_pts.view(B, N, 16)
        
        # Concatenate original coordinates and segment points
        x_augmented = torch.cat([x_t, sampled_pts_flat], dim=-1)  # (B, N, 18)
        
        # 2. Embed indices
        entity_type = torch.clamp(entity_type, 0, self.room_type_emb.num_embeddings - 1)
        entity_idx = torch.clamp(entity_idx, 0, self.room_idx_emb.num_embeddings - 1)
        corner_idx = torch.clamp(corner_idx, 0, self.corner_idx_emb.num_embeddings - 1)
        
        emb = self.coord_proj(x_augmented)
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
        
    def forward(self, x, outline_emb, csa_mask=None, gsa_mask=None, outline_mask=None):
        # 1. Component-wise Self-Attention (CSA): limit interactions to corners of the same room
        csa_out = self.csa_layer(self.csa_norm(x), mask=csa_mask)
        x = x + self.csa_dropout(csa_out)
        
        # 2. Global Self-Attention (GSA): global interaction between rooms
        gsa_out = self.gsa_layer(self.gsa_norm(x), mask=gsa_mask)
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
    outer apartment outlines instead of relational graphs, without discrete denoising.
    """
    def __init__(self, d_model=256, num_heads=8, d_ff=512, num_layers=6, dropout=0.1,
                 num_room_types=10, max_corners_per_room=512, max_rooms=512, max_outline_len=128):
        super().__init__()
        self.time_embed = TimestepEmbedding(d_model)
        self.corner_embed = CornerEmbedding(d_model, num_room_types, max_corners_per_room, max_rooms)
        self.outline_encoder = OutlineEncoder(d_model, max_outline_len)
        
        self.blocks = nn.ModuleList([
            HouseDiffusionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2)       # Predicts 2D noise / coordinates
        
    def forward(self, x_t, timesteps, entity_type, entity_idx, corner_idx, outline, outline_mask=None, corner_mask=None):
        """
        Args:
            x_t: (B, N, 2) Noisy room/door corner coordinates
            timesteps: (B,) Diffusion time steps
            entity_type: (B, N) Room/door category indices
            entity_idx: (B, N) Unique entity instance IDs
            corner_idx: (B, N) Indices of corners within room polygons
            outline: (B, M, 2) outer outline coordinates
            outline_mask: (B, M) Binary mask representing valid outline vertices (1 for active, 0 for padded)
            corner_mask: (B, N) Binary mask representing valid room corner tokens (1 for active, 0 for padded)
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
        
        # 5. Build GSA and token masking
        gsa_mask = None
        if corner_mask is not None:
            # Build valid pair mask of shape (B, N, N)
            valid_mask = (corner_mask.unsqueeze(2) * corner_mask.unsqueeze(1)).float()
            csa_mask = csa_mask * valid_mask
            gsa_mask = valid_mask
            
        # 6. Build cross-attention boundary mask
        cross_attn_mask = None
        if outline_mask is not None:
            # Shape: (B, N, M)
            cross_attn_mask = outline_mask.unsqueeze(1).repeat(1, N, 1)
            
        # 7. Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, outline_emb, csa_mask=csa_mask, gsa_mask=gsa_mask, outline_mask=cross_attn_mask)
            
        # 8. Project features back to output spaces
        x = self.out_norm(x)
        out_continuous = self.out_proj(x)       # (B, N, 2)
        
        return out_continuous


# =====================================================================
# 4. Room Condition Sampler
# =====================================================================
class RoomConditionSampler:
    """
    Constructs the input conditions probabilistically based on histograms
    derived from the training dataset (e.g., Modified Swiss Dwellings).
    """
    def __init__(self, max_total_corners=128):
        self.max_total_corners = max_total_corners
        
        # Empirical room counts per plan (mostly between 3 and 8 rooms)
        self.room_counts = [3, 4, 5, 6, 7, 8]
        self.room_count_weights = [0.1, 0.25, 0.3, 0.2, 0.1, 0.05]
        
        # All room types available in dataset (mapped to correct indices)
        # 0: Bedroom, 1: Livingroom, 2: Kitchen, 3: Dining, 4: Corridor,
        # 5: Stairs, 6: Storeroom, 7: Bathroom, 8: Balcony, 9: Structure
        self.room_types = list(range(10))
        self.room_type_weights = [0.25, 0.15, 0.15, 0.05, 0.15, 0.02, 0.03, 0.15, 0.08, 0.02]
        
        # Corner histograms per room type: {type: ([possible N corners], [probabilities])}
        self.corner_histograms = {
            0: ([4, 6], [0.95, 0.05]),         # Bedroom (usually rect)
            1: ([4, 6, 8], [0.70, 0.25, 0.05]), # Livingroom
            2: ([4, 6], [0.85, 0.15]),         # Kitchen
            3: ([4, 6], [0.90, 0.10]),         # Dining
            4: ([4, 6, 8, 10], [0.4, 0.4, 0.15, 0.05]), # Corridor (more complex shapes)
            5: ([4], [1.0]),                   # Stairs
            6: ([4], [1.0]),                   # Storeroom
            7: ([4], [1.0]),                   # Bathroom (usually rect)
            8: ([4, 6], [0.90, 0.10]),         # Balcony
            9: ([4], [1.0])                    # Structure
        }
        
    def sample(self, batch_size, device):
        batch_entity_type, batch_entity_idx = [], []
        batch_corner_idx, batch_corner_mask = [], []
        
        for b in range(batch_size):
            # 1. Sample total number of rooms
            num_rooms = random.choices(self.room_counts, weights=self.room_count_weights, k=1)[0]
            
            b_e_type, b_e_idx, b_c_idx = [], [], []
            
            # Ensure at least one Livingroom (1), Kitchen (2), and Bathroom (7). Randomize the rest.
            types_to_generate = [1, 2, 7]
            if num_rooms > 3:
                types_to_generate += random.choices(self.room_types, weights=self.room_type_weights, k=num_rooms-3)
            else:
                types_to_generate = types_to_generate[:num_rooms]
                
            for room_id, r_type in enumerate(types_to_generate):
                # 2. Sample number of corners for this specific room type
                c_opts, c_weights = self.corner_histograms[r_type]
                num_corners = random.choices(c_opts, weights=c_weights, k=1)[0]
                
                # 3. Populate sequences
                b_e_type.extend([r_type] * num_corners)
                b_e_idx.extend([room_id] * num_corners)
                b_c_idx.extend(list(range(num_corners)))
                
            # 4. Pad sequences to maximum length for batching
            actual_len = len(b_e_type)
            pad_len = max(0, self.max_total_corners - actual_len)
            
            # Mask is 1 for real corners, 0 for padded ones
            b_mask = [1] * actual_len + [0] * pad_len
            b_e_type = b_e_type + [0] * pad_len
            b_e_idx = b_e_idx + [0] * pad_len
            b_c_idx = b_c_idx + [0] * pad_len
            
            batch_entity_type.append(b_e_type[:self.max_total_corners])
            batch_entity_idx.append(b_e_idx[:self.max_total_corners])
            batch_corner_idx.append(b_c_idx[:self.max_total_corners])
            batch_corner_mask.append(b_mask[:self.max_total_corners])
            
        return {
            "entity_type": torch.tensor(batch_entity_type, dtype=torch.long, device=device),
            "entity_idx": torch.tensor(batch_entity_idx, dtype=torch.long, device=device),
            "corner_idx": torch.tensor(batch_corner_idx, dtype=torch.long, device=device),
            "corner_mask": torch.tensor(batch_corner_mask, dtype=torch.float32, device=device)
        }


# =====================================================================
# 5. Gaussian Diffusion Helper
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
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
    def p_losses(self, model, x_0, t, cond, noise=None):
        """
        Compute standard continuous MSE loss (noise prediction).
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        x_t = self.q_sample(x_0, t, noise)
        
        predicted_noise = model(
            x_t=x_t,
            timesteps=t,
            entity_type=cond["entity_type"],
            entity_idx=cond["entity_idx"],
            corner_idx=cond["corner_idx"],
            outline=cond["outline"],
            outline_mask=cond.get("outline_mask", None),
            corner_mask=cond.get("corner_mask", None)
        )
        
        # Continuous loss: mask out padded tokens
        loss_continuous = F.mse_loss(predicted_noise, noise, reduction='none')
        if cond.get("corner_mask", None) is not None:
            mask = cond["corner_mask"].unsqueeze(-1)
            loss_continuous = (loss_continuous * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss_continuous = loss_continuous.mean()
            
        return loss_continuous
        
    @torch.no_grad()
    def p_sample(self, model, x, t, cond):
        """
        Perform a single reverse step to denoise x_t into x_{t-1}.
        """
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        
        predicted_noise = model(
            x_t=x,
            timesteps=t,
            entity_type=cond["entity_type"],
            entity_idx=cond["entity_idx"],
            corner_idx=cond["corner_idx"],
            outline=cond["outline"],
            outline_mask=cond.get("outline_mask", None),
            corner_mask=cond.get("corner_mask", None)
        )
        
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
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
        """
        B = shape[0]
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.steps)):
            t = torch.full((B,), i, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, t, cond)
            
        return x


@torch.no_grad()
def generate_floorplan(model, diffusion, condition_sampler, outline, outline_mask=None, max_corners=128):
    """
    Generates a full vector floorplan conditioned only on the apartment outline.
    outline shape: (B, M, 2)
    """
    model.eval()
    B = outline.shape[0]
    device = outline.device
    
    # 1. Probabilistically sample internal room topology from dataset histograms
    cond = condition_sampler.sample(batch_size=B, device=device)
    cond["outline"] = outline
    cond["outline_mask"] = outline_mask
    
    # 2. Define the shape of the noise tensor to start diffusing
    shape = (B, max_corners, 2)
    
    # 3. Run the standard continuous reverse diffusion loop
    generated_coords = diffusion.p_sample_loop(
        model=model, 
        shape=shape, 
        cond=cond
    )
    
    # 4. Filter out padded corners using the generated mask
    final_layouts = []
    for b in range(B):
        mask = cond["corner_mask"][b].bool()
        coords = generated_coords[b][mask]
        types = cond["entity_type"][b][mask]
        room_ids = cond["entity_idx"][b][mask]
        
        final_layouts.append({
            "coordinates": coords, # Final 2D vector corners
            "room_types": types,   
            "room_ids": room_ids   # Group coordinates by room_ids to form polygons
        })
        
    return final_layouts
