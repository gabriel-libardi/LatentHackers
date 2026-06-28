import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Helper functions for 8-bit discrete coordinate branch
def coord_to_binary(coords, num_bits=8):
    """
    Converts continuous 2D coordinates in [0, 1] to a flattened 8-bit binary representation of size 2 * num_bits.
    """
    coords_scaled = torch.clamp(coords * (2**num_bits - 1), 0, 2**num_bits - 1).round().long()
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, dtype=torch.long, device=coords.device)
    mask_shape = [1] * coords_scaled.dim() + [num_bits]
    binary = (coords_scaled.unsqueeze(-1) & mask.view(mask_shape)).gt(0).float()
    return binary.view(*coords.shape[:-1], 2 * num_bits)

def binary_to_coord(binary_logits, num_bits=8):
    """
    Reconstructs continuous 2D coordinates in [0, 1] from predicted 8-bit binary logits.
    """
    binary_shape = list(binary_logits.shape[:-1]) + [2, num_bits]
    logits_reshaped = binary_logits.view(binary_shape)
    probs = torch.sigmoid(logits_reshaped)
    mask = 2 ** torch.arange(num_bits - 1, -1, -1, dtype=torch.float32, device=binary_logits.device)
    mask_shape = [1] * (logits_reshaped.dim() - 1) + [num_bits]
    coords_scaled = (probs * mask.view(mask_shape)).sum(dim=-1)
    return coords_scaled / (2**num_bits - 1)

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
    def __init__(self, d_model, num_room_types=32, max_corners_per_room=512, max_rooms=512, is_discrete=False):
        super().__init__()
        self.is_discrete = is_discrete
        # 18 from AU (2 + 8 * 2)
        # If discrete, add 16 for the 8-bit binary representation of the 2D coordinates
        in_features = 34 if is_discrete else 18 
        self.coord_proj = nn.Linear(in_features, d_model)
        
        self.room_type_emb = nn.Embedding(num_room_types, d_model)
        self.corner_idx_emb = nn.Embedding(max_corners_per_room, d_model)
        self.room_idx_emb = nn.Embedding(max_rooms, d_model)
        
    def forward(self, x, entity_type, entity_idx, corner_idx, time_emb, binary_rep=None):
        B, N, _ = x.shape
        
        # 1. Corner Augmentation (AU): sample 8 points along the segment connecting x to the next corner
        same_room = (entity_idx.unsqueeze(2) == entity_idx.unsqueeze(1))
        is_next = (corner_idx.unsqueeze(2) == (corner_idx.unsqueeze(1) + 1))
        is_wrap = (corner_idx.unsqueeze(2) == 0)
        
        match_next = same_room & is_next
        match_wrap = same_room & is_wrap
        
        has_next = match_next.any(dim=-1, keepdim=True)
        transition = torch.where(has_next, match_next.float(), match_wrap.float())
        
        x_next = torch.matmul(transition, x)
        
        # Sample 8 points along the segment connecting x and x_next
        lambdas = torch.linspace(0.0, 1.0, 8, device=x.device).view(1, 1, 8, 1)
        sampled_pts = (1.0 - lambdas) * x.unsqueeze(2) + lambdas * x_next.unsqueeze(2)
        sampled_pts_flat = sampled_pts.view(B, N, 16)
        
        # Concatenate original coordinates and segment points
        x_augmented = torch.cat([x, sampled_pts_flat], dim=-1)
        
        # If this is the discrete embedding, attach the int2bit binary representation
        if self.is_discrete and binary_rep is not None:
            x_augmented = torch.cat([x_augmented, binary_rep], dim=-1)
            
        # 2. Embed indices
        entity_type = torch.clamp(entity_type, 0, self.room_type_emb.num_embeddings - 1)
        entity_idx = torch.clamp(entity_idx, 0, self.room_idx_emb.num_embeddings - 1)
        corner_idx = torch.clamp(corner_idx, 0, self.corner_idx_emb.num_embeddings - 1)
        
        emb = self.coord_proj(x_augmented)
        emb = emb + self.room_type_emb(entity_type)
        emb = emb + self.corner_idx_emb(corner_idx)
        emb = emb + self.room_idx_emb(entity_idx)
        emb = emb + time_emb.unsqueeze(1) 
        
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


class HouseDiffusionModel(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, num_cont_layers=4, num_disc_layers=2, dropout=0.1,
                 num_room_types=32, max_corners_per_room=512, max_rooms=512, max_outline_len=128, num_layers=None):
        super().__init__()
        if num_layers is not None:
            num_cont_layers = max(1, int(num_layers * 2 // 3))
            num_disc_layers = max(1, num_layers - num_cont_layers)
            
        self.time_embed = TimestepEmbedding(d_model)
        self.outline_encoder = OutlineEncoder(d_model, max_outline_len)
        
        # Continuous Branch Modules
        self.cont_embed = CornerEmbedding(d_model, num_room_types, max_corners_per_room, max_rooms, is_discrete=False)
        self.cont_blocks = nn.ModuleList([
            HouseDiffusionBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_cont_layers)
        ])
        self.cont_norm = nn.LayerNorm(d_model)
        self.out_noise = nn.Linear(d_model, 2)
        
        # Discrete Branch Modules
        self.disc_embed = CornerEmbedding(d_model, num_room_types, max_corners_per_room, max_rooms, is_discrete=True)
        self.disc_blocks = nn.ModuleList([
            HouseDiffusionBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_disc_layers)
        ])
        self.disc_norm = nn.LayerNorm(d_model)
        self.out_discrete = nn.Linear(d_model, 16)
        
    def forward(self, x_t, timesteps, entity_type, entity_idx, corner_idx, outline, alphas_cumprod_t, outline_mask=None, corner_mask=None):
        B, N, _ = x_t.shape
        time_emb = self.time_embed(timesteps)
        outline_emb = self.outline_encoder(outline)
        
        # Shared Masks
        csa_mask = (entity_idx.unsqueeze(2) == entity_idx.unsqueeze(1)).float()
        gsa_mask = None
        if corner_mask is not None:
            valid_mask = (corner_mask.unsqueeze(2) * corner_mask.unsqueeze(1)).float()
            csa_mask = csa_mask * valid_mask
            gsa_mask = valid_mask
            
        cross_attn_mask = outline_mask.unsqueeze(1).repeat(1, N, 1) if outline_mask is not None else None

        # ==========================================
        # STAGE 1: Continuous Denoising
        # ==========================================
        h_cont = self.cont_embed(x_t, entity_type, entity_idx, corner_idx, time_emb)
        for block in self.cont_blocks:
            h_cont = block(h_cont, outline_emb, csa_mask=csa_mask, gsa_mask=gsa_mask, outline_mask=cross_attn_mask)
            
        h_cont = self.cont_norm(h_cont)
        pred_noise = self.out_noise(h_cont)
        
        # ==========================================
        # STAGE 2: Discrete Denoising
        # ==========================================
        # Calculate x_0 estimate from x_t and pred_noise
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t).view(-1, 1, 1)
        x_0_hat = (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
        
        # Convert estimate to 8-bit binary representation
        binary_rep = coord_to_binary(x_0_hat, num_bits=8)
        
        # Pass through discrete blocks
        h_disc = self.disc_embed(x_0_hat, entity_type, entity_idx, corner_idx, time_emb, binary_rep=binary_rep)
        for block in self.disc_blocks:
            h_disc = block(h_disc, outline_emb, csa_mask=csa_mask, gsa_mask=gsa_mask, outline_mask=cross_attn_mask)
            
        h_disc = self.disc_norm(h_disc)
        pred_discrete = self.out_discrete(h_disc)
        
        return pred_noise, pred_discrete


class RoomConditionSampler:
    """
    Constructs the input conditions probabilistically based on histograms
    derived from the training dataset (e.g., Modified Swiss Dwellings).
    """
    def __init__(self, max_total_corners=128):
        self.max_total_corners = max_total_corners
        
        # Simulated histogram: Number of rooms per apartment
        self.room_counts = [3, 4, 5, 6, 7]
        self.room_count_weights = [0.1, 0.3, 0.3, 0.2, 0.1]
        
        # Room Types: 0: Living, 1: Kitchen, 2: Bedroom, 3: Bath, 4: Corridor
        self.room_types = [0, 1, 2, 3, 4]
        
        # Corner histograms per room type: {type: ([possible N corners], [probabilities])}
        # e.g., Bathrooms are almost always 4 corners, Living rooms are often L-shaped (6)
        self.corner_histograms = {
            0: ([4, 6, 8], [0.6, 0.3, 0.1]),  # Living
            1: ([4, 6], [0.8, 0.2]),          # Kitchen
            2: ([4, 6], [0.9, 0.1]),          # Bedroom
            3: ([4], [1.0]),                  # Bath
            4: ([4, 6, 8, 10], [0.3, 0.4, 0.2, 0.1]) # Corridor
        }
        
    def sample(self, batch_size, device):
        batch_entity_type, batch_entity_idx = [], []
        batch_corner_idx, batch_corner_mask = [], []
        
        for b in range(batch_size):
            # 1. Sample total number of rooms
            num_rooms = random.choices(self.room_counts, weights=self.room_count_weights, k=1)[0]
            
            b_e_type, b_e_idx, b_c_idx = [], [], []
            
            # Force at least one Living (0), Kitchen (1), and Bath (3). Randomize the rest.
            types_to_generate = [0, 1, 3] + random.choices(self.room_types, k=num_rooms-3)
            
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


@torch.no_grad()
def generate_floorplan(model, diffusion, condition_sampler, outline, outline_mask=None, max_corners=128):
    """
    Generates a full vector floorplan conditioned only on the apartment outline.
    outline shape: (B, M, 2)
    """
    model.eval()
    B = outline.shape[0]
    device = outline.device
    
    # 1. Probabilistically generate the internal room graph/topology
    cond = condition_sampler.sample(batch_size=B, device=device)
    cond["outline"] = outline
    cond["outline_mask"] = outline_mask
    
    # 2. Define the shape of the noise tensor to start diffusing
    # We are generating 2D coordinates for max_corners
    shape = (B, max_corners, 2)
    
    # 3. Run the reverse diffusion loop (with discrete snapping in the final 32 steps)
    generated_coords = diffusion.p_sample_loop(
        model=model, 
        shape=shape, 
        cond=cond, 
        snapping_threshold=32 # Activates discrete branch at the end
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
        Compute continuous MSE loss (noise prediction) and discrete MSE loss (8-bit coordinate prediction).
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Forward pass: create noisy sample
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict the noise and discrete coordinates using model
        predicted_noise, predicted_discrete = model(
            x_t=x_t,
            timesteps=t,
            entity_type=cond["entity_type"],
            entity_idx=cond["entity_idx"],
            corner_idx=cond["corner_idx"],
            outline=cond["outline"],
            outline_mask=cond.get("outline_mask", None),
            corner_mask=cond.get("corner_mask", None)
        )
        
        # Continuous loss
        loss_continuous = F.mse_loss(predicted_noise, noise)
        
        # Discrete loss: target is 8-bit binary representation of coordinates
        target_binary = coord_to_binary(x_0, num_bits=8)
        loss_discrete = F.mse_loss(predicted_discrete, target_binary)
        
        return loss_continuous + loss_discrete
        
    @torch.no_grad()
    def p_sample(self, model, x, t, cond, snapping_threshold=32):
        """
        Perform a single reverse step to denoise x_t into x_{t-1}.
        If t < snapping_threshold, we use discrete snapping to align coordinates.
        """
        # Obtain alphas and betas for current timestep
        beta_t = self.betas[t].view(-1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        
        # Predict noise and discrete coordinates
        predicted_noise, predicted_discrete = model(
            x_t=x,
            timesteps=t,
            entity_type=cond["entity_type"],
            entity_idx=cond["entity_idx"],
            corner_idx=cond["corner_idx"],
            outline=cond["outline"],
            outline_mask=cond.get("outline_mask", None),
            corner_mask=cond.get("corner_mask", None)
        )
        
        # Snap coordinates if timestep is below threshold
        if t[0] < snapping_threshold:
            # Snap: reconstruct coordinates from binary representation
            x_0_pred = binary_to_coord(predicted_discrete, num_bits=8)
            # Posterior mean
            coeff_x_0 = torch.sqrt(alphas_cumprod_prev_t) * beta_t / (1.0 - alphas_cumprod_t)
            coeff_x_t = torch.sqrt(alpha_t) * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
            mean = coeff_x_0 * x_0_pred + coeff_x_t * x
        else:
            # Standard continuous step
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)
            
        if t[0] == 0:
            return mean
        else:
            variance = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise
            
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond, snapping_threshold=32):
        """
        Complete reverse process loop to generate vector floorplan corners from pure noise.
        """
        B = shape[0]
        # Start from isotropic Gaussian noise
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.steps)):
            t = torch.full((B,), i, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, t, cond, snapping_threshold=snapping_threshold)
            
        return x
