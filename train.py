import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset import FloorplanDataset, collate_fn
from house_diffusion import HouseDiffusionModel, GaussianDiffusion, coord_to_binary
import kagglehub

# Download latest version
path = kagglehub.dataset_download("caspervanengelenburg/modified-swiss-dwellings")

print("Path to dataset files:", path)
import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))

def train(args):
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load splits
    train_dataset = FloorplanDataset(
        csv_path=args.csv_path,
        index_file=args.train_index,
        max_total_corners=args.max_total_corners,
        max_outline_len=args.max_outline_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # 3. Model instantiation
    model = HouseDiffusionModel(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_room_types=32,            # Unique room categories in MSD (plus padding)
        max_corners_per_room=512,     # Padded room corners limit
        max_rooms=512,                # Limit on max rooms per plan
        max_outline_len=args.max_outline_len
    ).to(device)
    
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        
    # 4. Diffusion pipeline
    diffusion = GaussianDiffusion(steps=args.steps, device=device)
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 6. Training Loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if not batch:
                continue
                
            # Extract inputs
            x_0 = batch["x_0"].to(device)
            entity_type = batch["entity_type"].to(device)
            entity_idx = batch["entity_idx"].to(device)
            corner_idx = batch["corner_idx"].to(device)
            corner_mask = batch["corner_mask"].to(device)
            outline = batch["outline"].to(device)
            outline_mask = batch["outline_mask"].to(device)
            
            # Sample random timesteps
            B = x_0.shape[0]
            t = torch.randint(0, args.steps, (B,), device=device)
            
            # Forward diffusion and calculate loss
            optimizer.zero_grad()
            
            # Predict noise and calculate MSE loss
            noise = torch.randn_like(x_0)
            x_t = diffusion.q_sample(x_0, t, noise)
            
            # Model forward pass
            predicted_noise, predicted_discrete = model(
                x_t=x_t,
                timesteps=t,
                entity_type=entity_type,
                entity_idx=entity_idx,
                corner_idx=corner_idx,
                outline=outline,
                outline_mask=outline_mask,
                corner_mask=corner_mask
            )
            
            # Mask out continuous loss on padded corner tokens
            loss_cont_elementwise = (predicted_noise - noise) ** 2
            loss_continuous = (loss_cont_elementwise.mean(dim=-1) * corner_mask).sum() / corner_mask.sum()
            
            # Mask out discrete coordinate loss
            target_binary = coord_to_binary(x_0, num_bits=8)
            loss_disc_elementwise = (predicted_discrete - target_binary) ** 2
            loss_discrete = (loss_disc_elementwise.mean(dim=-1) * corner_mask).sum() / corner_mask.sum()
            
            # Combined Loss
            loss = loss_continuous + loss_discrete
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.5f}")
                
        scheduler.step()
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"=== Epoch {epoch} Complete | Avg Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f} ===")
        
        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    # Save final model
    torch.save(model.state_dict(), "model_final.pth")
    print("Training finished! Final model saved to model_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Outline-Conditioned HouseDiffusion Model")
    parser.add_argument("--csv_path", type=str, default="mds_V2_5.372k.csv", help="Path to room geometries CSV")
    parser.add_argument("--train_index", type=str, default="train_indices.txt", help="Path to train split index file")
    parser.add_argument("--epochs", type=str, default="50", help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer d_model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Transformer FFN dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of Transformer block layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_total_corners", type=int, default=800, help="Maximum corners limit to process in dataset")
    parser.add_argument("--max_outline_len", type=int, default=128, help="Outline sequence downsample limit")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--save_every", type=int, default=10, help="Epoch frequency to save checkpoint")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint weights to resume from")
    
    # Simple trick to handle command line args correctly in ipython / CLI
    import sys
    # If running in notebooks/interactive, default arguments are used
    if len(sys.argv) == 1:
        args = parser.parse_args(args=[])
    else:
        # Convert epochs to integer before training
        args = parser.parse_args()
    
    args.epochs = int(args.epochs)
    train(args)
