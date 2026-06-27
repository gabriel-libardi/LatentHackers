import argparse
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import FloorplanDataset, collate_fn
from house_flow import HouseFlowModel, FlowMatching

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataset = FloorplanDataset(csv_path=args.csv_path, index_file=args.train_index, max_total_corners=args.max_total_corners, max_outline_len=args.max_outline_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    model = HouseFlowModel(d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff, num_layers=args.num_layers, dropout=args.dropout, num_room_types=10, max_corners_per_room=512, max_rooms=512, max_outline_len=args.max_outline_len).to(device)
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    flow = FlowMatching(steps=args.sample_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if not batch:
                continue
            data = batch["x_0"].to(device)
            entity_type = batch["entity_type"].to(device)
            entity_idx = batch["entity_idx"].to(device)
            corner_idx = batch["corner_idx"].to(device)
            corner_mask = batch["corner_mask"].to(device)
            outline = batch["outline"].to(device)
            outline_mask = batch["outline_mask"].to(device)
            B = data.shape[0]
            noise = torch.randn_like(data)
            t = torch.rand(B, device=device)
            x_t = flow.add_noise(data, noise, t)
            target = flow.velocity(data, noise)
            optimizer.zero_grad()
            pred = model(x_t=x_t, timesteps=t, entity_type=entity_type, entity_idx=entity_idx, corner_idx=corner_idx, outline=outline, outline_mask=outline_mask)
            diff = (pred - target) ** 2
            loss = (diff.mean(dim=-1) * corner_mask).sum() / corner_mask.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.5f}")
        scheduler.step()
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"=== Epoch {epoch} Complete | Avg Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f} ===")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = f"flow_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint to {ckpt}")
    torch.save(model.state_dict(), "flow_final.pth")
    print("Training finished! Final model saved to flow_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Outline-Conditioned HouseFlowMatching Model")
    parser.add_argument("--csv_path", type=str, default="mds_V2_5.372k.csv")
    parser.add_argument("--train_index", type=str, default="train_indices.txt")
    parser.add_argument("--epochs", type=str, default="50")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_total_corners", type=int, default=800)
    parser.add_argument("--max_outline_len", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    if len(sys.argv) == 1:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    args.epochs = int(args.epochs)
    train(args)
