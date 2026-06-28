from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from floorplan_gen.config import DEFAULT_SEED
from floorplan_gen.decoding import decode_room_geometry
from floorplan_gen.losses import flow_matching_losses, sample_flow_batch
from floorplan_gen.models import ConditionalRoomFlow, WallGraphFlow
from floorplan_gen.prepared_dataset import PreparedFloorPlanDataset
from floorplan_gen.sampling import sample_room_geometry
from floorplan_gen.wall_graph_losses import sample_junction_flow_batch, wall_graph_losses


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_train_validation(dataset, val_fraction: float, seed: int, max_train_items: int | None, max_val_items: int | None):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_fraction))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    if max_train_items is not None:
        train_indices = train_indices[:max_train_items]
    if max_val_items is not None:
        val_indices = val_indices[:max_val_items]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def cap_dataset(dataset, max_items: int | None):
    if max_items is None:
        return dataset
    return Subset(dataset, list(range(min(len(dataset), max_items))))


def subset_dataset(dataset, max_items: int | None):
    if max_items is None:
        return dataset
    return Subset(dataset, list(range(min(len(dataset), max_items))))


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _dataset_arrays(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return base.room_type_ids[indices], base.room_masks[indices], base.room_counts[indices]
    return dataset.room_type_ids, dataset.room_masks, dataset.room_counts


def compute_type_weights(dataset, num_classes: int) -> torch.Tensor:
    type_ids, masks, _ = _dataset_arrays(dataset)
    counts = np.bincount(type_ids[masks].reshape(-1), minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = (counts[present].mean() / counts[present]) ** 0.5
    weights[0] = 0.0
    return torch.from_numpy(weights)


def count_distribution(dataset) -> dict[str, float | int]:
    _, _, counts = _dataset_arrays(dataset)
    if len(counts) == 0:
        return {"min": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0, "mean": 0.0}
    return {
        "min": int(np.min(counts)),
        "p50": float(np.percentile(counts, 50)),
        "p90": float(np.percentile(counts, 90)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(np.max(counts)),
        "mean": float(np.mean(counts)),
    }


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    generator: torch.Generator | None,
    epoch: int,
    progress_every: int,
    type_weights: torch.Tensor | None,
    loss_weights: dict[str, float],
):
    model.train(train)
    rows: list[dict[str, float]] = []
    label = f"{'train' if train else 'val'} epoch {epoch}"
    total_batches = len(loader)
    print(f"{label}: starting {total_batches} batches", flush=True)
    progress = tqdm(loader, desc=label, unit="batch", file=sys.stdout, dynamic_ncols=True)
    for batch_index, batch in enumerate(progress, start=1):
        boundary = batch["boundary_points"].to(device=device, dtype=torch.float32)
        geometry = batch["room_geometry"].to(device=device, dtype=torch.float32)
        presence = batch["room_presence"].to(device=device, dtype=torch.float32)
        type_ids = batch["room_type_ids"].to(device=device, dtype=torch.long)
        mask = batch["room_mask"].to(device=device, dtype=torch.bool)
        room_count = batch["room_count"].to(device=device, dtype=torch.long)
        noisy, t, target_flow = sample_flow_batch(geometry, generator=generator)
        with torch.set_grad_enabled(train):
            outputs = model(noisy, t, boundary, room_mask=mask)
            losses = flow_matching_losses(
                outputs,
                target_flow,
                geometry,
                mask,
                presence,
                type_ids,
                room_count=room_count,
                type_weights=type_weights,
                noisy_geometry=noisy,
                t=t,
                **loss_weights,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        row = {key: float(value.detach().cpu()) for key, value in losses.items()}
        rows.append(row)
        progress.set_postfix(loss=f"{row['loss']:.4f}", flow=f"{row['flow_loss']:.4f}")
        if progress_every > 0 and (batch_index % progress_every == 0 or batch_index == total_batches):
            running = mean_metrics(rows)
            progress.write(
                f"{label}: batch {batch_index}/{total_batches} "
                f"loss={running['loss']:.4f} flow={running['flow_loss']:.4f} "
                f"presence={running['presence_loss']:.4f} type={running['type_loss']:.4f} "
                f"count={running['count_loss']:.4f} area={running['area_loss']:.4f}",
            )
    metrics = mean_metrics(rows)
    print(f"{label}: finished {json.dumps(metrics, sort_keys=True)}", flush=True)
    return metrics


def run_wall_graph_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    generator: torch.Generator | None,
    epoch: int,
    progress_every: int,
    loss_weights: dict[str, float],
    use_gt_edges: bool,
    mask_mode: str = "gt",
    seed: int = DEFAULT_SEED,
    clean_coordinates: bool = False,
):
    model.train(train)
    rows: list[dict[str, float]] = []
    label = f"{'train' if train else 'val'} wall_graph epoch {epoch}"
    progress = tqdm(loader, desc=label, unit="batch", file=sys.stdout, dynamic_ncols=True)
    for batch_index, batch in enumerate(progress, start=1):
        boundary = batch["boundary_points"].to(device=device, dtype=torch.float32)
        junction_xy = batch["junction_xy"].to(device=device, dtype=torch.float32)
        junction_mask = batch["junction_mask"].to(device=device, dtype=torch.bool)
        edge_index = batch["edge_index"].to(device=device, dtype=torch.long)
        edge_mask = batch["edge_mask"].to(device=device, dtype=torch.bool)
        if train and clean_coordinates:
            noisy = junction_xy
            t = torch.ones((junction_xy.shape[0],), device=device)
            target_flow = torch.zeros_like(junction_xy)
        elif train:
            noisy, t, target_flow = sample_junction_flow_batch(junction_xy, generator=generator)
        else:
            fixed_rows = []
            for t_value in [0.1, 0.5, 0.9]:
                val_gen = torch.Generator(device=device) if device.type != "mps" else None
                if val_gen is not None:
                    val_gen.manual_seed(seed + epoch * 1009 + batch_index)
                noise = torch.randn(junction_xy.shape, device=device, generator=val_gen)
                t = torch.full((junction_xy.shape[0],), t_value, device=device)
                noisy = (1.0 - t_value) * noise + t_value * junction_xy
                target_flow = junction_xy - noise
                with torch.no_grad():
                    attention_mask = make_wall_graph_attention_mask(
                        model,
                        noisy,
                        t,
                        boundary,
                        junction_mask,
                        mask_mode,
                    )
                    outputs = model(noisy, t, boundary, junction_mask=attention_mask)
                    if mask_mode == "predicted":
                        topology_outputs = model(noisy, t, boundary, junction_mask=None)
                        outputs = dict(outputs)
                        outputs["junction_presence_logits"] = topology_outputs["junction_presence_logits"]
                        outputs["edge_logits"] = topology_outputs["edge_logits"]
                    losses = wall_graph_losses(
                        outputs,
                        target_flow,
                        junction_xy,
                        junction_mask,
                        edge_index,
                        edge_mask,
                        noisy_junction_xy=noisy,
                        t=t,
                        use_gt_edges=use_gt_edges,
                        **loss_weights,
                    )
                fixed_rows.append({key: float(value.detach().cpu()) for key, value in losses.items()})
            rows.append(mean_metrics(fixed_rows))
            row = rows[-1]
            progress.set_postfix(loss=f"{row['loss']:.4f}", endpoint=f"{row['junction_endpoint_loss']:.4f}", edge=f"{row['edge_loss']:.4f}")
            continue
        attention_mask = make_wall_graph_attention_mask(model, noisy, t, boundary, junction_mask, mask_mode)
        with torch.set_grad_enabled(train):
            outputs = model(noisy, t, boundary, junction_mask=attention_mask)
            if mask_mode == "predicted":
                topology_outputs = model(noisy, t, boundary, junction_mask=None)
                outputs = dict(outputs)
                outputs["junction_presence_logits"] = topology_outputs["junction_presence_logits"]
                outputs["edge_logits"] = topology_outputs["edge_logits"]
            losses = wall_graph_losses(
                outputs,
                target_flow,
                junction_xy,
                junction_mask,
                edge_index,
                edge_mask,
                noisy_junction_xy=noisy,
                t=t,
                use_gt_edges=use_gt_edges,
                **loss_weights,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        row = {key: float(value.detach().cpu()) for key, value in losses.items()}
        rows.append(row)
        progress.set_postfix(loss=f"{row['loss']:.4f}", endpoint=f"{row['junction_endpoint_loss']:.4f}", edge=f"{row['edge_loss']:.4f}")
        if progress_every > 0 and (batch_index % progress_every == 0 or batch_index == len(loader)):
            running = mean_metrics(rows)
            progress.write(
                f"{label}: batch {batch_index}/{len(loader)} loss={running['loss']:.4f} "
                f"flow={running['junction_flow_loss']:.4f} endpoint={running['junction_endpoint_loss']:.4f} "
                f"presence={running['junction_presence_loss']:.4f} edge={running['edge_loss']:.4f} edge_f1={running['edge_f1']:.4f}",
            )
    metrics = mean_metrics(rows)
    print(f"{label}: finished {json.dumps(metrics, sort_keys=True)}", flush=True)
    return metrics


@torch.no_grad()
def make_wall_graph_attention_mask(model, noisy, t, boundary, target_mask, mask_mode: str):
    if mask_mode == "gt":
        return target_mask
    if mask_mode == "none":
        return None
    if mask_mode != "predicted":
        raise ValueError("mask_mode must be gt, predicted, or none.")
    prelim = model(noisy, t, boundary, junction_mask=None)
    probs = torch.sigmoid(prelim["junction_presence_logits"])
    attention_mask = probs >= 0.5
    target_counts = target_mask.sum(dim=1).clamp_min(1)
    for row_index in range(attention_mask.shape[0]):
        if not attention_mask[row_index].any():
            topk = torch.topk(probs[row_index], int(target_counts[row_index].item())).indices
            attention_mask[row_index, topk] = True
    return attention_mask


def wall_graph_mask_mode(epoch: int, warmup_epochs: int, transition_epochs: int) -> str:
    if epoch < warmup_epochs:
        return "gt"
    if transition_epochs <= 0:
        return "predicted"
    if epoch < warmup_epochs + transition_epochs:
        return "gt" if (epoch - warmup_epochs) % 2 == 0 else "predicted"
    return "predicted"


def wall_graph_selection_metric(metrics: dict[str, float]) -> float:
    return (
        float(metrics.get("junction_endpoint_loss", 1e6))
        + (1.0 - float(metrics.get("edge_f1", 0.0)))
        + float(metrics.get("junction_presence_loss", 1e6))
    )


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, args, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
            "metadata": metadata,
            "model_config": {
                "geometry_dim": int(metadata.get("geometry_dim", metadata.get("vertex_count", 16) * 2)),
                "max_rooms": int(metadata["max_rooms"]),
                "num_room_types": len(metadata["type_to_id"]),
                "point_hidden_dim": args.point_hidden_dim,
                "cond_dim": args.cond_dim,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "encoder_layers": args.layers if args.layers is not None else args.encoder_layers,
                "decoder_layers": args.layers if args.layers is not None else args.decoder_layers,
                "dim_feedforward": args.ffn_dim,
                "dropout": args.dropout,
            },
        },
        path,
    )


def save_wall_graph_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, args, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["training_edge_threshold"] = float(args.edge_threshold)
    checkpoint_metadata["wall_graph_objective"] = {
        "junction_velocity": float(args.lambda_junction_flow),
        "junction_endpoint": float(args.lambda_junction_endpoint),
        "junction_presence": float(args.lambda_junction_presence),
        "edge_connectivity": 1.0 if args.lambda_edge is None else float(args.lambda_edge),
        "edge_loss": args.edge_loss,
        "edge_pos_weight_max": float(args.edge_pos_weight_max),
    }
    checkpoint_metadata["unsupported_wall_graph_features"] = [
        "room_type_generation",
        "edge_crossing_loss",
        "outline_contact_loss",
        "shared_wall_consistency_metric",
    ]
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
            "metadata": checkpoint_metadata,
            "model_config": {
                "model_type": "wall_graph",
                "max_junctions": int(metadata["max_junctions"]),
                "num_room_types": len(metadata["type_to_id"]),
                "d_model": args.d_model,
                "nhead": args.nhead,
                "encoder_layers": args.layers if args.layers is not None else args.encoder_layers,
                "decoder_layers": args.layers if args.layers is not None else args.decoder_layers,
                "dim_feedforward": args.ffn_dim,
                "dropout": args.dropout,
            },
        },
        path,
    )


@torch.no_grad()
def save_overfit_previews(
    model,
    dataset,
    output_dir: Path,
    metadata: dict,
    device,
    epoch: int,
    seed: int,
    steps: int,
    preview_repair: bool,
    limit: int = 4,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from shapely.geometry import Polygon
        try:
            from shapely.errors import GEOSException
        except ImportError:
            GEOSException = Exception
    except ImportError:
        return

    from floorplan_gen.decoding import iter_polygons, repair_geometry
    from floorplan_gen.plotting import room_color

    model.eval()
    preview_dir = output_dir / "overfit_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    type_to_id = {str(key): int(value) for key, value in metadata["type_to_id"].items()}
    type_id_to_label = {value: key for key, value in type_to_id.items()}
    representation = metadata.get("geometry_representation", "polygon")
    max_rooms = int(metadata["max_rooms"])
    for index in range(min(limit, len(dataset))):
        item = dataset[index]
        plan_id = str(item.get("plan_id", index))
        boundary = item["boundary_points"].to(device=device, dtype=torch.float32)
        try:
            outline = repair_geometry(Polygon(boundary.cpu().numpy()))
            geometry, outputs = sample_room_geometry(
                model,
                boundary,
                max_rooms=max_rooms,
                steps=steps,
                seed=seed + epoch * 1000 + index,
                device=device,
            )
            type_logits = outputs["type_logits"]
            type_ids = type_logits[..., 1:].argmax(dim=-1) + 1 if type_logits.shape[-1] > 1 else type_logits.argmax(dim=-1)
            presence = torch.sigmoid(outputs["presence_logits"])
            try:
                rooms = decode_room_geometry(
                    geometry[0],
                    presence[0],
                    type_ids[0],
                    type_id_to_label,
                    outline,
                    fill_gaps=preview_repair,
                    repair=preview_repair,
                    representation=representation,
                )
            except GEOSException as exc:
                print(
                    f"preview epoch {epoch} plan {plan_id} sample {index}: GEOSException during repaired decode: {exc}; retrying raw",
                    flush=True,
                )
                rooms = decode_room_geometry(
                    geometry[0],
                    presence[0],
                    type_ids[0],
                    type_id_to_label,
                    outline,
                    fill_gaps=False,
                    repair=False,
                    representation=representation,
                )
            except Exception as exc:
                print(
                    f"preview epoch {epoch} plan {plan_id} sample {index}: decode failed: {type(exc).__name__}: {exc}; retrying raw",
                    flush=True,
                )
                rooms = decode_room_geometry(
                    geometry[0],
                    presence[0],
                    type_ids[0],
                    type_id_to_label,
                    outline,
                    fill_gaps=False,
                    repair=False,
                    representation=representation,
                )
        except Exception as exc:
            print(
                f"preview epoch {epoch} plan {plan_id} sample {index}: skipped after {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        for room in rooms:
            for polygon in iter_polygons(room["geometry"]):
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.55, edgecolor="white", linewidth=0.6, color=room_color(room["type"]))
        for polygon in iter_polygons(outline):
            x, y = polygon.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"epoch {epoch} item {index}")
        fig.tight_layout()
        fig.savefig(preview_dir / f"epoch_{epoch:04d}_item_{index:03d}.png", dpi=150)
        plt.close(fig)


def str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditional flow-matching room-token baseline.")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--overfit-samples", "--overfit_samples", type=int, default=None)
    parser.add_argument("--overfit-preview-every", type=int, default=1)
    parser.add_argument("--preview-repair", type=str_to_bool, default=False)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--geometry-representation", choices=["polygon", "partition", "wall_graph"], default=None)
    parser.add_argument("--wall-graph-gt-mask", action="store_true")
    parser.add_argument("--wall-graph-gt-edges", action="store_true")
    parser.add_argument("--wall-graph-mask-warmup-epochs", type=int, default=20)
    parser.add_argument("--wall-graph-mask-transition-epochs", type=int, default=20)
    parser.add_argument("--wall-graph-clean-coordinates", action="store_true", help="Diagnostic edge-overfit mode: feed clean target junction coordinates.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=None, help="Deprecated alias for encoder/decoder layers.")
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--point-hidden-dim", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--velocity-loss", "--velocity_loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--presence-weight", type=float, default=0.2)
    parser.add_argument("--type-weight", type=float, default=0.3)
    parser.add_argument("--count-weight", type=float, default=0.3)
    parser.add_argument("--area-weight", type=float, default=0.1)
    parser.add_argument("--lambda-edge", "--lambda_edge", type=float, default=None)
    parser.add_argument("--lambda-short-edge", "--lambda_short_edge", type=float, default=0.0)
    parser.add_argument("--lambda-area", "--lambda_area", type=float, default=0.0)
    parser.add_argument("--lambda-outside", "--lambda_outside", type=float, default=0.0)
    parser.add_argument("--lambda-junction-flow", "--lambda_junction_flow", type=float, default=1.0)
    parser.add_argument("--lambda-junction-endpoint", "--lambda_junction_endpoint", type=float, default=1.0)
    parser.add_argument("--lambda-junction-presence", "--lambda_junction_presence", type=float, default=0.2)
    parser.add_argument("--edge-loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--edge-pos-weight-max", type=float, default=50.0)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--min-edge-length", type=float, default=0.02)
    parser.add_argument("--min-signed-area", type=float, default=1e-3)
    parser.add_argument("--outside-margin", type=float, default=1.1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a persistent progress line every N batches. Use 0 to disable batch progress prints.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_train = PreparedFloorPlanDataset(args.prepared_dir, split="train")
    prepared_val = PreparedFloorPlanDataset(args.prepared_dir, split="val")
    if args.overfit_samples is not None:
        train_dataset = subset_dataset(full_train, args.overfit_samples)
        val_dataset = subset_dataset(full_train, args.overfit_samples)
    elif len(prepared_val):
        train_dataset = cap_dataset(full_train, args.max_train_items)
        val_dataset = cap_dataset(prepared_val, args.max_val_items)
    else:
        train_dataset, val_dataset = split_train_validation(
            full_train,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_train_items=args.max_train_items,
            max_val_items=args.max_val_items,
        )
    metadata = full_train.metadata
    representation = str(metadata.get("geometry_representation", "partition"))
    if args.geometry_representation is not None and args.geometry_representation != representation:
        raise ValueError(
            f"Prepared data is {representation}, but --geometry-representation={args.geometry_representation} was requested."
        )
    if representation == "wall_graph":
        wall_model_config = {
            "max_junctions": int(metadata["max_junctions"]),
            "num_room_types": len(metadata["type_to_id"]),
            "d_model": args.d_model,
            "nhead": args.nhead,
            "encoder_layers": args.layers if args.layers is not None else args.encoder_layers,
            "decoder_layers": args.layers if args.layers is not None else args.decoder_layers,
            "dim_feedforward": args.ffn_dim,
            "dropout": args.dropout,
        }
        model = WallGraphFlow(**wall_model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_epoch = 0
        best_val = float("inf")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            if checkpoint.get("metadata", {}).get("geometry_representation") != "wall_graph":
                raise ValueError("Cannot resume wall_graph training from a non-wall_graph checkpoint.")
            model.load_state_dict(checkpoint["model_state"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint.get("best_val", best_val))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.overfit_samples is None)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if len(val_dataset) else None
        generator = None if device.type == "mps" else torch.Generator(device=device)
        if generator is not None:
            generator.manual_seed(args.seed)
        wall_loss_weights = {
            "lambda_junction_flow": args.lambda_junction_flow,
            "lambda_junction_endpoint": args.lambda_junction_endpoint,
            "lambda_junction_presence": args.lambda_junction_presence,
            "lambda_edge": 1.0 if args.lambda_edge is None else args.lambda_edge,
            "edge_loss": args.edge_loss,
            "edge_pos_weight_max": args.edge_pos_weight_max,
        }
        log_path = output_dir / "train_log.jsonl"
        print(
            json.dumps(
                {
                    "device": str(device),
                    "geometry_representation": "wall_graph",
                    "train_items": len(train_dataset),
                    "val_items": len(val_dataset),
                    "train_batches": len(train_loader),
                    "val_batches": len(val_loader) if val_loader else 0,
                    "start_epoch": start_epoch,
                    "epochs": args.epochs,
                    "max_junctions": int(metadata["max_junctions"]),
                    "max_edges": int(metadata["max_edges"]),
                    "loss_weights": wall_loss_weights,
                    "wall_graph_gt_mask": args.wall_graph_gt_mask,
                    "wall_graph_gt_edges": args.wall_graph_gt_edges,
                    "mask_warmup_epochs": args.wall_graph_mask_warmup_epochs,
                    "mask_transition_epochs": args.wall_graph_mask_transition_epochs,
                    "clean_coordinates": args.wall_graph_clean_coordinates,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for epoch in range(start_epoch, args.epochs):
            mask_mode = "gt" if args.wall_graph_gt_mask else wall_graph_mask_mode(
                epoch,
                args.wall_graph_mask_warmup_epochs,
                args.wall_graph_mask_transition_epochs,
            )
            train_metrics = run_wall_graph_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train=True,
                generator=generator,
                epoch=epoch,
                progress_every=args.progress_every,
                loss_weights=wall_loss_weights,
                use_gt_edges=args.wall_graph_gt_edges,
                mask_mode=mask_mode,
                seed=args.seed,
                clean_coordinates=args.wall_graph_clean_coordinates,
            )
            val_generator = None
            if device.type != "mps":
                val_generator = torch.Generator(device=device)
                val_generator.manual_seed(args.seed + 100_000)
            val_metrics_gt = (
                run_wall_graph_epoch(
                    model,
                    val_loader,
                    optimizer,
                    device,
                    train=False,
                    generator=val_generator,
                    epoch=epoch,
                    progress_every=args.progress_every,
                    loss_weights=wall_loss_weights,
                    use_gt_edges=args.wall_graph_gt_edges,
                    mask_mode="gt",
                    seed=args.seed,
                    clean_coordinates=args.wall_graph_clean_coordinates,
                )
                if val_loader
                else {}
            )
            val_metrics_pred = (
                run_wall_graph_epoch(
                    model,
                    val_loader,
                    optimizer,
                    device,
                    train=False,
                    generator=val_generator,
                    epoch=epoch,
                    progress_every=args.progress_every,
                    loss_weights=wall_loss_weights,
                    use_gt_edges=args.wall_graph_gt_edges,
                    mask_mode="predicted",
                    seed=args.seed + 17,
                    clean_coordinates=args.wall_graph_clean_coordinates,
                )
                if val_loader and not args.wall_graph_gt_mask
                else val_metrics_gt
            )
            primary_val = val_metrics_gt if args.wall_graph_gt_mask else val_metrics_pred
            record = {"epoch": epoch, "train": train_metrics, "val": primary_val, "val_gt_mask": val_metrics_gt, "val_pred_mask": val_metrics_pred}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            val_loss = wall_graph_selection_metric(primary_val) if primary_val else wall_graph_selection_metric(train_metrics)
            save_wall_graph_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val, args, metadata)
            if val_loss <= best_val:
                best_val = val_loss
                save_wall_graph_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val, args, metadata)
        return

    model_config = {
        "token_dim": 6,
        "geometry_dim": int(metadata.get("geometry_dim", metadata.get("vertex_count", 16) * 2)),
        "max_rooms": int(metadata["max_rooms"]),
        "num_room_types": len(metadata["type_to_id"]),
        "point_hidden_dim": args.point_hidden_dim,
        "cond_dim": args.cond_dim,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "encoder_layers": args.layers if args.layers is not None else args.encoder_layers,
        "decoder_layers": args.layers if args.layers is not None else args.decoder_layers,
        "dim_feedforward": args.ffn_dim,
        "dropout": args.dropout,
    }
    model = ConditionalRoomFlow(**model_config).to(device)
    type_weights = compute_type_weights(train_dataset, model_config["num_room_types"] + 1).to(device)
    loss_weights = {
        "velocity_loss": args.velocity_loss,
        "flow_weight": args.flow_weight,
        "presence_weight": args.presence_weight,
        "type_weight": args.type_weight,
        "count_weight": args.count_weight,
        "area_weight": args.area_weight,
        "lambda_edge": 0.0 if args.lambda_edge is None else args.lambda_edge,
        "lambda_short_edge": args.lambda_short_edge,
        "lambda_area": args.lambda_area,
        "lambda_outside": args.lambda_outside,
        "min_edge_length": args.min_edge_length,
        "min_signed_area": args.min_signed_area,
        "outside_margin": args.outside_margin,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.overfit_samples is None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if len(val_dataset) else None
    log_path = output_dir / "train_log.jsonl"
    generator = None if device.type == "mps" else torch.Generator(device=device)
    if generator is not None:
        generator.manual_seed(args.seed)

    print(
        json.dumps(
            {
                "device": str(device),
                "train_items": len(train_dataset),
                "val_items": len(val_dataset),
                "train_batches": len(train_loader),
                "val_batches": len(val_loader) if val_loader else 0,
                "start_epoch": start_epoch,
                "epochs": args.epochs,
                "train_room_count_distribution": count_distribution(train_dataset),
                "val_room_count_distribution": count_distribution(val_dataset) if len(val_dataset) else {},
                "type_weights": [float(value) for value in type_weights.detach().cpu().tolist()],
                "loss_weights": loss_weights,
                "overfit_samples": args.overfit_samples,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch}/{args.epochs - 1}: begin", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            generator=generator,
            epoch=epoch,
            progress_every=args.progress_every,
            type_weights=type_weights,
            loss_weights=loss_weights,
        )
        val_generator = None
        if device.type != "mps":
            val_generator = torch.Generator(device=device)
            val_generator.manual_seed(args.seed + 100_000)
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                generator=val_generator,
                epoch=epoch,
                progress_every=args.progress_every,
                type_weights=type_weights,
                loss_weights=loss_weights,
            )
            if val_loader
            else {}
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        val_loss = val_metrics.get("loss", train_metrics["loss"])
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val, args, metadata)
        if val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val, args, metadata)
        if args.overfit_samples is not None and args.overfit_preview_every > 0 and epoch % args.overfit_preview_every == 0:
            save_overfit_previews(
                model,
                train_dataset,
                output_dir,
                metadata,
                device,
                epoch=epoch,
                seed=args.seed,
                steps=args.sample_steps,
                preview_repair=args.preview_repair,
            )


if __name__ == "__main__":
    main()
