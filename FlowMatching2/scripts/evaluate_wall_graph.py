from __future__ import annotations

import argparse
import csv
import json
import os
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
from tqdm import tqdm

from floorplan_gen.evaluation import evaluate_layout
from floorplan_gen.models import WallGraphFlow
from floorplan_gen.prepared_dataset import PreparedFloorPlanDataset
from floorplan_gen.sampling import sample_wall_graph
from floorplan_gen.wall_graph import decode_wall_graph, edge_index_to_adjacency


FIXED_T_VALUES = (0.1, 0.5, 0.9)


def greedy_match(pred: np.ndarray, target: np.ndarray, max_distance: float = 0.05):
    if len(pred) == 0 or len(target) == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        cost = np.linalg.norm(pred[:, None, :] - target[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= max_distance]
    except Exception:
        pairs = []
        used = set()
        for i, point in enumerate(pred):
            distances = np.linalg.norm(target - point[None, :], axis=-1)
            order = np.argsort(distances)
            for j in order:
                if int(j) not in used and float(distances[j]) <= max_distance:
                    used.add(int(j))
                    pairs.append((i, int(j), float(distances[j])))
                    break
        return pairs


def edge_set_from_dense(edge_probs: np.ndarray, active: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    active = [int(i) for i in active]
    edges = set()
    for i in active:
        for j in active:
            if j > i and edge_probs[i, j] >= threshold:
                edges.add((i, j))
    return edges


def count_crossings(points: np.ndarray, edges: set[tuple[int, int]]) -> int:
    try:
        from shapely.geometry import LineString
    except ImportError:
        return 0
    lines = [(edge, LineString([tuple(points[edge[0]]), tuple(points[edge[1]])])) for edge in edges]
    crossings = 0
    for i, (edge_a, line_a) in enumerate(lines):
        for edge_b, line_b in lines[i + 1 :]:
            if set(edge_a) & set(edge_b):
                continue
            if line_a.crosses(line_b):
                crossings += 1
    return crossings


def precision_recall_f1(tp: int, pred: int, target: int) -> tuple[float, float, float]:
    precision = tp / max(pred, 1)
    recall = tp / max(target, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


@torch.no_grad()
def predicted_attention_mask(model, noisy, t, boundary, fallback_mask):
    prelim = model(noisy, t, boundary, junction_mask=None)
    probs = torch.sigmoid(prelim["junction_presence_logits"])
    mask = probs >= 0.5
    fallback_counts = fallback_mask.sum(dim=1).clamp_min(1)
    for row_index in range(mask.shape[0]):
        if not mask[row_index].any():
            topk = torch.topk(probs[row_index], int(fallback_counts[row_index].item())).indices
            mask[row_index, topk] = True
    return mask


@torch.no_grad()
def endpoint_diagnostics(model, boundary, target_xy, target_mask, seed: int, device) -> dict[str, float]:
    boundary = boundary.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_xy = target_xy.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_mask = target_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
    generator = None if torch.device(device).type == "mps" else torch.Generator(device=device)
    if generator is not None:
        generator.manual_seed(seed)
    rows = []
    for t_value in FIXED_T_VALUES:
        noise = torch.randn(target_xy.shape, device=device, generator=generator)
        t = torch.full((1,), float(t_value), device=device)
        noisy = (1.0 - float(t_value)) * noise + float(t_value) * target_xy
        target_flow = target_xy - noise
        gt_outputs = model(noisy, t, boundary, junction_mask=target_mask)
        pred_mask = predicted_attention_mask(model, noisy, t, boundary, target_mask)
        pred_outputs = model(noisy, t, boundary, junction_mask=pred_mask)
        mask = target_mask.to(torch.float32).unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0)
        gt_x1 = noisy + (1.0 - float(t_value)) * gt_outputs["flow"]
        pred_x1 = noisy + (1.0 - float(t_value)) * pred_outputs["flow"]
        rows.append(
            {
                "endpoint_mse_gt_mask": float((((gt_x1 - target_xy).pow(2) * mask).sum() / denom).detach().cpu()),
                "endpoint_mse_pred_mask": float((((pred_x1 - target_xy).pow(2) * mask).sum() / denom).detach().cpu()),
                "flow_mse_gt_mask": float((((gt_outputs["flow"] - target_flow).pow(2) * mask).sum() / denom).detach().cpu()),
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def save_graph_plot(path: Path, target_points, target_edges, pred_points, pred_edges, rooms, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, points, edges, label in [(axes[0], target_points, target_edges, "target"), (axes[1], pred_points, pred_edges, "pred graph")]:
        for a, b in edges:
            ax.plot([points[a, 0], points[b, 0]], [points[a, 1], points[b, 1]], color="#222222", linewidth=1.0)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=12, color="#e45756")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.axis("off")
    for room in rooms:
        geom = room["geometry"]
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        else:
            polys = [part for part in getattr(geom, "geoms", []) if part.geom_type == "Polygon"]
        for poly in polys:
            x, y = poly.exterior.xy
            axes[2].fill(x, y, alpha=0.45, edgecolor="white", linewidth=0.5)
    axes[2].set_title("reconstructed faces")
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experimental wall-graph checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-plans", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--edge-thresholds", default=None, help="Comma-separated threshold sweep, e.g. 0.1,0.2,0.3,0.4,0.5.")
    parser.add_argument("--junction-threshold", type=float, default=0.5)
    parser.add_argument("--wall-graph-gt-mask", action="store_true")
    parser.add_argument("--wall-graph-gt-edges", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if checkpoint.get("metadata", {}).get("geometry_representation") != "wall_graph":
        raise ValueError("evaluate_wall_graph.py requires a wall_graph checkpoint.")
    model = WallGraphFlow(**{k: v for k, v in checkpoint["model_config"].items() if k != "model_type"}).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    dataset = PreparedFloorPlanDataset(args.prepared_dir, split=args.split)
    rows = []
    threshold_rows = []
    failures = []
    thresholds = [args.edge_threshold]
    if args.edge_thresholds:
        thresholds = [float(value) for value in args.edge_thresholds.split(",") if value.strip()]
    for index in tqdm(range(min(args.num_plans, len(dataset))), desc="wall graph eval", unit="plan"):
        item = dataset[index]
        plan_id = str(item["plan_id"])
        boundary = item["boundary_points"]
        gt_mask = item["junction_mask"].bool()
        gt_points = item["junction_xy"].numpy()[gt_mask.numpy()]
        junction_mask = item["junction_mask"] if args.wall_graph_gt_mask else None
        try:
            pred_xy, outputs = sample_wall_graph(
                model,
                boundary,
                max_junctions=int(checkpoint["metadata"]["max_junctions"]),
                steps=args.steps,
                seed=args.seed + index,
                device=args.device,
                junction_mask=junction_mask,
            )
            pred_presence = torch.sigmoid(outputs["junction_presence_logits"])[0].detach().cpu().numpy()
            if args.wall_graph_gt_mask:
                pred_presence = item["junction_mask"].numpy().astype(np.float32)
            pred_points_all = pred_xy[0].detach().cpu().numpy()
            pred_active = np.flatnonzero(pred_presence >= args.junction_threshold)
            pred_points = pred_points_all[pred_active]
            matches = greedy_match(pred_points, gt_points)
            jp, jr, jf = precision_recall_f1(len(matches), len(pred_points), len(gt_points))
            coord_error = float(np.mean([match[2] for match in matches])) if matches else 1.0
            edge_probs = 1.0 / (1.0 + np.exp(-outputs["edge_logits"][0].detach().cpu().numpy()))
            if args.wall_graph_gt_edges:
                dense = edge_index_to_adjacency(item["edge_index"].numpy(), item["edge_mask"].numpy(), pred_points_all.shape[0])
                edge_probs = dense
            gt_dense = edge_index_to_adjacency(item["edge_index"].numpy(), item["edge_mask"].numpy(), pred_points_all.shape[0])
            gt_edges = edge_set_from_dense(gt_dense, np.flatnonzero(item["junction_mask"].numpy()), 0.5)
            for threshold in thresholds:
                sweep_edges = edge_set_from_dense(edge_probs, pred_active, threshold)
                sp, sr, sf = precision_recall_f1(len(sweep_edges & gt_edges), len(sweep_edges), len(gt_edges))
                threshold_rows.append(
                    {
                        "plan_id": plan_id,
                        "edge_threshold": threshold,
                        "edge_precision": sp,
                        "edge_recall": sr,
                        "edge_f1": sf,
                        "pred_edges": len(sweep_edges),
                        "target_edges": len(gt_edges),
                    }
                )
            pred_edges = edge_set_from_dense(edge_probs, pred_active, args.edge_threshold)
            ep, er, ef = precision_recall_f1(len(pred_edges & gt_edges), len(pred_edges), len(gt_edges))
            endpoint = endpoint_diagnostics(
                model,
                boundary,
                item["junction_xy"],
                item["junction_mask"],
                seed=args.seed + 10_000 + index,
                device=args.device,
            )
            try:
                from shapely.geometry import Polygon

                outline = Polygon(boundary.numpy())
            except Exception:
                outline = None
            rooms, decode_info = decode_wall_graph(
                pred_points_all,
                pred_presence,
                torch.from_numpy(np.log(np.clip(edge_probs, 1e-6, 1 - 1e-6) / np.clip(1 - edge_probs, 1e-6, 1))),
                outline,
                edge_threshold=args.edge_threshold,
                junction_threshold=args.junction_threshold,
            )
            layout_metrics = evaluate_layout(rooms, outline) if outline is not None else {}
            row = {
                "plan_id": plan_id,
                "junction_precision": jp,
                "junction_recall": jr,
                "junction_f1": jf,
                "junction_coord_error": coord_error,
                "endpoint_mse_gt_mask": endpoint["endpoint_mse_gt_mask"],
                "endpoint_mse_pred_mask": endpoint["endpoint_mse_pred_mask"],
                "flow_mse_gt_mask": endpoint["flow_mse_gt_mask"],
                "edge_precision": ep,
                "edge_recall": er,
                "edge_f1": ef,
                "crossing_walls": count_crossings(pred_points_all, pred_edges),
                "reconstructed_faces": decode_info.get("faces", 0),
                "room_count_error": abs(float(layout_metrics.get("room_count", 0)) - float(item["wall_room_mask"].sum())),
                "invalid_face_rate": layout_metrics.get("invalid_polygon_rate", 0.0),
                "overlap_ratio": layout_metrics.get("overlap_ratio", 0.0),
                "uncovered_outline_ratio": layout_metrics.get("uncovered_ratio", 1.0),
                "outside_outline_ratio": layout_metrics.get("outside_outline_ratio", 0.0),
                "decode_rejected_crossings": decode_info.get("rejected_crossings", 0),
            }
            rows.append(row)
            if index < 8:
                save_graph_plot(
                    output_dir / "plots" / f"{index:03d}_{plan_id}.png",
                    item["junction_xy"].numpy(),
                    gt_edges,
                    pred_points_all,
                    pred_edges,
                    rooms,
                    f"{plan_id} jf1={jf:.2f} ef1={ef:.2f}",
                )
        except Exception as exc:
            failures.append({"plan_id": plan_id, "reason": f"{type(exc).__name__}: {exc}"})
    if rows:
        with (output_dir / "metrics_per_plan.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if threshold_rows:
        with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(threshold_rows)
    with (output_dir / "wall_graph_failures.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["plan_id", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    threshold_summary = {}
    selected_threshold = args.edge_threshold
    if threshold_rows:
        for threshold in thresholds:
            subset = [row for row in threshold_rows if row["edge_threshold"] == threshold]
            threshold_summary[str(threshold)] = {
                "edge_precision": float(np.mean([row["edge_precision"] for row in subset])),
                "edge_recall": float(np.mean([row["edge_recall"] for row in subset])),
                "edge_f1": float(np.mean([row["edge_f1"] for row in subset])),
            }
        selected_threshold = max(thresholds, key=lambda value: threshold_summary[str(value)]["edge_f1"])
    summary = {
        "plans": len(rows),
        "failures": len(failures),
        "selected_edge_threshold": selected_threshold,
        "threshold_sweep": threshold_summary,
        "unsupported_features": ["room_type_generation", "shared_wall_consistency_metric"],
        "averages": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "plan_id"} if rows else {},
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_dir)


if __name__ == "__main__":
    main()
