from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from tqdm import tqdm

from floorplan_gen.config import CSV_PATH, GEOM_COL, PLAN_ID_COL, ROOM_TYPE_COL
from floorplan_gen.dataset import load_area_frame
from floorplan_gen.decoding import FloorPlanGenerator, iter_polygons
from floorplan_gen.evaluation import diagnostic_score, evaluate_layout, repair_change_ratio, sample_diversity, type_distribution_error
from floorplan_gen.geometry import build_apartment_outline, load_wkt_geometries
from floorplan_gen.models import ConditionalRoomFlow
from floorplan_gen.plotting import room_color
from floorplan_gen.raster_metrics import (
    RasterProtocol,
    density_coverage,
    distribution_summary,
    frechet_distance,
    nearest_real_distances,
    raster_features,
    rasterize_layout,
)


def load_plan_ids(prepared_dir: str | None, split: str, csv_path: str) -> list[str]:
    if prepared_dir is not None:
        path = Path(prepared_dir) / f"{split}.npz"
        if path.exists():
            return np.load(path, allow_pickle=True)["plan_ids"].astype(str).tolist()
    area_df = load_area_frame(csv_path)
    return sorted(area_df[PLAN_ID_COL].astype(str).unique().tolist())


def load_real_layout(area_df, plan_id: str):
    plan_df = area_df[area_df[PLAN_ID_COL] == str(plan_id)]
    if plan_df.empty:
        raise ValueError(f"No area rows for plan_id={plan_id}")
    geometries = load_wkt_geometries(plan_df[GEOM_COL].tolist())
    outline = build_apartment_outline(geometries)
    rooms = [
        {"type": str(room_type), "geometry": geometry}
        for room_type, geometry in zip(plan_df[ROOM_TYPE_COL].tolist(), geometries)
    ]
    return outline, rooms


def save_comparison_plot(path: Path, outline, real_rooms, raw_rooms, repaired_rooms, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, rooms, label in zip(axes, [real_rooms, raw_rooms, repaired_rooms], ["real", "raw generation", "repaired generation"]):
        for room in rooms:
            for polygon in iter_polygons(room["geometry"]):
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.55, linewidth=0.7, edgecolor="white", color=room_color(room["type"]))
        for polygon in iter_polygons(outline):
            x, y = polygon.exterior.xy
            ax.plot(x, y, color="black", linewidth=1.5)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_distribution_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    nearest = [float(row["nearest_real_distance"]) for row in rows]
    diagnostic = [float(row["diagnostic_score"]) for row in rows]
    raw_coverage = [float(row["raw_outline_coverage"]) for row in rows]
    repaired_coverage = [float(row["repaired_outline_coverage"]) for row in rows]
    raw_overlap = [float(row["raw_overlap_ratio"]) for row in rows]
    repaired_overlap = [float(row["repaired_overlap_ratio"]) for row in rows]
    room_counts = [float(row["repaired_room_count"]) for row in rows]
    count_errors = [float(row["room_count_error"]) for row in rows]
    tiny_cells = [float(row["repaired_tiny_cell_fraction_0_01"]) for row in rows]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plots = [
        ("nearest real distance", nearest),
        ("diagnostic score", diagnostic),
        ("room count error", count_errors),
        ("repaired room count", room_counts),
        ("tiny cell fraction <1%", tiny_cells),
        ("repaired overlap", repaired_overlap),
    ]
    for ax, (title, values) in zip(axes.ravel(), plots):
        ax.hist(values, bins=20, color="#4c78a8", alpha=0.85)
        ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def select_examples(rows: list[dict[str, object]], per_group: int) -> dict[str, list[dict[str, object]]]:
    ordered = sorted(rows, key=lambda row: float(row["diagnostic_score"]))
    if not ordered:
        return {"best": [], "medium": [], "worst": []}
    midpoint = len(ordered) // 2
    half = max(per_group // 2, 1)
    return {
        "best": ordered[:per_group],
        "medium": ordered[max(0, midpoint - half) : max(0, midpoint - half) + per_group],
        "worst": ordered[-per_group:],
    }


def parse_step_sweep(value: str | None, default_steps: int) -> list[int]:
    if not value:
        return [default_steps]
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("--step-sweep must contain positive integers.")
    return steps


def safe_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float | None:
    try:
        return float(frechet_distance(real_features, fake_features))
    except ValueError:
        return None


def safe_density_coverage(real_features: np.ndarray, fake_features: np.ndarray, k: int) -> dict[str, float | None]:
    try:
        return density_coverage(real_features, fake_features, k=k)
    except ValueError:
        return {"density": None, "coverage": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated layouts with FID, density, and coverage.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-plans", type=int, default=32)
    parser.add_argument("--samples-per-plan", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--step-sweep", default=None, help="Comma-separated Euler step counts; uses identical seeds across counts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--examples-per-group", type=int, default=4)
    parser.add_argument("--selection-mode", choices=["predicted_count", "threshold", "fixed_topk"], default="predicted_count")
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--fixed-room-count", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = RasterProtocol(
        resolution=args.resolution,
        feature_dim=args.feature_dim,
        seed=args.seed,
        knn=args.knn,
    )

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = ConditionalRoomFlow(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    type_to_id = {str(key): int(value) for key, value in checkpoint["metadata"]["type_to_id"].items()}
    type_id_to_label = {value: key for key, value in type_to_id.items()}
    generator = FloorPlanGenerator(
        model,
        type_id_to_label,
        max_rooms=int(checkpoint["metadata"]["max_rooms"]),
        boundary_points=int(checkpoint["metadata"]["num_boundary_points"]),
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        representation=checkpoint["metadata"].get("geometry_representation", "polygon"),
        selection_mode=args.selection_mode,
        presence_threshold=args.presence_threshold,
        fixed_room_count=args.fixed_room_count,
    )
    step_values = parse_step_sweep(args.step_sweep, args.steps)

    area_df = load_area_frame(args.csv)
    area_df[PLAN_ID_COL] = area_df[PLAN_ID_COL].astype(str)
    plan_ids = load_plan_ids(args.prepared_dir, args.split, args.csv)[: args.num_plans]
    real_rasters = []
    fake_rasters = []
    fake_rasters_by_step: dict[int, list[np.ndarray]] = {step: [] for step in step_values}
    rows: list[dict[str, object]] = []
    real_type_counts: Counter[str] = Counter()
    generated_type_counts: Counter[str] = Counter()
    real_room_counts: list[int] = []
    generated_room_counts: list[int] = []
    layouts: dict[tuple[str, int, int], tuple[object, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]] = {}
    generated_samples_by_plan_step: dict[tuple[str, int], list[list[dict[str, object]]]] = {}

    for plan_index, plan_id in enumerate(tqdm(plan_ids, desc="evaluating plans", unit="plan")):
        outline, real_rooms = load_real_layout(area_df, plan_id)
        real_metrics = evaluate_layout(real_rooms, outline)
        real_type_counts.update(real_metrics["type_distribution"])
        real_room_counts.append(int(real_metrics["room_count"]))
        real_rasters.append(rasterize_layout(real_rooms, outline, type_to_id, resolution=protocol.resolution))
        for step_count in step_values:
            generator.steps = step_count
            for sample_index in tqdm(
                range(args.samples_per_plan),
                desc=f"plan {plan_id} steps {step_count} samples",
                unit="sample",
                leave=False,
            ):
                seed = args.seed + plan_index * args.samples_per_plan + sample_index
                result = generator.generate_with_diagnostics(outline, seed=seed)
                repaired = result["rooms"]
                raw = result["raw_rooms"]
                fake_raster = rasterize_layout(repaired, outline, type_to_id, resolution=protocol.resolution)
                fake_rasters.append(fake_raster)
                fake_rasters_by_step[step_count].append(fake_raster)
                raw_metrics = evaluate_layout(raw, outline)
                repaired_metrics = evaluate_layout(repaired, outline)
                generated_type_counts.update(repaired_metrics["type_distribution"])
                generated_room_counts.append(int(repaired_metrics["room_count"]))
                generated_samples_by_plan_step.setdefault((str(plan_id), step_count), []).append(repaired)
                repair_change = repair_change_ratio(raw, repaired, outline)
                real_count = len([room for room in real_rooms if not room["geometry"].is_empty])
                type_error = type_distribution_error(repaired_metrics["type_distribution"], real_metrics["type_distribution"])
                row = {
                    "plan_id": plan_id,
                    "sample": sample_index,
                    "steps": step_count,
                    "seed": seed,
                    "raw_outline_coverage": raw_metrics["outline_coverage"],
                    "raw_overlap_ratio": raw_metrics["overlap_ratio"],
                    "raw_outside_outline_area": raw_metrics["outside_outline_area"],
                    "raw_outside_outline_ratio": raw_metrics["outside_outline_ratio"],
                    "raw_invalid_polygon_rate": raw_metrics["invalid_polygon_rate"],
                    "raw_self_intersection_rate": raw_metrics["self_intersection_rate"],
                    "raw_valid_fraction": raw_metrics["valid_fraction"],
                    "raw_room_count": raw_metrics["room_count"],
                    "raw_uncovered_ratio": raw_metrics["uncovered_ratio"],
                    "raw_uncovered_outline_area": raw_metrics["uncovered_outline_area"],
                    "raw_tiny_fragment_count": raw_metrics["tiny_fragment_count"],
                    "raw_tiny_cell_fraction_0_005": raw_metrics["tiny_cell_fraction_0_005"],
                    "raw_tiny_cell_fraction_0_01": raw_metrics["tiny_cell_fraction_0_01"],
                    "raw_tiny_cell_fraction_0_02": raw_metrics["tiny_cell_fraction_0_02"],
                    "raw_very_small_room_count": raw_metrics["very_small_room_count"],
                    "raw_very_thin_room_count": raw_metrics["very_thin_room_count"],
                    "raw_sliver_count": raw_metrics["sliver_count"],
                    "repaired_outline_coverage": repaired_metrics["outline_coverage"],
                    "repaired_overlap_ratio": repaired_metrics["overlap_ratio"],
                    "repaired_overlap_area": repaired_metrics["overlap_area"],
                    "repaired_outside_outline_area": repaired_metrics["outside_outline_area"],
                    "repaired_outside_outline_ratio": repaired_metrics["outside_outline_ratio"],
                    "repaired_invalid_polygon_rate": repaired_metrics["invalid_polygon_rate"],
                    "repaired_self_intersection_rate": repaired_metrics["self_intersection_rate"],
                    "repaired_valid_fraction": repaired_metrics["valid_fraction"],
                    "repaired_room_count": repaired_metrics["room_count"],
                    "repaired_uncovered_ratio": repaired_metrics["uncovered_ratio"],
                    "repaired_uncovered_outline_area": repaired_metrics["uncovered_outline_area"],
                    "repaired_disconnected_extra_components": repaired_metrics["disconnected_extra_components"],
                    "repaired_tiny_fragment_count": repaired_metrics["tiny_fragment_count"],
                    "repaired_tiny_cell_fraction_0_005": repaired_metrics["tiny_cell_fraction_0_005"],
                    "repaired_tiny_cell_fraction_0_01": repaired_metrics["tiny_cell_fraction_0_01"],
                    "repaired_tiny_cell_fraction_0_02": repaired_metrics["tiny_cell_fraction_0_02"],
                    "repaired_very_small_room_count": repaired_metrics["very_small_room_count"],
                    "repaired_very_thin_room_count": repaired_metrics["very_thin_room_count"],
                    "repaired_sliver_count": repaired_metrics["sliver_count"],
                    "repaired_perimeter_area_mean": repaired_metrics["perimeter_area_mean"],
                    "repaired_orthogonality_error_mean": repaired_metrics["orthogonality_error_mean"],
                    "repair_change_ratio": repair_change,
                    "real_room_count": real_count,
                    "room_count_error": abs(repaired_metrics["room_count"] - real_count),
                    "type_distribution_error": type_error,
                    "predicted_count": result.get("diagnostics", {}).get("predicted_count"),
                    "active_count": result.get("diagnostics", {}).get("active_count"),
                    "presence_mean": result.get("diagnostics", {}).get("presence_mean"),
                    "presence_max": result.get("diagnostics", {}).get("presence_max"),
                }
                row["diagnostic_score"] = diagnostic_score(raw_metrics, repaired_metrics, repair_change, real_count)
                rows.append(row)
                layouts[(plan_id, sample_index, step_count)] = (outline, real_rooms, raw, repaired)

    real_features = raster_features(np.asarray(real_rasters), feature_dim=protocol.feature_dim, seed=protocol.seed)
    fake_features = raster_features(np.asarray(fake_rasters), feature_dim=protocol.feature_dim, seed=protocol.seed)
    fid = safe_fid(real_features, fake_features)
    dc = safe_density_coverage(real_features, fake_features, k=protocol.knn)
    nearest = nearest_real_distances(real_features, fake_features)
    for row, distance in zip(rows, nearest):
        row["nearest_real_distance"] = float(distance)
        row["feature_distance"] = float(distance)

    step_summaries = {}
    for step_count, rasters in fake_rasters_by_step.items():
        step_rows = [row for row in rows if int(row["steps"]) == step_count]
        step_features = raster_features(np.asarray(rasters), feature_dim=protocol.feature_dim, seed=protocol.seed)
        step_summaries[str(step_count)] = {
            "fid": safe_fid(real_features, step_features),
            **safe_density_coverage(real_features, step_features, k=protocol.knn),
            "room_count_error": distribution_summary([row["room_count_error"] for row in step_rows]),
            "overlap_ratio": distribution_summary([row["repaired_overlap_ratio"] for row in step_rows]),
            "coverage": distribution_summary([row["repaired_outline_coverage"] for row in step_rows]),
            "invalid_polygon_rate": distribution_summary([row["repaired_invalid_polygon_rate"] for row in step_rows]),
            "self_intersection_rate": distribution_summary([row["repaired_self_intersection_rate"] for row in step_rows]),
            "very_small_room_count": distribution_summary([row["repaired_very_small_room_count"] for row in step_rows]),
            "very_thin_room_count": distribution_summary([row["repaired_very_thin_room_count"] for row in step_rows]),
        }
    diversity_rows = [
        {"plan_id": plan_id, "steps": step_count, **sample_diversity(samples)}
        for (plan_id, step_count), samples in generated_samples_by_plan_step.items()
    ]

    summary = {
        "protocol": protocol.__dict__,
        "split": args.split,
        "num_plans": len(plan_ids),
        "samples_per_plan": args.samples_per_plan,
        "step_values": step_values,
        "selection_mode": args.selection_mode,
        "presence_threshold": args.presence_threshold,
        "fixed_room_count": args.fixed_room_count,
        "fid": fid,
        **dc,
        "real_room_count_distribution": distribution_summary(real_room_counts),
        "generated_room_count_distribution": distribution_summary(generated_room_counts),
        "real_type_distribution": dict(real_type_counts),
        "generated_type_distribution": dict(generated_type_counts),
        "type_distribution_error_total": type_distribution_error(dict(generated_type_counts), dict(real_type_counts)),
        "step_summaries": step_summaries,
        "diversity": {
            "count_std": distribution_summary([row["count_std"] for row in diversity_rows]),
            "mean_signature_distance": distribution_summary([row["mean_signature_distance"] for row in diversity_rows]),
        },
        "distributions": {
            "nearest_real_distance": distribution_summary([row["nearest_real_distance"] for row in rows]),
            "diagnostic_score": distribution_summary([row["diagnostic_score"] for row in rows]),
            "repair_change_ratio": distribution_summary([row["repair_change_ratio"] for row in rows]),
            "raw_outline_coverage": distribution_summary([row["raw_outline_coverage"] for row in rows]),
            "repaired_outline_coverage": distribution_summary([row["repaired_outline_coverage"] for row in rows]),
            "raw_overlap_ratio": distribution_summary([row["raw_overlap_ratio"] for row in rows]),
            "repaired_overlap_ratio": distribution_summary([row["repaired_overlap_ratio"] for row in rows]),
            "repaired_overlap_area": distribution_summary([row["repaired_overlap_area"] for row in rows]),
            "repaired_outside_outline_area": distribution_summary([row["repaired_outside_outline_area"] for row in rows]),
            "repaired_outside_outline_ratio": distribution_summary([row["repaired_outside_outline_ratio"] for row in rows]),
            "repaired_uncovered_outline_area": distribution_summary([row["repaired_uncovered_outline_area"] for row in rows]),
            "repaired_invalid_polygon_rate": distribution_summary([row["repaired_invalid_polygon_rate"] for row in rows]),
            "repaired_self_intersection_rate": distribution_summary([row["repaired_self_intersection_rate"] for row in rows]),
            "repaired_room_count": distribution_summary([row["repaired_room_count"] for row in rows]),
            "room_count_error": distribution_summary([row["room_count_error"] for row in rows]),
            "type_distribution_error": distribution_summary([row["type_distribution_error"] for row in rows]),
            "repaired_tiny_cell_fraction_0_005": distribution_summary([row["repaired_tiny_cell_fraction_0_005"] for row in rows]),
            "repaired_tiny_cell_fraction_0_01": distribution_summary([row["repaired_tiny_cell_fraction_0_01"] for row in rows]),
            "repaired_tiny_cell_fraction_0_02": distribution_summary([row["repaired_tiny_cell_fraction_0_02"] for row in rows]),
            "repaired_very_small_room_count": distribution_summary([row["repaired_very_small_room_count"] for row in rows]),
            "repaired_very_thin_room_count": distribution_summary([row["repaired_very_thin_room_count"] for row in rows]),
            "active_count": distribution_summary([row["active_count"] for row in rows]),
            "presence_mean": distribution_summary([row["presence_mean"] for row in rows]),
            "presence_max": distribution_summary([row["presence_max"] for row in rows]),
        },
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_distribution_plot(output_dir / "metric_distributions.png", rows)

    examples = select_examples(rows, args.examples_per_group)
    for group, group_rows in examples.items():
        for index, row in enumerate(group_rows):
            key = (str(row["plan_id"]), int(row["sample"]), int(row["steps"]))
            outline, real_rooms, raw_rooms, generated_rooms = layouts[key]
            save_comparison_plot(
                output_dir / group / f"{index:03d}_plan_{row['plan_id']}_sample_{row['sample']}_steps_{row['steps']}.png",
                outline,
                real_rooms,
                raw_rooms,
                generated_rooms,
                (
                    f"{group}: plan {row['plan_id']} sample {row['sample']} steps {row['steps']} "
                    f"real/raw/rep rooms {row['real_room_count']}/{row['raw_room_count']}/{row['repaired_room_count']} "
                    f"diag {row['diagnostic_score']:.3f} cov {row['repaired_outline_coverage']:.2f} "
                    f"ov {row['repaired_overlap_ratio']:.3f} tiny1 {row['repaired_tiny_cell_fraction_0_01']:.2f}"
                ),
            )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_dir)


if __name__ == "__main__":
    main()
