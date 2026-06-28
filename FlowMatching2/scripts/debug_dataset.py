from __future__ import annotations

import argparse
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
from tqdm import tqdm

from floorplan_gen.geometry import PlanTransform, denormalize_geometry


def polygon_from_vertices(vertices):
    from shapely.geometry import Polygon

    return Polygon(np.asarray(vertices, dtype=np.float64))


def iter_room_vertices(vertices: np.ndarray, masks: np.ndarray):
    for plan_index in range(vertices.shape[0]):
        for room_index in np.flatnonzero(masks[plan_index]):
            yield plan_index, room_index, vertices[plan_index, room_index]


def room_geometry_stats(vertices: np.ndarray) -> dict[str, float | bool | int]:
    polygon = polygon_from_vertices(vertices)
    closed = np.concatenate([vertices, vertices[:1]], axis=0)
    edges = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    unique_points = np.unique(np.round(vertices, decimals=6), axis=0)
    return {
        "valid": bool(polygon.is_valid and not polygon.is_empty and polygon.area > 1e-8),
        "self_intersecting": bool((not polygon.is_valid) or (not polygon.is_simple)),
        "repeated_vertex_count": int(vertices.shape[0] - unique_points.shape[0]),
        "zero_length_edge_count": int((edges < 1e-6).sum()),
        "fewer_than_three_points": bool(unique_points.shape[0] < 3),
        "area": float(abs(polygon.area)),
    }


def denormalize_vertices(vertices: np.ndarray, transform: PlanTransform) -> np.ndarray:
    out = vertices.copy().astype(np.float64)
    out[..., 0] = out[..., 0] * transform.scale + transform.center_x
    out[..., 1] = out[..., 1] * transform.scale + transform.center_y
    return out


def save_debug_plots(
    output_dir: Path,
    plan_ids: list[str],
    boundary_points: np.ndarray,
    room_vertices: np.ndarray,
    room_masks: np.ndarray,
    transforms: dict[str, dict[str, float]],
    count: int,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for index in tqdm(range(min(count, len(plan_ids))), desc="plotting debug samples", unit="plan"):
        plan_id = str(plan_ids[index])
        transform = PlanTransform.from_dict(transforms[plan_id]) if plan_id in transforms else None
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, mode in zip(axes, ["normalized", "reconstructed"]):
            boundary = boundary_points[index]
            if mode == "reconstructed" and transform is not None:
                boundary = denormalize_vertices(boundary, transform)
            ax.plot(
                np.r_[boundary[:, 0], boundary[0, 0]],
                np.r_[boundary[:, 1], boundary[0, 1]],
                color="black",
                linewidth=1.2,
            )
            for room_index in np.flatnonzero(room_masks[index]):
                vertices = room_vertices[index, room_index]
                if mode == "reconstructed" and transform is not None:
                    vertices = denormalize_vertices(vertices, transform)
                polygon = polygon_from_vertices(vertices)
                if mode == "reconstructed" and transform is not None:
                    polygon = denormalize_geometry(polygon_from_vertices(room_vertices[index, room_index]), transform)
                if polygon.is_empty:
                    continue
                color = "#4c78a8" if polygon.is_valid else "#e45756"
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.35, edgecolor="white", linewidth=0.4, color=color)
            ax.set_title(mode)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle(f"plan {plan_id}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{index:03d}_plan_{plan_id}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prepared floor-plan geometry targets.")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-plots", type=int, default=20)
    parser.add_argument("--max-plans", type=int, default=None)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(prepared_dir / f"{args.split}.npz", allow_pickle=True)
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    transforms_path = prepared_dir / "inverse_transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8")) if transforms_path.exists() else {}

    plan_ids = data["plan_ids"].astype(str).tolist()
    if args.max_plans is not None:
        plan_ids = plan_ids[: args.max_plans]
    n = len(plan_ids)
    room_vertices = data["room_vertices"][:n].astype(np.float32)
    room_masks = data["room_masks"][:n].astype(bool)
    boundary_points = data["boundary_points"][:n].astype(np.float32)
    room_geometry = data["room_geometry"][:n].astype(np.float32)

    stats = [room_geometry_stats(vertices) for _, _, vertices in iter_room_vertices(room_vertices, room_masks)]
    total_rooms = max(len(stats), 1)
    padded_coords = int((~room_masks).sum()) * room_vertices.shape[2] * 2
    total_coords = int(np.prod(room_vertices.shape))
    valid_coords = room_vertices[room_masks]
    geometry_values = room_geometry.reshape(-1)
    report = {
        "split": args.split,
        "plans": n,
        "geometry_representation": metadata.get("geometry_representation"),
        "invalid_polygon_rate": float(sum(not row["valid"] for row in stats) / total_rooms),
        "self_intersection_rate": float(sum(row["self_intersecting"] for row in stats) / total_rooms),
        "mean_number_of_rooms": float(room_masks.sum(axis=1).mean()) if n else 0.0,
        "mean_corners_per_room": float(room_vertices.shape[2]),
        "fraction_padded_coordinates": float(padded_coords / max(total_coords, 1)),
        "coordinate_min": float(valid_coords.min()) if valid_coords.size else 0.0,
        "coordinate_max": float(valid_coords.max()) if valid_coords.size else 0.0,
        "room_geometry_min": float(geometry_values.min()) if geometry_values.size else 0.0,
        "room_geometry_max": float(geometry_values.max()) if geometry_values.size else 0.0,
        "repeated_vertex_rate": float(sum(row["repeated_vertex_count"] > 0 for row in stats) / total_rooms),
        "zero_length_edge_rate": float(sum(row["zero_length_edge_count"] > 0 for row in stats) / total_rooms),
        "fewer_than_three_valid_points_rate": float(sum(row["fewer_than_three_points"] for row in stats) / total_rooms),
    }
    (output_dir / "dataset_debug_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    save_debug_plots(output_dir, plan_ids, boundary_points, room_vertices, room_masks, transforms, args.num_plots)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(output_dir)


if __name__ == "__main__":
    main()
