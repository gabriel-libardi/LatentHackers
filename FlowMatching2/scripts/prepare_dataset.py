from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from floorplan_gen.config import (
    CSV_PATH,
    DEFAULT_BOUNDARY_POINTS,
    DEFAULT_MAX_ROOMS,
    DEFAULT_SEED,
    DEFAULT_VERTEX_COUNT,
    FLOOR_ID_COL,
    GEOM_COL,
    ORIGINAL_SPLIT_DIR,
    OUTLINE_BUFFER_METERS,
    PLAN_ID_COL,
    ROOM_TYPE_COL,
)
from floorplan_gen.dataset import load_area_frame
from floorplan_gen.geometry import (
    build_apartment_outline,
    load_wkt_geometries,
    normalize_geometry,
    sample_boundary_points,
)
from floorplan_gen.splits import make_original_plan_splits, make_plan_splits
from floorplan_gen.tokens import RoomRecord, build_room_type_vocab, make_room_tokens
from floorplan_gen.representations import make_room_partition_targets, make_room_polygon_targets
from floorplan_gen.evaluation import evaluate_layout
from floorplan_gen.wall_graph import convert_rooms_to_wall_graph, decode_wall_graph, edge_index_to_adjacency


def percentile_summary(values):
    import numpy as np

    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def greedy_mean_iou(decoded_rooms, target_rooms) -> float:
    if not decoded_rooms or not target_rooms:
        return 0.0
    pairs = []
    for pred_index, pred in enumerate(decoded_rooms):
        pred_geom = pred["geometry"]
        for target_index, room in enumerate(target_rooms):
            target_geom = room.geometry
            try:
                union = pred_geom.union(target_geom).area
                iou = pred_geom.intersection(target_geom).area / union if union > 0 else 0.0
            except Exception:
                iou = 0.0
            pairs.append((iou, pred_index, target_index))
    used_pred = set()
    used_target = set()
    matched = []
    for iou, pred_index, target_index in sorted(pairs, reverse=True):
        if pred_index in used_pred or target_index in used_target:
            continue
        used_pred.add(pred_index)
        used_target.add(target_index)
        matched.append(iou)
    return float(sum(matched) / max(len(target_rooms), 1))


def wall_graph_roundtrip(plan_id, wall_graph, rooms, outline, max_face_count_error: int, max_uncovered_ratio: float, min_mean_iou: float):
    dense = edge_index_to_adjacency(wall_graph.edge_index, wall_graph.edge_mask, wall_graph.junction_xy.shape[0])
    logits = np.where(dense > 0.5, 12.0, -12.0).astype(np.float32)
    decoded, info = decode_wall_graph(
        wall_graph.junction_xy,
        wall_graph.junction_mask.astype(np.float32),
        logits,
        outline,
        edge_threshold=0.5,
        junction_threshold=0.5,
    )
    layout_metrics = evaluate_layout(decoded, outline)
    face_count_error = abs(len(decoded) - len(rooms))
    mean_iou = greedy_mean_iou(decoded, rooms)
    row = {
        "plan_id": str(plan_id),
        "reconstructed_face_count": len(decoded),
        "target_room_count": len(rooms),
        "face_count_error": face_count_error,
        "overlap_ratio": float(layout_metrics.get("overlap_ratio", 0.0)),
        "uncovered_outline_ratio": float(layout_metrics.get("uncovered_ratio", 1.0)),
        "outside_outline_ratio": float(layout_metrics.get("outside_outline_ratio", 0.0)),
        "mean_matched_room_iou": mean_iou,
        "decode_faces": int(info.get("faces", 0)),
        "decode_segments": int(info.get("segments", 0)),
    }
    ok = (
        face_count_error <= max_face_count_error
        and row["uncovered_outline_ratio"] <= max_uncovered_ratio
        and mean_iou >= min_mean_iou
    )
    reason = ""
    if not ok:
        reason = (
            f"roundtrip failed: face_count_error={face_count_error}, "
            f"uncovered={row['uncovered_outline_ratio']:.4f}, mean_iou={mean_iou:.4f}"
        )
    return ok, row, reason


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize normalized baseline arrays.")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-points", type=int, default=DEFAULT_BOUNDARY_POINTS)
    parser.add_argument("--max-rooms", type=int, default=DEFAULT_MAX_ROOMS)
    parser.add_argument("--vertex-count", type=int, default=DEFAULT_VERTEX_COUNT)
    parser.add_argument(
        "--representation",
        "--geometry-representation",
        dest="representation",
        choices=["partition", "polygon", "wall_graph"],
        default="partition",
        help="Geometry target: partition, polygon, or experimental wall_graph.",
    )
    parser.add_argument("--max-junctions", type=int, default=256)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--snap-tolerance", type=float, default=1e-3)
    parser.add_argument("--wall-simplify-tolerance", type=float, default=0.0)
    parser.add_argument("--scan-wall-graph-stats", action="store_true", help="Only scan graph-size distributions; do not write prepared arrays.")
    parser.add_argument("--disable-wall-graph-roundtrip-filter", action="store_true")
    parser.add_argument("--roundtrip-max-face-count-error", type=int, default=50)
    parser.add_argument("--roundtrip-max-uncovered-ratio", type=float, default=0.70)
    parser.add_argument("--roundtrip-min-mean-iou", type=float, default=0.10)
    parser.add_argument("--outline-buffer", type=float, default=OUTLINE_BUFFER_METERS)
    parser.add_argument("--split-dir", default=ORIGINAL_SPLIT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-plans", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    import numpy as np
    from tqdm import tqdm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    area_df = load_area_frame(args.csv)
    area_df[PLAN_ID_COL] = area_df[PLAN_ID_COL].astype(str)
    area_df[ROOM_TYPE_COL] = area_df[ROOM_TYPE_COL].astype(str)

    type_to_id = build_room_type_vocab(area_df[ROOM_TYPE_COL].tolist())
    rooms_per_plan = area_df.groupby(PLAN_ID_COL).size()
    truncation_stats = {
        "max_rooms": args.max_rooms,
        "p99_rooms": int(np.ceil(np.quantile(rooms_per_plan.to_numpy(), 0.99))),
        "truncated_plans": int((rooms_per_plan > args.max_rooms).sum()),
        "total_plans": int(len(rooms_per_plan)),
        "rooms_lost": int(np.maximum(rooms_per_plan.to_numpy() - args.max_rooms, 0).sum()),
    }
    if args.split_dir and Path(args.split_dir).exists():
        floor_pairs = area_df[[FLOOR_ID_COL, PLAN_ID_COL]].drop_duplicates().astype(str)
        floor_to_plan = dict(zip(floor_pairs[FLOOR_ID_COL], floor_pairs[PLAN_ID_COL]))
        splits = make_original_plan_splits(
            floor_to_plan,
            split_dir=args.split_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        split_source = "original_floor_id"
    else:
        splits = make_plan_splits(
            area_df[PLAN_ID_COL].unique().tolist(),
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        split_source = "hash_plan_id"

    if args.limit_plans is not None:
        keep_ids = sorted(area_df[PLAN_ID_COL].unique().tolist())[: args.limit_plans]
        area_df = area_df[area_df[PLAN_ID_COL].isin(keep_ids)].copy()
    plan_to_split = {plan_id: split for split, ids in splits.items() for plan_id in ids}
    transforms: dict[str, dict[str, float]] = {}

    split_arrays = {
        split: {
            "plan_ids": [],
            "boundary_points": [],
            "room_tokens": [],
            "room_vertices": [],
            "room_presence": [],
            "room_type_ids": [],
            "room_masks": [],
            "room_geometry": [],
            "junction_xy": [],
            "junction_mask": [],
            "edge_index": [],
            "edge_mask": [],
            "edge_is_exterior": [],
            "edge_room_ids": [],
            "wall_room_types": [],
            "wall_room_mask": [],
        }
        for split in splits
    }
    wall_graph_stats = []
    wall_graph_failures = []
    wall_graph_roundtrip_rows = []
    wall_graph_roundtrip_failures = []
    grouped = list(area_df.groupby(PLAN_ID_COL, sort=True))
    for plan_id, plan_df in tqdm(grouped, desc="preparing plans", unit="plan"):
        geometries = load_wkt_geometries(plan_df[GEOM_COL].tolist())
        outline = build_apartment_outline(geometries, buffer_distance=args.outline_buffer)
        normalized_outline, transform = normalize_geometry(outline)
        normalized_rooms = [normalize_geometry(geometry, transform)[0] for geometry in geometries]
        rooms = [
            RoomRecord(room_type=str(room_type), geometry=geometry)
            for room_type, geometry in zip(plan_df[ROOM_TYPE_COL].tolist(), normalized_rooms)
        ]
        boundary_points = sample_boundary_points(normalized_outline, args.boundary_points)
        room_tokens, room_mask = make_room_tokens(rooms, type_to_id, args.max_rooms)
        polygon_targets = make_room_polygon_targets(rooms, type_to_id, args.max_rooms, args.vertex_count)
        partition_targets = make_room_partition_targets(rooms, type_to_id, args.max_rooms)
        wall_graph = None
        if args.representation == "wall_graph":
            try:
                wall_graph = convert_rooms_to_wall_graph(
                    rooms,
                    type_to_id,
                    max_junctions=max(args.max_junctions, 10000) if args.scan_wall_graph_stats else args.max_junctions,
                    max_edges=max(args.max_edges, 40000) if args.scan_wall_graph_stats else args.max_edges,
                    max_rooms=args.max_rooms,
                    snap_tolerance=args.snap_tolerance,
                    simplify_tolerance=args.wall_simplify_tolerance,
                    outline=normalized_outline,
                )
                wall_graph_stats.append({"plan_id": str(plan_id), **wall_graph.stats})
                if args.scan_wall_graph_stats:
                    continue
                if not args.disable_wall_graph_roundtrip_filter:
                    ok, roundtrip_row, roundtrip_reason = wall_graph_roundtrip(
                        plan_id,
                        wall_graph,
                        rooms,
                        normalized_outline,
                        max_face_count_error=args.roundtrip_max_face_count_error,
                        max_uncovered_ratio=args.roundtrip_max_uncovered_ratio,
                        min_mean_iou=args.roundtrip_min_mean_iou,
                    )
                    wall_graph_roundtrip_rows.append(roundtrip_row)
                    if not ok:
                        wall_graph_roundtrip_failures.append({"plan_id": str(plan_id), "reason": roundtrip_reason})
                        continue
            except Exception as exc:
                wall_graph_failures.append({"plan_id": str(plan_id), "reason": f"{type(exc).__name__}: {exc}"})
                continue
            room_geometry = wall_graph.junction_xy
        elif args.representation == "partition":
            room_geometry = partition_targets.params
        else:
            room_geometry = polygon_targets.vertices.reshape(args.max_rooms, -1)

        split = plan_to_split[str(plan_id)]
        split_arrays[split]["plan_ids"].append(str(plan_id))
        split_arrays[split]["boundary_points"].append(boundary_points)
        split_arrays[split]["room_tokens"].append(room_tokens)
        split_arrays[split].setdefault("room_geometry", []).append(room_geometry)
        split_arrays[split]["room_vertices"].append(polygon_targets.vertices)
        split_arrays[split]["room_presence"].append(polygon_targets.presence)
        split_arrays[split]["room_type_ids"].append(polygon_targets.type_ids)
        split_arrays[split]["room_masks"].append(room_mask)
        if wall_graph is not None:
            split_arrays[split]["junction_xy"].append(wall_graph.junction_xy)
            split_arrays[split]["junction_mask"].append(wall_graph.junction_mask)
            split_arrays[split]["edge_index"].append(wall_graph.edge_index)
            split_arrays[split]["edge_mask"].append(wall_graph.edge_mask)
            split_arrays[split]["edge_is_exterior"].append(wall_graph.edge_is_exterior)
            split_arrays[split]["edge_room_ids"].append(wall_graph.edge_room_ids)
            split_arrays[split]["wall_room_types"].append(wall_graph.room_types)
            split_arrays[split]["wall_room_mask"].append(wall_graph.room_mask)
        transforms[str(plan_id)] = transform.to_dict()

    if args.representation == "wall_graph" and args.scan_wall_graph_stats:
        if wall_graph_failures:
            with (output_dir / "wall_graph_failures.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["plan_id", "reason"])
                writer.writeheader()
                writer.writerows(wall_graph_failures)
        if wall_graph_stats:
            with (output_dir / "wall_graph_stats.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(wall_graph_stats[0].keys()))
                writer.writeheader()
                writer.writerows(wall_graph_stats)
        summary = {
            "converted": len(wall_graph_stats),
            "failed": len(wall_graph_failures),
            "success_rate": len(wall_graph_stats) / max(len(wall_graph_stats) + len(wall_graph_failures), 1),
            "junctions": percentile_summary([row["junctions"] for row in wall_graph_stats]),
            "edges": percentile_summary([row["walls"] for row in wall_graph_stats]),
            "rooms": percentile_summary([row["rooms"] for row in wall_graph_stats]),
            "internal_junctions": percentile_summary([row.get("internal_junctions", row["junctions"]) for row in wall_graph_stats]),
            "recommendation": "Choose max_junctions and max_edges at or above p99 if dense edge memory is feasible.",
        }
        (output_dir / "wall_graph_scan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(output_dir)
        return

    for split, values in split_arrays.items():
        plan_count = len(values["plan_ids"])
        geometry_dim = 2 if args.representation == "wall_graph" else (4 if args.representation == "partition" else args.vertex_count * 2)
        geometry_slots = args.max_junctions if args.representation == "wall_graph" else args.max_rooms
        arrays = {
            "plan_ids": np.asarray(values["plan_ids"]),
            "boundary_points": np.asarray(values["boundary_points"], dtype=np.float32).reshape(plan_count, args.boundary_points, 2),
            "room_tokens": np.asarray(values["room_tokens"], dtype=np.float32).reshape(plan_count, args.max_rooms, 6),
            "room_geometry": np.asarray(values["room_geometry"], dtype=np.float32).reshape(plan_count, geometry_slots, geometry_dim),
            "room_vertices": np.asarray(values["room_vertices"], dtype=np.float32).reshape(plan_count, args.max_rooms, args.vertex_count, 2),
            "room_presence": np.asarray(values["room_presence"], dtype=np.float32).reshape(plan_count, args.max_rooms),
            "room_type_ids": np.asarray(values["room_type_ids"], dtype=np.int64).reshape(plan_count, args.max_rooms),
            "room_masks": np.asarray(values["room_masks"], dtype=bool).reshape(plan_count, args.max_rooms),
        }
        if args.representation == "wall_graph":
            arrays.update(
                {
                    "junction_xy": np.asarray(values["junction_xy"], dtype=np.float32).reshape(plan_count, args.max_junctions, 2),
                    "junction_mask": np.asarray(values["junction_mask"], dtype=bool).reshape(plan_count, args.max_junctions),
                    "edge_index": np.asarray(values["edge_index"], dtype=np.int64).reshape(plan_count, args.max_edges, 2),
                    "edge_mask": np.asarray(values["edge_mask"], dtype=bool).reshape(plan_count, args.max_edges),
                    "edge_is_exterior": np.asarray(values["edge_is_exterior"], dtype=bool).reshape(plan_count, args.max_edges),
                    "edge_room_ids": np.asarray(values["edge_room_ids"], dtype=np.int64).reshape(plan_count, args.max_edges, 2),
                    "wall_room_types": np.asarray(values["wall_room_types"], dtype=np.int64).reshape(plan_count, args.max_rooms),
                    "wall_room_mask": np.asarray(values["wall_room_mask"], dtype=bool).reshape(plan_count, args.max_rooms),
                }
            )
        np.savez_compressed(output_dir / f"{split}.npz", **arrays)

    retained_by_split = {split: len(values["plan_ids"]) for split, values in split_arrays.items()}

    if wall_graph_failures:
        with (output_dir / "wall_graph_failures.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["plan_id", "reason"])
            writer.writeheader()
            writer.writerows(wall_graph_failures)
    else:
        (output_dir / "wall_graph_failures.csv").write_text("plan_id,reason\n", encoding="utf-8")
    if wall_graph_stats:
        with (output_dir / "wall_graph_stats.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(wall_graph_stats[0].keys()))
            writer.writeheader()
            writer.writerows(wall_graph_stats)
    if args.representation == "wall_graph":
        if wall_graph_roundtrip_rows:
            with (output_dir / "wall_graph_roundtrip_per_plan.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(wall_graph_roundtrip_rows[0].keys()))
                writer.writeheader()
                writer.writerows(wall_graph_roundtrip_rows)
        else:
            (output_dir / "wall_graph_roundtrip_per_plan.csv").write_text("plan_id\n", encoding="utf-8")
        with (output_dir / "wall_graph_roundtrip_failures.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["plan_id", "reason"])
            writer.writeheader()
            writer.writerows(wall_graph_roundtrip_failures)
        roundtrip_summary = {
            "checked": len(wall_graph_roundtrip_rows),
            "failed": len(wall_graph_roundtrip_failures),
            "success_rate": 1.0 - len(wall_graph_roundtrip_failures) / max(len(wall_graph_roundtrip_rows), 1),
            "averages": {
                key: float(np.mean([row[key] for row in wall_graph_roundtrip_rows]))
                for key in wall_graph_roundtrip_rows[0]
                if key != "plan_id"
            }
            if wall_graph_roundtrip_rows
            else {},
        }
        (output_dir / "wall_graph_roundtrip_summary.json").write_text(
            json.dumps(roundtrip_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    metadata = {
        "type_to_id": type_to_id,
        "splits": splits,
        "split_source": split_source,
        "num_boundary_points": args.boundary_points,
        "max_rooms": args.max_rooms,
        "vertex_count": args.vertex_count,
        "geometry_representation": args.representation,
        "geometry_dim": 2 if args.representation == "wall_graph" else (4 if args.representation == "partition" else args.vertex_count * 2),
        "max_junctions": args.max_junctions if args.representation == "wall_graph" else None,
        "max_edges": args.max_edges if args.representation == "wall_graph" else None,
        "edge_threshold": 0.5 if args.representation == "wall_graph" else None,
        "snap_tolerance": args.snap_tolerance if args.representation == "wall_graph" else None,
        "wall_simplify_tolerance": args.wall_simplify_tolerance if args.representation == "wall_graph" else None,
        "wall_graph_conversion": {
            "successful_plans": len(wall_graph_stats),
            "failed_plans": len(wall_graph_failures),
            "success_rate": len(wall_graph_stats) / max(len(wall_graph_stats) + len(wall_graph_failures), 1),
            "roundtrip_failed_plans": len(wall_graph_roundtrip_failures),
            "retained_by_split": retained_by_split,
            "junction_limit": args.max_junctions if args.representation == "wall_graph" else None,
            "edge_limit": args.max_edges if args.representation == "wall_graph" else None,
        },
        "wall_graph_roundtrip_filter": {
            "enabled": args.representation == "wall_graph" and not args.disable_wall_graph_roundtrip_filter,
            "max_face_count_error": args.roundtrip_max_face_count_error,
            "max_uncovered_ratio": args.roundtrip_max_uncovered_ratio,
            "min_mean_iou": args.roundtrip_min_mean_iou,
        },
        "truncation_stats": truncation_stats,
        "outline_buffer": args.outline_buffer,
        "transform": "original_xy = normalized_xy * scale + [center_x, center_y]",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "inverse_transforms.json").write_text(
        json.dumps(transforms, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
