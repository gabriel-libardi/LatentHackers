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

import torch

from tqdm import tqdm

from floorplan_gen.config import CSV_PATH, GEOM_COL, PLAN_ID_COL
from floorplan_gen.dataset import load_area_frame
from floorplan_gen.decoding import FloorPlanGenerator
from floorplan_gen.evaluation import diagnostic_score, evaluate_layout, outline_response, repair_change_ratio, sample_diversity
from floorplan_gen.geometry import build_apartment_outline, load_wkt_geometries
from floorplan_gen.models import ConditionalRoomFlow
from floorplan_gen.plotting import room_color


def iter_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from iter_polygons(part)


def load_outline(csv_path: str, plan_id: str):
    area_df = load_area_frame(csv_path)
    area_df[PLAN_ID_COL] = area_df[PLAN_ID_COL].astype(str)
    plan_df = area_df[area_df[PLAN_ID_COL] == str(plan_id)]
    if plan_df.empty:
        raise ValueError(f"No plan_id={plan_id} in {csv_path}")
    return build_apartment_outline(load_wkt_geometries(plan_df[GEOM_COL].tolist()))


def save_plot(path: Path, outline, rooms: list[dict[str, object]], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for room in rooms:
        for polygon in iter_polygons(room["geometry"]):
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.55, linewidth=0.8, edgecolor="white", color=room_color(room["type"]))
    for polygon in iter_polygons(outline):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color="black", linewidth=1.8)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample generated room layouts for one outline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--compare-plan-id", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-mode", choices=["predicted_count", "threshold", "fixed_topk"], default="predicted_count")
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--fixed-room-count", type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = ConditionalRoomFlow(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    type_to_id = checkpoint["metadata"]["type_to_id"]
    type_id_to_label = {int(value): key for key, value in type_to_id.items()}
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
    outline = load_outline(args.csv, args.plan_id)
    output_dir = Path(args.output_dir)
    metrics = []
    samples = []
    for index in tqdm(range(args.num_samples), desc="sampling layouts", unit="sample"):
        result = generator.generate_with_diagnostics(outline, seed=args.seed + index)
        rooms = result["rooms"]
        samples.append(rooms)
        raw_metrics = evaluate_layout(result["raw_rooms"], outline)
        repaired_metrics = evaluate_layout(rooms, outline)
        repair_change = repair_change_ratio(result["raw_rooms"], rooms, outline)
        row = {
            "sample": index,
            "raw": raw_metrics,
            "repaired": repaired_metrics,
            "repair_change_ratio": repair_change,
            "diagnostic_score": diagnostic_score(raw_metrics, repaired_metrics, repair_change),
            "diagnostics": result.get("diagnostics", {}),
        }
        metrics.append(row)
        save_plot(output_dir / f"sample_{index:03d}.png", outline, rooms, f"Plan {args.plan_id} sample {index}")
    summary = {"samples": metrics, "diversity": sample_diversity(samples)}
    if args.compare_plan_id is not None:
        compare_outline = load_outline(args.csv, args.compare_plan_id)
        first = generator.generate(outline, seed=args.seed)
        second = generator.generate(compare_outline, seed=args.seed)
        summary["outline_response"] = {
            "plan_id": str(args.plan_id),
            "compare_plan_id": str(args.compare_plan_id),
            **outline_response(first, second),
        }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
