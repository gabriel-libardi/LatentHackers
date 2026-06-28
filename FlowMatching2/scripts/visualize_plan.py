from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))

from floorplan_gen.config import CSV_PATH, GEOM_COL, PLAN_ID_COL, ROOM_TYPE_COL
from floorplan_gen.dataset import load_area_frame
from floorplan_gen.geometry import (
    build_apartment_outline,
    load_wkt_geometries,
    normalize_geometry,
    sample_boundary_points,
)
from floorplan_gen.tokens import RoomRecord, build_room_type_vocab, make_room_tokens


def iter_polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms
    else:
        for part in getattr(geometry, "geoms", []):
            yield from iter_polygons(part)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one MSD plan outline and room tokens.")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--output", default=None, help="PNG path. If omitted, show an interactive window.")
    parser.add_argument("--boundary-points", type=int, default=256)
    parser.add_argument("--max-rooms", type=int, default=128)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    area_df = load_area_frame(args.csv)
    area_df[PLAN_ID_COL] = area_df[PLAN_ID_COL].astype(str)
    plan_df = area_df[area_df[PLAN_ID_COL] == str(args.plan_id)].copy()
    if plan_df.empty:
        raise ValueError(f"No area rows found for plan_id={args.plan_id}.")

    geometries = load_wkt_geometries(plan_df[GEOM_COL].tolist())
    outline = build_apartment_outline(geometries)
    normalized_outline, transform = normalize_geometry(outline)
    normalized_rooms = [normalize_geometry(geometry, transform)[0] for geometry in geometries]
    points = sample_boundary_points(normalized_outline, args.boundary_points)
    vocab = build_room_type_vocab(area_df[ROOM_TYPE_COL].astype(str).tolist())
    rooms = [
        RoomRecord(room_type=str(room_type), geometry=geometry)
        for room_type, geometry in zip(plan_df[ROOM_TYPE_COL].tolist(), normalized_rooms)
    ]
    tokens, mask = make_room_tokens(rooms, vocab, args.max_rooms)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    for geom in normalized_rooms:
        for polygon in iter_polygons(geom):
            x, y = polygon.exterior.xy
            ax_left.fill(x, y, alpha=0.45, linewidth=0.8, edgecolor="white")
    for polygon in iter_polygons(normalized_outline):
        x, y = polygon.exterior.xy
        ax_left.plot(x, y, color="black", linewidth=1.8)
    ax_left.scatter(points[:, 0], points[:, 1], s=8, color="black")
    ax_left.set_title(f"Plan {args.plan_id}: normalized rooms and outline")
    ax_left.set_aspect("equal")
    ax_left.axis("off")

    present = tokens[mask]
    ax_right.scatter(present[:, 2], present[:, 3], s=28, c=present[:, 1], cmap="tab10")
    for token in present:
        cx, cy, width, height = token[2], token[3], token[4], token[5]
        rect = plt.Rectangle(
            (cx - width / 2.0, cy - height / 2.0),
            width,
            height,
            fill=False,
            linewidth=0.8,
            alpha=0.7,
        )
        ax_right.add_patch(rect)
    ax_right.set_title("Room tokens: centroid and bbox")
    ax_right.set_aspect("equal")
    ax_right.axis("off")

    fig.tight_layout()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
