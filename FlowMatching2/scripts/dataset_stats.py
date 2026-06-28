from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan_gen.config import (
    CSV_PATH,
    DEFAULT_MAX_ROOMS,
    ENTITY_TYPE_COL,
    FLOOR_ID_COL,
    GEOM_COL,
    ORIGINAL_SPLIT_DIR,
    PLAN_ID_COL,
    ROOM_TYPE_COL,
)
from floorplan_gen.dataset import load_area_frame, load_csv_frame
from floorplan_gen.splits import make_original_plan_splits, make_plan_splits
from floorplan_gen.tokens import build_room_type_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MSD floor-plan dataset statistics.")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--split-dir", default=ORIGINAL_SPLIT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    full_df = load_csv_frame(args.csv)
    area_df = full_df[full_df[ENTITY_TYPE_COL] == "area"].copy()
    room_counts = Counter(area_df[ROOM_TYPE_COL].astype(str).tolist())
    rooms_per_plan = area_df.groupby(PLAN_ID_COL).size()
    if args.split_dir and Path(args.split_dir).exists():
        floor_pairs = area_df[[FLOOR_ID_COL, PLAN_ID_COL]].drop_duplicates().astype(str)
        floor_to_plan = dict(zip(floor_pairs[FLOOR_ID_COL], floor_pairs[PLAN_ID_COL]))
        split_ids = make_original_plan_splits(
            floor_to_plan,
            split_dir=args.split_dir,
            val_fraction=args.val_fraction,
        )
        split_source = "original_floor_id"
    else:
        split_ids = make_plan_splits(area_df[PLAN_ID_COL].astype(str).unique().tolist())
        split_source = "hash_plan_id"
    geom_tags = Counter(str(value).split(" ", 1)[0] for value in full_df[GEOM_COL].tolist())
    stats = {
        "rows": int(len(full_df)),
        "area_rows": int(len(area_df)),
        "columns": list(full_df.columns),
        "entity_type_counts": dict(Counter(full_df[ENTITY_TYPE_COL].astype(str).tolist()).most_common()),
        "plan_count": int(area_df[PLAN_ID_COL].nunique()),
        "geometry_tags": dict(sorted(geom_tags.items())),
        "roomtype_counts": dict(room_counts.most_common()),
        "room_type_vocab": build_room_type_vocab(area_df[ROOM_TYPE_COL].astype(str).tolist()),
        "rooms_per_plan": {
            "min": int(rooms_per_plan.min()),
            "max": int(rooms_per_plan.max()),
            "mean": float(rooms_per_plan.mean()),
            "median": float(rooms_per_plan.median()),
            "p90": float(rooms_per_plan.quantile(0.90)),
            "p95": float(rooms_per_plan.quantile(0.95)),
            "p99": float(rooms_per_plan.quantile(0.99)),
        },
        "max_rooms_default": DEFAULT_MAX_ROOMS,
        "default_truncation": {
            "truncated_plans": int((rooms_per_plan > DEFAULT_MAX_ROOMS).sum()),
            "rooms_lost": int((rooms_per_plan - DEFAULT_MAX_ROOMS).clip(lower=0).sum()),
        },
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "split_source": split_source,
    }

    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    print(f"rows: {stats['rows']}")
    print(f"area rows: {stats['area_rows']}")
    print(f"entity types: {stats['entity_type_counts']}")
    print(f"plans: {stats['plan_count']}")
    print(f"geometry tags: {stats['geometry_tags']}")
    print(f"rooms per plan: {stats['rooms_per_plan']}")
    print(f"default max rooms: {stats['max_rooms_default']} -> {stats['default_truncation']}")
    print(f"splits: {stats['split_counts']}")
    print(f"split source: {stats['split_source']}")
    print("room types:")
    for label, count in stats["roomtype_counts"].items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
