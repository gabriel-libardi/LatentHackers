from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan_gen.config import CSV_PATH, FLOOR_ID_COL, ORIGINAL_SPLIT_DIR, PLAN_ID_COL
from floorplan_gen.dataset import load_area_frame
from floorplan_gen.splits import make_original_plan_splits, make_plan_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Write deterministic plan_id splits.")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-dir", default=ORIGINAL_SPLIT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    area_df = load_area_frame(args.csv)
    if args.split_dir and Path(args.split_dir).exists():
        floor_pairs = area_df[[FLOOR_ID_COL, PLAN_ID_COL]].drop_duplicates().astype(str)
        floor_to_plan = dict(zip(floor_pairs[FLOOR_ID_COL], floor_pairs[PLAN_ID_COL]))
        splits = make_original_plan_splits(
            floor_to_plan,
            split_dir=args.split_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    else:
        splits = make_plan_splits(
            area_df[PLAN_ID_COL].astype(str).unique().tolist(),
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
