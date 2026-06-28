from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_BOUNDARY_POINTS,
    DEFAULT_MAX_ROOMS,
    DEFAULT_SEED,
    ENTITY_TYPE_COL,
    FLOOR_ID_COL,
    GEOM_COL,
    ORIGINAL_SPLIT_DIR,
    OUTLINE_BUFFER_METERS,
    PLAN_ID_COL,
    ROOM_ENTITY_TYPE,
    ROOM_TYPE_COL,
)
from .geometry import (
    build_apartment_outline,
    load_wkt_geometries,
    normalize_geometry,
    sample_boundary_points,
)
from .splits import make_original_plan_splits, make_plan_splits
from .tokens import RoomRecord, build_room_type_vocab, make_room_tokens


@dataclass(frozen=True)
class PlanExample:
    plan_id: str
    boundary_points: Any
    room_tokens: Any
    room_mask: Any
    inverse_transform: dict[str, float]


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to load the MSD CSV.") from exc
    return pd


def load_area_frame(csv_path: str | Path):
    pd = _require_pandas()
    df = pd.read_csv(csv_path)
    required = {PLAN_ID_COL, ENTITY_TYPE_COL, GEOM_COL, ROOM_TYPE_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df[df[ENTITY_TYPE_COL] == ROOM_ENTITY_TYPE].copy()


def load_csv_frame(csv_path: str | Path):
    pd = _require_pandas()
    return pd.read_csv(csv_path)


def room_vocab_from_csv(csv_path: str | Path) -> dict[str, int]:
    area_df = load_area_frame(csv_path)
    return build_room_type_vocab(area_df[ROOM_TYPE_COL].dropna().astype(str).tolist())


def plan_ids_by_split(
    csv_path: str | Path,
    split: str,
    val_fraction: float = 0.0,
    test_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
    split_dir: str | Path | None = ORIGINAL_SPLIT_DIR,
) -> list[str]:
    area_df = load_area_frame(csv_path)
    if split_dir and Path(split_dir).exists():
        floor_to_plan_id = _floor_to_plan_id(area_df)
        splits = make_original_plan_splits(
            floor_to_plan_id,
            split_dir=split_dir,
            val_fraction=val_fraction,
            seed=seed,
        )
    else:
        splits = make_plan_splits(
            area_df[PLAN_ID_COL].dropna().astype(str).unique().tolist(),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    return splits[split]


def _floor_to_plan_id(area_df) -> dict[str, str]:
    required = {FLOOR_ID_COL, PLAN_ID_COL}
    missing = sorted(required - set(area_df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns for original split mapping: {missing}")
    pairs = area_df[[FLOOR_ID_COL, PLAN_ID_COL]].drop_duplicates().astype(str)
    if pairs[FLOOR_ID_COL].duplicated().any():
        raise ValueError("Expected each floor_id to map to exactly one plan_id.")
    return dict(zip(pairs[FLOOR_ID_COL], pairs[PLAN_ID_COL]))


class FloorPlanDataset:
    """PyTorch dataset for boundary-conditioned room-token baselines."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        num_boundary_points: int = DEFAULT_BOUNDARY_POINTS,
        max_rooms: int = DEFAULT_MAX_ROOMS,
        type_to_id: dict[str, int] | None = None,
        val_fraction: float = 0.0,
        test_fraction: float = 0.1,
        seed: int = DEFAULT_SEED,
        outline_buffer: float = OUTLINE_BUFFER_METERS,
        split_dir: str | Path | None = ORIGINAL_SPLIT_DIR,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("PyTorch is required to use FloorPlanDataset.") from exc

        self.csv_path = Path(csv_path)
        self.split = split
        self.num_boundary_points = num_boundary_points
        self.max_rooms = max_rooms
        self.outline_buffer = outline_buffer

        area_df = load_area_frame(self.csv_path)
        area_df[PLAN_ID_COL] = area_df[PLAN_ID_COL].astype(str)
        area_df[ROOM_TYPE_COL] = area_df[ROOM_TYPE_COL].astype(str)

        self.type_to_id = type_to_id or build_room_type_vocab(area_df[ROOM_TYPE_COL].tolist())
        if split_dir and Path(split_dir).exists():
            splits = make_original_plan_splits(
                _floor_to_plan_id(area_df),
                split_dir=split_dir,
                val_fraction=val_fraction,
                seed=seed,
            )
        else:
            splits = make_plan_splits(
                area_df[PLAN_ID_COL].unique().tolist(),
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )
        selected_plan_ids = set(splits[split])
        self.area_df = area_df[area_df[PLAN_ID_COL].isin(selected_plan_ids)].copy()
        self.plan_ids = sorted(self.area_df[PLAN_ID_COL].unique().tolist())

    def __len__(self) -> int:
        return len(self.plan_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        plan_id = self.plan_ids[index]
        plan_df = self.area_df[self.area_df[PLAN_ID_COL] == plan_id]
        geometries = load_wkt_geometries(plan_df[GEOM_COL].tolist())
        outline = build_apartment_outline(geometries, buffer_distance=self.outline_buffer)
        normalized_outline, transform = normalize_geometry(outline)
        normalized_rooms = [normalize_geometry(geometry, transform)[0] for geometry in geometries]
        rooms = [
            RoomRecord(room_type=str(room_type), geometry=geometry)
            for room_type, geometry in zip(plan_df[ROOM_TYPE_COL].tolist(), normalized_rooms)
        ]
        tokens, mask = make_room_tokens(rooms, self.type_to_id, self.max_rooms)
        boundary_points = sample_boundary_points(normalized_outline, self.num_boundary_points)

        return {
            "plan_id": plan_id,
            "boundary_points": torch.from_numpy(boundary_points),
            "room_tokens": torch.from_numpy(tokens),
            "room_mask": torch.from_numpy(mask),
            "inverse_transform": transform.to_dict(),
        }
