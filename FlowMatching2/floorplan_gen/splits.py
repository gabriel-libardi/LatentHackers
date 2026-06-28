from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def stable_unit_interval(value: object, seed: int = 42) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def split_for_plan_id(
    plan_id: object,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> str:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    value = stable_unit_interval(plan_id, seed=seed)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def make_plan_splits(
    plan_ids: Iterable[object],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    splits = {"train": [], "val": [], "test": []}
    for plan_id in sorted({str(plan_id) for plan_id in plan_ids}):
        split = split_for_plan_id(
            plan_id,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        splits[split].append(plan_id)
    return splits


def load_original_floor_id_splits(split_dir: str | Path) -> dict[str, list[str]]:
    """Read the dataset-provided split from train/test struct_in filenames."""

    split_dir = Path(split_dir)
    splits: dict[str, list[str]] = {}
    for split in ("train", "test"):
        struct_dir = split_dir / split / "struct_in"
        if not struct_dir.exists():
            raise FileNotFoundError(f"Missing original split directory: {struct_dir}")
        ids = sorted(path.stem for path in struct_dir.glob("*.npy"))
        if not ids:
            raise ValueError(f"No .npy files found in {struct_dir}")
        splits[split] = ids

    overlap = set(splits["train"]) & set(splits["test"])
    if overlap:
        raise ValueError(f"Original train/test splits overlap on {len(overlap)} floor IDs.")
    return splits


def make_original_plan_splits(
    floor_to_plan_id: dict[str, str],
    split_dir: str | Path,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Map the dataset-provided floor_id split to plan_id.

    The original data only provides train/test. If ``val_fraction`` is positive,
    validation IDs are carved deterministically from the original train split;
    the original test split is never changed.
    """

    original = load_original_floor_id_splits(split_dir)
    missing = sorted((set(original["train"]) | set(original["test"])) - set(floor_to_plan_id))
    if missing:
        raise ValueError(
            f"{len(missing)} original split floor IDs are absent from the CSV; "
            f"first examples: {missing[:5]}"
        )

    train_plan_ids = sorted({floor_to_plan_id[floor_id] for floor_id in original["train"]})
    test_plan_ids = sorted({floor_to_plan_id[floor_id] for floor_id in original["test"]})
    if len(train_plan_ids) != len(original["train"]):
        raise ValueError("Original train floor IDs do not map one-to-one to plan IDs.")
    if len(test_plan_ids) != len(original["test"]):
        raise ValueError("Original test floor IDs do not map one-to-one to plan IDs.")

    if val_fraction <= 0.0:
        return {"train": train_plan_ids, "val": [], "test": test_plan_ids}
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    train: list[str] = []
    val: list[str] = []
    for plan_id in train_plan_ids:
        if stable_unit_interval(plan_id, seed=seed) < val_fraction:
            val.append(plan_id)
        else:
            train.append(plan_id)
    return {"train": train, "val": val, "test": test_plan_ids}
