from floorplan_gen.splits import (
    load_original_floor_id_splits,
    make_original_plan_splits,
    make_plan_splits,
    split_for_plan_id,
)


def test_split_for_plan_id_is_deterministic():
    first = split_for_plan_id("1054", seed=123)
    second = split_for_plan_id("1054", seed=123)
    assert first == second


def test_make_plan_splits_is_order_independent():
    plan_ids = ["3", "1", "2", "1"]
    forward = make_plan_splits(plan_ids, seed=7)
    backward = make_plan_splits(reversed(plan_ids), seed=7)
    assert forward == backward
    assert sorted(forward) == ["test", "train", "val"]


def test_original_floor_id_splits_map_to_plan_id(tmp_path):
    for split, ids in {"train": ["floor_a", "floor_b"], "test": ["floor_c"]}.items():
        struct_dir = tmp_path / split / "struct_in"
        struct_dir.mkdir(parents=True)
        for floor_id in ids:
            (struct_dir / f"{floor_id}.npy").write_bytes(b"")

    floor_splits = load_original_floor_id_splits(tmp_path)
    plan_splits = make_original_plan_splits(
        {
            "floor_a": "plan_1",
            "floor_b": "plan_2",
            "floor_c": "plan_3",
        },
        split_dir=tmp_path,
    )

    assert floor_splits == {"train": ["floor_a", "floor_b"], "test": ["floor_c"]}
    assert plan_splits == {"train": ["plan_1", "plan_2"], "val": [], "test": ["plan_3"]}
