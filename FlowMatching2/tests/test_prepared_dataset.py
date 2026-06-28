import json

import pytest

np = pytest.importorskip("numpy")

from floorplan_gen.prepared_dataset import PreparedFloorPlanDataset


def test_prepared_dataset_room_count_targets_and_type_mapping(tmp_path):
    metadata = {"type_to_id": {"Bedroom": 1, "Kitchen": 2}, "max_rooms": 3}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split in ["train", "val"]:
        np.savez_compressed(
            tmp_path / f"{split}.npz",
            plan_ids=np.asarray(["p1"]),
            boundary_points=np.zeros((1, 4, 2), dtype=np.float32),
            room_tokens=np.zeros((1, 3, 6), dtype=np.float32),
            room_geometry=np.zeros((1, 3, 4), dtype=np.float32),
            room_vertices=np.zeros((1, 3, 4, 2), dtype=np.float32),
            room_presence=np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
            room_type_ids=np.asarray([[1, 2, 0]], dtype=np.int64),
            room_masks=np.asarray([[True, True, False]], dtype=bool),
        )

    dataset = PreparedFloorPlanDataset(tmp_path, "train")
    item = dataset[0]

    assert item["room_count"].item() == 2
    assert dataset.metadata["type_to_id"] == {"Bedroom": 1, "Kitchen": 2}
