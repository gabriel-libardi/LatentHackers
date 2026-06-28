import pytest

np = pytest.importorskip("numpy")
shapely_geometry = pytest.importorskip("shapely.geometry")
shapely_errors = pytest.importorskip("shapely.errors")

import floorplan_gen.decoding as decoding
from floorplan_gen.decoding import (
    GeneratedRoom,
    decode_room_geometry,
    decode_room_tokens,
    partition_geometry_to_cells,
    repair_geometry,
    repair_layout,
    select_active_queries,
)


def test_decode_repairs_clips_and_fills_outline():
    outline = shapely_geometry.box(-1, -1, 1, 1)
    tokens = np.asarray(
        [
            [0.9, 1, 0.0, 0.0, 3.0, 3.0],
            [0.1, 2, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    rooms = decode_room_tokens(tokens, {1: "Bedroom", 2: "Kitchen"}, outline)

    assert rooms
    assert all(room["geometry"].within(outline) or room["geometry"].equals(outline) for room in rooms)
    assert all(room["geometry"].is_valid for room in rooms)


def test_repair_geometry_returns_valid_geometry():
    invalid = shapely_geometry.Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])

    repaired = repair_geometry(invalid)

    assert repaired.is_valid
    assert not repaired.is_empty


def test_repair_geometry_keeps_only_polygonal_components_from_geometry_collection():
    collection = shapely_geometry.GeometryCollection(
        [
            shapely_geometry.LineString([(0, 0), (1, 1)]),
            shapely_geometry.Point(5, 5),
            shapely_geometry.box(0, 0, 1, 1),
        ]
    )

    repaired = repair_geometry(collection)

    assert repaired.geom_type == "Polygon"
    assert repaired.area == pytest.approx(1.0)


def test_repair_geometry_handles_invalid_hole_polygon():
    invalid_hole = shapely_geometry.Polygon(
        shell=[(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
        holes=[[(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)]],
    )

    repaired = repair_geometry(invalid_hole)

    assert repaired.is_valid
    assert not repaired.is_empty


def test_repair_geometry_handles_empty_geometry():
    repaired = repair_geometry(shapely_geometry.GeometryCollection())

    assert repaired.is_empty


def test_decode_room_geometry_uses_polygon_vertices():
    outline = shapely_geometry.box(-1, -1, 1, 1)
    geometry = np.asarray(
        [
            [[-0.8, -0.8], [0.3, -0.8], [0.3, -0.2], [-0.2, -0.2], [-0.2, 0.5], [-0.8, 0.5]],
        ],
        dtype=np.float32,
    )
    presence = np.asarray([0.9], dtype=np.float32)
    type_ids = np.asarray([1], dtype=np.int64)

    rooms = decode_room_geometry(geometry, presence, type_ids, {1: "Bedroom"}, outline, fill_gaps=False)

    assert len(rooms) == 1
    assert rooms[0]["type"] == "Bedroom"
    assert rooms[0]["geometry"].is_valid
    assert rooms[0]["geometry"].within(outline)


def test_gap_filling_assigns_gap_to_neighboring_room():
    outline = shapely_geometry.box(0, 0, 2, 1)
    rooms = [GeneratedRoom("Bedroom", shapely_geometry.box(0, 0, 1, 1), 1.0)]

    repaired = repair_layout(rooms, outline, fill_gaps=True)

    assert len(repaired) == 1
    assert repaired[0].room_type == "Bedroom"
    assert repaired[0].geometry.equals(outline)


def test_repair_layout_handles_overlapping_and_self_intersecting_rooms():
    outline = shapely_geometry.box(0, 0, 2, 2)
    bowtie = shapely_geometry.Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    rooms = [
        GeneratedRoom("Bedroom", shapely_geometry.box(0, 0, 1.5, 2), 1.0),
        GeneratedRoom("Kitchen", shapely_geometry.box(0.5, 0, 2, 2), 0.9),
        GeneratedRoom("Bathroom", bowtie, 0.8),
    ]

    repaired = repair_layout(rooms, outline, fill_gaps=True)

    assert repaired
    assert all(room.geometry.is_valid for room in repaired)
    assert all(room.geometry.intersection(outline).area == pytest.approx(room.geometry.area) for room in repaired)


def test_repair_layout_returns_partial_result_when_gap_filling_fails(monkeypatch):
    outline = shapely_geometry.box(0, 0, 2, 1)
    rooms = [GeneratedRoom("Bedroom", shapely_geometry.box(0, 0, 1, 1), 1.0)]
    original_safe_difference = decoding.safe_difference

    def raising_safe_difference(first, second):
        if first.equals(outline):
            raise shapely_errors.GEOSException("forced gap failure")
        return original_safe_difference(first, second)

    monkeypatch.setattr(decoding, "safe_difference", raising_safe_difference)

    repaired = repair_layout(rooms, outline, fill_gaps=True)

    assert len(repaired) == 1
    assert repaired[0].room_type == "Bedroom"


def test_select_active_queries_keeps_exact_top_k():
    presence = np.asarray([0.2, 0.9, 0.5, 0.7], dtype=np.float32)
    type_ids = np.asarray([1, 1, 0, 2], dtype=np.int64)
    count_logits = np.asarray([-5.0, -2.0, 4.0, -1.0, -3.0], dtype=np.float32)

    active = select_active_queries(presence, type_ids, count_logits=count_logits, mode="predicted_count")

    assert active.tolist() == [False, True, False, True]


def test_inactive_queries_do_not_create_partition_cells():
    outline = shapely_geometry.box(0, 0, 1, 1)
    geometry = np.asarray([[0.25, 0.5, 0.5, 0.0], [0.75, 0.5, 0.5, 0.0]], dtype=np.float32)
    presence = np.asarray([1.0, 1.0], dtype=np.float32)
    type_ids = np.asarray([1, 2], dtype=np.int64)

    rooms = partition_geometry_to_cells(
        geometry,
        presence,
        type_ids,
        {1: "Bedroom", 2: "Kitchen"},
        outline,
        active_mask=np.asarray([False, False]),
    )

    assert rooms == []
