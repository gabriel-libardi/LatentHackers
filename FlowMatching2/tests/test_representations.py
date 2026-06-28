import pytest

np = pytest.importorskip("numpy")
shapely_geometry = pytest.importorskip("shapely.geometry")

from floorplan_gen.representations import make_room_partition_targets, make_room_polygon_targets, sample_polygon_vertices
from floorplan_gen.tokens import RoomRecord


def signed_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def test_polygon_resampling_is_deterministic_with_ccw_orientation_and_start():
    polygon = shapely_geometry.Polygon([(1, 1), (0, 0), (1, 0), (0, 1)])

    first = sample_polygon_vertices(polygon.convex_hull, 8)
    second = sample_polygon_vertices(polygon.convex_hull, 8)

    np.testing.assert_allclose(first, second)
    assert signed_area(first) > 0
    assert tuple(first[0]) == tuple(first[np.lexsort((first[:, 1], first[:, 0]))[0]])


def test_targets_share_room_ordering_and_masks():
    rooms = [
        RoomRecord("Kitchen", shapely_geometry.box(2, 0, 3, 1)),
        RoomRecord("Bedroom", shapely_geometry.box(0, 0, 1, 1)),
    ]
    vocab = {"Bedroom": 1, "Kitchen": 2}

    polygon = make_room_polygon_targets(rooms, vocab, max_rooms=4, vertex_count=8)
    partition = make_room_partition_targets(rooms, vocab, max_rooms=4)

    assert polygon.mask.tolist() == partition.mask.tolist() == [True, True, False, False]
    assert polygon.type_ids.tolist() == partition.type_ids.tolist() == [1, 2, 0, 0]
    np.testing.assert_allclose(polygon.legacy_tokens[:, 1], partition.legacy_tokens[:, 1])
