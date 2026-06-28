import pytest

np = pytest.importorskip("numpy")
shapely_wkt = pytest.importorskip("shapely.wkt")

from floorplan_gen.geometry import (
    denormalize_geometry,
    normalize_geometry,
    sample_boundary_points,
)


def test_normalize_denormalize_round_trip_area():
    geom = shapely_wkt.loads("POLYGON ((2 2, 6 2, 6 4, 2 4, 2 2))")

    normalized, transform = normalize_geometry(geom)
    restored = denormalize_geometry(normalized, transform)

    assert restored.symmetric_difference(geom).area < 1e-8


def test_sample_boundary_points_shape():
    geom = shapely_wkt.loads("POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))")

    points = sample_boundary_points(geom, 8)

    assert points.shape == (8, 2)
    assert points.dtype == np.float32

