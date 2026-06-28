import pytest

np = pytest.importorskip("numpy")
shapely_geometry = pytest.importorskip("shapely.geometry")

from floorplan_gen.raster_metrics import (
    density_coverage,
    frechet_distance,
    raster_features,
    rasterize_layout,
)


def test_rasterize_layout_shape_and_channels():
    outline = shapely_geometry.box(0, 0, 1, 1)
    rooms = [{"type": "Bedroom", "geometry": shapely_geometry.box(0, 0, 0.5, 1)}]

    raster = rasterize_layout(rooms, outline, {"Bedroom": 1}, resolution=16)

    assert raster.shape == (2, 16, 16)
    assert raster[1].sum() > 0


def test_fid_is_near_zero_for_identical_features():
    outline = shapely_geometry.box(0, 0, 1, 1)
    rooms = [{"type": "Bedroom", "geometry": outline}]
    rasters = np.asarray([rasterize_layout(rooms, outline, {"Bedroom": 1}, resolution=8) for _ in range(3)])
    features = raster_features(rasters, feature_dim=8, seed=42)

    fid = frechet_distance(features, features)

    assert fid == pytest.approx(0.0, abs=1e-6)


def test_density_coverage_positive_for_matching_features():
    real = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    fake = real.copy()

    metrics = density_coverage(real, fake, k=1)

    assert metrics["density"] > 0
    assert metrics["coverage"] == pytest.approx(1.0)
