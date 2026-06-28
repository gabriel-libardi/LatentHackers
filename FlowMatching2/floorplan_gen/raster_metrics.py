from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RasterProtocol:
    resolution: int = 64
    feature_dim: int = 128
    seed: int = 42
    knn: int = 5


def _require_shapely_xy():
    try:
        from shapely import contains_xy
    except ImportError:
        contains_xy = None
    return contains_xy


def _geometry_mask(geometry, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    contains_xy = _require_shapely_xy()
    if geometry.is_empty:
        return np.zeros(xx.shape, dtype=bool)
    if contains_xy is not None:
        return contains_xy(geometry, xx, yy)

    from shapely.geometry import Point

    flat = [geometry.contains(Point(float(x), float(y))) for x, y in zip(xx.ravel(), yy.ravel())]
    return np.asarray(flat, dtype=bool).reshape(xx.shape)


def rasterize_layout(
    rooms: list[dict[str, object]],
    outline,
    type_to_id: dict[str, int],
    resolution: int = 64,
) -> np.ndarray:
    """Rasterize a typed vector layout into one-hot channels.

    Channel 0 is empty/background. Room channels use ``type_to_id``.
    """

    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    channels = max(type_to_id.values(), default=0) + 1
    min_x, min_y, max_x, max_y = outline.bounds
    if max_x <= min_x or max_y <= min_y:
        raise ValueError(f"Degenerate outline bounds: {outline.bounds}")

    xs = np.linspace(min_x, max_x, resolution, endpoint=False, dtype=np.float32)
    ys = np.linspace(max_y, min_y, resolution, endpoint=False, dtype=np.float32)
    xs = xs + (max_x - min_x) / (2.0 * resolution)
    ys = ys - (max_y - min_y) / (2.0 * resolution)
    xx, yy = np.meshgrid(xs, ys)

    labels = np.zeros((resolution, resolution), dtype=np.int16)
    outline_mask = _geometry_mask(outline, xx, yy)
    labels[~outline_mask] = 0
    for room in rooms:
        type_id = int(type_to_id.get(str(room["type"]), 0))
        if type_id <= 0:
            continue
        geometry = room["geometry"].intersection(outline)
        mask = _geometry_mask(geometry, xx, yy) & outline_mask
        labels[mask] = type_id

    raster = np.zeros((channels, resolution, resolution), dtype=np.float32)
    for channel in range(channels):
        raster[channel] = labels == channel
    return raster


def deterministic_projection(input_dim: int, feature_dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((input_dim, feature_dim)).astype(np.float32) / np.sqrt(feature_dim)).astype(np.float32)


def raster_features(rasters: np.ndarray, feature_dim: int = 128, seed: int = 42) -> np.ndarray:
    rasters = np.asarray(rasters, dtype=np.float32)
    flat = rasters.reshape(rasters.shape[0], -1)
    projection = deterministic_projection(flat.shape[1], feature_dim, seed=seed)
    return (flat @ projection).astype(np.float64)


def _sqrtm_product(cov_a: np.ndarray, cov_b: np.ndarray) -> np.ndarray:
    try:
        from scipy.linalg import sqrtm

        value = sqrtm(cov_a @ cov_b)
        if np.iscomplexobj(value):
            value = value.real
        return value
    except Exception:
        eigvals, eigvecs = np.linalg.eig(cov_a @ cov_b)
        eigvals = np.clip(eigvals.real, 0.0, None)
        eigvecs = eigvecs.real
        inv = np.linalg.pinv(eigvecs)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ inv


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray, eps: float = 1e-6) -> float:
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    if len(real_features) < 2 or len(fake_features) < 2:
        raise ValueError("FID requires at least two real and two generated feature vectors.")
    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    cov_real = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * eps
    cov_fake = np.cov(fake_features, rowvar=False) + np.eye(fake_features.shape[1]) * eps
    cov_mean = _sqrtm_product(cov_real, cov_fake)
    diff = mu_real - mu_fake
    fid = diff @ diff + np.trace(cov_real + cov_fake - 2.0 * cov_mean)
    return float(np.real(fid))


def density_coverage(real_features: np.ndarray, fake_features: np.ndarray, k: int = 5) -> dict[str, float]:
    from sklearn.metrics import pairwise_distances

    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    if len(real_features) < 2 or len(fake_features) == 0:
        return {"density": 0.0, "coverage": 0.0}
    k = max(1, min(int(k), len(real_features) - 1))
    real_to_real = pairwise_distances(real_features, real_features)
    radii = np.sort(real_to_real, axis=1)[:, k]
    real_to_fake = pairwise_distances(real_features, fake_features)
    within = real_to_fake <= radii[:, None]
    density = within.sum(axis=0).mean() / float(k)
    coverage = within.any(axis=1).mean()
    return {"density": float(density), "coverage": float(coverage)}


def nearest_real_distances(real_features: np.ndarray, fake_features: np.ndarray) -> np.ndarray:
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(np.asarray(fake_features), np.asarray(real_features))
    return distances.min(axis=1)


def distribution_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }
