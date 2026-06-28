from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PlanTransform:
    """Forward normalization and inverse metadata for one plan."""

    center_x: float
    center_y: float
    scale: float

    def to_dict(self) -> dict[str, float]:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "PlanTransform":
        return cls(
            center_x=float(values["center_x"]),
            center_y=float(values["center_y"]),
            scale=float(values["scale"]),
        )


def _require_shapely():
    try:
        import shapely  # noqa: F401
        from shapely import wkt
        from shapely.affinity import affine_transform
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError(
            "Shapely is required for geometry processing. Install dependencies from "
            "requirements.txt in the environment used for data preparation."
        ) from exc
    return wkt, affine_transform, unary_union


def load_wkt_geometry(value: str):
    """Parse one WKT geometry string."""

    wkt, _, _ = _require_shapely()
    return wkt.loads(value)


def load_wkt_geometries(values: Iterable[str]) -> list:
    """Parse WKT geometry strings into Shapely geometries."""

    wkt, _, _ = _require_shapely()
    return [wkt.loads(value) for value in values]


def build_apartment_outline(room_geometries: Iterable, buffer_distance: float = 0.3):
    """Build the apartment outline with buffer-union-buffer."""

    _, _, unary_union = _require_shapely()
    buffered = [geom.buffer(buffer_distance) for geom in room_geometries if not geom.is_empty]
    if not buffered:
        raise ValueError("Cannot build an outline from an empty room geometry list.")
    outline = unary_union(buffered).buffer(-buffer_distance)
    if outline.is_empty:
        raise ValueError("Buffer-union-buffer produced an empty outline.")
    return outline


def plan_transform_from_bounds(bounds: tuple[float, float, float, float]) -> PlanTransform:
    """Create a transform that maps the longest plan side to length 2."""

    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height) / 2.0
    if scale <= 0:
        raise ValueError(f"Degenerate plan bounds: {bounds}")
    return PlanTransform(
        center_x=(min_x + max_x) / 2.0,
        center_y=(min_y + max_y) / 2.0,
        scale=scale,
    )


def normalize_geometry(geometry, transform: PlanTransform | None = None):
    """Normalize a geometry to a centered, meter-scale-free coordinate frame."""

    _, affine_transform, _ = _require_shapely()
    if transform is None:
        transform = plan_transform_from_bounds(geometry.bounds)
    matrix = [
        1.0 / transform.scale,
        0.0,
        0.0,
        1.0 / transform.scale,
        -transform.center_x / transform.scale,
        -transform.center_y / transform.scale,
    ]
    return affine_transform(geometry, matrix), transform


def denormalize_geometry(geometry, transform: PlanTransform):
    """Map a normalized geometry back to original plan coordinates."""

    _, affine_transform, _ = _require_shapely()
    matrix = [
        transform.scale,
        0.0,
        0.0,
        transform.scale,
        transform.center_x,
        transform.center_y,
    ]
    return affine_transform(geometry, matrix)


def normalize_xy(x: float, y: float, transform: PlanTransform) -> tuple[float, float]:
    return (
        (float(x) - transform.center_x) / transform.scale,
        (float(y) - transform.center_y) / transform.scale,
    )


def denormalize_xy(x: float, y: float, transform: PlanTransform) -> tuple[float, float]:
    return (
        float(x) * transform.scale + transform.center_x,
        float(y) * transform.scale + transform.center_y,
    )


def sample_boundary_points(geometry, num_points: int):
    """Sample evenly spaced points from a geometry boundary."""

    import numpy as np

    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    boundary = geometry.boundary
    if boundary.is_empty or boundary.length <= 0:
        raise ValueError("Cannot sample boundary points from an empty boundary.")

    distances = np.linspace(0.0, boundary.length, num_points, endpoint=False)
    points = [boundary.interpolate(float(distance)) for distance in distances]
    return np.asarray([[point.x, point.y] for point in points], dtype=np.float32)

