from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .config import PAD_TYPE_ID
from .tokens import RoomRecord, make_room_tokens


@dataclass(frozen=True)
class PolygonTargets:
    vertices: np.ndarray
    presence: np.ndarray
    type_ids: np.ndarray
    mask: np.ndarray
    legacy_tokens: np.ndarray


@dataclass(frozen=True)
class PartitionTargets:
    params: np.ndarray
    presence: np.ndarray
    type_ids: np.ndarray
    mask: np.ndarray
    legacy_tokens: np.ndarray


def _polygon_for_sampling(geometry):
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        return max(geometry.geoms, key=lambda geom: geom.area)
    return geometry.convex_hull


def sample_polygon_vertices(geometry, vertex_count: int) -> np.ndarray:
    """Sample a fixed-length ordered exterior ring for irregular rooms."""

    if vertex_count < 3:
        raise ValueError("vertex_count must be at least 3.")
    polygon = _polygon_for_sampling(geometry)
    try:
        from shapely.geometry.polygon import orient

        polygon = orient(polygon, sign=1.0)
    except Exception:
        pass
    ring = polygon.exterior
    distances = np.linspace(0.0, ring.length, vertex_count, endpoint=False)
    points = np.asarray(
        [[ring.interpolate(float(distance)).x, ring.interpolate(float(distance)).y] for distance in distances],
        dtype=np.float32,
    )
    start = np.lexsort((points[:, 1], points[:, 0]))[0]
    return np.roll(points, -int(start), axis=0)


def make_room_polygon_targets(
    rooms: Iterable[RoomRecord],
    type_to_id: Mapping[str, int],
    max_rooms: int,
    vertex_count: int,
) -> PolygonTargets:
    rooms = sorted(
        list(rooms),
        key=lambda room: (room.room_type, room.geometry.centroid.x, room.geometry.centroid.y),
    )
    vertices = np.zeros((max_rooms, vertex_count, 2), dtype=np.float32)
    presence = np.zeros((max_rooms,), dtype=np.float32)
    type_ids = np.zeros((max_rooms,), dtype=np.int64)
    mask = np.zeros((max_rooms,), dtype=bool)
    legacy_tokens, _ = make_room_tokens(rooms, type_to_id, max_rooms)

    for index, room in enumerate(rooms[:max_rooms]):
        vertices[index] = sample_polygon_vertices(room.geometry, vertex_count)
        presence[index] = 1.0
        type_ids[index] = int(type_to_id.get(room.room_type, PAD_TYPE_ID))
        mask[index] = True

    return PolygonTargets(
        vertices=vertices,
        presence=presence,
        type_ids=type_ids,
        mask=mask,
        legacy_tokens=legacy_tokens,
    )


def make_room_partition_targets(
    rooms: Iterable[RoomRecord],
    type_to_id: Mapping[str, int],
    max_rooms: int,
) -> PartitionTargets:
    """Create compact partition targets: centroid x/y, sqrt area, aspect."""

    rooms = sorted(
        list(rooms),
        key=lambda room: (room.room_type, room.geometry.centroid.x, room.geometry.centroid.y),
    )
    params = np.zeros((max_rooms, 4), dtype=np.float32)
    presence = np.zeros((max_rooms,), dtype=np.float32)
    type_ids = np.zeros((max_rooms,), dtype=np.int64)
    mask = np.zeros((max_rooms,), dtype=bool)
    legacy_tokens, _ = make_room_tokens(rooms, type_to_id, max_rooms)
    for index, room in enumerate(rooms[:max_rooms]):
        min_x, min_y, max_x, max_y = room.geometry.bounds
        width = max(float(max_x - min_x), 1e-4)
        height = max(float(max_y - min_y), 1e-4)
        centroid = room.geometry.centroid
        params[index] = np.asarray(
            [
                float(centroid.x),
                float(centroid.y),
                float(np.sqrt(max(room.geometry.area, 1e-6))),
                float(np.log(width / height)),
            ],
            dtype=np.float32,
        )
        presence[index] = 1.0
        type_ids[index] = int(type_to_id.get(room.room_type, PAD_TYPE_ID))
        mask[index] = True
    return PartitionTargets(params=params, presence=presence, type_ids=type_ids, mask=mask, legacy_tokens=legacy_tokens)


def legacy_tokens_to_vertices(room_tokens: np.ndarray, vertex_count: int) -> np.ndarray:
    room_tokens = np.asarray(room_tokens, dtype=np.float32)
    vertices = np.zeros((room_tokens.shape[0], vertex_count, 2), dtype=np.float32)
    corners = np.asarray(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )
    distances = np.linspace(0.0, 4.0, vertex_count, endpoint=False)
    box_points = []
    for distance in distances:
        edge = int(distance) % 4
        alpha = distance - int(distance)
        box_points.append((1.0 - alpha) * corners[edge] + alpha * corners[(edge + 1) % 4])
    box_points = np.asarray(box_points, dtype=np.float32)
    for index, token in enumerate(room_tokens):
        cx, cy, width, height = token[2], token[3], abs(token[4]), abs(token[5])
        vertices[index, :, 0] = cx + box_points[:, 0] * width
        vertices[index, :, 1] = cy + box_points[:, 1] * height
    return vertices
