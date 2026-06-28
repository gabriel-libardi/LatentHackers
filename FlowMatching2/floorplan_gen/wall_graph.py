from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import PAD_TYPE_ID


@dataclass(frozen=True)
class WallGraphTargets:
    junction_xy: np.ndarray
    junction_mask: np.ndarray
    edge_index: np.ndarray
    edge_mask: np.ndarray
    edge_is_exterior: np.ndarray
    edge_room_ids: np.ndarray
    room_types: np.ndarray
    room_mask: np.ndarray
    stats: dict[str, float | int | str]


def _require_shapely():
    try:
        from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon
        from shapely.ops import polygonize, unary_union
        try:
            from shapely.validation import make_valid
        except ImportError:
            make_valid = None
    except ImportError as exc:
        raise ImportError("Shapely is required for wall-graph conversion.") from exc
    return GeometryCollection, LineString, MultiLineString, Point, Polygon, polygonize, unary_union, make_valid


def _repair_polygon(geometry):
    GeometryCollection, _, _, _, _, _, _, make_valid = _require_shapely()
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    try:
        repaired = make_valid(geometry) if make_valid is not None else geometry.buffer(0)
    except Exception:
        repaired = geometry.buffer(0)
    if repaired.is_empty:
        return GeometryCollection()
    if repaired.geom_type == "Polygon":
        return repaired
    if repaired.geom_type == "MultiPolygon":
        return max(repaired.geoms, key=lambda geom: geom.area)
    if hasattr(repaired, "geoms"):
        polygons = [part for part in repaired.geoms if part.geom_type == "Polygon" and not part.is_empty]
        return max(polygons, key=lambda geom: geom.area) if polygons else GeometryCollection()
    return repaired.convex_hull


def _snap_key(point: tuple[float, float], tolerance: float) -> tuple[int, int]:
    return (int(round(float(point[0]) / tolerance)), int(round(float(point[1]) / tolerance)))


def _segment_t(point, start, end, eps: float = 1e-8) -> float | None:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    direction = end - start
    length2 = float(np.dot(direction, direction))
    if length2 <= eps:
        return None
    t = float(np.dot(point - start, direction) / length2)
    if t <= eps or t >= 1.0 - eps:
        return None
    projected = start + t * direction
    if float(np.linalg.norm(projected - point)) > max(1e-7, eps):
        return None
    return t


def _ring_edges(polygon) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    coords = list(polygon.exterior.coords)
    return [
        ((float(coords[i][0]), float(coords[i][1])), (float(coords[i + 1][0]), float(coords[i + 1][1])))
        for i in range(len(coords) - 1)
    ]


def _component_count(num_nodes: int, edges: list[tuple[int, int]]) -> int:
    if num_nodes == 0:
        return 0
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return len({find(i) for i in range(num_nodes)})


def convert_rooms_to_wall_graph(
    rooms,
    type_to_id: Mapping[str, int],
    max_junctions: int,
    max_edges: int,
    max_rooms: int,
    snap_tolerance: float = 1e-3,
    simplify_tolerance: float | None = None,
    outline=None,
) -> WallGraphTargets:
    """Convert normalized room polygons to a padded shared wall graph."""

    _, _, _, _, _, polygonize, unary_union, _ = _require_shapely()
    clean_rooms = []
    for room_id, room in enumerate(rooms[:max_rooms]):
        polygon = _repair_polygon(room.geometry)
        if simplify_tolerance is not None and simplify_tolerance > 0 and not polygon.is_empty:
            polygon = _repair_polygon(polygon.simplify(float(simplify_tolerance), preserve_topology=True))
        if polygon.is_empty or polygon.area <= 1e-8:
            continue
        clean_rooms.append((room_id, room.room_type, polygon))
    if not clean_rooms:
        raise ValueError("No valid room polygons for wall graph conversion.")

    raw_segments: list[dict[str, object]] = []
    point_by_key: dict[tuple[int, int], list[float]] = {}
    point_count: dict[tuple[int, int], int] = {}

    def snap(point):
        key = _snap_key(point, snap_tolerance)
        if key not in point_by_key:
            point_by_key[key] = [float(point[0]), float(point[1])]
            point_count[key] = 1
        else:
            count = point_count[key]
            point_by_key[key][0] = (point_by_key[key][0] * count + float(point[0])) / (count + 1)
            point_by_key[key][1] = (point_by_key[key][1] * count + float(point[1])) / (count + 1)
            point_count[key] = count + 1
        return key

    for room_id, room_type, polygon in clean_rooms:
        for start, end in _ring_edges(polygon):
            start_key = snap(start)
            end_key = snap(end)
            if start_key != end_key:
                raw_segments.append({"a": start_key, "b": end_key, "rooms": {int(room_id)}, "exterior": False})

    if outline is not None:
        outline_polygon = _repair_polygon(outline)
        if simplify_tolerance is not None and simplify_tolerance > 0 and not outline_polygon.is_empty:
            outline_polygon = _repair_polygon(outline_polygon.simplify(float(simplify_tolerance), preserve_topology=True))
        if not outline_polygon.is_empty:
            for start, end in _ring_edges(outline_polygon):
                start_key = snap(start)
                end_key = snap(end)
                if start_key != end_key:
                    raw_segments.append({"a": start_key, "b": end_key, "rooms": set(), "exterior": True})

    key_to_point = {key: tuple(value) for key, value in point_by_key.items()}
    split_segments = []
    for segment in raw_segments:
        a = segment["a"]
        b = segment["b"]
        start = key_to_point[a]
        end = key_to_point[b]
        cuts = [(0.0, a), (1.0, b)]
        for key, point in key_to_point.items():
            if key in {a, b}:
                continue
            t = _segment_t(point, start, end)
            if t is not None:
                cuts.append((t, key))
        cuts = sorted(cuts, key=lambda item: item[0])
        for (_, left), (_, right) in zip(cuts, cuts[1:]):
            if left != right:
                split_segments.append({"a": left, "b": right, "rooms": set(segment["rooms"]), "exterior": bool(segment["exterior"])})

    edge_map: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, object]] = {}
    duplicate_edges = 0
    for segment in split_segments:
        a, b = segment["a"], segment["b"]
        key = tuple(sorted((a, b)))
        if key in edge_map:
            duplicate_edges += 1
            edge_map[key]["rooms"].update(segment["rooms"])
            edge_map[key]["exterior"] = bool(edge_map[key]["exterior"] or segment["exterior"])
        else:
            edge_map[key] = {"rooms": set(segment["rooms"]), "exterior": bool(segment["exterior"])}

    used_keys = sorted({key for edge in edge_map for key in edge}, key=lambda key: (key_to_point[key][0], key_to_point[key][1]))
    if len(used_keys) > max_junctions:
        raise ValueError(f"Wall graph has {len(used_keys)} junctions, max_junctions={max_junctions}.")
    key_to_index = {key: index for index, key in enumerate(used_keys)}
    ordered_edges = sorted(
        ((key_to_index[a], key_to_index[b], value) for (a, b), value in edge_map.items()),
        key=lambda item: (min(item[0], item[1]), max(item[0], item[1])),
    )
    if len(ordered_edges) > max_edges:
        raise ValueError(f"Wall graph has {len(ordered_edges)} edges, max_edges={max_edges}.")

    junction_xy = np.zeros((max_junctions, 2), dtype=np.float32)
    junction_mask = np.zeros((max_junctions,), dtype=bool)
    for key, index in key_to_index.items():
        junction_xy[index] = np.asarray(key_to_point[key], dtype=np.float32)
        junction_mask[index] = True

    edge_index = np.zeros((max_edges, 2), dtype=np.int64)
    edge_mask = np.zeros((max_edges,), dtype=bool)
    edge_is_exterior = np.zeros((max_edges,), dtype=bool)
    edge_room_ids = np.full((max_edges, 2), -1, dtype=np.int64)
    edge_pairs = []
    internal_edge_pairs = []
    for edge_id, (a, b, value) in enumerate(ordered_edges):
        edge_index[edge_id] = (a, b)
        edge_mask[edge_id] = True
        edge_is_exterior[edge_id] = bool(value["exterior"])
        rooms_for_edge = sorted(value["rooms"])[:2]
        edge_room_ids[edge_id, : len(rooms_for_edge)] = rooms_for_edge
        edge_pairs.append((a, b))
        if rooms_for_edge:
            internal_edge_pairs.append((a, b))

    room_types = np.zeros((max_rooms,), dtype=np.int64)
    room_mask = np.zeros((max_rooms,), dtype=bool)
    for room_id, room_type, _ in clean_rooms:
        if room_id < max_rooms:
            room_types[room_id] = int(type_to_id.get(room_type, PAD_TYPE_ID))
            room_mask[room_id] = True

    lines = []
    try:
        from shapely.geometry import LineString

        for a, b in edge_pairs:
            lines.append(LineString([tuple(junction_xy[a]), tuple(junction_xy[b])]))
        faces = list(polygonize(unary_union(lines))) if lines else []
        invalid_faces = sum(1 for face in faces if not face.is_valid or face.area <= 1e-8)
    except Exception:
        invalid_faces = -1
    stats = {
        "junctions": int(junction_mask.sum()),
        "internal_junctions": int(len({node for edge in internal_edge_pairs for node in edge})),
        "walls": int(edge_mask.sum()),
        "rooms": int(room_mask.sum()),
        "duplicate_wall_rate": float(duplicate_edges / max(len(split_segments), 1)),
        "disconnected_component_count": int(_component_count(int(junction_mask.sum()), edge_pairs)),
        "disconnected_component_rate": float(max(_component_count(int(junction_mask.sum()), edge_pairs) - 1, 0) / max(int(junction_mask.sum()), 1)),
        "invalid_face_count": int(invalid_faces),
        "status": "ok",
    }
    return WallGraphTargets(
        junction_xy=junction_xy,
        junction_mask=junction_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        edge_is_exterior=edge_is_exterior,
        edge_room_ids=edge_room_ids,
        room_types=room_types,
        room_mask=room_mask,
        stats=stats,
    )


def edge_index_to_adjacency(edge_index: np.ndarray, edge_mask: np.ndarray, max_junctions: int) -> np.ndarray:
    adjacency = np.zeros((max_junctions, max_junctions), dtype=np.float32)
    for edge, valid in zip(edge_index, edge_mask):
        if not valid:
            continue
        a, b = int(edge[0]), int(edge[1])
        if a != b and 0 <= a < max_junctions and 0 <= b < max_junctions:
            adjacency[a, b] = 1.0
            adjacency[b, a] = 1.0
    return adjacency


def decode_wall_graph(
    junction_xy,
    junction_presence,
    edge_logits,
    outline,
    type_id_to_label: Mapping[int, str] | None = None,
    edge_threshold: float = 0.5,
    junction_threshold: float = 0.5,
    min_edge_length: float = 1e-4,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Polygonize a predicted wall graph.

    Wall-graph v1 is geometry/topology only. Reconstructed faces use the neutral
    type label ``Room``; semantic room-type generation is intentionally unsupported.
    """

    _, LineString, MultiLineString, _, _, polygonize, unary_union, _ = _require_shapely()
    if hasattr(junction_xy, "detach"):
        junction_xy = junction_xy.detach().cpu().numpy()
    if hasattr(junction_presence, "detach"):
        junction_presence = junction_presence.detach().cpu().numpy()
    if hasattr(edge_logits, "detach"):
        edge_logits = edge_logits.detach().cpu().numpy()
    junction_xy = np.asarray(junction_xy, dtype=np.float32)
    junction_presence = np.asarray(junction_presence, dtype=np.float32)
    edge_probs = 1.0 / (1.0 + np.exp(-np.asarray(edge_logits, dtype=np.float32)))
    active = np.flatnonzero(junction_presence >= junction_threshold)
    active_set = set(int(i) for i in active)
    segments = []
    rejected_crossings = 0
    for i in active:
        for j in active:
            if j <= i:
                continue
            if edge_probs[i, j] < edge_threshold:
                continue
            a = junction_xy[i]
            b = junction_xy[j]
            if float(np.linalg.norm(a - b)) < min_edge_length:
                continue
            line = LineString([tuple(a), tuple(b)])
            if line.is_empty:
                continue
            crosses = False
            for existing in segments:
                if line.crosses(existing):
                    crosses = True
                    rejected_crossings += 1
                    break
            if not crosses:
                segments.append(line)
    rooms: list[dict[str, object]] = []
    try:
        graph = unary_union(segments) if segments else MultiLineString([])
        faces = list(polygonize(graph))
        outline_clean = _repair_polygon(outline)
        for face in faces:
            clipped = _repair_polygon(face.intersection(outline_clean))
            if clipped.is_empty or clipped.area <= 1e-6:
                continue
            if outline_clean.area > 0 and clipped.area / outline_clean.area > 0.995:
                continue
            rooms.append({"type": "Room", "geometry": clipped})
    except Exception as exc:
        return [], {"error": str(exc), "segments": len(segments), "active_junctions": len(active), "rejected_crossings": rejected_crossings}
    return rooms, {"segments": len(segments), "active_junctions": len(active), "rejected_crossings": rejected_crossings, "faces": len(rooms)}
