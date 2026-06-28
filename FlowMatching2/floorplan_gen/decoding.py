from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import DEFAULT_BOUNDARY_POINTS, DEFAULT_MAX_ROOMS, DEFAULT_SEED, PAD_TYPE_ID
from .geometry import PlanTransform, denormalize_geometry, normalize_geometry, sample_boundary_points
from .sampling import sample_room_geometry


@dataclass(frozen=True)
class GeneratedRoom:
    room_type: str
    geometry: object
    score: float = 1.0


def _require_shapely():
    try:
        from shapely.geometry import GeometryCollection, MultiPolygon, box
        from shapely.ops import unary_union
        try:
            from shapely.errors import GEOSException
        except ImportError:
            GEOSException = Exception
        try:
            from shapely.validation import make_valid
        except ImportError:
            make_valid = None
    except ImportError as exc:
        raise ImportError("Shapely is required for decoding generated rooms.") from exc
    return box, unary_union, GeometryCollection, MultiPolygon, make_valid, GEOSException


def repair_geometry(geometry):
    _, _, GeometryCollection, MultiPolygon, make_valid, GEOSException = _require_shapely()
    if geometry is None:
        return GeometryCollection()
    if geometry.is_empty:
        return GeometryCollection()
    try:
        repaired = make_valid(geometry) if make_valid is not None else geometry.buffer(0)
    except GEOSException:
        try:
            repaired = geometry.buffer(0)
        except GEOSException:
            return GeometryCollection()
    if repaired.is_empty:
        return GeometryCollection()
    polygons = list(iter_polygons(repaired))
    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        polygon = polygons[0]
        if not polygon.is_valid:
            try:
                polygon = polygon.buffer(0)
            except GEOSException:
                return GeometryCollection()
        return polygon if not polygon.is_empty else GeometryCollection()
    return MultiPolygon([polygon for polygon in polygons if not polygon.is_empty])


def iter_polygons(geometry):
    if geometry is None:
        return
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from iter_polygons(part)


def token_rectangles(
    tokens,
    type_id_to_label: Mapping[int, str],
    presence_threshold: float = 0.5,
    min_size: float = 0.02,
) -> list[GeneratedRoom]:
    box, _, _, _, _, _ = _require_shapely()
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu().numpy()
    tokens = np.asarray(tokens, dtype=np.float32)
    rooms: list[GeneratedRoom] = []
    for token in tokens:
        score = float(token[0])
        if score < presence_threshold:
            continue
        type_id = int(np.rint(token[1]))
        if type_id == PAD_TYPE_ID:
            continue
        room_type = type_id_to_label.get(type_id, f"type_{type_id}")
        cx, cy = float(token[2]), float(token[3])
        width = max(abs(float(token[4])), min_size)
        height = max(abs(float(token[5])), min_size)
        geom = box(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
        rooms.append(GeneratedRoom(room_type=room_type, geometry=geom, score=score))
    return rooms


def geometry_vectors_to_polygons(
    geometry,
    presence,
    type_ids,
    type_id_to_label: Mapping[int, str],
    presence_threshold: float = 0.5,
    min_area: float = 1e-5,
) -> list[GeneratedRoom]:
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise ImportError("Shapely is required for polygon decoding.") from exc
    if hasattr(geometry, "detach"):
        geometry = geometry.detach().cpu().numpy()
    if hasattr(presence, "detach"):
        presence = presence.detach().cpu().numpy()
    if hasattr(type_ids, "detach"):
        type_ids = type_ids.detach().cpu().numpy()
    geometry = np.asarray(geometry, dtype=np.float32)
    presence = np.asarray(presence, dtype=np.float32)
    type_ids = np.asarray(type_ids)
    if geometry.ndim == 2:
        geometry = geometry.reshape(geometry.shape[0], -1, 2)

    rooms: list[GeneratedRoom] = []
    for vertices, score, type_id in zip(geometry, presence, type_ids):
        score = float(score)
        type_id = int(type_id)
        if score < presence_threshold or type_id == PAD_TYPE_ID:
            continue
        polygon = repair_geometry(Polygon(vertices))
        if polygon.is_empty:
            continue
        room_type = type_id_to_label.get(type_id, f"type_{type_id}")
        for part in iter_polygons(polygon):
            if part.area >= min_area:
                rooms.append(GeneratedRoom(room_type, part, score))
    return rooms


def select_active_queries(
    presence,
    type_ids=None,
    count_logits=None,
    mode: str = "threshold",
    threshold: float = 0.5,
    fixed_k: int | None = None,
    allow_pad_type: bool = False,
):
    if hasattr(presence, "detach"):
        presence = presence.detach().cpu().numpy()
    if hasattr(type_ids, "detach"):
        type_ids = type_ids.detach().cpu().numpy()
    if hasattr(count_logits, "detach"):
        count_logits = count_logits.detach().cpu().numpy()
    presence = np.asarray(presence, dtype=np.float32)
    valid = np.ones(presence.shape, dtype=bool)
    if type_ids is not None and not allow_pad_type:
        valid &= np.asarray(type_ids) != PAD_TYPE_ID
    if mode == "threshold":
        return (presence >= float(threshold)) & valid
    if mode == "fixed_topk":
        if fixed_k is None:
            raise ValueError("fixed_k is required for fixed_topk selection.")
        k = int(fixed_k)
    elif mode == "predicted_count":
        if count_logits is None:
            raise ValueError("count_logits is required for predicted_count selection.")
        k = int(np.argmax(np.asarray(count_logits)))
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    k = max(0, min(k, int(valid.sum())))
    active = np.zeros(presence.shape, dtype=bool)
    if k == 0:
        return active
    candidate_indices = np.flatnonzero(valid)
    ordered = candidate_indices[np.argsort(-presence[candidate_indices], kind="mergesort")]
    active[ordered[:k]] = True
    return active


def _halfplane_for_site(site, other, bounds_scale: float, site_weight: float = 0.0, other_weight: float = 0.0):
    from shapely.geometry import Polygon

    site = np.asarray(site, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    direction = other - site
    norm = np.linalg.norm(direction)
    if norm <= 1e-8:
        return None
    direction = direction / norm
    tangent = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    midpoint = (site + other) / 2.0 + direction * ((float(site_weight) - float(other_weight)) / (2.0 * norm))
    keep_sign = np.sign(np.dot(site - midpoint, direction))
    if keep_sign == 0:
        keep_sign = -1.0
    q1 = midpoint + tangent * bounds_scale
    q2 = midpoint - tangent * bounds_scale
    q3 = q2 + direction * keep_sign * bounds_scale * 2.0
    q4 = q1 + direction * keep_sign * bounds_scale * 2.0
    return Polygon([tuple(q1), tuple(q2), tuple(q3), tuple(q4)])


def partition_geometry_to_cells(
    geometry,
    presence,
    type_ids,
    type_id_to_label: Mapping[int, str],
    outline,
    presence_threshold: float = 0.5,
    min_area: float = 1e-5,
    active_mask=None,
    use_area_weights: bool = True,
) -> list[GeneratedRoom]:
    if hasattr(geometry, "detach"):
        geometry = geometry.detach().cpu().numpy()
    if hasattr(presence, "detach"):
        presence = presence.detach().cpu().numpy()
    if hasattr(type_ids, "detach"):
        type_ids = type_ids.detach().cpu().numpy()
    geometry = np.asarray(geometry, dtype=np.float32)
    presence = np.asarray(presence, dtype=np.float32)
    type_ids = np.asarray(type_ids)
    sites = geometry[:, :2]
    weights = np.zeros((geometry.shape[0],), dtype=np.float32)
    if use_area_weights and geometry.shape[-1] >= 3:
        weights = np.square(np.clip(geometry[:, 2], 0.0, 2.0)).astype(np.float32)
    if active_mask is not None:
        if hasattr(active_mask, "detach"):
            active_mask = active_mask.detach().cpu().numpy()
        active = [index for index, is_active in enumerate(np.asarray(active_mask, dtype=bool)) if is_active]
    else:
        active_mask = select_active_queries(presence, type_ids, mode="threshold", threshold=presence_threshold)
        active = [index for index, is_active in enumerate(active_mask) if is_active]
    if not active:
        return []
    outline = repair_geometry(outline)
    min_x, min_y, max_x, max_y = outline.bounds
    bounds_scale = max(max_x - min_x, max_y - min_y, 1.0) * 8.0
    rooms: list[GeneratedRoom] = []
    for index in active:
        site = sites[index]
        cell = outline
        for other_index in active:
            if other_index == index:
                continue
            halfplane = _halfplane_for_site(
                site,
                sites[other_index],
                bounds_scale,
                site_weight=float(weights[index]),
                other_weight=float(weights[other_index]),
            )
            if halfplane is None:
                continue
            cell = safe_intersection(cell, halfplane)
            if cell.is_empty:
                break
        cell = repair_geometry(cell)
        room_type = type_id_to_label.get(int(type_ids[index]), f"type_{int(type_ids[index])}")
        for part in iter_polygons(cell):
            if part.area >= min_area:
                rooms.append(GeneratedRoom(room_type, part, float(presence[index])))
    return rooms


def repair_layout(
    rooms: list[GeneratedRoom],
    outline,
    gap_room_type: str = "Structure",
    fill_gaps: bool = True,
    min_area: float = 1e-5,
) -> list[GeneratedRoom]:
    """Clip to outline, remove overlaps in order, assign leftover gaps to neighbors."""

    _, unary_union, GeometryCollection, _, _, GEOSException = _require_shapely()
    outline = repair_geometry(outline)
    if outline.is_empty:
        return []
    repaired: list[GeneratedRoom] = []
    occupied = None
    for room in sorted(rooms, key=lambda item: item.score, reverse=True):
        geom = safe_intersection(repair_geometry(room.geometry), outline)
        if occupied is not None and not occupied.is_empty:
            geom = safe_difference(geom, occupied)
        geom = repair_geometry(geom)
        parts = [part for part in iter_polygons(geom) if part.area >= min_area]
        for part in parts:
            repaired.append(GeneratedRoom(room.room_type, part, room.score))
        if parts:
            union_parts = safe_unary_union(parts)
            occupied = repair_geometry(union_parts if occupied is None else safe_unary_union([occupied, union_parts]))

    if fill_gaps:
        try:
            occupied_geom = repair_geometry(occupied) if occupied is not None else None
            gap = outline if occupied_geom is None or occupied_geom.is_empty else safe_difference(outline, occupied_geom)
            gap = repair_geometry(gap)
            for part in iter_polygons(gap):
                if part.area >= min_area:
                    target_index = best_gap_neighbor(part, repaired)
                    if target_index is None:
                        repaired.append(GeneratedRoom(gap_room_type, part, 1.0))
                    else:
                        target = repaired[target_index]
                        merged = repair_geometry(safe_union(target.geometry, part))
                        if not merged.is_empty:
                            repaired[target_index] = GeneratedRoom(target.room_type, merged, target.score)
        except GEOSException:
            return repaired
    return repaired


def safe_intersection(first, second):
    _, _, GeometryCollection, _, _, GEOSException = _require_shapely()
    first = repair_geometry(first)
    second = repair_geometry(second)
    if first.is_empty or second.is_empty:
        return GeometryCollection()
    try:
        return repair_geometry(first.intersection(second))
    except GEOSException:
        return GeometryCollection()


def safe_difference(first, second):
    _, _, GeometryCollection, _, _, GEOSException = _require_shapely()
    first = repair_geometry(first)
    second = repair_geometry(second)
    if first.is_empty:
        return GeometryCollection()
    if second.is_empty:
        return first
    try:
        return repair_geometry(first.difference(second))
    except GEOSException:
        return GeometryCollection()


def safe_union(first, second):
    _, _, GeometryCollection, _, _, GEOSException = _require_shapely()
    first = repair_geometry(first)
    second = repair_geometry(second)
    if first.is_empty:
        return second
    if second.is_empty:
        return first
    try:
        return repair_geometry(first.union(second))
    except GEOSException:
        return GeometryCollection()


def safe_unary_union(geometries):
    _, unary_union, GeometryCollection, _, _, GEOSException = _require_shapely()
    polygonal = [repair_geometry(geometry) for geometry in geometries]
    polygonal = [geometry for geometry in polygonal if not geometry.is_empty]
    if not polygonal:
        return GeometryCollection()
    try:
        return repair_geometry(unary_union(polygonal))
    except GEOSException:
        merged = GeometryCollection()
        for geometry in polygonal:
            merged = safe_union(merged, geometry)
        return repair_geometry(merged)


def best_gap_neighbor(gap, rooms: list[GeneratedRoom]) -> int | None:
    best_index = None
    best_score = 0.0
    for index, room in enumerate(rooms):
        try:
            shared = gap.boundary.intersection(room.geometry.boundary).length
        except Exception:
            shared = 0.0
        if shared <= 0.0:
            shared = safe_intersection(gap.buffer(1e-4), room.geometry).area
        if shared > best_score:
            best_score = float(shared)
            best_index = index
    return best_index if best_score > 0.0 else None


def decode_room_tokens(
    tokens,
    type_id_to_label: Mapping[int, str],
    outline,
    transform: PlanTransform | dict[str, float] | None = None,
    presence_threshold: float = 0.5,
    fill_gaps: bool = True,
) -> list[dict[str, object]]:
    if isinstance(transform, dict):
        transform = PlanTransform.from_dict(transform)
    outline = repair_geometry(outline)
    rooms = token_rectangles(tokens, type_id_to_label, presence_threshold=presence_threshold)
    repaired = repair_layout(rooms, outline, fill_gaps=fill_gaps)
    decoded: list[dict[str, object]] = []
    for room in repaired:
        geom = denormalize_geometry(room.geometry, transform) if transform is not None else room.geometry
        decoded.append({"type": room.room_type, "geometry": repair_geometry(geom)})
    return decoded


def decode_room_geometry(
    geometry,
    presence,
    type_ids,
    type_id_to_label: Mapping[int, str],
    outline,
    transform: PlanTransform | dict[str, float] | None = None,
    presence_threshold: float = 0.5,
    fill_gaps: bool = True,
    repair: bool = True,
    representation: str = "polygon",
    active_mask=None,
) -> list[dict[str, object]]:
    if isinstance(transform, dict):
        transform = PlanTransform.from_dict(transform)
    outline = repair_geometry(outline)
    if representation == "partition":
        rooms = partition_geometry_to_cells(
            geometry,
            presence,
            type_ids,
            type_id_to_label,
            outline,
            presence_threshold=presence_threshold,
            active_mask=active_mask,
        )
    else:
        rooms = geometry_vectors_to_polygons(
            geometry,
            presence,
            type_ids,
            type_id_to_label,
            presence_threshold=presence_threshold,
        )
    if repair:
        rooms = repair_layout(rooms, outline, fill_gaps=fill_gaps)
    decoded: list[dict[str, object]] = []
    for room in rooms:
        geom = denormalize_geometry(room.geometry, transform) if transform is not None else room.geometry
        decoded.append({"type": room.room_type, "geometry": repair_geometry(geom)})
    return decoded


class FloorPlanGenerator:
    def __init__(
        self,
        model,
        type_id_to_label: Mapping[int, str],
        max_rooms: int = DEFAULT_MAX_ROOMS,
        boundary_points: int = DEFAULT_BOUNDARY_POINTS,
        steps: int = 32,
        seed: int = DEFAULT_SEED,
        device: str | None = None,
        representation: str = "polygon",
        selection_mode: str = "predicted_count",
        presence_threshold: float = 0.5,
        fixed_room_count: int | None = None,
    ) -> None:
        self.model = model
        self.type_id_to_label = dict(type_id_to_label)
        self.max_rooms = max_rooms
        self.boundary_points = boundary_points
        self.steps = steps
        self.seed = seed
        self.device = device
        self.representation = representation
        self.selection_mode = selection_mode
        self.presence_threshold = presence_threshold
        self.fixed_room_count = fixed_room_count

    def generate_with_diagnostics(self, outline, seed: int | None = None) -> dict[str, object]:
        import torch

        self.model.eval()
        normalized_outline, transform = normalize_geometry(outline)
        points = sample_boundary_points(normalized_outline, self.boundary_points)
        boundary = torch.from_numpy(points)
        geometry, outputs = sample_room_geometry(
            self.model,
            boundary,
            max_rooms=self.max_rooms,
            steps=self.steps,
            seed=self.seed if seed is None else seed,
            device=self.device,
        )
        type_logits = outputs["type_logits"]
        if type_logits.shape[-1] > 1:
            type_ids = type_logits[..., 1:].argmax(dim=-1) + 1
        else:
            type_ids = type_logits.argmax(dim=-1)
        presence = torch.sigmoid(outputs["presence_logits"])
        active_mask = select_active_queries(
            presence[0],
            type_ids[0],
            count_logits=outputs.get("count_logits", None)[0] if outputs.get("count_logits", None) is not None else None,
            mode=self.selection_mode,
            threshold=self.presence_threshold,
            fixed_k=self.fixed_room_count,
        )
        raw = decode_room_geometry(
            geometry[0],
            presence[0],
            type_ids[0],
            self.type_id_to_label,
            normalized_outline,
            transform=transform,
            fill_gaps=False,
            repair=False,
            representation=self.representation,
            active_mask=active_mask,
        )
        repaired = decode_room_geometry(
            geometry[0],
            presence[0],
            type_ids[0],
            self.type_id_to_label,
            normalized_outline,
            transform=transform,
            fill_gaps=True,
            repair=True,
            representation=self.representation,
            active_mask=active_mask,
        )
        diagnostics = {
            "selection_mode": self.selection_mode,
            "predicted_count": int(outputs["count_logits"][0].argmax().detach().cpu()) if "count_logits" in outputs else None,
            "active_count": int(np.asarray(active_mask, dtype=bool).sum()),
            "presence_mean": float(presence[0].mean().detach().cpu()),
            "presence_max": float(presence[0].max().detach().cpu()),
        }
        return {"rooms": repaired, "raw_rooms": raw, "outline": outline, "diagnostics": diagnostics}

    def generate(self, outline, seed: int | None = None) -> list[dict[str, object]]:
        return self.generate_with_diagnostics(outline, seed=seed)["rooms"]


def generate(outline, model, type_id_to_label: Mapping[int, str], **kwargs):
    return FloorPlanGenerator(model, type_id_to_label, **kwargs).generate(outline)
