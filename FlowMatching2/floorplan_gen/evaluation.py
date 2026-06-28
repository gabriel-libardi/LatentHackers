from __future__ import annotations

from collections import Counter
import math


def _require_shapely():
    try:
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Shapely is required for layout evaluation.") from exc
    return unary_union


def _iter_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def _orthogonality_error(geometry) -> float:
    import math

    errors = []
    for polygon in _iter_polygons(geometry):
        coords = list(polygon.exterior.coords)
        for i in range(1, len(coords) - 1):
            ax, ay = coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1]
            bx, by = coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]
            norm_a = math.hypot(ax, ay)
            norm_b = math.hypot(bx, by)
            if norm_a < 1e-8 or norm_b < 1e-8:
                continue
            dot = max(-1.0, min(1.0, (ax * bx + ay * by) / (norm_a * norm_b)))
            angle = abs(math.degrees(math.acos(dot)))
            errors.append(min(abs(angle - 90.0), abs(angle - 180.0), abs(angle)))
    return sum(errors) / max(len(errors), 1)


def evaluate_layout(rooms: list[dict[str, object]], outline) -> dict[str, object]:
    unary_union = _require_shapely()
    geometries = [room["geometry"] for room in rooms if not room["geometry"].is_empty]
    type_counts = Counter(str(room["type"]) for room in rooms)
    valid_count = sum(1 for geom in geometries if geom.is_valid)
    self_intersection_count = sum(1 for geom in geometries if not geom.is_simple)
    union = unary_union(geometries) if geometries else None
    outline_area = max(float(outline.area), 1e-8)
    coverage = 0.0 if union is None else float(union.intersection(outline).area / outline_area)
    uncovered_ratio = 1.0 - min(coverage, 1.0)
    outside_area = 0.0 if union is None else float(union.difference(outline).area)

    overlap_area = 0.0
    for i, geom in enumerate(geometries):
        for other in geometries[i + 1 :]:
            overlap_area += float(geom.intersection(other).area)
    component_count = sum(sum(1 for _ in _iter_polygons(geom)) for geom in geometries)
    tiny_thresholds = {
        "tiny_cell_fraction_0_005": outline_area * 0.005,
        "tiny_cell_fraction_0_01": outline_area * 0.01,
        "tiny_cell_fraction_0_02": outline_area * 0.02,
    }
    tiny_counts = {key: 0 for key in tiny_thresholds}
    tiny_area_threshold = outline_area * 0.002
    tiny_fragments = 0
    very_small_rooms = 0
    very_thin_rooms = 0
    slivers = 0
    perimeter_area_values = []
    orthogonality_values = []
    for geom in geometries:
        area = max(float(geom.area), 1e-8)
        if area < outline_area * 0.005:
            very_small_rooms += 1
        perimeter_area_values.append(float(geom.length) / area)
        orthogonality_values.append(_orthogonality_error(geom))
        for part in _iter_polygons(geom):
            for key, threshold in tiny_thresholds.items():
                if part.area < threshold:
                    tiny_counts[key] += 1
            if part.area < tiny_area_threshold:
                tiny_fragments += 1
            min_x, min_y, max_x, max_y = part.bounds
            short = min(max_x - min_x, max_y - min_y)
            long = max(max_x - min_x, max_y - min_y)
            if long > 0 and short / long < 0.08:
                slivers += 1
                very_thin_rooms += 1

    return {
        "room_count": len(geometries),
        "type_distribution": dict(type_counts),
        "outline_coverage": coverage,
        "covered_outline_area": coverage * outline_area,
        "uncovered_ratio": uncovered_ratio,
        "uncovered_outline_area": uncovered_ratio * outline_area,
        "outside_outline_area": outside_area,
        "outside_outline_ratio": outside_area / outline_area,
        "overlap_ratio": overlap_area / outline_area,
        "overlap_area": overlap_area,
        "valid_fraction": valid_count / max(len(geometries), 1),
        "invalid_polygon_rate": 1.0 - valid_count / max(len(geometries), 1),
        "self_intersection_rate": self_intersection_count / max(len(geometries), 1),
        "component_count": component_count,
        "disconnected_extra_components": max(0, component_count - len(geometries)),
        "tiny_fragment_count": tiny_fragments,
        **{key: value / max(len(geometries), 1) for key, value in tiny_counts.items()},
        "sliver_count": slivers,
        "very_small_room_count": very_small_rooms,
        "very_thin_room_count": very_thin_rooms,
        "perimeter_area_mean": sum(perimeter_area_values) / max(len(perimeter_area_values), 1),
        "orthogonality_error_mean": sum(orthogonality_values) / max(len(orthogonality_values), 1),
    }


def type_distribution_error(generated: dict[str, int], reference: dict[str, int]) -> float:
    keys = set(generated) | set(reference)
    gen_total = max(sum(generated.values()), 1)
    ref_total = max(sum(reference.values()), 1)
    return float(
        0.5
        * sum(
            abs(float(generated.get(key, 0)) / gen_total - float(reference.get(key, 0)) / ref_total)
            for key in keys
        )
    )


def repair_change_ratio(raw_rooms: list[dict[str, object]], repaired_rooms: list[dict[str, object]], outline) -> float:
    unary_union = _require_shapely()
    raw = unary_union([room["geometry"] for room in raw_rooms]) if raw_rooms else None
    repaired = unary_union([room["geometry"] for room in repaired_rooms]) if repaired_rooms else None
    if raw is None and repaired is None:
        return 0.0
    if raw is None:
        return float(repaired.area / max(outline.area, 1e-8))
    if repaired is None:
        return float(raw.area / max(outline.area, 1e-8))
    return float(raw.symmetric_difference(repaired).area / max(outline.area, 1e-8))


def diagnostic_score(raw_metrics: dict[str, float], repaired_metrics: dict[str, float], repair_change: float, real_room_count: int | None = None) -> float:
    count_error = 0.0
    if real_room_count is not None:
        count_error = abs(float(repaired_metrics["room_count"]) - float(real_room_count)) / max(float(real_room_count), 1.0)
    return float(
        4.0 * raw_metrics.get("uncovered_ratio", 0.0)
        + 2.0 * raw_metrics.get("overlap_ratio", 0.0)
        + 1.5 * repair_change
        + 0.25 * repaired_metrics.get("disconnected_extra_components", 0.0)
        + 0.15 * repaired_metrics.get("tiny_fragment_count", 0.0)
        + 0.75 * repaired_metrics.get("tiny_cell_fraction_0_01", 0.0)
        + 0.5 * repaired_metrics.get("tiny_cell_fraction_0_02", 0.0)
        + 0.15 * repaired_metrics.get("sliver_count", 0.0)
        + 0.02 * repaired_metrics.get("orthogonality_error_mean", 0.0)
        + count_error
    )


def layout_signature(rooms: list[dict[str, object]]) -> dict[str, float]:
    geometries = [room["geometry"] for room in rooms if not room["geometry"].is_empty]
    if not geometries:
        return {"room_count": 0.0, "centroid_x": 0.0, "centroid_y": 0.0, "area_mean": 0.0}
    total_area = sum(float(geom.area) for geom in geometries)
    centroid_x = sum(float(geom.centroid.x) * float(geom.area) for geom in geometries) / max(total_area, 1e-8)
    centroid_y = sum(float(geom.centroid.y) * float(geom.area) for geom in geometries) / max(total_area, 1e-8)
    return {
        "room_count": float(len(geometries)),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "area_mean": total_area / max(len(geometries), 1),
    }


def sample_diversity(samples: list[list[dict[str, object]]]) -> dict[str, float]:
    if len(samples) < 2:
        return {"count_std": 0.0, "mean_signature_distance": 0.0}
    signatures = [layout_signature(sample) for sample in samples]
    counts = [sig["room_count"] for sig in signatures]
    distances = []
    for i, first in enumerate(signatures):
        for second in signatures[i + 1 :]:
            distances.append(
                math.sqrt(
                    (first["room_count"] - second["room_count"]) ** 2
                    + (first["centroid_x"] - second["centroid_x"]) ** 2
                    + (first["centroid_y"] - second["centroid_y"]) ** 2
                    + (first["area_mean"] - second["area_mean"]) ** 2
                )
            )
    mean = sum(counts) / len(counts)
    variance = sum((count - mean) ** 2 for count in counts) / len(counts)
    return {
        "count_std": math.sqrt(variance),
        "mean_signature_distance": sum(distances) / max(len(distances), 1),
    }


def outline_response(first: list[dict[str, object]], second: list[dict[str, object]]) -> dict[str, float]:
    a = layout_signature(first)
    b = layout_signature(second)
    return {
        "signature_distance": math.sqrt(
            (a["room_count"] - b["room_count"]) ** 2
            + (a["centroid_x"] - b["centroid_x"]) ** 2
            + (a["centroid_y"] - b["centroid_y"]) ** 2
            + (a["area_mean"] - b["area_mean"]) ** 2
        )
    }
