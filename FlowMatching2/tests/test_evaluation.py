import pytest

shapely_geometry = pytest.importorskip("shapely.geometry")

from floorplan_gen.evaluation import outline_response, sample_diversity


def test_sample_diversity_detects_different_room_counts():
    samples = [
        [{"type": "Bedroom", "geometry": shapely_geometry.box(0, 0, 1, 1)}],
        [
            {"type": "Bedroom", "geometry": shapely_geometry.box(0, 0, 0.5, 1)},
            {"type": "Kitchen", "geometry": shapely_geometry.box(0.5, 0, 1, 1)},
        ],
    ]

    metrics = sample_diversity(samples)

    assert metrics["count_std"] > 0
    assert metrics["mean_signature_distance"] > 0


def test_outline_response_detects_different_layouts():
    first = [{"type": "Bedroom", "geometry": shapely_geometry.box(0, 0, 1, 1)}]
    second = [{"type": "Bedroom", "geometry": shapely_geometry.box(0, 0, 2, 1)}]

    metrics = outline_response(first, second)

    assert metrics["signature_distance"] > 0
