import pytest

torch = pytest.importorskip("torch")

from floorplan_gen.models import PointNetEncoder


def test_pointnet_output_shape():
    model = PointNetEncoder(output_dim=32)
    points = torch.randn(4, 16, 2)

    encoded = model(points)

    assert encoded.shape == (4, 32)
