import pytest

torch = pytest.importorskip("torch")

from floorplan_gen.models import ConditionalRoomFlow


def test_flow_model_shapes():
    model = ConditionalRoomFlow(
        geometry_dim=6,
        max_rooms=5,
        num_room_types=3,
        point_hidden_dim=16,
        cond_dim=16,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    tokens = torch.randn(2, 5, 6)
    t = torch.rand(2)
    boundary = torch.randn(2, 8, 2)

    outputs = model(tokens, t, boundary)

    assert outputs["flow"].shape == (2, 5, 6)
    assert outputs["presence_logits"].shape == (2, 5)
    assert outputs["type_logits"].shape == (2, 5, 4)
    assert outputs["count_logits"].shape == (2, 6)
    assert outputs["area_pred"].shape == (2, 5)
    assert outputs["boundary_tokens"].shape == (2, 8, 32)
