import pytest

torch = pytest.importorskip("torch")

from floorplan_gen.models import ConditionalRoomFlow
from floorplan_gen.sampling import sample_room_tokens


def test_sampling_is_deterministic_for_seed():
    model = ConditionalRoomFlow(
        max_rooms=4,
        num_room_types=2,
        point_hidden_dim=8,
        cond_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model.eval()
    boundary = torch.randn(16, 2)

    first, _ = sample_room_tokens(model, boundary, max_rooms=4, steps=2, seed=123, device="cpu")
    second, _ = sample_room_tokens(model, boundary, max_rooms=4, steps=2, seed=123, device="cpu")

    torch.testing.assert_close(first, second)


def test_sampling_changes_with_seed():
    model = ConditionalRoomFlow(
        max_rooms=4,
        num_room_types=2,
        point_hidden_dim=8,
        cond_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model.eval()
    boundary = torch.randn(16, 2)

    first, _ = sample_room_tokens(model, boundary, max_rooms=4, steps=2, seed=123, device="cpu")
    second, _ = sample_room_tokens(model, boundary, max_rooms=4, steps=2, seed=124, device="cpu")

    assert not torch.allclose(first, second)


def test_sampling_responds_to_different_outlines():
    model = ConditionalRoomFlow(
        max_rooms=4,
        num_room_types=2,
        point_hidden_dim=8,
        cond_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model.eval()
    boundary_a = torch.zeros(16, 2)
    boundary_b = torch.ones(16, 2)

    _, outputs_a = sample_room_tokens(model, boundary_a, max_rooms=4, steps=2, seed=123, device="cpu")
    _, outputs_b = sample_room_tokens(model, boundary_b, max_rooms=4, steps=2, seed=123, device="cpu")

    assert not torch.allclose(outputs_a["outline_embedding"], outputs_b["outline_embedding"])
