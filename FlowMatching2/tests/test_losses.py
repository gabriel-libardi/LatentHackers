import pytest

torch = pytest.importorskip("torch")

from floorplan_gen.losses import flow_matching_losses, sample_flow_batch


def test_masked_flow_loss_ignores_padded_room_flow():
    target = torch.zeros(1, 2, 6)
    pred = torch.zeros(1, 2, 6)
    pred[:, 1, :] = 1000.0
    tokens = torch.tensor([[[1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    outputs = {
        "flow": pred,
        "presence_logits": torch.zeros(1, 2),
        "type_logits": torch.zeros(1, 2, 3),
    }

    losses = flow_matching_losses(outputs, target, tokens, mask, presence_weight=0.0, type_weight=0.0)

    assert losses["loss"].item() == pytest.approx(0.0)


def test_sample_flow_batch_is_fixed_when_generator_seed_is_reset():
    geometry = torch.randn(2, 3, 4)
    first_generator = torch.Generator(device="cpu")
    first_generator.manual_seed(42)
    second_generator = torch.Generator(device="cpu")
    second_generator.manual_seed(42)

    first = sample_flow_batch(geometry, generator=first_generator)
    second = sample_flow_batch(geometry, generator=second_generator)

    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


def test_weighted_loss_ignores_padding_type_weight_and_uses_count_target():
    target = torch.zeros(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    outputs = {
        "flow": torch.zeros_like(target),
        "presence_logits": torch.zeros(2, 3),
        "type_logits": torch.zeros(2, 3, 4),
        "count_logits": torch.tensor([[-5.0, 0.0, 5.0, -5.0], [-5.0, 5.0, 0.0, -5.0]]),
        "area_pred": torch.zeros(2, 3),
    }
    type_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    weights = torch.tensor([0.0, 1.0, 2.0, 3.0])

    losses = flow_matching_losses(
        outputs,
        target,
        target,
        mask,
        room_type_ids=type_ids,
        room_count=torch.tensor([2, 1]),
        type_weights=weights,
        flow_weight=0.0,
        presence_weight=0.0,
        type_weight=0.0,
        area_weight=0.0,
        count_weight=1.0,
    )

    assert losses["count_loss"].item() < 0.02
