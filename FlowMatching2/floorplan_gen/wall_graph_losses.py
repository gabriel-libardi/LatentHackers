from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_junction_flow_batch(junction_xy: torch.Tensor, generator: torch.Generator | None = None):
    batch = junction_xy.shape[0]
    t = torch.rand(batch, device=junction_xy.device, generator=generator)
    noise = torch.randn(junction_xy.shape, device=junction_xy.device, generator=generator)
    t_view = t.view(batch, *([1] * (junction_xy.ndim - 1)))
    noisy = (1.0 - t_view) * noise + t_view * junction_xy
    target = junction_xy - noise
    return noisy, t, target


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def edge_index_to_dense(edge_index: torch.Tensor, edge_mask: torch.Tensor, max_junctions: int) -> torch.Tensor:
    dense = torch.zeros(edge_index.shape[0], max_junctions, max_junctions, device=edge_index.device, dtype=torch.float32)
    for batch in range(edge_index.shape[0]):
        valid_edges = edge_index[batch][edge_mask[batch].bool()]
        if valid_edges.numel() == 0:
            continue
        a = valid_edges[:, 0].long().clamp(0, max_junctions - 1)
        b = valid_edges[:, 1].long().clamp(0, max_junctions - 1)
        dense[batch, a, b] = 1.0
        dense[batch, b, a] = 1.0
    return dense


def upper_pair_mask(junction_mask: torch.Tensor) -> torch.Tensor:
    n = junction_mask.shape[1]
    valid = junction_mask.bool()
    pair = valid[:, :, None] & valid[:, None, :]
    upper = torch.triu(torch.ones(n, n, device=junction_mask.device, dtype=torch.bool), diagonal=1)
    return pair & upper[None, :, :]


def wall_graph_losses(
    outputs: dict[str, torch.Tensor],
    target_flow: torch.Tensor,
    junction_xy: torch.Tensor,
    junction_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    noisy_junction_xy: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    lambda_junction_flow: float = 1.0,
    lambda_junction_endpoint: float = 1.0,
    lambda_junction_presence: float = 0.2,
    lambda_edge: float = 1.0,
    use_gt_edges: bool = False,
    edge_loss: str = "bce",
    edge_pos_weight_max: float = 50.0,
) -> dict[str, torch.Tensor]:
    junction_mask = junction_mask.bool()
    flow_loss = masked_mean((outputs["flow"] - target_flow).pow(2), junction_mask)
    if noisy_junction_xy is not None and t is not None:
        t_view = t.view(t.shape[0], *([1] * (noisy_junction_xy.ndim - 1)))
        x1_pred = noisy_junction_xy + (1.0 - t_view) * outputs["flow"]
        endpoint_loss = masked_mean((x1_pred - junction_xy).pow(2), junction_mask)
    else:
        endpoint_loss = flow_loss * 0.0
    presence_loss = F.binary_cross_entropy_with_logits(
        outputs["junction_presence_logits"],
        junction_mask.to(outputs["junction_presence_logits"].dtype),
    )
    edge_targets = edge_index_to_dense(edge_index, edge_mask.bool(), junction_xy.shape[1])
    pair_mask = upper_pair_mask(junction_mask)
    if pair_mask.any() and not use_gt_edges:
        positives = edge_targets[pair_mask].sum().clamp_min(1.0)
        negatives = (1.0 - edge_targets[pair_mask]).sum().clamp_min(1.0)
        pos_weight = (negatives / positives).detach().clamp(1.0, float(edge_pos_weight_max))
        logits = outputs["edge_logits"][pair_mask]
        targets = edge_targets[pair_mask]
        if edge_loss == "bce":
            edge_loss_value = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        elif edge_loss == "focal":
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
            probs = torch.sigmoid(logits)
            pt = torch.where(targets > 0.5, probs, 1.0 - probs)
            edge_loss_value = ((1.0 - pt).pow(2.0) * bce).mean()
        else:
            raise ValueError("edge_loss must be 'bce' or 'focal'.")
    else:
        edge_loss_value = outputs["edge_logits"].sum() * 0.0
        targets = edge_targets[pair_mask]
        logits = outputs["edge_logits"][pair_mask]
    with torch.no_grad():
        if pair_mask.any():
            pred = torch.sigmoid(outputs["edge_logits"][pair_mask]) >= 0.5
            true = edge_targets[pair_mask] >= 0.5
            tp = (pred & true).sum().to(torch.float32)
            precision = tp / pred.sum().clamp_min(1)
            recall = tp / true.sum().clamp_min(1)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        else:
            precision = recall = f1 = outputs["edge_logits"].sum() * 0.0
    total = (
        lambda_junction_flow * flow_loss
        + lambda_junction_endpoint * endpoint_loss
        + lambda_junction_presence * presence_loss
        + lambda_edge * edge_loss_value
    )
    return {
        "loss": total,
        "junction_flow_loss": flow_loss.detach(),
        "junction_endpoint_loss": endpoint_loss.detach(),
        "junction_presence_loss": presence_loss.detach(),
        "edge_loss": edge_loss_value.detach(),
        "edge_precision": precision.detach(),
        "edge_recall": recall.detach(),
        "edge_f1": f1.detach(),
    }
