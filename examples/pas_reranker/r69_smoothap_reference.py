#!/usr/bin/env python3
"""Pure-PyTorch reference objective for r69 metric-aligned list losses."""

from __future__ import annotations

from typing import Literal

import torch


Reduction = Literal["none", "mean", "sum"]


def _validate(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    group_size: int,
) -> None:
    if group_size <= 1:
        raise ValueError("group_size must exceed one")
    if scores.ndim != 2 or scores.shape[1] != group_size:
        raise ValueError(
            f"r69 requires [batch,{group_size}] scores, got {tuple(scores.shape)}"
        )
    if labels.shape != scores.shape:
        raise ValueError("labels must have the same [batch,20] shape as scores")
    if not scores.is_floating_point():
        raise TypeError("scores must be floating point")
    if temperature <= 0.0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("temperature must be finite and positive")
    if labels.requires_grad:
        raise ValueError("hard labels must not require gradients")
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError("r69 accepts binary hard labels only")
    positives = labels.sum(dim=-1)
    if not bool(torch.all((positives > 0) & (positives < scores.shape[1]))):
        raise ValueError("every K20 group must contain positive and negative labels")


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unsupported reduction {reduction!r}")


def smooth_average_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    group_size: int = 20,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Differentiable AP using sigmoid-relaxed pairwise ranks.

    For candidate i, ``soft_rank`` counts all candidates above it, while
    ``soft_positive_rank`` counts only positives above it. Their ratio is the
    relaxed precision at i; averaging it over hard-label positives gives AP.
    """

    _validate(scores, labels, temperature, group_size)
    work_scores = scores.float()
    work_labels = labels.to(dtype=work_scores.dtype)
    # pairwise[b,i,j] = s_j - s_i: probability that j ranks above i.
    pairwise = work_scores.unsqueeze(1) - work_scores.unsqueeze(2)
    above = torch.sigmoid(pairwise / float(temperature))
    diagonal = torch.eye(scores.shape[1], dtype=torch.bool, device=scores.device)
    above = above.masked_fill(diagonal.unsqueeze(0), 0.0)
    soft_rank = 1.0 + above.sum(dim=-1)
    soft_positive_rank = 1.0 + (above * work_labels.unsqueeze(1)).sum(dim=-1)
    precision = soft_positive_rank / soft_rank
    ap = (precision * work_labels).sum(dim=-1) / work_labels.sum(dim=-1)
    return _reduce(ap, reduction)


def soft_rank1_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    group_size: int = 20,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Negative log probability that a Plackett-Luce top-1 is hard-positive."""

    _validate(scores, labels, temperature, group_size)
    log_prob = torch.log_softmax(scores.float() / float(temperature), dim=-1)
    positive_log_mass = torch.logsumexp(
        log_prob.masked_fill(labels == 0, -torch.inf), dim=-1
    )
    return _reduce(-positive_log_mass, reduction)


def r69_metric_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    smoothap_temperature: float,
    soft_r1_temperature: float,
    smoothap_weight: float = 1.0,
    soft_r1_weight: float = 0.25,
    group_size: int = 20,
    reduction: Reduction = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the scalar r69 loss and detached per-group audit components."""

    if smoothap_weight < 0.0 or soft_r1_weight < 0.0:
        raise ValueError("r69 loss weights must be non-negative")
    ap = smooth_average_precision(
        scores,
        labels,
        temperature=smoothap_temperature,
        group_size=group_size,
        reduction="none",
    )
    r1 = soft_rank1_loss(
        scores,
        labels,
        temperature=soft_r1_temperature,
        group_size=group_size,
        reduction="none",
    )
    per_group = smoothap_weight * (1.0 - ap) + soft_r1_weight * r1
    loss = _reduce(per_group, reduction)
    return loss, {
        "smooth_ap": ap.detach(),
        "smooth_ap_loss": (1.0 - ap).detach(),
        "soft_r1_loss": r1.detach(),
        "total_per_group": per_group.detach(),
    }
