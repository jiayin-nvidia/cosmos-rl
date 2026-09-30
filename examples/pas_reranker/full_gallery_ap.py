"""Metric-faithful AP utilities for fixed-depth PAS reranking.

The deployed reranker permutes only the retriever's first ``K`` candidates.  If
the remaining gallery order is held fixed, its *exact* change in full-gallery
AP is

``(AP-numerator(new top K) - AP-numerator(old top K)) / total_relevant``.

The tail contribution does not change: a top-K permutation preserves both the
tail ranks and the number of relevant candidates before every tail item.  This
module intentionally works with that counterfactual delta instead of the
commonly reported within-top-K AP, whose denominator is only the number of
positives retrieved in the head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def ap_numerator(labels_in_rank_order: torch.Tensor) -> torch.Tensor:
    """Return the unnormalised sum of precisions at relevant ranks."""

    labels = labels_in_rank_order.float().reshape(-1)
    if labels.numel() == 0:
        return labels.new_zeros(())
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("AP labels must be binary")
    ranks = torch.arange(1, labels.numel() + 1, device=labels.device, dtype=labels.dtype)
    return ((labels.cumsum(0) / ranks) * labels).sum()


def head_ap_contribution(
    labels_in_rank_order: torch.Tensor, total_relevant: int | torch.Tensor
) -> torch.Tensor:
    """Contribution of the reordered head to full-gallery AP."""

    denominator = torch.as_tensor(
        total_relevant,
        device=labels_in_rank_order.device,
        dtype=torch.float32,
    )
    if denominator.numel() != 1 or not bool(denominator > 0):
        raise ValueError("total_relevant must be one positive scalar")
    head_positives = labels_in_rank_order.float().sum()
    if bool(head_positives > denominator):
        raise ValueError("total_relevant cannot be smaller than head positives")
    return ap_numerator(labels_in_rank_order) / denominator


def _stable_score_order(scores: torch.Tensor) -> torch.Tensor:
    return torch.argsort(scores.reshape(-1), descending=True, stable=True)


@dataclass(frozen=True)
class CounterfactualAPTelemetry:
    """Exact AP movement attributable to a top-K permutation."""

    retriever_head_contribution: torch.Tensor
    reranked_head_contribution: torch.Tensor
    ap_delta: torch.Tensor
    oracle_head_contribution: torch.Tensor
    oracle_ap_gain: torch.Tensor
    oracle_regret: torch.Tensor
    head_recall: torch.Tensor


def counterfactual_ap_telemetry(
    scores: torch.Tensor,
    labels_in_retriever_order: torch.Tensor,
    total_relevant: int | torch.Tensor,
) -> CounterfactualAPTelemetry:
    """Measure exact full-gallery AP delta while treating the tail as fixed."""

    scores = scores.float().reshape(-1)
    labels = labels_in_retriever_order.float().reshape(-1).to(scores.device)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have identical shapes")
    reranked = labels[_stable_score_order(scores)]
    # Stable sorting leaves retriever order intact within the two oracle classes.
    oracle = labels[torch.argsort(labels, descending=True, stable=True)]
    old = head_ap_contribution(labels, total_relevant)
    new = head_ap_contribution(reranked, total_relevant)
    best = head_ap_contribution(oracle, total_relevant)
    denominator = torch.as_tensor(total_relevant, device=scores.device, dtype=torch.float32)
    return CounterfactualAPTelemetry(
        retriever_head_contribution=old,
        reranked_head_contribution=new,
        ap_delta=new - old,
        oracle_head_contribution=best,
        oracle_ap_gain=best - old,
        oracle_regret=best - new,
        head_recall=labels.sum() / denominator,
    )


def exact_ap_swap_weights(
    scores: torch.Tensor,
    labels: torch.Tensor,
    total_relevant: int | torch.Tensor,
    *,
    rank1_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact local swap weights for AP plus optional Rank-1 utility.

    Deltas are evaluated at the model's current detached ranking, as in
    LambdaRank.  Rows of the returned weight matrix correspond to positive
    item indices and columns to negative item indices in the original input.
    ``rank1_weight`` adds the exact absolute change in binary Rank-1 success
    caused by each swap. This yields Lambda weights for the explicit utility
    ``full_gallery_AP + rank1_weight * Rank1`` instead of relying on a generic
    top-pair heuristic. ``K`` is 20 in PAS. All counterfactual swaps are materialised as one
    ``(positive * negative, K)`` tensor, avoiding Python loops and GPU
    synchronisation in the training path.
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1).to(scores.device)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have identical shapes")
    if rank1_weight < 0:
        raise ValueError("rank1_weight must be nonnegative")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("labels must be binary")
    positive_indices = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    negative_indices = torch.nonzero(labels < 0.5, as_tuple=False).flatten()
    if not positive_indices.numel() or not negative_indices.numel():
        raise ValueError("LambdaAP requires both positive and negative candidates")
    denominator = torch.as_tensor(total_relevant, device=scores.device, dtype=torch.float32)
    if denominator.numel() != 1 or not bool(denominator > 0):
        raise ValueError("total_relevant must be one positive scalar")
    if bool(labels.sum() > denominator):
        raise ValueError("total_relevant cannot be smaller than head positives")

    with torch.no_grad():
        order = _stable_score_order(scores.detach())
        position = torch.empty_like(order)
        position[order] = torch.arange(order.numel(), device=order.device)
        ranked_labels = labels[order]
        baseline = ap_numerator(ranked_labels)
        positive_positions = position[positive_indices]
        negative_positions = position[negative_indices]
        pair_count = positive_indices.numel() * negative_indices.numel()
        swapped = ranked_labels.expand(pair_count, -1).clone()
        pair_rows = torch.arange(pair_count, device=scores.device)
        swap_positive_positions = (
            positive_positions.unsqueeze(1)
            .expand(-1, negative_indices.numel())
            .reshape(-1)
        )
        swap_negative_positions = (
            negative_positions.unsqueeze(0)
            .expand(positive_indices.numel(), -1)
            .reshape(-1)
        )
        swapped[pair_rows, swap_positive_positions] = 0.0
        swapped[pair_rows, swap_negative_positions] = 1.0
        ranks = torch.arange(
            1,
            ranked_labels.numel() + 1,
            device=scores.device,
            dtype=swapped.dtype,
        )
        swapped_numerators = ((swapped.cumsum(1) / ranks) * swapped).sum(1)
        weights = ((swapped_numerators - baseline).abs() / denominator).reshape(
            positive_indices.numel(), negative_indices.numel()
        )
        if rank1_weight:
            changes_rank1 = (
                positive_positions.unsqueeze(1).eq(0)
                | negative_positions.unsqueeze(0).eq(0)
            )
            weights = weights + rank1_weight * changes_rank1.float()
    return positive_indices, negative_indices, weights


@dataclass(frozen=True)
class LambdaAPResult:
    loss: torch.Tensor
    mean_ap_delta: torch.Tensor
    mean_oracle_ap_gain: torch.Tensor
    mean_oracle_regret: torch.Tensor
    mean_head_recall: torch.Tensor
    lambda_weight_mass: torch.Tensor
    active_pair_fraction: torch.Tensor


def full_gallery_lambda_ap_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    total_relevant: torch.Tensor,
    *,
    group_size: int,
    temperature: float = 0.2,
    margin: float = 0.0,
    lambda_scale: float = 1.0,
    rank1_weight: float = 0.0,
    retriever_success_weight: float = 1.0,
    retriever_error_weight: float = 1.0,
    active_topk: int | None = None,
) -> LambdaAPResult:
    """Lambda objective aligned to full-gallery AP and optional Rank-1.

    Crucially, weights are *not normalised per query*.  Per-query
    normalisation would cancel the ``1 / total_relevant`` factor and once
    again optimise a different metric.  ``lambda_scale`` is a single global
    optimiser-scale knob and therefore does not alter relative query weights.
    The outer temperature keeps gradient scale stable when temperature changes.
    """

    if group_size <= 1 or scores.numel() % group_size:
        raise ValueError("scores must contain complete groups with group_size > 1")
    if (
        temperature <= 0
        or lambda_scale < 0
        or rank1_weight < 0
        or retriever_success_weight < 0
        or retriever_error_weight < 0
    ):
        raise ValueError(
            "temperature must be positive; lambda_scale and rank1_weight "
            "and retriever weights must be nonnegative"
        )
    if active_topk is not None and active_topk <= 0:
        raise ValueError("active_topk must be positive when provided")
    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1).to(scores.device)
    totals = total_relevant.float().reshape(-1).to(scores.device)
    group_count = scores.numel() // group_size
    if totals.numel() != group_count:
        raise ValueError("total_relevant must contain exactly one value per group")

    losses = []
    delta_values = []
    oracle_gains = []
    oracle_regrets = []
    recalls = []
    weight_masses = []
    active_fractions = []
    for group_scores, group_labels, group_total in zip(
        scores.split(group_size), labels.split(group_size), totals, strict=True
    ):
        positive, negative, weights = exact_ap_swap_weights(
            group_scores,
            group_labels,
            group_total,
            rank1_weight=rank1_weight,
        )
        raw_violations = (
            group_scores[negative].unsqueeze(0)
            + margin
            - group_scores[positive].unsqueeze(1)
        )
        # Row zero is the frozen retriever's top-1 candidate. Weighting by
        # its ground-truth correctness lets residual fine-tuning spend more
        # capacity on queries that can improve over the retriever, without
        # discarding any of the K candidates or changing the deployed score.
        query_weight = (
            retriever_success_weight
            if bool(group_labels[0] > 0.5)
            else retriever_error_weight
        )
        if active_topk is None:
            pair_losses = temperature * F.softplus(raw_violations / temperature)
            selected_weights = weights
            active_fractions.append(weights.new_ones(()))
        else:
            # Update only currently inverted pairs, prioritizing the swaps
            # with the largest exact full-gallery AP consequence. Once a pair
            # is correctly ordered its gradient is exactly zero.
            inverted = raw_violations.detach() > 0
            flat_priority = torch.where(
                inverted.reshape(-1),
                weights.reshape(-1),
                weights.new_full((weights.numel(),), -torch.inf),
            )
            selected = torch.zeros_like(flat_priority, dtype=torch.bool)
            active_count = int(inverted.sum().item())
            if active_count:
                chosen = torch.topk(
                    flat_priority,
                    k=min(active_topk, active_count),
                    sorted=False,
                ).indices
                selected[chosen] = True
            selected = selected.reshape_as(weights)
            selected_weights = weights * selected
            pair_losses = F.relu(raw_violations)
            active_fractions.append(selected.float().mean())
        losses.append(
            lambda_scale
            * query_weight
            * (selected_weights * pair_losses).sum()
        )
        telemetry = counterfactual_ap_telemetry(group_scores, group_labels, group_total)
        delta_values.append(telemetry.ap_delta)
        oracle_gains.append(telemetry.oracle_ap_gain)
        oracle_regrets.append(telemetry.oracle_regret)
        recalls.append(telemetry.head_recall)
        weight_masses.append(query_weight * selected_weights.sum())

    return LambdaAPResult(
        loss=torch.stack(losses).mean(),
        mean_ap_delta=torch.stack(delta_values).mean().detach(),
        mean_oracle_ap_gain=torch.stack(oracle_gains).mean().detach(),
        mean_oracle_regret=torch.stack(oracle_regrets).mean().detach(),
        mean_head_recall=torch.stack(recalls).mean().detach(),
        lambda_weight_mass=torch.stack(weight_masses).mean().detach(),
        active_pair_fraction=torch.stack(active_fractions).mean().detach(),
    )
