"""Fixed grouped CR3 reranker objective: probability-mass rank loss plus BCE."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from examples.pas_reranker.full_gallery_ap import (
    exact_ap_swap_weights,
    full_gallery_lambda_ap_loss,
)
from examples.pas_reranker.r69_smoothap_reference import r69_metric_loss


@dataclass(frozen=True)
class BinaryResponseSpec:
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]
    decision_offset: int
    positive_token_id: int
    negative_token_id: int


@dataclass(frozen=True)
class OrdinalResponseSpec:
    """Five response prefixes that differ at one grade decision token."""

    response_ids: tuple[tuple[int, ...], ...]
    decision_offset: int
    token_ids: tuple[int, ...]


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Response tokenized to an empty sequence: {text!r}")
    return tuple(int(token_id) for token_id in ids)


def build_binary_response_spec(
    tokenizer: Any,
    positive_response: str = "<answer>yes</answer>",
    negative_response: str = "<answer>no</answer>",
) -> BinaryResponseSpec:
    positive_ids = _token_ids(tokenizer, positive_response)
    negative_ids = _token_ids(tokenizer, negative_response)
    shared = min(len(positive_ids), len(negative_ids))
    decision_offset = next(
        (index for index in range(shared) if positive_ids[index] != negative_ids[index]),
        shared,
    )
    if decision_offset >= len(positive_ids) or decision_offset >= len(negative_ids):
        raise ValueError(
            "Positive and negative responses do not have distinct decision tokens: "
            f"positive_ids={positive_ids}, negative_ids={negative_ids}"
        )
    return BinaryResponseSpec(
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        decision_offset=decision_offset,
        positive_token_id=positive_ids[decision_offset],
        negative_token_id=negative_ids[decision_offset],
    )


def build_ordinal_response_spec(
    tokenizer: Any,
    responses: Sequence[str] = tuple(f"<score>{value}</score>" for value in range(1, 6)),
) -> OrdinalResponseSpec:
    """Build a robust 1--5 response spec without assuming standalone digit IDs."""

    encoded = tuple(_token_ids(tokenizer, response) for response in responses)
    if len(encoded) != 5:
        raise ValueError("Ordinal relevance requires exactly five responses")
    shared_length = min(len(ids) for ids in encoded)
    decision_offset = next(
        (
            index
            for index in range(shared_length)
            if len({ids[index] for ids in encoded}) > 1
        ),
        shared_length,
    )
    if decision_offset >= shared_length:
        raise ValueError(f"Ordinal responses have no distinct grade token: {encoded}")
    token_ids = tuple(ids[decision_offset] for ids in encoded)
    if len(set(token_ids)) != 5:
        raise ValueError(
            "Ordinal responses must have five unique decision tokens: "
            f"offset={decision_offset}, token_ids={token_ids}"
        )
    common_prefix = {ids[:decision_offset] for ids in encoded}
    if len(common_prefix) != 1:
        raise ValueError("Ordinal responses do not share one decision prefix")
    return OrdinalResponseSpec(encoded, decision_offset, token_ids)


def extract_ordinal_scores(
    output: torch.Tensor,
    labels: torch.LongTensor,
    response_spec: OrdinalResponseSpec,
    *,
    ignore_index: int = -100,
    projection_weight: torch.Tensor | None = None,
    score_readout: Literal["expected", "strict_logodds"] = "expected",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a five-grade score, exact-match labels, and grade CE.

    Grade 5 is the sole full-match class. Grades 1--4 encode increasingly
    small but still disqualifying mismatch counts. ``expected`` retains the
    complete ordinal distribution. ``strict_logodds`` ranks by the log odds
    of grade 5 against the union of every disqualifying grade, preventing
    good attributes from compensating for one failed requirement.
    """

    if score_readout not in {"expected", "strict_logodds"}:
        raise ValueError(f"Unsupported ordinal score readout: {score_readout!r}")

    if output.ndim != 3 or labels.ndim != 2 or output.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected output [B,L,D] and labels [B,L], got {output.shape=} "
            f"{labels.shape=}"
        )
    local_projection = projection_weight
    if local_projection is not None and hasattr(local_projection, "to_local"):
        local_projection = local_projection.to_local()
    scores: list[torch.Tensor] = []
    binary_labels: list[float] = []
    grade_losses: list[torch.Tensor] = []
    values = torch.arange(1, 6, device=output.device, dtype=torch.float32)
    for row_index in range(labels.shape[0]):
        supervised_positions = torch.nonzero(
            labels[row_index] != ignore_index, as_tuple=False
        ).flatten()
        if supervised_positions.numel() == 0:
            raise ValueError(f"Example {row_index} has no supervised assistant tokens")
        supervised_ids = labels[row_index, supervised_positions].tolist()
        matches = [
            grade_index
            for grade_index, response_ids in enumerate(response_spec.response_ids)
            if _starts_with(
                supervised_ids,
                response_ids[: response_spec.decision_offset + 1],
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Example {row_index} response has no unique ordinal grade: "
                f"supervised_ids={supervised_ids[:16]}"
            )
        grade_index = matches[0]
        target_position = int(
            supervised_positions[response_spec.decision_offset].item()
        )
        prediction_position = target_position - 1
        if prediction_position < 0:
            raise ValueError(f"Example {row_index} has no ordinal decision input token")
        decision_output = output[row_index, prediction_position].float()
        if local_projection is None or decision_output.numel() == local_projection.shape[0]:
            grade_logits = decision_output[list(response_spec.token_ids)]
        elif decision_output.numel() == local_projection.shape[1]:
            grade_weight = local_projection[list(response_spec.token_ids)].float()
            grade_logits = F.linear(decision_output, grade_weight)
        else:
            raise ValueError(
                "Ordinal projection shape mismatch: "
                f"output={decision_output.shape}, weight={local_projection.shape}"
            )
        if score_readout == "expected":
            probabilities = grade_logits.float().softmax(dim=0)
            scores.append((probabilities * values).sum())
        else:
            scores.append(
                grade_logits[4].float()
                - torch.logsumexp(grade_logits[:4].float(), dim=0)
            )
        binary_labels.append(float(grade_index == 4))
        grade_losses.append(
            F.cross_entropy(
                grade_logits.float().unsqueeze(0),
                torch.tensor([grade_index], device=output.device),
            )
        )
    return (
        torch.stack(scores),
        output.new_tensor(binary_labels, dtype=torch.float32),
        torch.stack(grade_losses).mean(),
    )


def _starts_with(values: Sequence[int], prefix: Sequence[int]) -> bool:
    return len(values) >= len(prefix) and tuple(values[: len(prefix)]) == tuple(prefix)


def bf16_ste(value: torch.Tensor) -> torch.Tensor:
    """Use BF16 values in the forward pass and identity gradients backward."""

    quantized = value.to(torch.bfloat16).to(value.dtype)
    return value + (quantized - value).detach()


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return the local value of a replicated DTensor parameter."""

    return value.to_local() if hasattr(value, "to_local") else value


def _selected_projection_logits(
    hidden: torch.Tensor,
    token_ids: Sequence[int],
    projection_weight: torch.Tensor,
    projection_module: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Project selected vocabulary rows, including an LM-head LoRA.

    PAS normally bypasses the full vocabulary projection and indexes the two
    frozen yes/no rows directly.  When ``lm_head`` itself is a LoRA target,
    using only ``module.weight`` would silently discard that adapter and give
    it no gradient.  This is the selected-row equivalent of
    ``LoraInjectedLinear.forward`` and avoids materializing [B, L, vocab].
    """

    ids = [int(value) for value in token_ids]
    weight = _local_tensor(projection_weight)
    logits = F.linear(hidden, weight[ids].float())
    if projection_module is None:
        return logits

    bias = getattr(projection_module, "bias", None)
    if bias is not None:
        logits = logits + _local_tensor(bias)[ids].float()

    lora_a_wrapper = getattr(projection_module, "lora_A", None)
    lora_b_wrapper = getattr(projection_module, "lora_B", None)
    if (
        lora_a_wrapper is None
        or lora_b_wrapper is None
        or bool(getattr(projection_module, "merged", False))
    ):
        return logits
    lora_a = _local_tensor(lora_a_wrapper.weight)
    lora_b = _local_tensor(lora_b_wrapper.weight)
    dropout = getattr(projection_module, "lora_dropout", None)
    lora_input = hidden if dropout is None else dropout(hidden)
    after_a = F.linear(lora_input.to(lora_a.dtype), lora_a)
    lora_logits = F.linear(after_a, lora_b[ids])
    return logits + float(getattr(projection_module, "scaling", 1.0)) * lora_logits.float()


def extract_ordinal_aux_loss(
    output: torch.Tensor,
    labels: torch.LongTensor,
    response_spec: OrdinalResponseSpec,
    *,
    ignore_index: int = -100,
    projection_weight: torch.Tensor | None = None,
    binary_response_spec: BinaryResponseSpec | None = None,
    binary_decision_position: Literal["prefix", "unique_suffix"] = "prefix",
) -> torch.Tensor:
    """Cross-entropy for a 1--5 suffix following another response.

    The grade target is parsed from a causal
    ``<answer>yes/no</answer><score>N</score>`` response. By default the grade
    logits are read at the suffix position for backward compatibility. When
    ``binary_response_spec`` is supplied, the five logits are instead read
    from the *same pre-decision hidden state* that predicts yes/no. This makes
    the auxiliary label-independent: that causal state cannot see either the
    ground-truth binary token or the later grade suffix.
    """

    if output.ndim != 3 or labels.ndim != 2 or output.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected output [B,L,D] and labels [B,L], got {output.shape=} "
            f"{labels.shape=}"
        )
    local_projection = projection_weight
    if local_projection is not None and hasattr(local_projection, "to_local"):
        local_projection = local_projection.to_local()
    predecision_positions = None
    if binary_response_spec is not None:
        predecision_positions, _ = _binary_decision_positions_and_labels(
            labels,
            binary_response_spec,
            ignore_index=ignore_index,
            decision_position=binary_decision_position,
        )
    losses: list[torch.Tensor] = []
    for row_index in range(labels.shape[0]):
        supervised_positions = torch.nonzero(
            labels[row_index] != ignore_index, as_tuple=False
        ).flatten()
        supervised_ids = labels[row_index, supervised_positions].tolist()
        matches: list[tuple[int, int]] = []
        for grade_index, response_ids in enumerate(response_spec.response_ids):
            prefix = response_ids[: response_spec.decision_offset + 1]
            for start in range(len(supervised_ids) - len(prefix) + 1):
                if tuple(supervised_ids[start : start + len(prefix)]) == tuple(prefix):
                    matches.append((grade_index, start))
        if len(matches) != 1:
            raise ValueError(
                f"Example {row_index} response has no unique ordinal suffix: "
                f"supervised_ids={supervised_ids[:24]}"
            )
        grade_index, start = matches[0]
        if predecision_positions is None:
            target_offset = start + response_spec.decision_offset
            target_position = int(supervised_positions[target_offset].item())
            prediction_position = target_position - 1
        else:
            prediction_position = predecision_positions[row_index]
        if prediction_position < 0:
            raise ValueError(f"Example {row_index} has no ordinal decision input token")
        decision_output = output[row_index, prediction_position].float()
        if local_projection is None or decision_output.numel() == local_projection.shape[0]:
            grade_logits = decision_output[list(response_spec.token_ids)]
        elif decision_output.numel() == local_projection.shape[1]:
            grade_weight = local_projection[list(response_spec.token_ids)].float()
            grade_logits = F.linear(decision_output, grade_weight)
        else:
            raise ValueError(
                "Ordinal projection shape mismatch: "
                f"output={decision_output.shape}, weight={local_projection.shape}"
            )
        losses.append(
            F.cross_entropy(
                grade_logits.float().unsqueeze(0),
                torch.tensor([grade_index], device=output.device),
            )
        )
    return torch.stack(losses).mean()


def _binary_decision_positions_and_labels(
    labels: torch.LongTensor,
    response_spec: BinaryResponseSpec,
    *,
    ignore_index: int = -100,
    decision_position: Literal["prefix", "unique_suffix"] = "prefix",
) -> tuple[list[int], list[float]]:
    """Locate the causal state that predicts each binary decision token."""

    if labels.ndim != 2:
        raise ValueError(f"Expected labels [B,L], got {labels.shape=}")
    prediction_positions: list[int] = []
    binary_labels: list[float] = []
    for row_index in range(labels.shape[0]):
        supervised_positions = torch.nonzero(
            labels[row_index] != ignore_index, as_tuple=False
        ).flatten()
        if supervised_positions.numel() == 0:
            raise ValueError(f"Example {row_index} has no supervised assistant tokens")
        supervised_ids = labels[row_index, supervised_positions].tolist()
        # Only the common prefix through the yes/no decision token is needed
        # to identify relevance. Auxiliary trajectories may legally follow it.
        positive_decision_prefix = response_spec.positive_ids[
            : response_spec.decision_offset + 1
        ]
        negative_decision_prefix = response_spec.negative_ids[
            : response_spec.decision_offset + 1
        ]
        if decision_position == "prefix":
            if _starts_with(supervised_ids, positive_decision_prefix):
                binary_label = 1.0
                response_start = 0
            elif _starts_with(supervised_ids, negative_decision_prefix):
                binary_label = 0.0
                response_start = 0
            else:
                raise ValueError(
                    f"Example {row_index} response is not configured yes/no: "
                    f"supervised_ids={supervised_ids[:16]}"
                )
        elif decision_position == "unique_suffix":
            matches: list[tuple[float, int]] = []
            for binary_label, prefix in (
                (1.0, positive_decision_prefix),
                (0.0, negative_decision_prefix),
            ):
                for start in range(len(supervised_ids) - len(prefix) + 1):
                    if tuple(supervised_ids[start : start + len(prefix)]) == tuple(prefix):
                        matches.append((binary_label, start))
            if len(matches) != 1:
                raise ValueError(
                    f"Example {row_index} response has no unique yes/no suffix: "
                    f"matches={matches}, supervised_ids={supervised_ids[:24]}"
                )
            binary_label, response_start = matches[0]
        else:
            raise ValueError(f"Unknown binary decision position: {decision_position}")

        target_position = int(
            supervised_positions[
                response_start + response_spec.decision_offset
            ].item()
        )
        prediction_position = target_position - 1
        if prediction_position < 0:
            raise ValueError(f"Example {row_index} has no decision input token")
        prediction_positions.append(prediction_position)
        binary_labels.append(binary_label)
    return prediction_positions, binary_labels


def extract_binary_decision_states(
    output: torch.Tensor,
    labels: torch.LongTensor,
    response_spec: BinaryResponseSpec,
    *,
    ignore_index: int = -100,
    decision_position: Literal["prefix", "unique_suffix"] = "prefix",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract final hidden states immediately before the yes/no token."""

    if output.ndim != 3 or labels.ndim != 2 or output.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected output [B,L,H] and labels [B,L], got {output.shape=} "
            f"{labels.shape=}"
        )
    positions, binary_labels = _binary_decision_positions_and_labels(
        labels,
        response_spec,
        ignore_index=ignore_index,
        decision_position=decision_position,
    )
    states = torch.stack(
        [output[row_index, position].float() for row_index, position in enumerate(positions)]
    )
    return states, output.new_tensor(binary_labels, dtype=torch.float32)


def within_query_supervised_contrastive_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    group_size: int,
    temperature: float = 0.1,
    center: bool = True,
) -> torch.Tensor:
    """Supervised contrastive loss over candidates belonging to the same query.

    The query mean is removed before cosine similarity so the auxiliary focuses
    on candidate-specific evidence rather than the shared query/prompt prefix.
    Only relevant candidates are positive anchors/peers. Negatives participate
    in the denominator but are deliberately not pulled toward one another,
    because distinct clothing failures need not share a semantic representation.
    """

    if hidden_states.ndim != 2 or labels.ndim != 1:
        raise ValueError("hidden_states and labels must be [N,H] and [N]")
    if hidden_states.shape[0] != labels.numel():
        raise ValueError("hidden_states and labels must have the same first dimension")
    if group_size <= 1 or hidden_states.shape[0] % group_size:
        raise ValueError("candidate count must be divisible by group_size > 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    num_groups = hidden_states.shape[0] // group_size
    grouped = hidden_states.float().reshape(num_groups, group_size, -1)
    grouped_labels = labels.reshape(num_groups, group_size)
    if center:
        grouped = grouped - grouped.mean(dim=1, keepdim=True)
    residual = F.normalize(grouped, dim=-1)
    similarity = torch.matmul(residual, residual.transpose(1, 2)) / float(temperature)

    diagonal = torch.eye(group_size, device=hidden_states.device, dtype=torch.bool)
    diagonal = diagonal.unsqueeze(0)
    relevant = grouped_labels > 0.5
    positive_mask = (
        relevant.unsqueeze(2) & relevant.unsqueeze(1) & ~diagonal
    )
    valid_anchor = relevant & positive_mask.any(dim=2)
    log_denominator = torch.logsumexp(
        similarity.masked_fill(diagonal, -torch.inf), dim=2
    )
    log_probability = similarity - log_denominator.unsqueeze(2)
    positive_count = positive_mask.sum(dim=2).clamp_min(1)
    anchor_loss = -(
        log_probability.masked_fill(~positive_mask, 0.0).sum(dim=2)
        / positive_count
    )
    valid_count = valid_anchor.sum(dim=1)
    valid_group = valid_count > 0
    if not bool(valid_group.any()):
        return hidden_states.sum() * 0.0
    group_loss = (
        anchor_loss.masked_fill(~valid_anchor, 0.0).sum(dim=1)
        / valid_count.clamp_min(1)
    )
    return group_loss[valid_group].mean()


def extract_binary_scores(
    output: torch.Tensor,
    labels: torch.LongTensor,
    response_spec: BinaryResponseSpec,
    *,
    ignore_index: int = -100,
    projection_weight: torch.Tensor | None = None,
    projection_module: torch.nn.Module | None = None,
    ordinal_token_ids: Sequence[int] | None = None,
    decision_position: Literal["prefix", "unique_suffix"] = "prefix",
    binary_readout: Literal["fp32", "bf16_ste"] = "fp32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``logit(yes)-logit(no)`` and the PAS relevance label per row.

    ``output`` is either full-vocabulary logits or final hidden states.  In the
    latter case only the yes/no rows of the frozen LM head are projected.
    """

    if output.ndim != 3 or labels.ndim != 2 or output.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected output [B,L,D] and labels [B,L], got {output.shape=} "
            f"{labels.shape=}"
        )
    if binary_readout not in {"fp32", "bf16_ste"}:
        raise ValueError(f"Unsupported binary readout: {binary_readout!r}")
    if ordinal_token_ids is not None and binary_readout != "fp32":
        raise ValueError("bf16_ste is defined only for the native yes/no readout")

    scores: list[torch.Tensor] = []
    local_projection = projection_weight
    if local_projection is not None and hasattr(local_projection, "to_local"):
        local_projection = local_projection.to_local()

    prediction_positions, binary_labels = _binary_decision_positions_and_labels(
        labels,
        response_spec,
        ignore_index=ignore_index,
        decision_position=decision_position,
    )
    for row_index, prediction_position in enumerate(prediction_positions):
        decision_output = output[row_index, prediction_position].float()
        if local_projection is None or decision_output.numel() == local_projection.shape[0]:
            if ordinal_token_ids is None:
                binary_logits = decision_output[
                    [response_spec.positive_token_id, response_spec.negative_token_id]
                ]
            else:
                ordinal_logits = decision_output[list(ordinal_token_ids)]
        elif decision_output.numel() == local_projection.shape[1]:
            if ordinal_token_ids is None:
                # Project the two token rows independently. Deployment emits
                # BF16 yes/no logits before subtracting them; projecting their
                # weight difference directly in FP32 hides that cancellation.
                binary_logits = _selected_projection_logits(
                    decision_output,
                    [response_spec.positive_token_id, response_spec.negative_token_id],
                    local_projection,
                    projection_module,
                )
            else:
                ordinal_weight = local_projection[list(ordinal_token_ids)].float()
                ordinal_logits = F.linear(decision_output, ordinal_weight)
        else:
            raise ValueError(
                "Binary projection shape mismatch: "
                f"output={decision_output.shape}, weight={local_projection.shape}"
            )
        if ordinal_token_ids is not None:
            # Match the deployed expected-token readout while retaining an
            # unbounded logit for pointwise BCE.  This monotonic transform does
            # not change candidate ordering.
            values = torch.arange(
                1, len(ordinal_token_ids) + 1, device=output.device, dtype=torch.float32
            )
            expected = (ordinal_logits.float().softmax(dim=0) * values).sum()
            probability = ((expected - 1.0) / max(len(ordinal_token_ids) - 1, 1)).clamp(
                1e-6, 1.0 - 1e-6
            )
            score = torch.logit(probability)
        else:
            if binary_readout == "bf16_ste":
                binary_logits = bf16_ste(binary_logits)
            score = binary_logits[0] - binary_logits[1]
        scores.append(score)

    return torch.stack(scores), output.new_tensor(binary_labels, dtype=torch.float32)


def _plackett_luce_log_prob(
    logits: torch.Tensor, permutations: torch.LongTensor
) -> torch.Tensor:
    """Log probability of complete rankings under a Plackett--Luce policy."""

    if logits.ndim != 1 or permutations.ndim != 2:
        raise ValueError("PL logits/permutations must be [K] and [N,K]")
    if permutations.shape[1] != logits.numel():
        raise ValueError("PL permutation width must equal the candidate count")
    expanded = logits.unsqueeze(0).expand(permutations.shape[0], -1)
    available = torch.ones_like(expanded, dtype=torch.bool)
    result = logits.new_zeros(permutations.shape[0])
    for position in range(permutations.shape[1]):
        action = permutations[:, position]
        result = result + expanded.gather(1, action[:, None]).squeeze(1)
        result = result - torch.logsumexp(
            expanded.masked_fill(~available, float("-inf")), dim=1
        )
        available = available.scatter(1, action[:, None], False)
    return result


def _ranking_rewards(
    permutations: torch.LongTensor,
    labels: torch.Tensor,
    *,
    ap_weight: float,
    r1_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact AP and R1 rewards for sampled or enumerated rankings."""

    ordered = labels[permutations].float()
    ranks = torch.arange(
        1, ordered.shape[1] + 1, device=ordered.device, dtype=torch.float32
    )
    ap = (
        ordered.cumsum(dim=1) / ranks.unsqueeze(0) * ordered
    ).sum(dim=1) / labels.float().sum().clamp_min(1.0)
    r1 = ordered[:, 0]
    reward = ap_weight * ap + r1_weight * r1
    return reward, ap, r1


def plackett_luce_policy_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    samples: int,
    exact_max_k: int,
    ap_reward_weight: float,
    r1_reward_weight: float,
    preserve: bool,
    preserve_anchor_weight: float,
    preserve_anchor_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize the ranking policy's hard-label AP/R1 reward.

    For small lists we sum over every permutation, producing the exact
    expected metric and its gradient.  Larger lists use exact PL samples
    (Gumbel top-k) and a leave-one-out, within-query reward baseline.  The
    Preserve groups receive a hard-label Rank-1 margin anchor.  No parent
    logits, teacher scores, soft labels, or deployment-time fusion are used.
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("PL score and label shapes must match")
    if temperature <= 0 or samples < 2 or exact_max_k < 2:
        raise ValueError("PL temperature/exact_max_k must be positive and samples >= 2")
    if ap_reward_weight < 0 or r1_reward_weight < 0:
        raise ValueError("PL reward weights must be nonnegative")
    if ap_reward_weight + r1_reward_weight <= 0:
        raise ValueError("At least one PL reward weight must be positive")
    if preserve_anchor_weight < 0 or preserve_anchor_margin < 0:
        raise ValueError("PL preserve anchor weight/margin must be nonnegative")

    logits = scores / temperature
    k = scores.numel()
    if k <= exact_max_k:
        permutations = torch.tensor(
            list(itertools.permutations(range(k))),
            device=scores.device,
            dtype=torch.long,
        )
        log_policy = _plackett_luce_log_prob(logits, permutations)
        log_policy = log_policy - torch.logsumexp(log_policy, dim=0)
        with torch.no_grad():
            reward, ap, r1 = _ranking_rewards(
                permutations,
                labels,
                ap_weight=ap_reward_weight,
                r1_weight=r1_reward_weight,
            )
        probability = log_policy.exp()
        expected_reward = (probability * reward).sum()
        expected_ap = (probability * ap).sum()
        expected_r1 = (probability * r1).sum()
        policy_gradient = -expected_reward
    else:
        uniform = torch.rand(
            samples, k, device=scores.device, dtype=torch.float32
        ).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        permutations = torch.argsort(
            logits.detach().unsqueeze(0) + gumbel,
            dim=1,
            descending=True,
            stable=True,
        )
        log_policy = _plackett_luce_log_prob(logits, permutations)
        with torch.no_grad():
            reward, ap, r1 = _ranking_rewards(
                permutations,
                labels,
                ap_weight=ap_reward_weight,
                r1_weight=r1_reward_weight,
            )
            leave_one_out = (reward.sum() - reward) / float(samples - 1)
            advantage = reward - leave_one_out
        policy_gradient = -(advantage * log_policy).mean()
        expected_reward = reward.mean()
        expected_ap = ap.mean()
        expected_r1 = r1.mean()
    positive = labels > 0.5
    negative = ~positive
    preserve_anchor = (
        F.relu(
            scores[negative].amax()
            + preserve_anchor_margin
            - scores[positive].amax()
        )
        if preserve
        else scores.sum() * 0.0
    )
    loss = policy_gradient + preserve_anchor_weight * preserve_anchor
    first_probability = F.softmax(logits, dim=0)
    entropy = -(first_probability * F.log_softmax(logits, dim=0)).sum()
    return loss, {
        "policy_reward": expected_reward.detach(),
        "policy_expected_ap": expected_ap.detach(),
        "policy_expected_r1": expected_r1.detach(),
        "policy_preserve_anchor": preserve_anchor.detach(),
        "policy_first_entropy": entropy.detach(),
        "policy_preserve_fraction": scores.new_tensor(float(preserve)),
    }


def rb_plackett_luce_expected_ap_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    outer_points: int,
    inner_points: int,
    ap_reward_weight: float,
    r1_reward_weight: float,
    preserve: bool,
    preserve_anchor_weight: float,
    preserve_anchor_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Deterministic expected AP/R1 under a Plackett--Luce policy.

    PL rankings are equivalent to ordering independent exponential arrival
    times with rates ``exp(score / temperature)``.  Conditional on positive
    item ``i`` arriving, the events that the other candidates arrived first
    are independent Bernoulli variables.  The identity
    ``1 / (1 + n) = integral_0^1 x**n dx`` then turns the expected AP
    contribution into a two-dimensional integral.  The inner integrand is a
    polynomial of degree K-1 (so 10-point Gauss--Legendre is exact for K=20);
    the outer integral is evaluated deterministically.  This is a
    Rao--Blackwellized policy objective: no sampled permutations, baselines,
    parent logits, teacher scores, or soft labels are involved.
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("RB-PL score and label shapes must match")
    if temperature <= 0 or outer_points < 2 or inner_points < 2:
        raise ValueError("RB-PL temperature/quadrature points must be positive")
    if ap_reward_weight < 0 or r1_reward_weight < 0:
        raise ValueError("RB-PL reward weights must be nonnegative")
    if ap_reward_weight + r1_reward_weight <= 0:
        raise ValueError("At least one RB-PL reward weight must be positive")

    # Mapping the exponential-race arrival u to z=exp(-u) makes the outer
    # measure uniform on [0,1], which is substantially more accurate than a
    # fixed Laguerre grid when score ratios are large.
    outer_nodes_np, outer_weights_np = np.polynomial.legendre.leggauss(outer_points)
    inner_nodes_np, inner_weights_np = np.polynomial.legendre.leggauss(inner_points)
    outer_z = scores.new_tensor((outer_nodes_np + 1.0) * 0.5)
    outer_weights = scores.new_tensor(outer_weights_np * 0.5)
    inner_x = scores.new_tensor((inner_nodes_np + 1.0) * 0.5)
    inner_weights = scores.new_tensor(inner_weights_np * 0.5)

    logits = scores / temperature
    candidates = torch.arange(scores.numel(), device=scores.device)
    positive_indices = torch.where(labels > 0.5)[0]
    contributions: list[torch.Tensor] = []
    for positive_index in positive_indices:
        other = candidates != positive_index
        rate_ratio = torch.exp(logits[other] - logits[positive_index])
        before_probability = 1.0 - torch.pow(
            outer_z[:, None], rate_ratio[None, :]
        )
        factors = (
            1.0
            - before_probability[:, :, None]
            + before_probability[:, :, None] * inner_x[None, None, :]
        )
        product = factors.prod(dim=1)
        other_positive = (labels[other] > 0.5).to(scores.dtype)
        numerator = product * (
            1.0
            + (
                before_probability[:, :, None]
                * inner_x[None, None, :]
                / factors
                * other_positive[None, :, None]
            ).sum(dim=1)
        )
        contributions.append(
            (outer_weights[:, None] * inner_weights[None, :] * numerator).sum()
        )
    expected_ap = torch.stack(contributions).mean()
    first_probability = F.softmax(logits, dim=0)
    expected_r1 = (first_probability * labels).sum()
    expected_reward = (
        ap_reward_weight * expected_ap + r1_reward_weight * expected_r1
    )

    positive = labels > 0.5
    negative = ~positive
    preserve_anchor = (
        F.relu(
            scores[negative].amax()
            + preserve_anchor_margin
            - scores[positive].amax()
        )
        if preserve
        else scores.sum() * 0.0
    )
    loss = -expected_reward + preserve_anchor_weight * preserve_anchor
    entropy = -(first_probability * F.log_softmax(logits, dim=0)).sum()
    return loss, {
        "policy_reward": expected_reward.detach(),
        "policy_expected_ap": expected_ap.detach(),
        "policy_expected_r1": expected_r1.detach(),
        "policy_preserve_anchor": preserve_anchor.detach(),
        "policy_first_entropy": entropy.detach(),
        "policy_preserve_fraction": scores.new_tensor(float(preserve)),
    }


def misordered_lambda_ap_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact-delta LambdaAP on only the student's currently wrong pairs.

    The current stable ranking and every swap utility are detached, matching
    LambdaRank's local-metric construction.  A pair is active exactly when an
    irrelevant candidate currently precedes a relevant candidate.  Its weight
    is the exact AP@K gain obtained by swapping those two ranks.  Normalising
    the active weight mass per query prevents queries with many positives (and
    therefore many possible pairs) from dominating the update.  The softplus
    surrogate is the stable logistic LambdaLoss; multiplying by temperature
    preserves its score-space gradient scale.
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1).to(scores.device)
    if scores.shape != labels.shape:
        raise ValueError("LambdaAP score and label shapes must match")
    if temperature <= 0:
        raise ValueError("LambdaAP temperature must be positive")
    positive, negative, swap_weight = exact_ap_swap_weights(
        scores,
        labels,
        labels.sum(),
    )
    with torch.no_grad():
        order = torch.argsort(scores.detach(), descending=True, stable=True)
        position = torch.empty_like(order)
        position[order] = torch.arange(order.numel(), device=order.device)
        misordered = position[negative].unsqueeze(0) < position[positive].unsqueeze(1)
        active_weight = swap_weight * misordered.to(swap_weight.dtype)
        weight_mass = active_weight.sum()
        active_fraction = misordered.float().mean()
    pair_loss = temperature * F.softplus(
        (scores[negative].unsqueeze(0) - scores[positive].unsqueeze(1))
        / temperature
    )
    loss = (active_weight * pair_loss).sum() / weight_mass.clamp_min(1e-12)
    return loss, {
        "safe_lambda_active_pair_fraction": active_fraction.detach(),
        "misordered_lambda_ap_weight_mass": weight_mass.detach(),
    }


def asymmetric_safe_residual_lambda_ap_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    retriever_scores: torch.Tensor,
    *,
    temperature: float,
    margin: float,
    repair_weight: float = 1.0,
    preservation_weight: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Repair uncertain retriever pairs without relearning its safe ordering.

    ``scores`` are the deployed fused scores.  The frozen retriever score is
    used only to choose the *direction of the hard-label objective*, never as
    a target: positive/negative pairs whose retriever z-margin is at most
    ``margin`` receive a LambdaAP logistic repair loss.  Pairs above that
    threshold receive exactly zero gradient while their fused margin remains
    safe, and a one-sided hinge only if the residual update erodes it below
    ``margin``.  This differs from ``misordered_lambda_ap`` (student-order
    active set) and ``incumbent_safe_lambda_ap`` (row-0/query-level guard).
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1).to(scores.device)
    retriever_scores = retriever_scores.float().reshape(-1).to(scores.device)
    if scores.shape != labels.shape or scores.shape != retriever_scores.shape:
        raise ValueError("safe residual score, label, and retriever shapes must match")
    if temperature <= 0 or margin < 0 or epsilon <= 0:
        raise ValueError("safe residual temperature/epsilon must be positive and margin nonnegative")
    if repair_weight < 0 or preservation_weight < 0:
        raise ValueError("safe residual repair/preservation weights must be nonnegative")
    if repair_weight + preservation_weight <= 0:
        raise ValueError("safe residual repair/preservation weights cannot both be zero")

    with torch.no_grad():
        retriever_centered = retriever_scores - retriever_scores.mean()
        retriever_z = retriever_centered / (
            retriever_scores.std(unbiased=False) + epsilon
        )
    positive, negative, swap_weight = exact_ap_swap_weights(
        scores, labels, labels.sum()
    )
    fused_margin = (
        scores[positive].unsqueeze(1) - scores[negative].unsqueeze(0)
    )
    with torch.no_grad():
        retriever_margin = (
            retriever_z[positive].unsqueeze(1)
            - retriever_z[negative].unsqueeze(0)
        )
        repair = retriever_margin <= margin
        preserve = ~repair
        weight_mass = swap_weight.sum().clamp_min(1e-12)

    repair_loss = temperature * F.softplus((margin - fused_margin) / temperature)
    # ReLU is intentionally exact here: a safe preserved pair has zero loss
    # and zero gradient rather than the nonzero tail of a logistic surrogate.
    preservation_hinge = F.relu(margin - fused_margin)
    weighted_repair = swap_weight * repair.to(swap_weight.dtype) * repair_loss
    weighted_preservation = (
        swap_weight * preserve.to(swap_weight.dtype) * preservation_hinge
    )
    loss = (
        repair_weight * weighted_repair.sum()
        + preservation_weight * weighted_preservation.sum()
    ) / weight_mass
    preserve_count = preserve.sum().clamp_min(1)
    return loss, {
        "safe_residual_repair_pair_fraction": repair.float().mean().detach(),
        "safe_residual_preserve_active_fraction": (
            (preserve & (fused_margin.detach() < margin)).sum() / preserve_count
        ).detach(),
        "safe_residual_repair_loss": (
            weighted_repair.sum() / weight_mass
        ).detach(),
        "safe_residual_preservation_loss": (
            weighted_preservation.sum() / weight_mass
        ).detach(),
    }


def robust_normalized_lambdaap_soft_r1_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    candidate_weights: torch.Tensor | None = None,
    pair_temperature: float,
    soft_r1_temperature: float,
    lambdaap_weight: float,
    soft_r1_weight: float,
    robust_pair_q: float,
    query_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """All-pair LambdaAP/soft-R1 with bounded candidate-level influence.

    Exact detached AP-swap utility weights cover every positive-negative pair.
    The highest-influence 10% of positive and negative candidates retain
    ``robust_pair_q`` mass instead of being deleted.  Normalizing inside every
    query prevents positive cardinality from changing its optimizer weight;
    averaging the returned losses then gives every query equal mass.  The R1
    term uses positive probability mass, never a weakest-positive boundary.
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1).to(scores.device)
    if scores.shape != labels.shape:
        raise ValueError("robust LambdaAP score and label shapes must match")
    if candidate_weights is not None:
        candidate_weights = candidate_weights.float().reshape(-1).to(scores.device)
        if candidate_weights.shape != scores.shape:
            raise ValueError("robust LambdaAP candidate-weight shape mismatch")
        if not bool(torch.all((candidate_weights == 0) | (candidate_weights == 1))):
            raise ValueError("robust LambdaAP requires binary candidate weights")
        retained = candidate_weights > 0
        scores = scores[retained]
        labels = labels[retained]
        if not bool((labels > 0.5).any()) or not bool((labels < 0.5).any()):
            raise ValueError(
                "masked robust LambdaAP requires a retained positive and negative"
            )
    if pair_temperature <= 0 or soft_r1_temperature <= 0:
        raise ValueError("robust LambdaAP temperatures must be positive")
    if lambdaap_weight < 0 or soft_r1_weight < 0 or lambdaap_weight + soft_r1_weight <= 0:
        raise ValueError("robust LambdaAP objective weights are invalid")
    if not 0 < robust_pair_q <= 1:
        raise ValueError("robust_pair_q must lie in (0,1]")
    if query_weight < 0:
        raise ValueError("robust LambdaAP query_weight must be nonnegative")
    positive, negative, swap_weight = exact_ap_swap_weights(
        scores, labels, labels.sum()
    )
    pair_loss = pair_temperature * F.softplus(
        (scores[negative].unsqueeze(0) - scores[positive].unsqueeze(1))
        / pair_temperature
    )
    with torch.no_grad():
        influence = swap_weight * pair_loss.detach()
        positive_factor = torch.ones(len(positive), device=scores.device)
        negative_factor = torch.ones(len(negative), device=scores.device)
        if len(positive) > 1:
            trim = max(1, math.ceil(0.10 * len(positive)))
            outliers = influence.mean(dim=1).topk(trim).indices
            positive_factor[outliers] = robust_pair_q
        if len(negative) > 1:
            trim = max(1, math.ceil(0.10 * len(negative)))
            outliers = influence.mean(dim=0).topk(trim).indices
            negative_factor[outliers] = robust_pair_q
        robust_weight = positive_factor.unsqueeze(1) * negative_factor.unsqueeze(0)
        pair_weight = swap_weight * robust_weight
        pair_weight = pair_weight / pair_weight.sum().clamp_min(1e-12)
    lambda_loss = (pair_weight * pair_loss).sum()
    log_probability = F.log_softmax(scores / soft_r1_temperature, dim=0)
    soft_r1 = -torch.logsumexp(
        log_probability.masked_fill(labels < 0.5, -torch.inf), dim=0
    )
    loss = query_weight * (
        lambdaap_weight * lambda_loss + soft_r1_weight * soft_r1
    )
    return loss, {
        "robust_lambdaap": lambda_loss.detach(),
        "robust_soft_r1": soft_r1.detach(),
        "robust_pair_weight_mass": pair_weight.sum().detach(),
        "robust_positive_trim_fraction": (positive_factor < 1).float().mean(),
        "robust_negative_trim_fraction": (negative_factor < 1).float().mean(),
        "robust_query_weight": scores.new_tensor(query_weight).detach(),
    }


def retriever_residual_fused_scores(
    cr3_scores: torch.Tensor,
    retriever_scores: torch.Tensor,
    *,
    group_size: int,
    alpha: float | torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Return the exact deployed within-query z-score fusion.

    The retriever branch is frozen metadata; gradients flow only through the
    CR3 score. Population variance (``unbiased=False``) matches NumPy's
    ``std`` used by the frozen R537 evaluator. Normalising each K-way query
    asks the LoRA to learn only the residual ordering missing from SigLIP2.
    """

    cr3_scores = cr3_scores.float().reshape(-1)
    retriever_scores = retriever_scores.float().reshape(-1).to(cr3_scores.device)
    if cr3_scores.shape != retriever_scores.shape:
        raise ValueError("CR3 and retriever score shapes must match")
    if group_size <= 1 or cr3_scores.numel() % group_size:
        raise ValueError("Residual fusion requires complete K-way query groups")
    if isinstance(alpha, torch.Tensor):
        if not bool(torch.isfinite(alpha).all()) or bool((alpha < 0).any()):
            raise ValueError("Residual fusion alpha must be finite and nonnegative")
    elif not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Residual fusion alpha must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Residual fusion epsilon must be finite and positive")

    def query_zscore(values: torch.Tensor) -> torch.Tensor:
        grouped = values.reshape(-1, group_size)
        centered = grouped - grouped.mean(dim=1, keepdim=True)
        scale = grouped.std(dim=1, keepdim=True, unbiased=False)
        return (centered / (scale + epsilon)).reshape(-1)

    # Retriever scores are data, never supervision, and must not accidentally
    # acquire a gradient if a caller supplies a tensor with requires_grad.
    retriever_z = query_zscore(retriever_scores.detach())
    cr3_z = query_zscore(cr3_scores)
    grouped_retriever = retriever_z.reshape(-1, group_size)
    grouped_cr3 = cr3_z.reshape(-1, group_size)
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(device=cr3_scores.device, dtype=cr3_scores.dtype)
        if alpha.numel() == 1:
            alpha = alpha.reshape(1, 1)
        elif alpha.numel() == grouped_cr3.shape[0]:
            alpha = alpha.reshape(-1, 1)
        else:
            raise ValueError(
                "Residual fusion alpha tensor must be scalar or one value per query"
            )
    else:
        alpha = float(alpha)
    return (grouped_retriever + alpha * grouped_cr3).reshape(-1)


class QueryAdaptiveResidualGate(torch.nn.Module):
    """Predict one bounded fusion weight from label-independent query geometry.

    The gate sees the complete top-K score distributions, never candidate
    labels.  Its features are detached so CR3 cannot game its own confidence
    estimate; LoRA still receives the ordinary gradient through the fused CR3
    scores.  Eight trainable scalars are enough to learn when the two rankers
    agree or disagree without fitting candidate identities.
    """

    FEATURE_NAMES = (
        "retriever_margin",
        "retriever_entropy",
        "cr3_margin",
        "cr3_entropy",
        "ranker_correlation",
        "soft_top_agreement",
        "retriever_top_probability",
    )
    LEGACY_FEATURE_VERSION = "r539_linear_v1"
    INTERACTION_FEATURE_VERSION = "r542_interactions_v1"
    INTERACTION_FEATURE_NAMES = FEATURE_NAMES + (
        "top1_disagreement",
        "ranker_correlation_squared",
        "cr3_margin_squared",
        "top1_disagreement_x_ranker_correlation",
        "top1_disagreement_x_ranker_correlation_squared",
        "top1_disagreement_x_cr3_margin",
    )

    def __init__(
        self,
        *,
        initial_alpha: float,
        minimum: float = 0.0,
        maximum: float = 8.0,
        feature_version: str = LEGACY_FEATURE_VERSION,
    ) -> None:
        super().__init__()
        if not (math.isfinite(minimum) and math.isfinite(maximum)) or minimum >= maximum:
            raise ValueError("Residual gate bounds must be finite and increasing")
        if not math.isfinite(initial_alpha) or not minimum < initial_alpha < maximum:
            raise ValueError("Initial residual alpha must lie strictly inside its bounds")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.initial_alpha = float(initial_alpha)
        if feature_version == self.LEGACY_FEATURE_VERSION:
            self.feature_names = self.FEATURE_NAMES
        elif feature_version == self.INTERACTION_FEATURE_VERSION:
            self.feature_names = self.INTERACTION_FEATURE_NAMES
        else:
            raise ValueError(f"Unsupported residual gate feature version: {feature_version!r}")
        self.feature_version = feature_version
        self.linear = torch.nn.Linear(len(self.feature_names), 1)
        fraction = (initial_alpha - minimum) / (maximum - minimum)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(math.log(fraction / (1.0 - fraction)))

    @staticmethod
    def _zscore(values: torch.Tensor, epsilon: float) -> torch.Tensor:
        centered = values - values.mean(dim=1, keepdim=True)
        return centered / (values.std(dim=1, keepdim=True, unbiased=False) + epsilon)

    @staticmethod
    def _normalized_entropy(values: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(values, dim=1)
        entropy = -(probability * torch.log_softmax(values, dim=1)).sum(dim=1)
        return entropy / math.log(values.shape[1])

    def features(
        self,
        cr3_scores: torch.Tensor,
        retriever_scores: torch.Tensor,
        *,
        group_size: int,
        epsilon: float,
    ) -> torch.Tensor:
        if group_size <= 1 or cr3_scores.numel() % group_size:
            raise ValueError("Residual gate requires complete K-way query groups")
        cr3 = cr3_scores.float().reshape(-1, group_size)
        retriever = retriever_scores.detach().float().reshape(-1, group_size).to(cr3.device)
        cr3_z = self._zscore(cr3, epsilon)
        retriever_z = self._zscore(retriever, epsilon)
        cr3_top2 = torch.topk(cr3_z, 2, dim=1).values
        retriever_top2 = torch.topk(retriever_z, 2, dim=1).values
        cr3_probability = torch.softmax(cr3_z, dim=1)
        retriever_probability = torch.softmax(retriever_z, dim=1)
        base_features = torch.stack(
            (
                torch.tanh(retriever_top2[:, 0] - retriever_top2[:, 1]),
                self._normalized_entropy(retriever_z),
                torch.tanh(cr3_top2[:, 0] - cr3_top2[:, 1]),
                self._normalized_entropy(cr3_z),
                (retriever_z * cr3_z).mean(dim=1).clamp(-1.0, 1.0),
                (group_size * (retriever_probability * cr3_probability).sum(dim=1) - 1.0)
                / (group_size - 1.0),
                retriever_probability.amax(dim=1),
            ),
            dim=1,
        )
        if self.feature_version == self.LEGACY_FEATURE_VERSION:
            return base_features.detach()

        rho = base_features[:, 4]
        cr3_gap = base_features[:, 2]
        disagreement = (retriever_z.argmax(dim=1) != cr3_z.argmax(dim=1)).to(rho.dtype)
        rho_squared = rho.square()
        interaction_features = torch.stack(
            (
                disagreement,
                rho_squared,
                cr3_gap.square(),
                disagreement * rho,
                disagreement * rho_squared,
                disagreement * cr3_gap,
            ),
            dim=1,
        )
        return torch.cat((base_features, interaction_features), dim=1).detach()

    def forward(
        self,
        cr3_scores: torch.Tensor,
        retriever_scores: torch.Tensor,
        *,
        group_size: int,
        epsilon: float,
    ) -> torch.Tensor:
        features = self.features(
            cr3_scores,
            retriever_scores,
            group_size=group_size,
            epsilon=epsilon,
        )
        fraction = torch.sigmoid(self.linear(features.float())).reshape(-1)
        return self.minimum + (self.maximum - self.minimum) * fraction


def r535_semantic_and_aux_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    pair_temperature: float,
    soft_r1_temperature: float,
    lambdaap_weight: float,
    soft_r1_weight: float,
    robust_pair_q: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve a natural K20 while weakly learning signed clause effects.

    The first 20 rows are an ordinary natural retrieval list.  The final four
    rows are ``matched_full, matched_drop, veto_full, veto_drop`` for one
    shared clause, with each adjacent pair holding the physical image fixed.
    Constants are deliberately sealed to the R535 precommit: the auxiliary is
    only five percent of the primary natural-list objective.
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1).to(scores.device)
    if scores.shape != labels.shape or scores.numel() != 24:
        raise ValueError("R535 semantic-AND loss requires aligned K24 scores/labels")
    expected_aux = labels.new_tensor([1.0, 0.0, 0.0, 1.0])
    if not bool(torch.equal(labels[20:], expected_aux)):
        raise ValueError(
            "R535 auxiliary slots must have technical labels [1,0,0,1]"
        )
    natural_labels = labels[:20]
    if not bool((natural_labels > 0.5).any()) or not bool(
        (natural_labels < 0.5).any()
    ):
        raise ValueError("R535 natural K20 requires positive and negative candidates")

    natural_loss, natural_components = robust_normalized_lambdaap_soft_r1_loss(
        scores[:20],
        natural_labels,
        pair_temperature=pair_temperature,
        soft_r1_temperature=soft_r1_temperature,
        lambdaap_weight=lambdaap_weight,
        soft_r1_weight=soft_r1_weight,
        robust_pair_q=robust_pair_q,
    )
    matched_delta = scores[20] - scores[21]
    veto_delta = scores[22] - scores[23]
    temperature = scores.new_tensor(0.5)
    one_sided_margin = scores.new_tensor(0.25)
    contrast_margin = scores.new_tensor(0.5)
    clause_terms = torch.stack(
        (
            temperature
            * F.softplus((one_sided_margin - matched_delta) / temperature),
            temperature
            * F.softplus((one_sided_margin + veto_delta) / temperature),
            temperature
            * F.softplus(
                (contrast_margin - (matched_delta - veto_delta)) / temperature
            ),
        )
    )
    clause_loss = clause_terms.mean()
    loss = natural_loss + 0.05 * clause_loss
    return loss, {
        **natural_components,
        "r535_natural_loss": natural_loss.detach(),
        "r535_clause_aux_loss": clause_loss.detach(),
        "r535_matched_delta": matched_delta.detach(),
        "r535_veto_delta": veto_delta.detach(),
        "r535_delta_contrast": (matched_delta - veto_delta).detach(),
        "r535_matched_sign_accuracy": (matched_delta > 0).float().detach(),
        "r535_veto_sign_accuracy": (veto_delta < 0).float().detach(),
    }


def consensus_trimmed_lambdaap_soft_r1_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    pair_temperature: float,
    soft_r1_temperature: float,
    lambdaap_weight: float,
    soft_r1_weight: float,
    robust_pair_q: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """LambdaAP with an adaptive, positive-only consensus guard.

    Multi-positive retrieval packets can contain one view whose supervision is
    inconsistent with the remaining positive bag.  Fixed hard-example trimming
    cannot distinguish that case from a merely difficult query.  Here each
    positive's detached AP-weighted logistic risk is compared with the robust
    within-bag median and MAD.  Only when there are at least four positives and
    the single worst positive is a clear risk outlier is its influence reduced
    to ``robust_pair_q``.  All negatives and every non-outlying positive retain
    full AP-swap mass.  The soft-R1 positive-bag term still requires positive
    probability mass at the top, so the guard cannot turn into an ignore-hard-
    queries objective.
    """

    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1).to(scores.device)
    if scores.shape != labels.shape:
        raise ValueError("consensus LambdaAP score and label shapes must match")
    if pair_temperature <= 0 or soft_r1_temperature <= 0:
        raise ValueError("consensus LambdaAP temperatures must be positive")
    if lambdaap_weight < 0 or soft_r1_weight < 0 or lambdaap_weight + soft_r1_weight <= 0:
        raise ValueError("consensus LambdaAP objective weights are invalid")
    if not 0 < robust_pair_q <= 1:
        raise ValueError("robust_pair_q must lie in (0,1]")

    positive, negative, swap_weight = exact_ap_swap_weights(
        scores, labels, labels.sum()
    )
    pair_loss = pair_temperature * F.softplus(
        (scores[negative].unsqueeze(0) - scores[positive].unsqueeze(1))
        / pair_temperature
    )
    with torch.no_grad():
        positive_mass = swap_weight.sum(dim=1).clamp_min(1e-12)
        positive_risk = (swap_weight * pair_loss.detach()).sum(dim=1) / positive_mass
        positive_factor = torch.ones(len(positive), device=scores.device)
        trim_triggered = scores.new_tensor(0.0)
        outlier_ratio = scores.new_tensor(1.0)
        if len(positive) >= 4:
            median = positive_risk.median()
            mad = (positive_risk - median).abs().median()
            scale = mad.clamp_min(0.10 * pair_temperature)
            worst_value, worst_index = positive_risk.max(dim=0)
            threshold = median + 2.5 * scale
            outlier_ratio = worst_value / threshold.clamp_min(1e-12)
            if worst_value > threshold:
                positive_factor[worst_index] = robust_pair_q
                trim_triggered = scores.new_tensor(1.0)
        pair_weight = swap_weight * positive_factor.unsqueeze(1)
        pair_weight = pair_weight / pair_weight.sum().clamp_min(1e-12)

    lambda_loss = (pair_weight * pair_loss).sum()
    log_probability = F.log_softmax(scores / soft_r1_temperature, dim=0)
    soft_r1 = -torch.logsumexp(
        log_probability.masked_fill(labels < 0.5, -torch.inf), dim=0
    )
    loss = lambdaap_weight * lambda_loss + soft_r1_weight * soft_r1
    return loss, {
        "consensus_lambdaap": lambda_loss.detach(),
        "consensus_soft_r1": soft_r1.detach(),
        "consensus_pair_weight_mass": pair_weight.sum().detach(),
        "consensus_trim_triggered": trim_triggered.detach(),
        "consensus_outlier_ratio": outlier_ratio.detach(),
    }


def rank_point_loss_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    candidate_weights: torch.Tensor | None = None,
    rank_weight: float = 1.0,
    point_weight: float = 1.0,
    rank_temperature: float = 1.0,
    rank_mode: Literal[
        "probability_mass",
        "bag_top1_logsumexp",
        "triplet_alignment_logsumexp",
        "worst_positive_hardest_negative_logmeanexp",
        "pu_certified_logsumexp",
        "pu_certified_logmeanexp",
        "pu_bag_top1_logsumexp",
        "nnpu_certified_hybrid",
        "nnpu_certified_hybrid",
        "orthogonal_cycle",
        "deployment_weighted_robust_cycle",
        "preservation_constrained_cycle_block",
        "smoothap_soft_r1",
        "plackett_luce_policy",
        "rb_plackett_luce_expected_ap",
        "counterfactual_set5",
        "paired_adjacent",
        "dynamic_top20_smoothap_soft_r1",
        "incumbent_guarded_smoothap_soft_r1",
        "incumbent_safe_lambda_ap",
        "misordered_lambda_ap",
        "asymmetric_safe_residual_lambda_ap",
        "robust_normalized_lambdaap_soft_r1",
        "paired_view_robust_normalized_lambdaap_soft_r1",
        "consensus_trimmed_lambdaap_soft_r1",
        "r535_semantic_and_aux",
        "all_pairs",
        "active_margin_all_pairs",
        "fixed_mass_active_all_pairs",
        "cardinality_compensated_fixed_mass",
        "trimmed_active_all_pairs",
        "natural20_inverse_pair",
        "weak_veto_mil",
        "active_lambda_ap",
        "head_all_pairs",
        "lambda_ndcg_top1",
        "constrained_retriever_lambda_ndcg_top1",
        "smooth_top1",
        "top1_competitor",
        "topm_all_pairs",
        "robust_topm_gce",
        "topm_logsumexp",
        "hybrid_all_pairs",
        "partial_negatives",
        "partial_negatives_logmeanexp",
        "retriever_top1_guarded",
        "retriever_top1_competitor",
        "constrained_retriever_top1_competitor",
        "parent_top1_corrective",
        "parent_top1_competitor",
        "dual_top1_competitor",
        "constrained_dual_top1_competitor",
        "dual_multi_positive_competitor",
        "query_selective_regret",
        "cross_query_selective_override",
        "tail_risk_selective_override",
        "nested_tail_risk_selective_override",
        "global_nested_tail_risk_selective_override",
        "global_action_faithful_topm_ap",
        "full_gallery_lambda_ap",
    ] = "probability_mass",
    bag_positive_temperature: float = 0.1,
    bag_negative_temperature: float = 0.1,
    pu_class_prior: float = 0.2,
    pu_nn_weight: float = 0.25,
    partial_negative_ratio: float = 0.2,
    rank_margin: float = 0.05,
    natural20_rank_weight: float = 1.0,
    inverse_pair_weight: float = 1.0,
    inverse_pair_offsets: torch.Tensor | None = None,
    weak_veto_temperature: float = 0.2,
    weak_veto_full_weight: float = 1.0,
    weak_veto_mil_weight: float = 1.0,
    robust_pair_q: float = 0.3,
    all_pairs_aux_weight: float = 0.0,
    positive_tail_weight: float = 0.0,
    positive_tail_temperature: float = 0.2,
    retriever_success_weight: float = 1.0,
    retriever_error_weight: float = 1.0,
    near_miss_k: int | None = None,
    near_miss_weight: float = 1.0,
    parent_preservation_margin: float | None = None,
    parent_preservation_teacher_margin_slack: float | None = None,
    parent_preservation_weight: float = 1.0,
    topk_preservation_k: int | None = None,
    topk_preservation_margin: float = 0.0,
    topk_preservation_weight: float = 0.0,
    point_temperature: float = 1.0,
    point_mode: Literal[
        "all",
        "positive_only",
        "class_balanced",
        "query_centered_class_balanced",
    ] = "all",
    negative_point_weight: float = 1.0,
    retriever_reference_scores: torch.Tensor | None = None,
    teacher_targets: torch.Tensor | None = None,
    teacher_weight: float = 0.0,
    teacher_repair_group_weight: float = 1.0,
    teacher_temperature: float = 1.0,
    teacher_mode: Literal[
        "pointwise",
        "pairwise_margin",
        "pairwise_margin_clamped",
        "pairwise_logit_delta_clamped",
        "pairwise_logit_delta_correct_only",
    ] = "pointwise",
    group_size: int | None = None,
    query_consistency_weight: float = 0.0,
    query_cross_weight: float = 0.0,
    total_relevant: torch.Tensor | None = None,
    full_gallery_rank1_weight: float = 0.0,
    full_gallery_active_topk: int | None = None,
    transition_pooled_weight: float = 0.0,
    transition_pooled_topk: int = 4,
    transition_pooled_temperature: float = 0.2,
    transition_pooled_margin: float = 0.0,
    cycle_weights: torch.Tensor | None = None,
    cycle_block_metadata: torch.Tensor | None = None,
    robust_cycle_rho: float = 0.25,
    robust_cycle_lambda: float = 1.0,
    preservation_cycle_margin: float = 2.0,
    preservation_cycle_weight: float = 4.0,
    repair_cycle_ap_weight: float = 0.25,
    preserve_cycle_ap_weight: float = 0.10,
    preserve_cycle_robust_scale: float = 0.25,
    r69_smoothap_temperature: float = 0.384007173733197,
    r69_soft_r1_temperature: float = 2.076483289679061,
    r69_smoothap_weight: float = 1.0,
    r69_soft_r1_weight: float = 0.25,
    policy_samples: int = 16,
    policy_exact_max_k: int = 7,
    policy_ap_reward_weight: float = 1.0,
    policy_r1_reward_weight: float = 1.0,
    policy_preserve_anchor_weight: float = 1.0,
    policy_preserve_anchor_margin: float = 0.0,
    policy_preserve: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the deployed fixed-list objective to contiguous query groups."""

    if (
        rank_weight < 0
        or point_weight < 0
        or teacher_weight < 0
        or rank_weight + point_weight + teacher_weight <= 0
    ):
        raise ValueError(
            "rank_weight, point_weight, and teacher_weight must be nonnegative "
            "and not all zero"
        )
    if all_pairs_aux_weight < 0:
        raise ValueError("all_pairs_aux_weight must be nonnegative")
    if positive_tail_weight < 0:
        raise ValueError("positive_tail_weight must be nonnegative")
    if positive_tail_temperature <= 0:
        raise ValueError("positive_tail_temperature must be positive")
    if not 0.0 < pu_class_prior < 1.0:
        raise ValueError("PU class prior must lie strictly between zero and one")
    if pu_nn_weight < 0.0:
        raise ValueError("PU non-negative risk weight must be nonnegative")
    if query_consistency_weight < 0:
        raise ValueError("query_consistency_weight must be nonnegative")
    if query_cross_weight < 0:
        raise ValueError("query_cross_weight must be nonnegative")
    if full_gallery_rank1_weight < 0:
        raise ValueError("full_gallery_rank1_weight must be nonnegative")
    if full_gallery_rank1_weight and rank_mode != "full_gallery_lambda_ap":
        raise ValueError(
            "full_gallery_rank1_weight requires rank_mode=full_gallery_lambda_ap"
        )
    if full_gallery_active_topk is not None and full_gallery_active_topk <= 0:
        raise ValueError("full_gallery_active_topk must be positive")
    if full_gallery_active_topk is not None and rank_mode != "full_gallery_lambda_ap":
        raise ValueError(
            "full_gallery_active_topk requires rank_mode=full_gallery_lambda_ap"
        )
    if transition_pooled_weight < 0:
        raise ValueError("transition_pooled_weight must be nonnegative")
    if transition_pooled_topk <= 0:
        raise ValueError("transition_pooled_topk must be positive")
    if transition_pooled_temperature <= 0:
        raise ValueError("transition_pooled_temperature must be positive")
    if transition_pooled_margin < 0:
        raise ValueError("transition_pooled_margin must be nonnegative")
    if transition_pooled_weight and rank_mode != "full_gallery_lambda_ap":
        raise ValueError(
            "transition_pooled_weight requires rank_mode=full_gallery_lambda_ap"
        )
    if robust_cycle_rho <= 0:
        raise ValueError("robust_cycle_rho must be positive")
    if robust_cycle_lambda < 0:
        raise ValueError("robust_cycle_lambda must be nonnegative")
    if cycle_weights is not None and rank_mode != "deployment_weighted_robust_cycle":
        raise ValueError(
            "cycle_weights require rank_mode=deployment_weighted_robust_cycle"
        )
    if (
        cycle_block_metadata is not None
        and rank_mode != "preservation_constrained_cycle_block"
    ):
        raise ValueError(
            "cycle_block_metadata require "
            "rank_mode=preservation_constrained_cycle_block"
        )
    if preservation_cycle_margin < 0 or preservation_cycle_weight < 0:
        raise ValueError("preservation cycle margin/weight must be nonnegative")
    if repair_cycle_ap_weight < 0 or preserve_cycle_ap_weight < 0:
        raise ValueError("cycle AP weights must be nonnegative")
    if preserve_cycle_robust_scale < 0:
        raise ValueError("preserve_cycle_robust_scale must be nonnegative")
    if r69_smoothap_temperature <= 0 or r69_soft_r1_temperature <= 0:
        raise ValueError("r69 temperatures must be positive")
    if r69_smoothap_weight < 0 or r69_soft_r1_weight < 0:
        raise ValueError("r69 objective weights must be nonnegative")
    if policy_samples < 2 or policy_exact_max_k < 2:
        raise ValueError("policy_samples and policy_exact_max_k must be at least 2")
    if policy_ap_reward_weight < 0 or policy_r1_reward_weight < 0:
        raise ValueError("policy reward weights must be nonnegative")
    if policy_ap_reward_weight + policy_r1_reward_weight <= 0:
        raise ValueError("policy AP/R1 reward weights cannot both be zero")
    if policy_preserve_anchor_weight < 0 or policy_preserve_anchor_margin < 0:
        raise ValueError("policy preserve anchor weight/margin must be nonnegative")
    if (
        rank_mode
        in {
            "smoothap_soft_r1",
            "counterfactual_set5",
            "dynamic_top20_smoothap_soft_r1",
            "incumbent_guarded_smoothap_soft_r1",
            "incumbent_safe_lambda_ap",
        }
        and r69_smoothap_weight + r69_soft_r1_weight <= 0
    ):
        raise ValueError("r69 SmoothAP and soft-R1 weights cannot both be zero")
    if all_pairs_aux_weight and not rank_weight:
        raise ValueError("all_pairs_aux_weight requires nonzero rank_weight")
    if retriever_success_weight < 1.0:
        raise ValueError("retriever_success_weight must be at least 1")
    if rank_temperature <= 0 or point_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("loss temperatures must be positive")
    if bag_positive_temperature <= 0 or bag_negative_temperature <= 0:
        raise ValueError("bag positive/negative temperatures must be positive")
    if negative_point_weight <= 0:
        raise ValueError("negative_point_weight must be positive")
    if not 0.0 <= teacher_repair_group_weight <= 1.0:
        raise ValueError("teacher_repair_group_weight must lie in [0, 1]")
    if teacher_mode == "pointwise" and teacher_repair_group_weight != 1.0:
        raise ValueError(
            "teacher_repair_group_weight currently requires a pairwise teacher mode"
        )
    if not 0.0 < partial_negative_ratio <= 1.0:
        raise ValueError("partial_negative_ratio must lie in (0, 1]")
    if rank_margin < 0.0:
        raise ValueError("rank_margin must be nonnegative")
    if natural20_rank_weight < 0.0 or inverse_pair_weight < 0.0:
        raise ValueError("natural20/inverse-pair weights must be nonnegative")
    if (
        rank_mode == "natural20_inverse_pair"
        and natural20_rank_weight + inverse_pair_weight <= 0.0
    ):
        raise ValueError("natural20/inverse-pair weights cannot both be zero")
    if inverse_pair_offsets is not None and rank_mode != "natural20_inverse_pair":
        raise ValueError(
            "inverse_pair_offsets require rank_mode=natural20_inverse_pair"
        )
    if weak_veto_temperature <= 0.0:
        raise ValueError("weak_veto_temperature must be positive")
    if weak_veto_full_weight < 0.0 or weak_veto_mil_weight < 0.0:
        raise ValueError("weak-veto full/MIL weights must be nonnegative")
    if (
        rank_mode == "weak_veto_mil"
        and weak_veto_full_weight + weak_veto_mil_weight <= 0.0
    ):
        raise ValueError("weak-veto full/MIL weights cannot both be zero")
    if not 0.0 < robust_pair_q <= 1.0:
        raise ValueError("robust_pair_q must lie in (0, 1]")
    if retriever_error_weight < 0.0:
        raise ValueError("retriever_error_weight must be nonnegative")
    if near_miss_k is not None and near_miss_k < 2:
        raise ValueError("near_miss_k must be at least 2")
    if near_miss_weight < 1.0:
        raise ValueError("near_miss_weight must be at least 1")
    if near_miss_weight > 1.0 and near_miss_k is None:
        raise ValueError("near_miss_k is required when near_miss_weight exceeds 1")
    if parent_preservation_margin is not None and parent_preservation_margin < 0.0:
        raise ValueError("parent_preservation_margin must be nonnegative")
    if (
        parent_preservation_teacher_margin_slack is not None
        and parent_preservation_teacher_margin_slack < 0.0
    ):
        raise ValueError(
            "parent_preservation_teacher_margin_slack must be nonnegative"
        )
    if parent_preservation_weight < 1.0:
        raise ValueError("parent_preservation_weight must be at least 1")
    if topk_preservation_k is not None and topk_preservation_k < 1:
        raise ValueError("topk_preservation_k must be positive")
    if topk_preservation_margin < 0.0:
        raise ValueError("topk_preservation_margin must be nonnegative")
    if topk_preservation_weight < 0.0:
        raise ValueError("topk_preservation_weight must be nonnegative")
    if topk_preservation_weight and topk_preservation_k is None:
        raise ValueError(
            "topk_preservation_k is required when topk_preservation_weight is nonzero"
        )
    if rank_mode not in {
        "probability_mass",
        "bag_top1_logsumexp",
        "triplet_alignment_logsumexp",
        "worst_positive_hardest_negative_logmeanexp",
        "pu_certified_logsumexp",
        "pu_certified_logmeanexp",
        "pu_bag_top1_logsumexp",
        "nnpu_certified_hybrid",
        "orthogonal_cycle",
        "deployment_weighted_robust_cycle",
        "preservation_constrained_cycle_block",
        "smoothap_soft_r1",
        "plackett_luce_policy",
        "rb_plackett_luce_expected_ap",
        "counterfactual_set5",
        "paired_adjacent",
        "dynamic_top20_smoothap_soft_r1",
        "incumbent_guarded_smoothap_soft_r1",
        "incumbent_safe_lambda_ap",
        "misordered_lambda_ap",
        "asymmetric_safe_residual_lambda_ap",
        "robust_normalized_lambdaap_soft_r1",
        "paired_view_robust_normalized_lambdaap_soft_r1",
        "consensus_trimmed_lambdaap_soft_r1",
        "r535_semantic_and_aux",
        "all_pairs",
        "active_margin_all_pairs",
        "fixed_mass_active_all_pairs",
        "cardinality_compensated_fixed_mass",
        "trimmed_active_all_pairs",
        "natural20_inverse_pair",
        "weak_veto_mil",
        "active_lambda_ap",
        "head_all_pairs",
        "lambda_ndcg_top1",
        "constrained_retriever_lambda_ndcg_top1",
        "smooth_top1",
        "top1_competitor",
        "topm_all_pairs",
        "robust_topm_gce",
        "topm_logsumexp",
        "hybrid_all_pairs",
        "partial_negatives",
        "partial_negatives_logmeanexp",
        "retriever_top1_guarded",
        "retriever_top1_competitor",
        "constrained_retriever_top1_competitor",
        "parent_top1_corrective",
        "parent_top1_competitor",
        "dual_top1_competitor",
        "constrained_dual_top1_competitor",
        "dual_multi_positive_competitor",
        "query_selective_regret",
        "cross_query_selective_override",
        "tail_risk_selective_override",
        "nested_tail_risk_selective_override",
        "global_nested_tail_risk_selective_override",
        "global_action_faithful_topm_ap",
        "full_gallery_lambda_ap",
    }:
        raise ValueError(
            "rank_mode must be 'probability_mass', 'bag_top1_logsumexp', "
            "'triplet_alignment_logsumexp', "
            "'orthogonal_cycle', "
            "'deployment_weighted_robust_cycle', "
            "'preservation_constrained_cycle_block', 'all_pairs', "
            "'smoothap_soft_r1', 'plackett_luce_policy', "
            "'rb_plackett_luce_expected_ap', "
            "'counterfactual_set5', 'paired_adjacent', "
            "'dynamic_top20_smoothap_soft_r1', "
            "'incumbent_guarded_smoothap_soft_r1', "
            "'incumbent_safe_lambda_ap', "
            "'head_all_pairs', 'lambda_ndcg_top1', "
            "'constrained_retriever_lambda_ndcg_top1', 'smooth_top1', "
            "'top1_competitor', 'topm_all_pairs', 'robust_topm_gce', "
            "'topm_logsumexp', "
            "'hybrid_all_pairs', 'partial_negatives', "
            "'partial_negatives_logmeanexp', 'retriever_top1_guarded', "
            "'retriever_top1_competitor', 'parent_top1_corrective', or "
            "'parent_top1_competitor', 'dual_top1_competitor', "
            "'constrained_dual_top1_competitor', or "
            "'dual_multi_positive_competitor', 'query_selective_regret', "
            "'cross_query_selective_override', or "
            "'tail_risk_selective_override', or "
            "'nested_tail_risk_selective_override', or "
            "'global_nested_tail_risk_selective_override', or "
            "'global_action_faithful_topm_ap', or 'full_gallery_lambda_ap'"
        )
    if rank_mode in {
        "nested_tail_risk_selective_override",
        "global_nested_tail_risk_selective_override",
        "global_action_faithful_topm_ap",
        "topm_all_pairs",
        "robust_topm_gce",
        "topm_logsumexp",
    } and near_miss_k is None:
        raise ValueError(
            f"{rank_mode} requires near_miss_k"
        )
    if point_mode not in {
        "all",
        "positive_only",
        "class_balanced",
        "query_centered_class_balanced",
    }:
        raise ValueError(
            "point_mode must be 'all', 'positive_only', 'class_balanced', "
            "or 'query_centered_class_balanced'"
        )
    if teacher_mode not in {
        "pointwise",
        "pairwise_margin",
        "pairwise_margin_clamped",
        "pairwise_logit_delta_clamped",
        "pairwise_logit_delta_correct_only",
    }:
        raise ValueError(
            "teacher_mode must be 'pointwise', 'pairwise_margin', or "
            "'pairwise_margin_clamped', 'pairwise_logit_delta_clamped', or "
            "'pairwise_logit_delta_correct_only'"
        )
    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(f"Score/label shape mismatch: {scores.shape=} {labels.shape=}")
    if candidate_weights is not None:
        candidate_weights = candidate_weights.float().reshape(-1).to(scores.device)
        if candidate_weights.shape != scores.shape:
            raise ValueError(
                "Candidate weight shape mismatch: "
                f"{candidate_weights.shape=} {scores.shape=}"
            )
        if not bool(torch.isfinite(candidate_weights).all()) or bool(
            (candidate_weights < 0).any()
        ):
            raise ValueError("Candidate weights must be finite and nonnegative")
        if rank_mode not in {
            "all_pairs",
            "active_margin_all_pairs",
            "fixed_mass_active_all_pairs",
            "cardinality_compensated_fixed_mass",
            "trimmed_active_all_pairs",
            "pu_certified_logsumexp",
            "pu_certified_logmeanexp",
            "pu_bag_top1_logsumexp",
            "nnpu_certified_hybrid",
            "robust_normalized_lambdaap_soft_r1",
        }:
            raise ValueError(
                "Candidate weights currently require rank_mode='all_pairs', "
                "'active_margin_all_pairs', 'trimmed_active_all_pairs', "
                "'pu_certified_logsumexp', "
                "'pu_certified_logmeanexp', or "
                "'pu_bag_top1_logsumexp', 'nnpu_certified_hybrid', or "
                "'robust_normalized_lambdaap_soft_r1'"
            )
    parent_rank_modes = {
        "parent_top1_corrective",
        "parent_top1_competitor",
        "dual_top1_competitor",
        "constrained_dual_top1_competitor",
        "dual_multi_positive_competitor",
    }
    if teacher_weight or rank_mode in parent_rank_modes:
        if teacher_targets is None:
            raise ValueError(
                "teacher_targets are required when teacher_weight is nonzero or "
                "rank_mode is 'parent_top1_corrective' or "
                "'parent_top1_competitor', 'dual_top1_competitor', "
                "'constrained_dual_top1_competitor', or "
                "'dual_multi_positive_competitor'"
            )

        teacher_targets = teacher_targets.float().reshape(-1).to(scores.device)
        if teacher_targets.shape != scores.shape:
            raise ValueError(
                "Teacher target shape mismatch: "
                f"{teacher_targets.shape=} {scores.shape=}"
            )
        if not bool(((teacher_targets >= 0) & (teacher_targets <= 1)).all()):
            raise ValueError("teacher_targets must lie in [0, 1]")
    if group_size is None:
        group_size = scores.numel()
    if group_size <= 1 or scores.numel() % group_size:
        raise ValueError(
            f"Batch size {scores.numel()} must be divisible by group_size={group_size}"
        )
    if rank_mode in {
        "smoothap_soft_r1",
        "counterfactual_set5",
        "dynamic_top20_smoothap_soft_r1",
        "incumbent_guarded_smoothap_soft_r1",
        "incumbent_safe_lambda_ap",
        "bag_top1_logsumexp",
    }:
        if rank_mode != "smoothap_soft_r1" and group_size != 20:
            raise ValueError(
                f"{rank_mode} requires group_size=20"
            )
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError(
                f"{rank_mode} is isolated from point and teacher objectives"
            )
        if (
            all_pairs_aux_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError(
                f"{rank_mode} does not accept auxiliary ranking objectives"
            )
    if rank_mode == "misordered_lambda_ap":
        if group_size != 20:
            raise ValueError("misordered_lambda_ap requires group_size=20")
        if teacher_weight or teacher_targets is not None:
            raise ValueError("misordered_lambda_ap forbids teacher objectives")
        if (
            all_pairs_aux_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError("misordered_lambda_ap does not accept ranking auxiliaries")
    if rank_mode == "asymmetric_safe_residual_lambda_ap":
        if group_size != 20:
            raise ValueError("asymmetric_safe_residual_lambda_ap requires group_size=20")
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError(
                "asymmetric_safe_residual_lambda_ap forbids point/teacher objectives"
            )
        if retriever_reference_scores is None:
            raise ValueError(
                "asymmetric_safe_residual_lambda_ap requires frozen retriever scores"
            )
        if (
            all_pairs_aux_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError(
                "asymmetric_safe_residual_lambda_ap does not accept ranking auxiliaries"
            )
    if rank_mode == "robust_normalized_lambdaap_soft_r1":
        if group_size != 20:
            raise ValueError("robust_normalized_lambdaap_soft_r1 requires group_size=20")
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError("robust_normalized_lambdaap_soft_r1 forbids point/teacher objectives")
        if (
            all_pairs_aux_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError("robust_normalized_lambdaap_soft_r1 does not accept ranking auxiliaries")
    if rank_mode == "paired_view_robust_normalized_lambdaap_soft_r1":
        if group_size != 40:
            raise ValueError(
                "paired_view_robust_normalized_lambdaap_soft_r1 requires group_size=40"
            )
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError(
                "paired-view robust LambdaAP forbids point/teacher objectives"
            )
        if candidate_weights is not None:
            raise ValueError("paired-view robust LambdaAP forbids candidate masking")
        if (
            all_pairs_aux_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError(
                "paired-view robust LambdaAP does not accept ranking auxiliaries"
            )
    if rank_mode == "consensus_trimmed_lambdaap_soft_r1":
        if group_size != 20:
            raise ValueError("consensus_trimmed_lambdaap_soft_r1 requires group_size=20")
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError("consensus_trimmed_lambdaap_soft_r1 forbids point/teacher objectives")
        if (
            all_pairs_aux_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError("consensus_trimmed_lambdaap_soft_r1 does not accept ranking auxiliaries")
    if rank_mode == "r535_semantic_and_aux":
        if group_size != 24:
            raise ValueError("r535_semantic_and_aux requires group_size=24")
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError("r535_semantic_and_aux forbids point/teacher objectives")
        if (
            all_pairs_aux_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError("r535_semantic_and_aux does not accept external auxiliaries")
    if rank_mode == "natural20_inverse_pair":
        if group_size != 21:
            raise ValueError("natural20_inverse_pair requires group_size=21")
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError(
                "natural20_inverse_pair is isolated from point and teacher objectives"
            )
        if (
            all_pairs_aux_weight
            or positive_tail_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError(
                "natural20_inverse_pair does not accept auxiliary ranking objectives"
            )
        if near_miss_k is not None:
            raise ValueError(
                "natural20_inverse_pair does not use near_miss_k"
            )
    if rank_mode == "weak_veto_mil":
        if group_size < 6 or group_size % 2:
            raise ValueError(
                "weak_veto_mil requires one full pair and at least two attribute pairs"
            )
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError("weak_veto_mil is isolated from point/teacher objectives")
        if (
            all_pairs_aux_weight
            or positive_tail_weight
            or topk_preservation_weight
            or transition_pooled_weight
            or query_consistency_weight
            or query_cross_weight
        ):
            raise ValueError("weak_veto_mil does not accept auxiliary objectives")
        if near_miss_k is not None:
            raise ValueError("weak_veto_mil does not use near_miss_k")

    grouped_scores = list(scores.split(group_size))
    grouped_labels = list(labels.split(group_size))
    grouped_candidate_weights = (
        list(candidate_weights.split(group_size))
        if candidate_weights is not None
        else [None] * len(grouped_scores)
    )
    if rank_mode == "natural20_inverse_pair":
        if inverse_pair_offsets is None:
            raise ValueError(
                "natural20_inverse_pair requires one original offset per group"
            )
        inverse_pair_offsets = inverse_pair_offsets.reshape(-1).to(scores.device)
        if inverse_pair_offsets.numel() != len(grouped_scores):
            raise ValueError(
                "inverse-pair offset count must equal the candidate-group count"
            )
        if inverse_pair_offsets.dtype == torch.bool or not bool(
            torch.equal(inverse_pair_offsets, inverse_pair_offsets.long())
        ):
            raise ValueError("inverse-pair offsets must be integers")
        inverse_pair_offsets = inverse_pair_offsets.long()
        if bool(((inverse_pair_offsets < 0) | (inverse_pair_offsets >= 20)).any()):
            raise ValueError("inverse-pair offsets must lie in [0, 19]")
        for group_labels, original_offset in zip(
            grouped_labels, inverse_pair_offsets.tolist(), strict=True
        ):
            natural_labels = group_labels[:20]
            if not bool((natural_labels > 0.5).any()) or not bool(
                (natural_labels <= 0.5).any()
            ):
                raise ValueError(
                    "natural20_inverse_pair first 20 rows need positive and negative labels"
                )
            if bool(group_labels[original_offset] > 0.5) or not bool(
                group_labels[20] > 0.5
            ):
                raise ValueError(
                    "inverse pair must reference a natural negative and append a positive"
                )
    if rank_mode == "weak_veto_mil":
        expected = labels.new_tensor([1.0, 0.0]).repeat(group_size // 2)
        if any(
            not torch.equal(group_labels, expected)
            for group_labels in grouped_labels
        ):
            raise ValueError(
                "weak_veto_mil expects alternating [positive, negative] full/field pairs"
            )
    if rank_mode in {"plackett_luce_policy", "rb_plackett_luce_expected_ap"}:
        if point_weight or teacher_weight or teacher_targets is not None:
            raise ValueError(
                f"{rank_mode} uses only hard-label ranking rewards"
            )
        if policy_preserve is None:
            raise ValueError(
                f"{rank_mode} requires one hard preserve flag per group"
            )
        policy_preserve = policy_preserve.bool().reshape(-1).to(scores.device)
        if policy_preserve.numel() != len(grouped_scores):
            raise ValueError("PL preserve flag count must equal the group count")
    if rank_mode == "paired_adjacent":
        if group_size % 2:
            raise ValueError("paired_adjacent requires an even group_size")
        expected = labels.new_tensor([1.0, 0.0]).repeat(group_size // 2)
        if any(not torch.equal(group_labels, expected) for group_labels in grouped_labels):
            raise ValueError(
                "paired_adjacent expects contiguous [positive original query, "
                "negative one-field mutation] pairs"
            )
    if rank_mode == "counterfactual_set5":
        if group_size != 20:
            raise ValueError("counterfactual_set5 requires group_size=20")
        expected = labels.new_tensor([1.0, 0.0, 0.0, 0.0, 0.0]).repeat(4)
        if any(not torch.equal(group_labels, expected) for group_labels in grouped_labels):
            raise ValueError(
                "counterfactual_set5 expects four contiguous same-image K5 "
                "sets with label layout [1,0,0,0,0]"
            )
    positives = [group_labels > 0.5 for group_labels in grouped_labels]
    negatives = [~positive for positive in positives]
    if rank_mode == "natural20_inverse_pair":
        metric_grouped_scores = [group_scores[:20] for group_scores in grouped_scores]
        metric_grouped_labels = [group_labels[:20] for group_labels in grouped_labels]
        metric_positives = [group_labels > 0.5 for group_labels in metric_grouped_labels]
        metric_negatives = [~positive for positive in metric_positives]
    elif rank_mode == "weak_veto_mil":
        # Only the first pair is an ordinary full-query ranking decision.  The
        # atomic pairs are latent witness hypotheses and must not masquerade as
        # independently certified labels in AP/binary telemetry.
        metric_grouped_scores = [group_scores[:2] for group_scores in grouped_scores]
        metric_grouped_labels = [group_labels[:2] for group_labels in grouped_labels]
        metric_positives = [group_labels > 0.5 for group_labels in metric_grouped_labels]
        metric_negatives = [~positive for positive in metric_positives]
    else:
        metric_grouped_scores = grouped_scores
        metric_grouped_labels = grouped_labels
        metric_positives = positives
        metric_negatives = negatives
    full_gallery_lambda_result = None
    r69_components = None
    policy_components = None
    if rank_mode == "full_gallery_lambda_ap":
        if total_relevant is None:
            raise ValueError(
                "full_gallery_lambda_ap requires one total_relevant value per group"
            )
        total_relevant = total_relevant.float().reshape(-1).to(scores.device)
        if total_relevant.numel() != len(grouped_scores):
            raise ValueError(
                "full_gallery_lambda_ap total_relevant count must equal group count"
            )
    if any(
        not positive.any() or not negative.any()
        for positive, negative in zip(positives, negatives)
    ):
        raise ValueError(
            "Ranking requires at least one positive and one negative in every group"
        )
    zero = scores.sum() * 0.0
    inverse_pair_loss = zero
    inverse_pair_accuracy = zero.detach()
    inverse_pair_margin_mean = zero.detach()
    weak_veto_full_loss = zero
    weak_veto_mil_loss = zero
    weak_veto_full_margin_mean = zero.detach()
    weak_veto_smooth_witness_mean = zero.detach()
    weak_veto_any_field_accuracy = zero.detach()

    if rank_mode == "preservation_constrained_cycle_block":
        if group_size != 32:
            raise ValueError(
                "preservation_constrained_cycle_block requires group_size=32"
            )
        expected = labels.new_tensor([1.0, 0.0, 1.0, 0.0] * 8)
        if any(not torch.equal(group_labels, expected) for group_labels in grouped_labels):
            raise ValueError(
                "preservation_constrained_cycle_block expects eight contiguous "
                "[1,0,1,0] checkerboards"
            )
        if point_weight:
            raise ValueError(
                "preservation_constrained_cycle_block requires point_weight=0"
            )
        if cycle_block_metadata is None:
            raise ValueError(
                "preservation_constrained_cycle_block requires block metadata"
            )
        metadata = cycle_block_metadata.float().reshape(-1, 10).to(scores.device)
        if metadata.shape[0] != len(grouped_scores):
            raise ValueError(
                "Cycle-block metadata count mismatch: "
                f"metadata={metadata.shape[0]} groups={len(grouped_scores)}"
            )
        block_roles = metadata[:, 0].round().long()
        support_bits = metadata[:, 1].round().long()
        block_weights = metadata[:, 2:]
        if not bool(((block_roles >= 0) & (block_roles <= 2)).all()):
            raise ValueError("Cycle-block roles must be 0=R, 1=V, or 2=D")
        if not bool(((support_bits >= 0) & (support_bits <= 255)).all()):
            raise ValueError("Cycle-block support masks must be 8-bit integers")
        if not bool(torch.isfinite(block_weights).all()) or not bool(
            (block_weights > 0).all()
        ):
            raise ValueError("Cycle-block AP weights must be finite and positive")
        if not bool(torch.allclose(
            block_weights.sum(dim=1),
            torch.ones(metadata.shape[0], device=scores.device),
            atol=1e-5,
            rtol=0.0,
        )):
            raise ValueError("Cycle-block AP weights must sum to one per query")

        block_scores = scores.reshape(-1, 8, 4)
        block_edge_margins = torch.stack(
            (
                block_scores[:, :, 0] - block_scores[:, :, 1],
                block_scores[:, :, 2] - block_scores[:, :, 3],
            ),
            dim=2,
        )
        block_edge_losses = rank_temperature * F.softplus(
            (rank_margin - block_edge_margins) / rank_temperature
        )
        block_robust_losses = robust_cycle_rho * (
            torch.logsumexp(block_edge_losses / robust_cycle_rho, dim=2)
            - math.log(2.0)
        )
        support_positions = torch.arange(8, device=scores.device)
        support_masks = ((support_bits[:, None] >> support_positions) & 1).bool()
        correct_masks = block_roles != 0
        if bool((correct_masks & ~support_masks.any(dim=1)).any()):
            raise ValueError("Every correct block requires incumbent support")

        per_block_losses: list[torch.Tensor] = []
        preservation_hinges: list[torch.Tensor] = []
        preservation_passes: list[torch.Tensor] = []
        for index in range(metadata.shape[0]):
            q_losses = block_edge_losses[index, :, 0]
            robust_losses = block_robust_losses[index]
            weights = block_weights[index]
            if int(block_roles[index].item()) == 0:
                # Builder guarantees slot zero is the exact parent rank-1
                # negative versus the best parent-ranked positive.
                loss = (
                    q_losses[0]
                    + repair_cycle_ap_weight * (weights * q_losses).sum()
                    + robust_cycle_lambda * (weights * robust_losses).sum()
                )
            else:
                supported = block_edge_margins[index, :, 0][support_masks[index]]
                worst_margin = supported.amin()
                hinge = F.relu(preservation_cycle_margin - worst_margin)
                preservation_hinges.append(hinge)
                preservation_passes.append(
                    (worst_margin >= preservation_cycle_margin).float()
                )
                loss = (
                    preservation_cycle_weight * hinge
                    + preserve_cycle_ap_weight * (weights * q_losses).sum()
                    + preserve_cycle_robust_scale
                    * robust_cycle_lambda
                    * (weights * robust_losses).sum()
                )
            per_block_losses.append(loss)
        loss_rank = torch.stack(per_block_losses).mean()
        zero_cycle_metric = loss_rank.detach() * 0.0
        preservation_hinge_mean = (
            torch.stack(preservation_hinges).mean()
            if preservation_hinges
            else zero_cycle_metric
        )
        preservation_pass_rate = (
            torch.stack(preservation_passes).mean()
            if preservation_passes
            else zero_cycle_metric
        )
        repair_block_fraction = (block_roles == 0).float().mean()
        vulnerable_block_fraction = (block_roles == 1).float().mean()
        diversity_block_fraction = (block_roles == 2).float().mean()
    elif rank_mode in {"orthogonal_cycle", "deployment_weighted_robust_cycle"}:
        # Each group is the 2x2 query-image square
        #   [s(q,p), s(q,n), s(q_n,n), s(q_n,p)].
        # Optimize the two fixed-query edges separately. Query-only offsets
        # cancel inside each edge. An image-only offset helps one edge by the
        # exact amount it hurts the reversed edge, so it cannot solve both;
        # separate softplus terms also prevent one easy edge from compensating
        # an incorrect one.
        if group_size != 4:
            raise ValueError(f"{rank_mode} requires group_size=4")
        expected = labels.new_tensor([1.0, 0.0, 1.0, 0.0])
        if any(not torch.equal(group_labels, expected) for group_labels in grouped_labels):
            raise ValueError(f"{rank_mode} expects [1,0,1,0] group layout")
        if point_weight:
            raise ValueError(
                f"{rank_mode} requires point_weight=0 because slot 3 is "
                "an interaction control, not an asserted binary negative"
            )
        edge_margins = torch.stack(
            [
                torch.stack(
                    (
                        group_scores[0] - group_scores[1],
                        group_scores[2] - group_scores[3],
                    )
                )
                for group_scores in grouped_scores
            ]
        )
        edge_losses = rank_temperature * F.softplus(
            (rank_margin - edge_margins) / rank_temperature
        )
        if rank_mode == "orthogonal_cycle":
            loss_rank = edge_losses.mean()
        else:
            if cycle_weights is None:
                raise ValueError(
                    "deployment_weighted_robust_cycle requires one cycle weight "
                    "per four-row group"
                )
            cycle_weights = cycle_weights.float().reshape(-1).to(scores.device)
            if cycle_weights.numel() != len(grouped_scores):
                raise ValueError(
                    "Cycle-weight count mismatch: "
                    f"weights={cycle_weights.numel()} groups={len(grouped_scores)}"
                )
            if not bool(torch.isfinite(cycle_weights).all()) or not bool(
                (cycle_weights > 0).all()
            ):
                raise ValueError("cycle_weights must be finite and strictly positive")
            deployment_edge_losses = edge_losses[:, 0]
            robust_edge_losses = robust_cycle_rho * (
                torch.logsumexp(edge_losses / robust_cycle_rho, dim=1)
                - math.log(2.0)
            )
            per_cycle_losses = cycle_weights * (
                deployment_edge_losses
                + robust_cycle_lambda * robust_edge_losses
            )
            loss_rank = per_cycle_losses.mean()

            # Analytic absolute margin-gradient contributions.  Reporting both
            # means makes their ratio exactly aggregatable across DP ranks and
            # differently sized batches; averaging per-batch ratios would not.
            edge_gradient_mass = torch.sigmoid(
                (rank_margin - edge_margins) / rank_temperature
            )
            robust_edge_mix = torch.softmax(
                edge_losses / robust_cycle_rho, dim=1
            )
            q_gradient_mass = cycle_weights * edge_gradient_mass[:, 0] * (
                1.0 + robust_cycle_lambda * robust_edge_mix[:, 0]
            )
            qn_gradient_mass = (
                cycle_weights
                * edge_gradient_mass[:, 1]
                * robust_cycle_lambda
                * robust_edge_mix[:, 1]
            )
    elif rank_mode == "query_selective_regret":
        if group_size % 2:
            raise ValueError("query_selective_regret requires an even group_size")
        query_utilities: list[torch.Tensor] = []
        query_targets: list[torch.Tensor] = []
        consistency_losses: list[torch.Tensor] = []
        query_risk_losses: list[torch.Tensor] = []
        semantic_point_losses: list[torch.Tensor] = []
        query_margins: list[torch.Tensor] = []
        query_correct: list[torch.Tensor] = []
        repairable: list[bool] = []
        repairs: list[torch.Tensor] = []
        preservations: list[torch.Tensor] = []
        repair_confidences: list[torch.Tensor] = []
        safe_confidences: list[torch.Tensor] = []

        for group_scores, positive in zip(grouped_scores, positives):
            # Rows are [incumbent-first, challenger-first] for each challenger.
            # A positive score always means "choose image A", so the two
            # order-invariant challenger utilities are -s_even and +s_odd.
            incumbent_first = -group_scores[0::2]
            challenger_first = group_scores[1::2]
            replace_targets = ~positive[0::2]
            if not bool(torch.equal(positive[1::2], replace_targets)):
                raise ValueError(
                    "query_selective_regret expects complementary swapped labels "
                    "[keep: 1,0; replace: 0,1] for every challenger"
                )
            utilities = 0.5 * (incumbent_first + challenger_first)
            query_utilities.append(utilities)
            query_targets.append(replace_targets)
            consistency_losses.append(
                F.smooth_l1_loss(incumbent_first, challenger_first)
            )

            if bool(replace_targets.any()):
                positive_utilities = utilities[replace_targets]
                negative_utilities = utilities[~replace_targets]
                repair_confidences.append(positive_utilities.max())
                # Deployment replaces only when a relevant challenger beats
                # abstention (utility zero) and every irrelevant challenger.
                competitors = torch.cat((utilities.new_zeros(1), negative_utilities))
                best_positive = rank_temperature * torch.logsumexp(
                    positive_utilities / rank_temperature, dim=0
                )
                hardest_competitor = rank_temperature * torch.logsumexp(
                    competitors / rank_temperature, dim=0
                )
                margin = best_positive - hardest_competitor
                query_risk_losses.append(
                    retriever_error_weight
                    * F.softplus((rank_margin - margin) / rank_temperature)
                )
                positive_bce = F.binary_cross_entropy_with_logits(
                    positive_utilities / point_temperature,
                    torch.ones_like(positive_utilities),
                )
                if negative_utilities.numel():
                    negative_bce = F.binary_cross_entropy_with_logits(
                        negative_utilities / point_temperature,
                        torch.zeros_like(negative_utilities),
                    )
                    semantic_point_losses.append(
                        retriever_error_weight
                        * 0.5
                        * (positive_bce + negative_bce)
                    )
                else:
                    semantic_point_losses.append(
                        retriever_error_weight * positive_bce
                    )
                winner = int(torch.argmax(utilities).item())
                repaired = (utilities[winner] > 0) & replace_targets[winner]
                repairable.append(True)
                repairs.append(repaired.float())
                preservations.append(utilities.new_zeros(()))
                query_correct.append(repaired.float())
                query_margins.append(margin)
            else:
                # One false-positive among K-1 challengers breaks the query.
                # LogSumExp is a smooth family-wise maximum, matching that
                # max-selection risk instead of averaging it away.
                hardest_challenger = rank_temperature * torch.logsumexp(
                    utilities / rank_temperature, dim=0
                )
                safe_confidences.append(utilities.max())
                margin = -hardest_challenger
                query_risk_losses.append(
                    retriever_success_weight
                    * F.softplus((rank_margin - margin) / rank_temperature)
                )
                semantic_point_losses.append(
                    F.binary_cross_entropy_with_logits(
                        utilities / point_temperature,
                        torch.zeros_like(utilities),
                    )
                )
                preserved = (utilities.max() <= 0).float()
                repairable.append(False)
                repairs.append(utilities.new_zeros(()))
                preservations.append(preserved)
                query_correct.append(preserved)
                query_margins.append(margin)

        loss_rank = torch.stack(query_risk_losses).mean()
        loss_consistency = torch.stack(consistency_losses).mean()
        loss_rank = loss_rank + query_consistency_weight * loss_consistency
        # A single deployment threshold can separate repairs from safe keeps
        # only when their *query-level* maximum override confidences are
        # ordered across queries. Per-query losses alone do not enforce this.
        # The all-pairs term is a smooth AUC-style surrogate for ranking every
        # repairable query above every safe query observed in the same batch.
        if query_cross_weight and repair_confidences and safe_confidences:
            repair_confidence = torch.stack(repair_confidences)
            safe_confidence = torch.stack(safe_confidences)
            cross_violations = (
                safe_confidence[:, None]
                + rank_margin
                - repair_confidence[None, :]
            )
            loss_cross = F.softplus(cross_violations / rank_temperature).mean()
            loss_rank = loss_rank + query_cross_weight * loss_cross
        loss_point = torch.stack(semantic_point_losses).mean() if point_weight else zero
        loss_teacher = zero
        total = rank_weight * loss_rank + point_weight * loss_point

        with torch.no_grad():
            utilities = torch.cat(query_utilities)
            targets = torch.cat(query_targets)
            predicted = utilities > 0
            repair_mask = torch.tensor(repairable, device=scores.device, dtype=torch.bool)
            repairs_t = torch.stack(repairs)
            preservations_t = torch.stack(preservations)
            correct_t = torch.stack(query_correct)
            margins_t = torch.stack(query_margins)
            positive_utilities = utilities[targets]
            negative_utilities = utilities[~targets]
            zero_metric = zero.detach()
            repair_accuracy = (
                repairs_t[repair_mask].mean() if bool(repair_mask.any()) else zero_metric
            )
            preservation_accuracy = (
                preservations_t[~repair_mask].mean()
                if bool((~repair_mask).any())
                else zero_metric
            )
            positive_accuracy = (
                (positive_utilities > 0).float().mean()
                if positive_utilities.numel()
                else zero_metric
            )
            negative_accuracy = (
                (negative_utilities <= 0).float().mean()
                if negative_utilities.numel()
                else zero_metric
            )
            positive_mean = (
                positive_utilities.mean() if positive_utilities.numel() else zero_metric
            )
            negative_mean = (
                negative_utilities.mean() if negative_utilities.numel() else zero_metric
            )
            best_positive_mean = (
                torch.stack(
                    [u[t].max() for u, t in zip(query_utilities, query_targets) if bool(t.any())]
                ).mean()
                if bool(repair_mask.any())
                else zero_metric
            )
            hardest_negative_mean = torch.stack(
                [u[~t].max() if bool((~t).any()) else u.new_zeros(())
                 for u, t in zip(query_utilities, query_targets)]
            ).mean()
            slot_positive = scores[labels > 0.5]
            slot_negative = scores[labels <= 0.5]
            metrics = {
                "rank": loss_rank.detach(),
                "point": loss_point.detach(),
                "teacher": zero_metric,
                "teacher_target_mean": zero_metric,
                "teacher_probability_mae": zero_metric,
                "pairwise_accuracy": (predicted == targets).float().mean(),
                "positive_score_mean": positive_mean,
                "negative_score_mean": negative_mean,
                "best_positive_score_mean": best_positive_mean,
                "hardest_negative_score_mean": hardest_negative_mean,
                "top1_margin": margins_t.mean(),
                "top1_accuracy": correct_t.mean(),
                "retriever_top1_accuracy": (~repair_mask).float().mean(),
                "retriever_success_preservation_accuracy": preservation_accuracy,
                "retriever_error_repair_accuracy": repair_accuracy,
                "rank1_fix_rate": repairs_t.mean(),
                "rank1_break_rate": ((~repair_mask).float() * (1.0 - preservations_t)).mean(),
                "rank1_net_gain": repairs_t.mean()
                - ((~repair_mask).float() * (1.0 - preservations_t)).mean(),
                "binary_accuracy": ((scores > 0) == (labels > 0.5)).float().mean(),
                "positive_accuracy": positive_accuracy,
                "negative_accuracy": negative_accuracy,
                "balanced_binary_accuracy": 0.5 * (positive_accuracy + negative_accuracy),
                "hard_negative_accuracy": preservation_accuracy,
                "average_precision": correct_t.mean(),
                "rank5_accuracy": correct_t.mean(),
                "margin": margins_t.mean(),
            }
        return total, metrics

    if rank_mode in {
        "orthogonal_cycle",
        "deployment_weighted_robust_cycle",
        "preservation_constrained_cycle_block",
    }:
        # The specialized cycle loss was constructed above.  Keep it instead
        # of falling through to the generic all-pairs branch below.
        pass
    elif rank_weight and rank_mode == "paired_adjacent":
        # Each adjacent pair shares the exact same image.  The first row uses
        # the original fully-matching query and the second differs in exactly
        # one verified attribute.  Comparing only within each pair avoids the
        # invalid cross-query ordering imposed by a generic listwise loss.
        paired = scores.reshape(-1, 2)
        pair_margins = paired[:, 0] - paired[:, 1]
        loss_rank = (
            rank_temperature
            * F.softplus((rank_margin - pair_margins) / rank_temperature)
        ).mean()
    elif rank_weight and rank_mode == "counterfactual_set5":
        # Each raw K20 packet is four independent same-image counterfactual
        # K5 sets. Optimize only within those K5 sets: cross-source score
        # calibration is meaningless and a global confidence shift cannot
        # improve any of the four ranking objectives.
        per_set_loss, r69_components = r69_metric_loss(
            scores.reshape(-1, 5),
            labels.reshape(-1, 5),
            smoothap_temperature=r69_smoothap_temperature,
            soft_r1_temperature=r69_soft_r1_temperature,
            smoothap_weight=r69_smoothap_weight,
            soft_r1_weight=r69_soft_r1_weight,
            group_size=5,
            reduction="none",
        )
        loss_rank = per_set_loss.mean()
    elif rank_weight and rank_mode in {
        "smoothap_soft_r1",
        "dynamic_top20_smoothap_soft_r1",
        "incumbent_guarded_smoothap_soft_r1",
    }:
        # Optimize the two deployment metrics directly over each complete
        # query list. SmoothAP supplies dense gradients to every currently
        # misordered positive/negative relation, while soft-R1 concentrates a
        # second signal on the probability that the winner is relevant. Both
        # consume only the student's native yes-minus-no logit and hard labels.
        per_group_loss, r69_components = r69_metric_loss(
            scores.reshape(-1, group_size),
            labels.reshape(-1, group_size),
            smoothap_temperature=r69_smoothap_temperature,
            soft_r1_temperature=r69_soft_r1_temperature,
            smoothap_weight=r69_smoothap_weight,
            soft_r1_weight=r69_soft_r1_weight,
            group_size=group_size,
            reduction="none",
        )
        if rank_mode == "incumbent_guarded_smoothap_soft_r1":
            # The input list is sorted by the frozen parent before training,
            # so row zero is its incumbent winner.  Every query still gets the
            # dense, hard-label SmoothAP/R1 objective.  Parent errors receive
            # extra repair mass, while a correct incumbent additionally gets
            # an active-set barrier against all negatives.  The barrier is
            # exactly zero once the incumbent clears the requested margin;
            # this avoids spending most gradient on already-safe examples but
            # pushes back as soon as shared LoRA updates threaten a regression.
            grouped = scores.reshape(-1, 20)
            grouped_hard_labels = labels.reshape(-1, 20) > 0.5
            incumbent_correct = grouped_hard_labels[:, 0]
            base_weights = torch.where(
                incumbent_correct,
                torch.ones_like(per_group_loss),
                per_group_loss.new_full(per_group_loss.shape, retriever_error_weight),
            )
            guard_losses = []
            for group_scores, group_positive, is_correct in zip(
                grouped, grouped_hard_labels, incumbent_correct, strict=True
            ):
                if bool(is_correct):
                    violations = F.relu(
                        group_scores[~group_positive]
                        + rank_margin
                        - group_scores[0]
                    )
                    guard_losses.append(
                        retriever_success_weight
                        * (violations.amax() + violations.mean())
                    )
                else:
                    guard_losses.append(group_scores.sum() * 0.0)
            guard = torch.stack(guard_losses)
            loss_rank = (base_weights * per_group_loss + guard).mean()
            r69_components["incumbent_guard_loss"] = guard.detach()
            r69_components["incumbent_guard_pass"] = (
                (~incumbent_correct) | (guard.detach() == 0)
            ).float()
        else:
            loss_rank = per_group_loss.mean()
    elif rank_weight and rank_mode == "plackett_luce_policy":
        assert policy_preserve is not None
        policy_losses: list[torch.Tensor] = []
        component_rows: list[dict[str, torch.Tensor]] = []
        for group_scores, group_labels, preserve in zip(
            grouped_scores, grouped_labels, policy_preserve, strict=True
        ):
            group_loss, components = plackett_luce_policy_loss(
                group_scores,
                group_labels,
                temperature=rank_temperature,
                samples=policy_samples,
                exact_max_k=policy_exact_max_k,
                ap_reward_weight=policy_ap_reward_weight,
                r1_reward_weight=policy_r1_reward_weight,
                preserve=bool(preserve.item()),
                preserve_anchor_weight=policy_preserve_anchor_weight,
                preserve_anchor_margin=policy_preserve_anchor_margin,
            )
            policy_losses.append(group_loss)
            component_rows.append(components)
        loss_rank = torch.stack(policy_losses).mean()
        policy_components = {
            name: torch.stack([row[name] for row in component_rows])
            for name in component_rows[0]
        }
    elif rank_weight and rank_mode == "rb_plackett_luce_expected_ap":
        assert policy_preserve is not None
        policy_losses = []
        component_rows = []
        for group_scores, group_labels, preserve in zip(
            grouped_scores, grouped_labels, policy_preserve, strict=True
        ):
            group_loss, components = rb_plackett_luce_expected_ap_loss(
                group_scores,
                group_labels,
                temperature=rank_temperature,
                outer_points=policy_samples,
                inner_points=policy_exact_max_k,
                ap_reward_weight=policy_ap_reward_weight,
                r1_reward_weight=policy_r1_reward_weight,
                preserve=bool(preserve.item()),
                preserve_anchor_weight=policy_preserve_anchor_weight,
                preserve_anchor_margin=policy_preserve_anchor_margin,
            )
            policy_losses.append(group_loss)
            component_rows.append(components)
        loss_rank = torch.stack(policy_losses).mean()
        policy_components = {
            name: torch.stack([row[name] for row in component_rows])
            for name in component_rows[0]
        }
    elif rank_weight and rank_mode == "incumbent_safe_lambda_ap":
        # Exact-delta active-set ranking.  Lambda weights are the exact AP@20
        # change caused by swapping each positive/negative pair at the
        # student's current order.  Only pairs inside the requested margin
        # receive a hinge gradient, so safe tail relations do not dilute the
        # handful of head mistakes that can change deployment utility.  Row
        # zero is the frozen-parent winner in the parent-ordered train set.
        lambda_losses = []
        active_fractions = []
        guard_losses = []
        for group_scores, positive in zip(grouped_scores, positives, strict=True):
            group_labels = positive.float()
            positive_index, negative_index, swap_weight = exact_ap_swap_weights(
                group_scores,
                group_labels,
                group_labels.sum(),
                rank1_weight=r69_soft_r1_weight,
            )
            violations = F.relu(
                group_scores[negative_index].unsqueeze(0)
                + rank_margin
                - group_scores[positive_index].unsqueeze(1)
            )
            lambda_ap = r69_smoothap_weight * (swap_weight * violations).sum() / 20.0
            incumbent_correct = bool(positive[0])
            if incumbent_correct:
                incumbent_violations = F.relu(
                    group_scores[~positive] + rank_margin - group_scores[0]
                )
                guard = retriever_success_weight * (
                    incumbent_violations.amax() + incumbent_violations.mean()
                )
                query_loss = lambda_ap + guard
            else:
                guard = group_scores.sum() * 0.0
                head = F.relu(
                    group_scores[~positive].amax()
                    + rank_margin
                    - group_scores[positive].amax()
                )
                query_loss = retriever_error_weight * (
                    lambda_ap + r69_soft_r1_weight * head
                )
            lambda_losses.append(query_loss)
            guard_losses.append(guard.detach())
            active_fractions.append((violations > 0).float().mean().detach())
        loss_rank = torch.stack(lambda_losses).mean()
        r69_components = {
            "safe_lambda_active_pair_fraction": torch.stack(active_fractions),
            "incumbent_guard_loss": torch.stack(guard_losses),
            "incumbent_guard_pass": (
                torch.stack(guard_losses) == 0
            ).float(),
        }
    elif rank_weight and rank_mode == "misordered_lambda_ap":
        lambda_losses = []
        active_fractions = []
        weight_masses = []
        for group_scores, group_labels in zip(
            grouped_scores, grouped_labels, strict=True
        ):
            group_loss, components = misordered_lambda_ap_loss(
                group_scores,
                group_labels,
                temperature=rank_temperature,
            )
            lambda_losses.append(group_loss)
            active_fractions.append(
                components["safe_lambda_active_pair_fraction"]
            )
            weight_masses.append(components["misordered_lambda_ap_weight_mass"])
        loss_rank = torch.stack(lambda_losses).mean()
        r69_components = {
            "safe_lambda_active_pair_fraction": torch.stack(active_fractions),
            "misordered_lambda_ap_weight_mass": torch.stack(weight_masses),
        }
    elif rank_weight and rank_mode == "asymmetric_safe_residual_lambda_ap":
        assert retriever_reference_scores is not None
        grouped_retriever = retriever_reference_scores.float().reshape(-1).to(
            scores.device
        ).split(group_size)
        group_losses = []
        component_rows = []
        for group_scores, group_labels, group_retriever in zip(
            grouped_scores, grouped_labels, grouped_retriever, strict=True
        ):
            group_loss, components = asymmetric_safe_residual_lambda_ap_loss(
                group_scores,
                group_labels,
                group_retriever,
                temperature=rank_temperature,
                margin=rank_margin,
                repair_weight=retriever_error_weight,
                preservation_weight=retriever_success_weight,
            )
            group_losses.append(group_loss)
            component_rows.append(components)
        loss_rank = torch.stack(group_losses).mean()
        r69_components = {
            name: torch.stack([row[name] for row in component_rows])
            for name in component_rows[0]
        }
    elif rank_weight and rank_mode == "robust_normalized_lambdaap_soft_r1":
        group_losses = []
        component_rows = []
        for group_scores, group_labels, group_weights in zip(
            grouped_scores, grouped_labels, grouped_candidate_weights, strict=True
        ):
            # Candidate zero is the frozen retriever's rank-1 item because
            # annotations preserve candidate order.  This is the same
            # hard-label-only preservation classification used by the
            # full-gallery LambdaAP branch: no retriever score is a target.
            query_weight = (
                retriever_success_weight
                if bool(group_labels[0] > 0.5)
                else retriever_error_weight
            )
            group_loss, components = robust_normalized_lambdaap_soft_r1_loss(
                group_scores,
                group_labels,
                candidate_weights=group_weights,
                pair_temperature=rank_temperature,
                soft_r1_temperature=r69_soft_r1_temperature,
                lambdaap_weight=r69_smoothap_weight,
                soft_r1_weight=r69_soft_r1_weight,
                robust_pair_q=robust_pair_q,
                query_weight=query_weight,
            )
            group_losses.append(group_loss)
            component_rows.append(components)
        loss_rank = torch.stack(group_losses).mean()
        r69_components = {
            name: torch.stack([row[name] for row in component_rows])
            for name in component_rows[0]
        }
    elif (
        rank_weight
        and rank_mode == "paired_view_robust_normalized_lambdaap_soft_r1"
    ):
        # Each K40 packet contains the exact same K20 candidate list twice:
        # canonical pixels followed by a horizontal flip.  Both views consume
        # the unchanged hard labels.  The centered score penalty is translation
        # invariant and therefore regularizes only within-query ranking geometry.
        group_losses = []
        component_rows = []
        for group_scores, group_labels in zip(
            grouped_scores, grouped_labels, strict=True
        ):
            canonical_scores, flipped_scores = group_scores.split(20)
            canonical_labels, flipped_labels = group_labels.split(20)
            if not bool(torch.equal(canonical_labels, flipped_labels)):
                raise ValueError("paired-view K20 hard labels differ across views")
            canonical_loss, canonical_components = (
                robust_normalized_lambdaap_soft_r1_loss(
                    canonical_scores,
                    canonical_labels,
                    pair_temperature=rank_temperature,
                    soft_r1_temperature=r69_soft_r1_temperature,
                    lambdaap_weight=r69_smoothap_weight,
                    soft_r1_weight=r69_soft_r1_weight,
                    robust_pair_q=robust_pair_q,
                )
            )
            flipped_loss, flipped_components = (
                robust_normalized_lambdaap_soft_r1_loss(
                    flipped_scores,
                    flipped_labels,
                    pair_temperature=rank_temperature,
                    soft_r1_temperature=r69_soft_r1_temperature,
                    lambdaap_weight=r69_smoothap_weight,
                    soft_r1_weight=r69_soft_r1_weight,
                    robust_pair_q=robust_pair_q,
                )
            )
            hard_loss = 0.5 * (canonical_loss + flipped_loss)
            canonical_centered = canonical_scores - canonical_scores.mean()
            flipped_centered = flipped_scores - flipped_scores.mean()
            view_loss = F.smooth_l1_loss(
                (canonical_centered - flipped_centered) / 0.5,
                torch.zeros_like(canonical_centered),
            )
            # The configured EFLC coefficient is 0.25, with a detached budget
            # that prevents consistency from exceeding 10% of the hard loss.
            view_weight = torch.minimum(
                view_loss.new_tensor(0.25),
                0.10 * hard_loss.detach() / view_loss.detach().clamp_min(1e-8),
            )
            group_losses.append(hard_loss + view_weight * view_loss)
            component_rows.append(
                {
                    "robust_lambdaap": 0.5
                    * (
                        canonical_components["robust_lambdaap"]
                        + flipped_components["robust_lambdaap"]
                    ),
                    "robust_soft_r1": 0.5
                    * (
                        canonical_components["robust_soft_r1"]
                        + flipped_components["robust_soft_r1"]
                    ),
                    "robust_pair_weight_mass": 0.5
                    * (
                        canonical_components["robust_pair_weight_mass"]
                        + flipped_components["robust_pair_weight_mass"]
                    ),
                    "robust_positive_trim_fraction": 0.5
                    * (
                        canonical_components["robust_positive_trim_fraction"]
                        + flipped_components["robust_positive_trim_fraction"]
                    ),
                    "robust_negative_trim_fraction": 0.5
                    * (
                        canonical_components["robust_negative_trim_fraction"]
                        + flipped_components["robust_negative_trim_fraction"]
                    ),
                    "paired_view_consistency": view_loss.detach(),
                    "paired_view_effective_weight": view_weight.detach(),
                    "paired_view_contribution_fraction": (
                        view_weight.detach() * view_loss.detach()
                        / hard_loss.detach().clamp_min(1e-8)
                    ),
                    "paired_view_centered_abs_gap": (
                        canonical_centered.detach() - flipped_centered.detach()
                    ).abs().mean(),
                }
            )
        loss_rank = torch.stack(group_losses).mean()
        r69_components = {
            name: torch.stack([row[name] for row in component_rows])
            for name in component_rows[0]
        }
    elif rank_weight and rank_mode == "consensus_trimmed_lambdaap_soft_r1":
        group_losses = []
        component_rows = []
        for group_scores, group_labels in zip(
            grouped_scores, grouped_labels, strict=True
        ):
            group_loss, components = consensus_trimmed_lambdaap_soft_r1_loss(
                group_scores,
                group_labels,
                pair_temperature=rank_temperature,
                soft_r1_temperature=r69_soft_r1_temperature,
                lambdaap_weight=r69_smoothap_weight,
                soft_r1_weight=r69_soft_r1_weight,
                robust_pair_q=robust_pair_q,
            )
            group_losses.append(group_loss)
            component_rows.append(components)
        loss_rank = torch.stack(group_losses).mean()
        r69_components = {
            name: torch.stack([row[name] for row in component_rows])
            for name in component_rows[0]
        }
    elif rank_weight and rank_mode == "r535_semantic_and_aux":
        group_losses = []
        component_rows = []
        for group_scores, group_labels in zip(
            grouped_scores, grouped_labels, strict=True
        ):
            group_loss, components = r535_semantic_and_aux_loss(
                group_scores,
                group_labels,
                pair_temperature=rank_temperature,
                soft_r1_temperature=r69_soft_r1_temperature,
                lambdaap_weight=r69_smoothap_weight,
                soft_r1_weight=r69_soft_r1_weight,
                robust_pair_q=robust_pair_q,
            )
            group_losses.append(group_loss)
            component_rows.append(components)
        loss_rank = torch.stack(group_losses).mean()
        r69_components = {
            name: torch.stack([row[name] for row in component_rows])
            for name in component_rows[0]
        }
    elif rank_weight and rank_mode == "global_action_faithful_topm_ap":
        # Match both parts of deployment instead of constructing a fictitious
        # winner from the best relevant and best irrelevant candidates.
        #
        # (1) Full-list semantics: every relevant item must clear the top-m
        # currently dangerous irrelevants.  This supplies the relations AP
        # measures and gives several hard negatives gradient on every query.
        # (2) Selective action: deployment promotes exactly the highest-scored
        # challenger (rows 1..K-1).  Only a wrong->right winner is a benefit
        # and only a right->wrong winner is a harm.  Same-label replacements
        # are neutral.  The cross-query term orders observed benefit gaps over
        # observed harm gaps without any log-sum-exp cardinality offset.
        assert near_miss_k is not None
        semantic_losses: list[torch.Tensor] = []
        positive_tail_losses: list[torch.Tensor] = []
        action_losses: list[torch.Tensor] = []
        action_confidences: list[torch.Tensor] = []
        # 0=neutral, 1=benefit, 2=harm.  These labels describe the actual
        # deployed challenger selected by the current student.
        action_kinds: list[int] = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            positive_scores = group_scores[positive]
            negative_scores = group_scores[negative]
            hard_negatives = negative_scores.topk(
                min(near_miss_k, negative_scores.numel())
            ).values
            pair_violations = (
                hard_negatives.unsqueeze(0)
                + rank_margin
                - positive_scores.unsqueeze(1)
            )
            # Multiplying by tau preserves the hinge scale as temperature is
            # changed; omitting it made the old tau=.2 objective amplify every
            # rank gradient five-fold and saturate global gradient clipping.
            semantic_losses.append(
                rank_temperature
                * F.softplus(pair_violations / rank_temperature).mean()
            )
            if positive_tail_weight:
                # AP is bottlenecked by the lowest-scoring positive in a
                # multi-positive group, but the ordinary pair mean gives each
                # easy positive equal gradient.  A smooth minimum focuses a
                # bounded auxiliary term on that bottleneck while retaining
                # the top-k negative tail used by the deployment objective.
                positive_softmin = -positive_tail_temperature * torch.logsumexp(
                    -positive_scores / positive_tail_temperature, dim=0
                )
                negative_softmax = rank_temperature * torch.logsumexp(
                    hard_negatives / rank_temperature, dim=0
                )
                positive_tail_losses.append(
                    rank_temperature
                    * F.softplus(
                        (negative_softmax + rank_margin - positive_softmin)
                        / rank_temperature
                    )
                )
            else:
                positive_tail_losses.append(group_scores.sum() * 0.0)

            challenger_index = int(torch.argmax(group_scores[1:]).item()) + 1
            confidence = group_scores[challenger_index] - group_scores[0]
            incumbent_relevant = bool(positive[0])
            challenger_relevant = bool(positive[challenger_index])
            if not incumbent_relevant and challenger_relevant:
                action_kind = 1
                action_loss = (
                    retriever_error_weight
                    * rank_temperature
                    * F.softplus(
                        (rank_margin - confidence) / rank_temperature
                    )
                )
            elif incumbent_relevant and not challenger_relevant:
                action_kind = 2
                action_loss = (
                    retriever_success_weight
                    * rank_temperature
                    * F.softplus(
                        (confidence + rank_margin) / rank_temperature
                    )
                )
            else:
                action_kind = 0
                action_loss = confidence * 0.0
            action_losses.append(action_loss)
            action_confidences.append(confidence)
            action_kinds.append(action_kind)

        loss_rank = torch.stack(semantic_losses).mean() + torch.stack(
            action_losses
        ).mean()
        if positive_tail_weight:
            loss_rank = loss_rank + positive_tail_weight * torch.stack(
                positive_tail_losses
            ).mean()
        if query_cross_weight:
            local_confidences = torch.stack(action_confidences)
            local_kinds = torch.tensor(
                action_kinds, device=scores.device, dtype=torch.int8
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from torch.distributed.nn.functional import all_gather

                global_confidences = torch.cat(all_gather(local_confidences), dim=0)
                gathered_kinds = [
                    torch.empty_like(local_kinds)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(gathered_kinds, local_kinds)
                global_kinds = torch.cat(gathered_kinds, dim=0)
            else:
                global_confidences = local_confidences
                global_kinds = local_kinds
            benefit_confidences = global_confidences[global_kinds == 1]
            harm_confidences = global_confidences[global_kinds == 2]
            if benefit_confidences.numel() and harm_confidences.numel():
                cross_violations = (
                    harm_confidences.unsqueeze(1)
                    + rank_margin
                    - benefit_confidences.unsqueeze(0)
                )
                loss_rank = loss_rank + (
                    query_cross_weight
                    * rank_temperature
                    * F.softplus(
                        cross_violations / rank_temperature
                    ).mean()
                )
    elif rank_weight and rank_mode == "full_gallery_lambda_ap":
        assert total_relevant is not None
        full_gallery_lambda_result = full_gallery_lambda_ap_loss(
            scores,
            labels,
            total_relevant,
            group_size=group_size,
            temperature=rank_temperature,
            margin=rank_margin,
            rank1_weight=full_gallery_rank1_weight,
            retriever_success_weight=retriever_success_weight,
            retriever_error_weight=retriever_error_weight,
            active_topk=full_gallery_active_topk,
        )
        loss_rank = full_gallery_lambda_result.loss
    elif rank_weight and rank_mode in {
        "cross_query_selective_override",
        "tail_risk_selective_override",
        "nested_tail_risk_selective_override",
        "global_nested_tail_risk_selective_override",
    }:
        # Keep CR3 on its native single-image scoring task, but optimize the
        # *selective override* decision made at deployment. Row zero is the
        # frozen retriever's incumbent and rows 1..K-1 are challengers. A
        # correct incumbent is protected against its strongest irrelevant
        # challenger. When the incumbent is wrong, the strongest relevant
        # candidate must beat every irrelevant candidate (including row zero).
        #
        # The second term is deliberately cross-query: it orders the benefit
        # confidence of retriever repairs above the harm confidence of
        # retriever breaks. This makes one train-only calibrated threshold
        # usable at inference without combining CR3 and retriever scores.
        selective_losses = []
        repair_confidences = []
        safe_confidences = []
        cross_confidences = []
        cross_safe_masks = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            incumbent = group_scores[0]
            irrelevant_scores = group_scores[negative]
            if rank_mode in {
                "nested_tail_risk_selective_override",
                "global_nested_tail_risk_selective_override",
            }:
                # A hard maximum routes all gradient through one negative even
                # though deployment compares against K challengers.  The
                # log-sum-exp upper bound preserves the max-risk semantics but
                # spreads pressure over the top-m currently dangerous
                # negatives.  This is the candidate-level tail of the nested
                # candidate/query risk objective.
                assert near_miss_k is not None
                hard_negatives = irrelevant_scores.topk(
                    min(near_miss_k, irrelevant_scores.numel())
                ).values
                strongest_irrelevant = rank_temperature * torch.logsumexp(
                    hard_negatives / rank_temperature, dim=0
                )
            else:
                strongest_irrelevant = irrelevant_scores.amax()
            if bool(positive[0]):
                # Harm confidence is the amount by which the most dangerous
                # irrelevant challenger beats the known-good incumbent.
                safe_confidence = strongest_irrelevant - incumbent
                safe_confidences.append(safe_confidence)
                cross_confidences.append(safe_confidence)
                cross_safe_masks.append(True)
                violation = strongest_irrelevant + rank_margin - incumbent
                selective_losses.append(
                    retriever_success_weight
                    * F.softplus(violation / rank_temperature)
                )
            else:
                strongest_relevant = group_scores[positive].amax()
                # Benefit confidence is measured against the incumbent because
                # that is the CR3-only gate available during deployment. The
                # within-query loss additionally clears every wrong challenger.
                repair_confidence = strongest_relevant - incumbent
                repair_confidences.append(repair_confidence)
                cross_confidences.append(repair_confidence)
                cross_safe_masks.append(False)
                violation = (
                    strongest_irrelevant + rank_margin - strongest_relevant
                )
                selective_losses.append(
                    retriever_error_weight
                    * F.softplus(violation / rank_temperature)
                )
        loss_rank = torch.stack(selective_losses).mean()
        if query_cross_weight and rank_mode == "global_nested_tail_risk_selective_override":
            # One data-parallel rank sees only a handful of query groups, so a
            # local smooth maximum cannot estimate the rare harmful-override
            # tail permitted at deployment. Gather one confidence per query
            # across all DP replicas before constructing that risk boundary.
            # The functional all_gather is autograd-aware and routes gradients
            # back to the rank that produced each confidence.
            local_confidences = torch.stack(cross_confidences)
            local_safe_mask = torch.tensor(
                cross_safe_masks, device=scores.device, dtype=torch.bool
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from torch.distributed.nn.functional import all_gather

                global_confidences = torch.cat(all_gather(local_confidences), dim=0)
                gathered_masks = [
                    torch.empty_like(local_safe_mask)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(gathered_masks, local_safe_mask)
                global_safe_mask = torch.cat(gathered_masks, dim=0)
            else:
                global_confidences = local_confidences
                global_safe_mask = local_safe_mask
            repair_confidence = global_confidences[~global_safe_mask]
            safe_confidence = global_confidences[global_safe_mask]
            if repair_confidence.numel() and safe_confidence.numel():
                tail_harm = rank_temperature * torch.logsumexp(
                    safe_confidence / rank_temperature, dim=0
                )
                cross_loss = F.softplus(
                    (tail_harm + rank_margin - repair_confidence)
                    / rank_temperature
                ).mean()
                loss_rank = loss_rank + query_cross_weight * cross_loss
        elif query_cross_weight and repair_confidences and safe_confidences:
            repair_confidence = torch.stack(repair_confidences)
            safe_confidence = torch.stack(safe_confidences)
            if rank_mode in {
                "tail_risk_selective_override",
                "nested_tail_risk_selective_override",
                "global_nested_tail_risk_selective_override",
            }:
                # Deployment permits only a small break budget, so average
                # safe-query separation is the wrong statistic. A smooth
                # maximum estimates the upper tail of harmful override
                # confidence and makes repair benefits clear that boundary.
                tail_harm = rank_temperature * torch.logsumexp(
                    safe_confidence / rank_temperature, dim=0
                )
                cross_loss = F.softplus(
                    (tail_harm + rank_margin - repair_confidence)
                    / rank_temperature
                ).mean()
            else:
                cross_violations = (
                    safe_confidence[:, None]
                    + rank_margin
                    - repair_confidence[None, :]
                )
                cross_loss = F.softplus(
                    cross_violations / rank_temperature
                ).mean()
            loss_rank = loss_rank + query_cross_weight * cross_loss
    elif rank_weight and rank_mode in {
        "dual_top1_competitor",
        "constrained_dual_top1_competitor",
        "dual_multi_positive_competitor",
    }:
        # Preserve the union of the two deployed systems' wins without using
        # either score at inference: row zero is SigLIP2's winner, while the
        # teacher argmax is the frozen CR3 parent's winner. Prefer a correct
        # retriever winner, otherwise a correct parent winner, and repair only
        # when both systems are wrong.
        assert teacher_targets is not None
        grouped_parent_targets = list(teacher_targets.split(group_size))
        competitor_losses = []
        for group_scores, group_targets, positive, negative in zip(
            grouped_scores, grouped_parent_targets, positives, negatives
        ):
            retriever_top1_correct = bool(positive[0])
            parent_top1 = int(torch.argmax(group_targets).item())
            parent_top1_correct = bool(positive[parent_top1])
            if retriever_top1_correct:
                anchor_index = 0
                anchor_positive = group_scores[0]
            elif parent_top1_correct:
                anchor_index = parent_top1
                anchor_positive = group_scores[parent_top1]
            elif rank_mode == "dual_multi_positive_competitor":
                # On repair groups, distribute supervision across every valid
                # match while retaining a max-like boundary.  The log-mean-exp
                # count correction prevents queries with many positives from
                # receiving an artificial score advantage.
                positive_scores = group_scores[positive]
                anchor_positive = rank_temperature * (
                    torch.logsumexp(positive_scores / rank_temperature, dim=0)
                    - math.log(positive_scores.numel())
                )
            else:
                anchor_positive = group_scores[positive].amax()
            hardest_negative = group_scores[negative].amax()
            preservation_group = retriever_top1_correct or parent_top1_correct
            repair_weight = retriever_error_weight
            if not preservation_group and near_miss_k is not None:
                k = min(near_miss_k, group_size)
                retriever_near_miss = bool(positive[:k].any())
                parent_topk = torch.argsort(
                    group_targets, descending=True, stable=True
                )[:k]
                parent_near_miss = bool(positive[parent_topk].any())
                if retriever_near_miss or parent_near_miss:
                    repair_weight *= near_miss_weight
            effective_margin = (
                parent_preservation_margin
                if preservation_group and parent_preservation_margin is not None
                else rank_margin
            )
            if (
                preservation_group
                and parent_preservation_teacher_margin_slack is not None
            ):
                # Preserve the inherited model's query-specific winning
                # boundary instead of replacing every solved query with one
                # small global margin.  The stored teacher target is
                # sigmoid(parent_score), so logit recovers the parent's score
                # and therefore its exact anchor-vs-hardest-negative margin.
                eps = torch.finfo(group_targets.dtype).eps
                parent_logits = torch.logit(
                    group_targets.clamp(eps, 1.0 - eps)
                )
                inherited_margin = (
                    parent_logits[anchor_index]
                    - parent_logits[negative].amax()
                    - parent_preservation_teacher_margin_slack
                ).clamp_min(0.0)
                effective_margin = torch.maximum(
                    torch.as_tensor(
                        effective_margin,
                        device=group_scores.device,
                        dtype=group_scores.dtype,
                    ),
                    inherited_margin.to(group_scores.dtype),
                )
            violation = hardest_negative + effective_margin - anchor_positive
            if preservation_group and rank_mode == "constrained_dual_top1_competitor":
                # A true constraint: once an existing retriever/parent win has
                # the requested safety margin it contributes exactly zero
                # loss and gradient. Softplus never becomes zero and therefore
                # drifts already-correct groups while trying to repair errors.
                competitor_loss = F.relu(violation) * parent_preservation_weight
            else:
                competitor_loss = F.softplus(violation / rank_temperature)
                competitor_loss = competitor_loss * (
                    parent_preservation_weight
                    if preservation_group
                    else repair_weight
                )
            competitor_losses.append(competitor_loss)
        loss_rank = torch.stack(competitor_losses).mean()
    elif rank_weight and rank_mode == "parent_top1_competitor":
        # Rank-1 depends only on whether the strongest relevant candidate beats
        # the strongest irrelevant candidate.  Use the frozen parent's top-1
        # decision to distinguish preservation groups from repair groups.  On
        # a parent win, anchor that exact relevant candidate; on a parent
        # error, promote whichever relevant candidate the student currently
        # finds strongest.  This avoids diluting the gradient over negatives
        # that cannot affect the deployed top-1 decision.
        assert teacher_targets is not None
        grouped_parent_targets = list(teacher_targets.split(group_size))
        competitor_losses = []
        for group_scores, group_targets, positive, negative in zip(
            grouped_scores, grouped_parent_targets, positives, negatives
        ):
            parent_top1 = int(torch.argmax(group_targets).item())
            parent_top1_correct = bool(positive[parent_top1])
            anchor_positive = (
                group_scores[parent_top1]
                if parent_top1_correct
                else group_scores[positive].amax()
            )
            hardest_negative = group_scores[negative].amax()
            effective_margin = (
                rank_margin
                if not parent_top1_correct or parent_preservation_margin is None
                else parent_preservation_margin
            )
            competitor_loss = F.softplus(
                (hardest_negative + effective_margin - anchor_positive)
                / rank_temperature
            )
            if not parent_top1_correct:
                competitor_loss = competitor_loss * retriever_error_weight
            else:
                competitor_loss = competitor_loss * parent_preservation_weight
            competitor_losses.append(competitor_loss)
        loss_rank = torch.stack(competitor_losses).mean()
    elif rank_weight and rank_mode == "parent_top1_corrective":
        # Fine-tuning should be corrective relative to the strongest existing
        # CR3 checkpoint, not only relative to the stage-1 retriever.  The
        # frozen parent's probabilities are monotonic in its logits, so their
        # argmax identifies its deployed top-1 candidate exactly.  Preserve
        # that exact candidate when it is relevant; only when the parent is
        # wrong may any relevant candidate become the repair target.
        assert teacher_targets is not None
        grouped_parent_targets = list(teacher_targets.split(group_size))
        corrective_losses = []
        for group_scores, group_targets, positive, negative in zip(
            grouped_scores, grouped_parent_targets, positives, negatives
        ):
            parent_top1 = int(torch.argmax(group_targets).item())
            parent_top1_correct = bool(positive[parent_top1])
            positive_mass = (
                group_scores[parent_top1].reshape(1)
                if parent_top1_correct
                else group_scores[positive]
            )
            negative_scores = group_scores[negative] + rank_margin
            corrective_loss = (
                torch.logsumexp(
                    torch.cat((positive_mass, negative_scores))
                    / rank_temperature,
                    dim=0,
                )
                - torch.logsumexp(positive_mass / rank_temperature, dim=0)
            )
            # Parent errors are the only groups that can improve over the
            # deployed checkpoint.  Upweight their corrective gradient while
            # leaving parent-correct preservation constraints at unit weight.
            if not parent_top1_correct:
                corrective_loss = corrective_loss * retriever_error_weight
            corrective_losses.append(corrective_loss)
        loss_rank = torch.stack(corrective_losses).mean()
    elif rank_weight and rank_mode in {
        "retriever_top1_competitor",
        "constrained_retriever_top1_competitor",
    }:
        # Rank-1 changes only when the strongest valid positive crosses the
        # strongest negative.  Optimize that boundary directly.  Preserve a
        # correct retriever top-1 by anchoring row zero; when row zero is
        # wrong, permit whichever valid positive the model currently finds
        # easiest to promote.  Unlike a probability-mass denominator, this
        # objective does not keep pushing merely because a group contains
        # many already-separated negatives.
        competitor_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            anchor_positive = (
                group_scores[0]
                if bool(positive[0])
                else group_scores[positive].amax()
            )
            hardest_negative = group_scores[negative].amax()
            violation = hardest_negative + rank_margin - anchor_positive
            if (
                bool(positive[0])
                and rank_mode == "constrained_retriever_top1_competitor"
            ):
                # Stop updating a correct SigLIP2 winner after it clears the
                # requested boundary. This avoids drifting solved teacher wins
                # while label supervision repairs the teacher's misses.
                competitor_loss = F.relu(violation)
            else:
                competitor_loss = F.softplus(violation / rank_temperature)
            # Only a negative retriever top-1 offers a chance to improve on
            # the retriever's Rank-1. Correct groups remain unit-weight
            # preservation constraints; callers may emphasize repair groups.
            if not bool(positive[0]):
                competitor_loss = competitor_loss * retriever_error_weight
            competitor_losses.append(competitor_loss)
        loss_rank = torch.stack(competitor_losses).mean()
    elif rank_weight and rank_mode == "retriever_top1_guarded":
        # Candidate rows retain retriever order, so row zero is SigLIP2's
        # top-1 decision.  If it is correct, anchor that exact positive and
        # prevent CR3 from breaking the win.  If it is wrong, allow any PAS
        # positive to take over.  Other positives are excluded from the
        # correct-top1 denominator because their internal ordering is
        # irrelevant to Rank-1.
        guarded_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            negative_scores = group_scores[negative] + rank_margin
            if bool(positive[0]):
                positive_mass = group_scores[0].reshape(1)
            else:
                positive_mass = group_scores[positive]
            guarded_losses.append(
                torch.logsumexp(
                    torch.cat((positive_mass, negative_scores))
                    / rank_temperature,
                    dim=0,
                )
                - torch.logsumexp(positive_mass / rank_temperature, dim=0)
            )
        loss_rank = torch.stack(guarded_losses).mean()
    elif rank_weight and rank_mode == "top1_competitor":
        # Optimize the exact decision boundary used by deployed Rank-1. Both
        # competitors are selected from the student's current scores, so the
        # objective follows whichever negative becomes hardest as training
        # changes the ordering. No retriever or frozen-parent score is used.
        top1_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            top1_losses.append(
                F.softplus(
                    (
                        group_scores[negative].amax()
                        + rank_margin
                        - group_scores[positive].amax()
                    )
                    / rank_temperature
                )
            )
        loss_rank = torch.stack(top1_losses).mean()
    elif rank_weight and rank_mode == "topm_all_pairs":
        # Learn more than the single current winner.  For each query, train
        # every relevant candidate against the top-m negatives currently most
        # likely to displace it.  This retains online hard-negative selection
        # while supplying several independent visual/semantic relations per
        # group, which is especially important for atomic-constraint labels.
        # Multiplying by tau keeps the loss and gradient in score units; the
        # historical top1 objective omitted this factor, so tau < 1 silently
        # amplified its gradient and frequently saturated global clipping.
        assert near_miss_k is not None
        topm_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            positive_scores = group_scores[positive]
            hard_negatives = group_scores[negative].topk(
                min(near_miss_k, int(negative.sum().item()))
            ).values
            violations = (
                hard_negatives.unsqueeze(0)
                + rank_margin
                - positive_scores.unsqueeze(1)
            )
            topm_losses.append(
                rank_temperature
                * F.softplus(violations / rank_temperature).mean()
            )
        loss_rank = torch.stack(topm_losses).mean()
    elif rank_weight and rank_mode == "robust_topm_gce":
        # Generalized cross entropy on positive-vs-hard-negative relations.
        # Ordinary logistic ranking gives a nearly maximal gradient to an
        # extreme contradictory pair, which makes a single false negative get
        # mined forever.  GCE keeps its strongest gradient near the decision
        # boundary and attenuates pairs the current model considers extreme.
        # q -> 0 recovers logistic CE; larger q is increasingly noise robust.
        assert near_miss_k is not None
        robust_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            positive_scores = group_scores[positive]
            hard_negatives = group_scores[negative].topk(
                min(near_miss_k, int(negative.sum().item()))
            ).values
            pair_probability = torch.sigmoid(
                (
                    positive_scores.unsqueeze(1)
                    - hard_negatives.unsqueeze(0)
                    - rank_margin
                )
                / rank_temperature
            )
            robust_losses.append(
                rank_temperature
                * ((1.0 - pair_probability.pow(robust_pair_q)) / robust_pair_q).mean()
            )
        loss_rank = torch.stack(robust_losses).mean()
    elif rank_weight and rank_mode == "topm_logsumexp":
        # Optimize the upper tail created by having several hard competitors.
        # The smooth maximum gives the highest-scoring negative the largest
        # gradient while still updating near-tied alternatives.  Unlike an
        # average of pair losses, one dangerous outlier cannot be diluted by
        # three already-safe negatives.  Multiplication by tau keeps the total
        # score-gradient bounded independently of temperature.
        assert near_miss_k is not None
        tail_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            positive_scores = group_scores[positive]
            hard_negatives = group_scores[negative].topk(
                min(near_miss_k, int(negative.sum().item()))
            ).values
            smooth_negative_max = rank_temperature * torch.logsumexp(
                hard_negatives / rank_temperature, dim=0
            )
            violation = (
                smooth_negative_max
                + rank_margin
                - positive_scores.amax()
            )
            tail_losses.append(
                rank_temperature * F.softplus(violation / rank_temperature)
            )
        loss_rank = torch.stack(tail_losses).mean()
    elif rank_weight and rank_mode == "smooth_top1":
        # Directly optimize the boundary that controls deployed Rank-1 while
        # keeping the hard-example assignment smooth. Log-mean-exp approaches
        # max as the temperature falls, but gives every near-tied candidate a
        # gradient. The mean correction prevents groups with more positives or
        # negatives from receiving an artificial score bonus due only to count.
        smooth_top1_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            positive_scores = group_scores[positive]
            negative_scores = group_scores[negative]
            smooth_positive = rank_temperature * (
                torch.logsumexp(positive_scores / rank_temperature, dim=0)
                - math.log(positive_scores.numel())
            )
            smooth_negative = rank_temperature * (
                torch.logsumexp(negative_scores / rank_temperature, dim=0)
                - math.log(negative_scores.numel())
            )
            smooth_top1_losses.append(
                F.softplus(smooth_negative + rank_margin - smooth_positive)
            )
        loss_rank = torch.stack(smooth_top1_losses).mean()
    elif rank_weight and rank_mode in {
        "partial_negatives",
        "partial_negatives_logmeanexp",
    }:
        # PASA-inspired Partial-negative Alignment (PA), adapted from a
        # symmetric dual-encoder batch to CR3's fixed query-candidate groups.
        # Select a fraction of the negatives that the *current CR3 model*
        # scores highest, aggregate them with a smooth maximum, and require
        # the soft positive score to exceed that aggregate by a margin.
        partial_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            positive_scores = group_scores[positive]
            negative_scores = group_scores[negative]
            selected_count = max(
                1, math.ceil(partial_negative_ratio * negative_scores.numel())
            )
            selected_negatives = torch.topk(
                negative_scores, k=selected_count, sorted=False
            ).values
            positive_weights = torch.softmax(
                positive_scores / rank_temperature, dim=0
            )
            soft_positive = (positive_weights * positive_scores).sum()
            smooth_hard_negative = rank_temperature * torch.logsumexp(
                selected_negatives / rank_temperature, dim=0
            )
            if rank_mode == "partial_negatives_logmeanexp":
                # CR3 groups can contain different positive multiplicities, so
                # ceil(ratio * num_negatives) produces a variable selected_count.
                # Raw log-sum-exp then adds tau*log(k) solely because a group has
                # more selected negatives.  Subtracting log(k) makes this a
                # smooth maximum of the selected scores without that count bias.
                smooth_hard_negative = smooth_hard_negative - (
                    rank_temperature * math.log(selected_count)
                )
            partial_loss = F.relu(
                rank_margin - soft_positive + smooth_hard_negative
            )
            # Row zero is the frozen retriever's winner.  Error groups are the
            # only ones that can produce a net Rank-1 gain over that retriever,
            # so let train-only online selection emphasize their multi-negative
            # boundary while the separate preservation term protects solved
            # groups.  A value of one preserves the historical PA objective.
            if not bool(positive[0]):
                partial_loss = partial_loss * retriever_error_weight
            partial_losses.append(partial_loss)
        loss_rank = torch.stack(partial_losses).mean()
    elif rank_weight and rank_mode in {
        "lambda_ndcg_top1",
        "constrained_retriever_lambda_ndcg_top1",
    }:
        # LambdaRank-style pair weighting: swapping a relevant/irrelevant pair
        # near the head changes NDCG much more than swapping a pair near the
        # tail.  Compute those metric deltas from the current (detached) order,
        # then use them to weight differentiable pairwise softplus gradients.
        # Add the exact best-positive/hardest-negative boundary because Rank-1
        # is the primary deployed decision and can otherwise be diluted when a
        # query contains many relevant candidates.
        lambda_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives
        ):
            with torch.no_grad():
                order = torch.argsort(group_scores, descending=True, stable=True)
                ranks = torch.empty_like(order)
                ranks[order] = torch.arange(group_size, device=scores.device)
                discounts = 1.0 / torch.log2(ranks.float() + 2.0)
                positive_count = int(positive.sum().item())
                ideal_discounts = 1.0 / torch.log2(
                    torch.arange(
                        positive_count, device=scores.device, dtype=torch.float32
                    )
                    + 2.0
                )
                ideal_dcg = ideal_discounts.sum()
                delta_ndcg = (
                    discounts[positive].unsqueeze(1)
                    - discounts[negative].unsqueeze(0)
                ).abs() / ideal_dcg
            pair_losses = F.softplus(
                (
                    group_scores[negative].unsqueeze(0)
                    + rank_margin
                    - group_scores[positive].unsqueeze(1)
                )
                / rank_temperature
            )
            weighted_pairs = (delta_ndcg * pair_losses).sum() / delta_ndcg.sum()
            head_loss = F.softplus(
                (
                    group_scores[negative].amax()
                    + rank_margin
                    - group_scores[positive].amax()
                )
                / rank_temperature
            )
            lambda_loss = weighted_pairs + head_loss
            if rank_mode == "constrained_retriever_lambda_ndcg_top1":
                if bool(positive[0]):
                    # Row zero is the frozen retriever's correct winner.  Use
                    # every negative as a true preservation constraint, but do
                    # not dilute the single boundary that controls Rank-1 by
                    # averaging it with up to K-2 already-safe negatives.  The
                    # head term protects the exact decision; the mean term also
                    # clears every violating negative.  Once all boundaries
                    # clear the margin this solved group has exactly zero
                    # gradient.
                    preservation_violations = F.relu(
                        group_scores[negative]
                        + rank_margin
                        - group_scores[0]
                    )
                    lambda_loss = (
                        preservation_violations.amax()
                        + preservation_violations.mean()
                    ) * retriever_success_weight
                else:
                    # Only retriever errors can produce a net Rank-1 gain.  On
                    # them retain every NDCG-weighted positive/negative swap
                    # and the exact best-positive/hardest-negative boundary.
                    lambda_loss = lambda_loss * retriever_error_weight
            lambda_losses.append(lambda_loss)
        loss_rank = torch.stack(lambda_losses).mean()
    elif rank_weight and rank_mode in {
        "pu_certified_logsumexp",
        "pu_certified_logmeanexp",
    }:
        # Positive-unlabeled listwise ranking. Candidate weights are a hard
        # certification mask: strict positives and independently verified
        # attribute-veto negatives have weight one; candidates whose relation
        # to the query is unknown have weight zero and therefore exactly zero
        # gradient. Each strict positive competes against the *set* of
        # certified negatives through one smooth log-sum-exp energy, without
        # treating the other strict positives as competitors.
        pu_losses = []
        for group_scores, positive, negative, group_weights in zip(
            grouped_scores,
            positives,
            negatives,
            grouped_candidate_weights,
            strict=True,
        ):
            if group_weights is None:
                certified_positive = positive
                certified_negative = negative
            else:
                if not bool(torch.all((group_weights == 0) | (group_weights == 1))):
                    raise ValueError(
                        f"{rank_mode} requires binary candidate weights"
                    )
                certified = group_weights > 0
                certified_positive = positive & certified
                certified_negative = negative & certified
            if not bool(certified_positive.any()) or not bool(
                certified_negative.any()
            ):
                raise ValueError(
                    "Every PU group needs a certified positive and negative"
                )
            negative_energy = rank_temperature * torch.logsumexp(
                group_scores[certified_negative] / rank_temperature, dim=0
            )
            if rank_mode == "pu_certified_logmeanexp":
                # A plain logsumexp adds T*log(N) even when every certified
                # negative has the same score.  N varies by source/domain in
                # masked K20 curricula, so that offset is an accidental margin
                # and source weight.  Log-mean-exp retains smooth hard-negative
                # focus while making the boundary invariant to duplicating an
                # otherwise identical certified negative.
                negative_energy = negative_energy - rank_temperature * math.log(
                    int(certified_negative.sum().item())
                )
            pu_losses.append(
                F.softplus(
                    (
                        negative_energy
                        + rank_margin
                        - group_scores[certified_positive]
                    )
                    / rank_temperature
                ).mean()
            )
        loss_rank = torch.stack(pu_losses).mean()
    elif rank_weight and rank_mode == "pu_bag_top1_logsumexp":
        # Positive-unlabeled bag Rank-1 boundary. Only independently certified
        # candidates enter either energy; ambiguous K20 context has exactly
        # zero gradient. A low-temperature positive logsumexp allows any one
        # certified positive to establish the head boundary instead of forcing
        # every possibly noisy positive to win, while the negative logsumexp
        # retains pressure from the complete certified competitor set.
        pu_bag_losses = []
        pu_tail_losses = []
        for group_scores, positive, negative, group_weights in zip(
            grouped_scores,
            positives,
            negatives,
            grouped_candidate_weights,
            strict=True,
        ):
            if group_weights is None:
                certified_positive = positive
                certified_negative = negative
            else:
                if not bool(torch.all((group_weights == 0) | (group_weights == 1))):
                    raise ValueError(
                        "pu_bag_top1_logsumexp requires binary candidate weights"
                    )
                certified = group_weights > 0
                certified_positive = positive & certified
                certified_negative = negative & certified
            if not bool(certified_positive.any()) or not bool(certified_negative.any()):
                raise ValueError(
                    "Every PU bag group needs a certified positive and negative"
                )
            positive_energy = bag_positive_temperature * torch.logsumexp(
                group_scores[certified_positive] / bag_positive_temperature, dim=0
            )
            negative_energy = bag_negative_temperature * torch.logsumexp(
                group_scores[certified_negative] / bag_negative_temperature, dim=0
            )
            pu_bag_losses.append(
                F.softplus(
                    (negative_energy + rank_margin - positive_energy)
                    / rank_temperature
                )
            )
            if positive_tail_weight:
                positive_softmin = -positive_tail_temperature * torch.logsumexp(
                    -group_scores[certified_positive] / positive_tail_temperature,
                    dim=0,
                )
                pu_tail_losses.append(
                    F.softplus(
                        (negative_energy + rank_margin - positive_softmin)
                        / rank_temperature
                    )
                )
        loss_rank = torch.stack(pu_bag_losses).mean()
        if positive_tail_weight:
            loss_rank = loss_rank + positive_tail_weight * torch.stack(
                pu_tail_losses
            ).mean()
    elif rank_weight and rank_mode == "nnpu_certified_hybrid":
        # Certified constraints give precise local ordering, while the
        # remaining Top-20 candidates are positive-unlabeled rather than
        # assumed negatives.  The non-negative PU correction learns from the
        # *distribution* of ambiguous candidates without assigning any one of
        # them a false-negative target.  This is the key difference from both
        # hard-negative training and simply discarding the unknown rows.
        hybrid_losses = []
        for group_scores, positive, negative, group_weights in zip(
            grouped_scores,
            positives,
            negatives,
            grouped_candidate_weights,
            strict=True,
        ):
            if group_weights is None or not bool(
                torch.all((group_weights == 0) | (group_weights == 1))
            ):
                raise ValueError(
                    "nnpu_certified_hybrid requires binary candidate weights; "
                    f"received {None if group_weights is None else group_weights.detach().cpu().tolist()}"
                )
            certified = group_weights > 0
            certified_positive = positive & certified
            certified_negative = negative & certified
            unlabeled = ~certified
            if not (
                bool(certified_positive.any())
                and bool(certified_negative.any())
                and bool(unlabeled.any())
            ):
                raise ValueError(
                    "Every nnPU group needs certified positives, certified "
                    "negatives, and unlabeled candidates"
                )

            positive_scores = group_scores[certified_positive]
            negative_scores = group_scores[certified_negative]
            unlabeled_scores = group_scores[unlabeled]
            negative_energy = rank_temperature * torch.logsumexp(
                negative_scores / rank_temperature, dim=0
            )
            certified_rank = F.softplus(
                (
                    negative_energy
                    + rank_margin
                    - positive_scores
                )
                / rank_temperature
            ).mean()

            # Logistic nnPU risk (Kiryo et al.).  The correction subtracts the
            # positive component already present in the unlabeled mixture;
            # clamping prevents a flexible model from driving the unbiased
            # negative-risk estimate below zero and overfitting label noise.
            positive_risk = F.softplus(
                -positive_scores / rank_temperature
            ).mean()
            positive_as_negative = F.softplus(
                positive_scores / rank_temperature
            ).mean()
            unlabeled_negative = F.softplus(
                unlabeled_scores / rank_temperature
            ).mean()
            corrected_negative = torch.clamp_min(
                unlabeled_negative - pu_class_prior * positive_as_negative,
                0.0,
            )
            certified_negative_risk = F.softplus(
                negative_scores / rank_temperature
            ).mean()
            nnpu_risk = (
                pu_class_prior * positive_risk
                + corrected_negative
                + certified_negative_risk
            )
            hybrid_losses.append(certified_rank + pu_nn_weight * nnpu_risk)
        loss_rank = torch.stack(hybrid_losses).mean()
    elif (
        rank_weight
        and rank_mode == "worst_positive_hardest_negative_logmeanexp"
    ):
        # Smooth the two extremes that determine a strict multi-positive
        # ranking boundary.  logmeanexp (rather than logsumexp) makes the
        # objective invariant to duplicating an entire positive or negative
        # set, so groups with different positive cardinality keep equal mass.
        boundary_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives, strict=True
        ):
            positive_scores = group_scores[positive]
            negative_scores = group_scores[negative]
            smooth_worst_positive = -rank_temperature * (
                torch.logsumexp(-positive_scores / rank_temperature, dim=0)
                - math.log(positive_scores.numel())
            )
            smooth_hardest_negative = rank_temperature * (
                torch.logsumexp(negative_scores / rank_temperature, dim=0)
                - math.log(negative_scores.numel())
            )
            boundary_losses.append(
                F.softplus(
                    (
                        rank_margin
                        + smooth_hardest_negative
                        - smooth_worst_positive
                    )
                    / rank_temperature
                )
            )
        loss_rank = torch.stack(boundary_losses).mean()
    elif rank_weight and rank_mode == "triplet_alignment_logsumexp":
        # RDE/TAL-style positive-to-negative-set alignment.  Unlike the bag
        # objective below, every reliable positive independently has to clear
        # the smooth collective negative energy; positives are never pooled.
        # This is particularly appropriate for a single-certain-positive K20.
        tal_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives, strict=True
        ):
            negative_energy = rank_temperature * torch.logsumexp(
                group_scores[negative] / rank_temperature, dim=0
            )
            tal_losses.append(
                F.relu(rank_margin - group_scores[positive] + negative_energy).mean()
            )
        loss_rank = torch.stack(tal_losses).mean()
    elif rank_weight and rank_mode == "bag_top1_logsumexp":
        # Multi-positive bag-to-bag Rank-1 boundary. Both sides deliberately
        # use unnormalized logsumexp (no log-count subtraction): multiple
        # valid positives increase the probability that at least one wins,
        # while every hard negative contributes collective displacement risk.
        # Unlike all-pairs/AP losses, noisy positives are not each required to
        # outrank every negative.
        bag_losses = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives, strict=True
        ):
            positive_energy = bag_positive_temperature * torch.logsumexp(
                group_scores[positive] / bag_positive_temperature, dim=0
            )
            negative_energy = bag_negative_temperature * torch.logsumexp(
                group_scores[negative] / bag_negative_temperature, dim=0
            )
            bag_losses.append(
                F.softplus(
                    (negative_energy - positive_energy + rank_margin)
                    / rank_temperature
                )
            )
        loss_rank = torch.stack(bag_losses).mean()
    elif rank_weight and rank_mode == "probability_mass":
        loss_rank = torch.stack(
            [
                torch.logsumexp(group_scores / rank_temperature, dim=0)
                - torch.logsumexp(
                    (group_scores / rank_temperature)[positive], dim=0
                )
                for group_scores, positive in zip(grouped_scores, positives)
            ]
        ).mean()
    elif rank_weight and rank_mode == "weak_veto_mil":
        full_losses = []
        mil_losses = []
        full_margins = []
        smooth_witnesses = []
        any_field_correct = []
        num_fields = group_size // 2 - 1
        log_field_count = math.log(num_fields)
        for group_scores in grouped_scores:
            paired = group_scores.reshape(-1, 2)
            pair_margins_for_group = paired[:, 0] - paired[:, 1]
            full_margin = pair_margins_for_group[0]
            field_margins = pair_margins_for_group[1:]
            # Normalized smooth maximum: unlike raw logsumexp it cannot improve
            # merely because a packet contains more duplicate field hypotheses.
            smooth_witness = weak_veto_temperature * (
                torch.logsumexp(field_margins / weak_veto_temperature, dim=0)
                - log_field_count
            )
            full_losses.append(F.relu(rank_margin - full_margin))
            mil_losses.append(F.relu(rank_margin - smooth_witness))
            full_margins.append(full_margin)
            smooth_witnesses.append(smooth_witness)
            any_field_correct.append(field_margins.amax() > 0)
        weak_veto_full_loss = torch.stack(full_losses).mean()
        weak_veto_mil_loss = torch.stack(mil_losses).mean()
        weak_veto_full_margin_mean = torch.stack(full_margins).detach().mean()
        weak_veto_smooth_witness_mean = (
            torch.stack(smooth_witnesses).detach().mean()
        )
        weak_veto_any_field_accuracy = (
            torch.stack(any_field_correct).float().mean()
        )
        loss_rank = (
            weak_veto_full_weight * weak_veto_full_loss
            + weak_veto_mil_weight * weak_veto_mil_loss
        )
    elif rank_weight and rank_mode == "natural20_inverse_pair":
        assert inverse_pair_offsets is not None
        natural_terms = []
        inverse_terms = []
        inverse_margins = []
        for group_scores, group_labels, original_offset in zip(
            grouped_scores,
            grouped_labels,
            inverse_pair_offsets.tolist(),
            strict=True,
        ):
            natural_scores = group_scores[:20]
            natural_positive = group_labels[:20] > 0.5
            natural_negative = ~natural_positive
            violations = F.relu(
                natural_scores[natural_negative].unsqueeze(0)
                + rank_margin
                - natural_scores[natural_positive].unsqueeze(1)
            )
            active = (violations.detach() > 0).to(violations.dtype)
            natural_terms.append(
                (violations * active).sum() / active.sum().clamp_min(1.0)
            )
            inverse_margin = group_scores[20] - group_scores[original_offset]
            inverse_margins.append(inverse_margin)
            inverse_terms.append(F.relu(rank_margin - inverse_margin))
        natural_loss = torch.stack(natural_terms).mean()
        inverse_pair_loss = torch.stack(inverse_terms).mean()
        inverse_margin_tensor = torch.stack(inverse_margins)
        inverse_pair_accuracy = (inverse_margin_tensor.detach() > 0).float().mean()
        inverse_pair_margin_mean = inverse_margin_tensor.detach().mean()
        loss_rank = (
            natural20_rank_weight * natural_loss
            + inverse_pair_weight * inverse_pair_loss
        )
    elif rank_weight and rank_mode in {
        "active_margin_all_pairs",
        "fixed_mass_active_all_pairs",
        "cardinality_compensated_fixed_mass",
    }:
        # True active-set margin ranking.  Unlike the historical ``all_pairs``
        # branch, this uses rank_margin and gives exactly zero gradient to a
        # positive/negative relation after the requested boundary is clear.
        # The historical mode normalizes by the still-active relations; the
        # fixed-mass variant retains the packet's original pair denominator.
        active_margin_terms = []
        for group_scores, positive, negative, group_weights in zip(
            grouped_scores,
            positives,
            negatives,
            grouped_candidate_weights,
            strict=True,
        ):
            violations = F.relu(
                group_scores[negative].unsqueeze(0)
                + rank_margin
                - group_scores[positive].unsqueeze(1)
            )
            active = (violations.detach() > 0).to(violations.dtype)
            if group_weights is None:
                active_weight = active
            else:
                pair_weights = (
                    group_weights[positive].unsqueeze(1)
                    * group_weights[negative].unsqueeze(0)
                )
                if not bool(pair_weights.sum() > 0):
                    raise ValueError(
                        "Every group needs positive weighted rank-pair mass"
                    )
                active_weight = active * pair_weights
            if rank_mode == "active_margin_all_pairs":
                # Historical behavior: every group with any unresolved edge
                # keeps unit mass.  Late in training this can amplify one
                # contradictory edge to the weight of a genuinely unsolved
                # packet.
                normalizer = active_weight.sum()
            else:
                # Preserve the packet's original pair mass.  Solved edges
                # still have exactly zero gradient, but the remaining edge is
                # not automatically enlarged as the active set shrinks.  This
                # is the hinge analogue of averaging label noise into the
                # broad natural distribution instead of hard-mining it.
                normalizer = (
                    pair_weights.sum()
                    if group_weights is not None
                    else violations.numel()
                )
            term = (
                (violations * active_weight).sum()
                / torch.as_tensor(normalizer, device=violations.device).clamp_min(1.0)
            )
            if rank_mode == "cardinality_compensated_fixed_mass":
                # With P positives and N negatives, ordinary pair averaging
                # gives each positive only about 1/P of a packet's gradient
                # mass.  The frozen train-only audit localizes essentially all
                # residual AP deficit to P=2..4, while P>=5 already clears the
                # target.  Restore one unit per-positive mass only for those
                # weak cardinalities, while retaining fixed-mass decay so a
                # single late contradictory edge is never amplified.
                positive_count = int(positive.sum().item())
                if 2 <= positive_count <= 4:
                    term = term * positive_count
            active_margin_terms.append(term)
        loss_rank = torch.stack(active_margin_terms).mean()
    elif rank_weight and rank_mode == "trimmed_active_all_pairs":
        # Label-preserving robust ranking for mined natural K20 packets.
        # The active negative with the largest mean violation is precisely the
        # candidate that ordinary hard mining repeats most aggressively.  Train
        # all of its positive/negative relations with a bounded residual weight
        # (robust_pair_q), while leaving all labels and all other active
        # relations unchanged.  If a group has only one active negative, retain
        # full weight so supervision never vanishes.
        trimmed_terms = []
        for group_scores, positive, negative, group_weights in zip(
            grouped_scores,
            positives,
            negatives,
            grouped_candidate_weights,
            strict=True,
        ):
            violations = F.relu(
                group_scores[negative].unsqueeze(0)
                + rank_margin
                - group_scores[positive].unsqueeze(1)
            )
            active = (violations.detach() > 0).to(violations.dtype)
            if group_weights is None:
                pair_weights = torch.ones_like(violations)
            else:
                pair_weights = (
                    group_weights[positive].unsqueeze(1)
                    * group_weights[negative].unsqueeze(0)
                )
                if not bool(pair_weights.sum() > 0):
                    raise ValueError(
                        "Every group needs positive weighted rank-pair mass"
                    )
            active_weight = active * pair_weights
            # Aggregate over positives first so a single suspect candidate is
            # the robust unit.  Otherwise one mislabeled negative is repeated
            # P times in a P-positive packet and can still dominate after
            # merely trimming one edge.
            active_negative = (active_weight.detach().sum(dim=0) > 0)
            if int(active_negative.sum().item()) > 1:
                aggregate = (
                    violations.detach() * active_weight.detach()
                ).sum(dim=0) / active_weight.detach().sum(dim=0).clamp_min(1.0)
                aggregate = aggregate.masked_fill(~active_negative, -torch.inf)
                outlier_negative = int(aggregate.argmax().item())
                robust_weight = torch.ones_like(violations)
                robust_weight[:, outlier_negative] = robust_pair_q
                active_weight = active_weight * robust_weight
            trimmed_terms.append(
                (violations * active_weight).sum()
                / active_weight.sum().clamp_min(1.0)
            )
        loss_rank = torch.stack(trimmed_terms).mean()
    elif rank_weight and rank_mode == "active_lambda_ap":
        # Query-local LambdaAP with a true inactive region.  The hard labels
        # define exact AP/Rank-1 swap values at the student's current order;
        # those values weight only positive/negative pairs still inside the
        # requested margin.  This keeps the supervision dense enough for K9
        # packets while concentrating updates on relations that can actually
        # change the deployment metric.
        lambda_terms = []
        active_fractions = []
        for group_scores, positive, negative in zip(
            grouped_scores, positives, negatives, strict=True
        ):
            positive_index, negative_index, swap_weight = exact_ap_swap_weights(
                group_scores,
                positive.float(),
                positive.float().sum(),
                rank1_weight=r69_soft_r1_weight,
            )
            violations = F.relu(
                group_scores[negative_index].unsqueeze(0)
                + rank_margin
                - group_scores[positive_index].unsqueeze(1)
            )
            active = (violations.detach() > 0).to(violations.dtype)
            active_weight = swap_weight * active
            lambda_terms.append(
                (active_weight * violations).sum()
                / active_weight.sum().clamp_min(torch.finfo(violations.dtype).eps)
            )
            active_fractions.append(active.mean().detach())
        loss_rank = torch.stack(lambda_terms).mean()
        r69_components = {
            "safe_lambda_active_pair_fraction": torch.stack(active_fractions),
        }
    elif rank_weight:
        # Unlike probability-mass ranking, this supplies a gradient for every
        # PAS positive/negative relation in the deployed candidate list.  It
        # is therefore aligned with AP's requirement that all relevant items
        # outrank all irrelevant items rather than merely lifting one positive.
        all_pairs_terms = []
        for group_scores, positive, negative, group_weights in zip(
            grouped_scores,
            positives,
            negatives,
            grouped_candidate_weights,
            strict=True,
        ):
            pair_losses = F.softplus(
                (
                    group_scores[negative].unsqueeze(0)
                    - group_scores[positive].unsqueeze(1)
                )
                / rank_temperature
            )
            if group_weights is None:
                all_pairs_terms.append(pair_losses.mean())
                continue
            pair_weights = (
                group_weights[positive].unsqueeze(1)
                * group_weights[negative].unsqueeze(0)
            )
            if not bool(pair_weights.sum() > 0):
                raise ValueError("Every group needs positive weighted rank-pair mass")
            all_pairs_terms.append(
                (pair_losses * pair_weights).sum() / pair_weights.sum()
            )
        all_pairs_loss = torch.stack(all_pairs_terms).mean()
        if rank_mode == "head_all_pairs":
            # AP needs every positive/negative relation, while Rank-1 is
            # controlled by the best positive and hardest negative.  The
            # extra head term prevents the many already-correct easy pairs in
            # a K-many group from diluting the one inversion that determines
            # whether reranking fixes the retrieved top item.
            head_loss = torch.stack(
                [
                    F.softplus(
                        (
                            group_scores[negative].amax()
                            - group_scores[positive].amax()
                        )
                        / rank_temperature
                    )
                    for group_scores, positive, negative in zip(
                        grouped_scores, positives, negatives
                    )
                ]
            ).mean()
            loss_rank = all_pairs_loss + head_loss
        elif rank_mode == "hybrid_all_pairs":
            # All-pairs is aligned with AP but can improve the average pair
            # while sacrificing the strongest positive, which controls the
            # deployed Rank-1/Rank-5 behavior.  Retain the original
            # multi-positive probability-mass term as a preservation term.
            probability_mass_loss = torch.stack(
                [
                    torch.logsumexp(group_scores / rank_temperature, dim=0)
                    - torch.logsumexp(
                        (group_scores / rank_temperature)[positive], dim=0
                    )
                    for group_scores, positive in zip(grouped_scores, positives)
                ]
            ).mean()
            loss_rank = all_pairs_loss + probability_mass_loss
        else:
            loss_rank = all_pairs_loss
    else:
        loss_rank = zero
    if rank_weight and all_pairs_aux_weight:
        # The primary rank mode protects the deployed head decision.  This
        # auxiliary term supplies supervision for every relevant/irrelevant
        # relation, which is the part of the ordering measured by AP.  It is
        # opt-in so existing objectives and checkpoints remain unchanged.
        all_pairs_aux = rank_temperature * torch.stack(
            [
                F.softplus(
                    (
                        group_scores[negative].unsqueeze(0)
                        - group_scores[positive].unsqueeze(1)
                    )
                    / rank_temperature
                ).mean()
                for group_scores, positive, negative in zip(
                    grouped_scores, positives, negatives
                )
            ]
        ).mean()
        loss_rank = loss_rank + all_pairs_aux_weight * all_pairs_aux
    if not point_weight:
        loss_point = zero
    elif point_mode == "all":
        point_losses = F.binary_cross_entropy_with_logits(
            scores / point_temperature, labels, reduction="none"
        )
        point_weights = torch.where(
            labels > 0.5,
            torch.ones_like(labels),
            torch.full_like(labels, negative_point_weight),
        )
        if candidate_weights is not None:
            point_weights = point_weights * candidate_weights
        # Normalize by the applied weight mass so changing class balance does
        # not silently change the point objective's scale relative to ranking.
        loss_point = (point_losses * point_weights).sum() / point_weights.sum()
    elif point_mode == "positive_only":
        positive_mask = labels > 0.5
        positive_losses = F.binary_cross_entropy_with_logits(
            scores[positive_mask] / point_temperature,
            torch.ones_like(scores[positive_mask]),
            reduction="none",
        )
        if candidate_weights is None:
            loss_point = positive_losses.mean()
        else:
            positive_weights = candidate_weights[positive_mask]
            loss_point = (
                (positive_losses * positive_weights).sum()
                / positive_weights.sum().clamp_min(1e-12)
            )
    else:
        # Give every query equal weight, then give its positive and negative
        # classes equal mass. This prevents variable K-many class counts from
        # silently changing the pointwise class prior.  The query-centered
        # variant also removes each packet's score offset before BCE.  A
        # reranker is invariant to that offset, so this supplies smooth
        # candidate-level gradients without spending LoRA capacity on a
        # global yes/no calibration shared by visually unequal queries.
        grouped_point_scores = list(scores.split(group_size))
        grouped_point_labels = list(labels.split(group_size))
        group_point_terms = []
        for group_scores, group_labels, positive, negative, group_weights in zip(
            grouped_point_scores,
            grouped_point_labels,
            positives,
            negatives,
            grouped_candidate_weights,
            strict=True,
        ):
            if point_mode == "query_centered_class_balanced":
                if group_weights is None:
                    center = group_scores.mean()
                else:
                    center = (
                        (group_scores * group_weights).sum()
                        / group_weights.sum().clamp_min(1e-12)
                    )
                group_scores = group_scores - center
            group_losses = F.binary_cross_entropy_with_logits(
                group_scores / point_temperature,
                group_labels,
                reduction="none",
            )
            if group_weights is None:
                positive_loss = group_losses[positive].mean()
                negative_loss = group_losses[negative].mean()
            else:
                positive_weights = group_weights[positive]
                negative_weights = group_weights[negative]
                if not bool(positive_weights.sum() > 0) or not bool(
                    negative_weights.sum() > 0
                ):
                    raise ValueError(
                        "Every group needs positive and negative point-weight mass"
                    )
                positive_loss = (
                    group_losses[positive] * positive_weights
                ).sum() / positive_weights.sum()
                negative_loss = (
                    group_losses[negative] * negative_weights
                ).sum() / negative_weights.sum()
            group_point_terms.append(
                (positive_loss + negative_point_weight * negative_loss)
                / (1.0 + negative_point_weight)
            )
        loss_point = torch.stack(group_point_terms).mean()
    # Rank-k succeeds when some positive outranks the kth-hardest negative.
    # Preserve that boundary for every query where either the retriever or
    # frozen parent already succeeds, while leaving joint misses untouched for
    # the corrective top-1 objective.
    if topk_preservation_weight:
        assert topk_preservation_k is not None
        grouped_parent_targets = (
            list(teacher_targets.split(group_size))
            if teacher_targets is not None
            else [None] * len(grouped_scores)
        )
        topk_losses = []
        for group_scores, group_targets, positive, negative in zip(
            grouped_scores,
            grouped_parent_targets,
            positives,
            negatives,
        ):
            k = min(topk_preservation_k, group_size)
            retriever_topk = torch.arange(k, device=group_scores.device)
            preserved_sources = [retriever_topk[positive[retriever_topk]]]
            if group_targets is not None:
                parent_topk = torch.argsort(
                    group_targets, descending=True, stable=True
                )[:k]
                preserved_sources.append(parent_topk[positive[parent_topk]])
            preserved_positive = torch.cat(preserved_sources).unique()
            if not preserved_positive.numel():
                topk_losses.append(zero)
                continue
            anchor_positive = group_scores[preserved_positive].amax()
            negative_scores = group_scores[negative]
            negative_k = min(k, negative_scores.numel())
            kth_hardest_negative = torch.topk(
                negative_scores, negative_k
            ).values[-1]
            topk_losses.append(
                F.relu(
                    kth_hardest_negative
                    + topk_preservation_margin
                    - anchor_positive
                )
            )
        loss_rank = loss_rank + topk_preservation_weight * torch.stack(
            topk_losses
        ).mean()

    grouped_teacher_targets = (
        list(teacher_targets.split(group_size)) if teacher_targets is not None else None
    )
    teacher_pair_targets = None
    teacher_pair_scores = None
    if teacher_weight and teacher_mode == "pointwise":
        assert teacher_targets is not None
        loss_teacher = F.binary_cross_entropy_with_logits(
            scores / teacher_temperature,
            teacher_targets,
        )
    elif teacher_targets is not None and teacher_mode != "pointwise":
        # Parent-aware rank modes need the frozen-parent targets even when the
        # teacher imitation term is disabled.  Build the pairwise tensors for
        # telemetry and rank-mode bookkeeping; multiplying by a zero teacher
        # weight still keeps them out of the optimization objective.
        assert grouped_teacher_targets is not None
        teacher_pair_scores = torch.cat(
            [
                (
                    group_scores[positive].unsqueeze(1)
                    - group_scores[negative].unsqueeze(0)
                ).reshape(-1)
                for group_scores, positive, negative in zip(
                    grouped_scores, positives, negatives
                )
            ]
        )
        # The frozen parent is useful as a stability target on groups where
        # either it or the retriever already wins. On a joint-error group,
        # however, imitating all of the parent's otherwise-correct pairwise
        # relations can pin the shared representation and oppose changing the
        # single winner that controls Rank-1. Allow that imitation pressure to
        # be attenuated while the explicit rank and top-k guard terms repair
        # and protect the deployed boundaries.
        teacher_pair_weights = torch.cat(
            [
                torch.full(
                    (
                        int(positive.sum().item())
                        * int(negative.sum().item()),
                    ),
                    (
                        1.0
                        if bool(positive[0])
                        or bool(positive[int(torch.argmax(group_targets).item())])
                        else teacher_repair_group_weight
                    ),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                for group_targets, positive, negative in zip(
                    grouped_teacher_targets, positives, negatives
                )
            ]
        )
        teacher_pair_differences = torch.cat(
            [
                (
                    group_targets[positive].unsqueeze(1)
                    - group_targets[negative].unsqueeze(0)
                ).reshape(-1)
                for group_targets, positive, negative in zip(
                    grouped_teacher_targets, positives, negatives
                )
            ]
        )
        if teacher_mode in {
            "pairwise_logit_delta_clamped",
            "pairwise_logit_delta_correct_only",
        }:
            # A frozen-parent row stores t = sigmoid(s_parent / tau_teacher).
            # Converting each probability back to a logit makes the pair
            # target exactly sigmoid((s_pos - s_neg) / tau_teacher), so every
            # PAS-consistent parent pair has zero teacher gradient at step 0.
            # Inverted pairs are clamped to a tie; PAS rank/point losses then
            # provide the correction direction without teacher label inversion.
            eps = torch.finfo(teacher_pair_differences.dtype).eps
            teacher_pair_logit_differences = torch.cat(
                [
                    (
                        torch.logit(group_targets[positive].clamp(eps, 1.0 - eps)).unsqueeze(1)
                        - torch.logit(group_targets[negative].clamp(eps, 1.0 - eps)).unsqueeze(0)
                    ).reshape(-1)
                    for group_targets, positive, negative in zip(
                        grouped_teacher_targets, positives, negatives
                    )
                ]
            )
            if teacher_mode == "pairwise_logit_delta_clamped":
                teacher_pair_targets = torch.sigmoid(
                    teacher_pair_logit_differences.clamp_min(0.0)
                )
            else:
                # Preserve only relations that the frozen parent already gets
                # right.  An inverted parent relation must be corrected solely
                # by PAS supervision: using the current student's detached
                # probability as its target gives that relation exactly zero
                # teacher gradient without dropping it from distributed metric
                # aggregation or changing the objective scale between batches.
                parent_consistent = teacher_pair_logit_differences > 0.0
                current_detached_targets = torch.sigmoid(
                    (teacher_pair_scores / teacher_temperature).detach()
                )
                teacher_pair_targets = torch.where(
                    parent_consistent,
                    torch.sigmoid(teacher_pair_logit_differences),
                    current_detached_targets,
                )
        elif teacher_mode == "pairwise_margin_clamped":
            teacher_pair_differences = teacher_pair_differences.clamp(0.0, 1.0)
        elif not bool(
            ((teacher_pair_differences > 0.0) & (teacher_pair_differences <= 1.0)).all()
        ):
            raise ValueError(
                "pairwise_margin teacher requires every PAS positive target to "
                "exceed every negative target in its group"
            )
        if teacher_pair_targets is None:
            teacher_pair_targets = 0.5 + 0.45 * teacher_pair_differences
        teacher_pair_losses = F.binary_cross_entropy_with_logits(
            teacher_pair_scores / teacher_temperature,
            teacher_pair_targets,
            reduction="none",
        )
        teacher_pair_weight_sum = teacher_pair_weights.sum()
        loss_teacher = (
            (teacher_pair_losses * teacher_pair_weights).sum()
            / teacher_pair_weight_sum
            if bool(teacher_pair_weight_sum > 0)
            else zero
        )
    else:
        loss_teacher = zero
    if transition_pooled_weight:
        transition_losses = []
        transition_correct = []
        for group_scores, group_labels in zip(grouped_scores, grouped_labels, strict=True):
            group_positive = group_labels > 0.5
            group_negative = ~group_positive
            negative_scores = group_scores[group_negative]
            hard_negative_scores = torch.topk(
                negative_scores,
                k=min(transition_pooled_topk, negative_scores.numel()),
            ).values
            pooled_negative = transition_pooled_temperature * torch.logsumexp(
                hard_negative_scores / transition_pooled_temperature, dim=0
            )
            target_positive = (
                group_scores[0]
                if bool(group_positive[0])
                else group_scores[group_positive].amax()
            )
            transition_losses.append(
                transition_pooled_temperature
                * F.softplus(
                    (
                        pooled_negative
                        - target_positive
                        + transition_pooled_margin
                    )
                    / transition_pooled_temperature
                )
            )
            transition_correct.append(target_positive > negative_scores.amax())
        loss_transition_pooled = torch.stack(transition_losses).mean()
        transition_pooled_accuracy = torch.stack(transition_correct).float().mean()
    else:
        loss_transition_pooled = zero
        transition_pooled_accuracy = zero

    total = (
        rank_weight * (loss_rank + transition_pooled_weight * loss_transition_pooled)
        + point_weight * loss_point
        + teacher_weight * loss_teacher
    )

    with torch.no_grad():
        metric_scores = torch.cat(metric_grouped_scores)
        metric_labels = torch.cat(metric_grouped_labels)
        pair_margins = torch.cat(
            [
                (
                    group_scores[positive].unsqueeze(1)
                    - group_scores[negative].unsqueeze(0)
                ).reshape(-1)
                for group_scores, positive, negative in zip(
                    metric_grouped_scores, metric_positives, metric_negatives
                )
            ]
        )
        positive_scores = torch.cat(
            [
                group_scores[positive]
                for group_scores, positive in zip(
                    metric_grouped_scores, metric_positives
                )
            ]
        )
        negative_scores = torch.cat(
            [
                group_scores[negative]
                for group_scores, negative in zip(
                    metric_grouped_scores, metric_negatives
                )
            ]
        )
        best_positive_scores = torch.stack(
            [
                group_scores[positive].amax()
                for group_scores, positive in zip(
                    metric_grouped_scores, metric_positives
                )
            ]
        )
        hardest_negative_scores = torch.stack(
            [
                group_scores[negative].amax()
                for group_scores, negative in zip(
                    metric_grouped_scores, metric_negatives
                )
            ]
        )
        top1_margins = best_positive_scores - hardest_negative_scores
        bag_positive_energies = torch.stack(
            [
                bag_positive_temperature
                * torch.logsumexp(
                    group_scores[positive] / bag_positive_temperature, dim=0
                )
                for group_scores, positive in zip(
                    metric_grouped_scores, metric_positives
                )
            ]
        )
        bag_negative_energies = torch.stack(
            [
                bag_negative_temperature
                * torch.logsumexp(
                    group_scores[negative] / bag_negative_temperature, dim=0
                )
                for group_scores, negative in zip(
                    metric_grouped_scores, metric_negatives
                )
            ]
        )
        bag_energy_margins = bag_positive_energies - bag_negative_energies
        student_top1_correct = top1_margins > 0
        retriever_top1_correct = torch.stack(
            [positive[0] for positive in metric_positives]
        )
        retriever_error = ~retriever_top1_correct
        rank1_fixes = retriever_error & student_top1_correct
        rank1_breaks = retriever_top1_correct & ~student_top1_correct
        retriever_correct_count = retriever_top1_correct.sum()
        retriever_error_count = retriever_error.sum()
        zero_metric = zero.detach()
        preservation_accuracy = (
            (student_top1_correct & retriever_top1_correct).float().sum()
            / retriever_correct_count
            if bool(retriever_correct_count)
            else zero_metric
        )
        repair_accuracy = (
            rank1_fixes.float().sum() / retriever_error_count
            if bool(retriever_error_count)
            else zero_metric
        )
        if teacher_targets is None:
            teacher_target_mean = zero.detach()
            teacher_probability_mae = zero.detach()
        elif teacher_mode in {
            "pairwise_margin",
            "pairwise_margin_clamped",
            "pairwise_logit_delta_clamped",
            "pairwise_logit_delta_correct_only",
        }:
            assert teacher_pair_scores is not None
            assert teacher_pair_targets is not None
            student_probabilities = torch.sigmoid(
                teacher_pair_scores / teacher_temperature
            )
            teacher_target_mean = teacher_pair_targets.mean()
            teacher_probability_mae = (
                student_probabilities - teacher_pair_targets
            ).abs().mean()
        else:
            student_probabilities = torch.sigmoid(scores / teacher_temperature)
            teacher_target_mean = teacher_targets.mean()
            teacher_probability_mae = (
                student_probabilities - teacher_targets
            ).abs().mean()
        average_precision = torch.stack(
            [
                (
                    ordered_positive.cumsum(0)
                    / torch.arange(
                        1,
                        ordered_positive.numel() + 1,
                        device=ordered_positive.device,
                        dtype=torch.float32,
                    )
                    * ordered_positive
                ).sum()
                / ordered_positive.sum()
                for group_scores, positive in zip(
                    metric_grouped_scores, metric_positives
                )
                for ordered_positive in [
                    positive[
                        torch.argsort(group_scores, descending=True, stable=True)
                    ].float()
                ]
            ]
        ).mean()
        metrics = {
            "rank": loss_rank.detach(),
            "point": loss_point.detach(),
            "teacher": loss_teacher.detach(),
            "teacher_target_mean": teacher_target_mean.detach(),
            "teacher_probability_mae": teacher_probability_mae.detach(),
            "pairwise_accuracy": (pair_margins > 0).float().mean(),
            "positive_score_mean": positive_scores.mean(),
            "negative_score_mean": negative_scores.mean(),
            "best_positive_score_mean": best_positive_scores.mean(),
            "hardest_negative_score_mean": hardest_negative_scores.mean(),
            "top1_margin": top1_margins.mean(),
            "bag_positive_energy": bag_positive_energies.mean(),
            "bag_negative_energy": bag_negative_energies.mean(),
            "bag_energy_margin": bag_energy_margins.mean(),
            "bag_energy_accuracy": (bag_energy_margins > 0).float().mean(),
            "top1_accuracy": student_top1_correct.float().mean(),
            "retriever_top1_accuracy": retriever_top1_correct.float().mean(),
            "retriever_success_preservation_accuracy": preservation_accuracy,
            "retriever_error_repair_accuracy": repair_accuracy,
            "rank1_fix_rate": rank1_fixes.float().mean(),
            "rank1_break_rate": rank1_breaks.float().mean(),
            "rank1_net_gain": (
                rank1_fixes.float().mean() - rank1_breaks.float().mean()
            ),
            "binary_accuracy": (
                (metric_scores > 0) == (metric_labels > 0.5)
            ).float().mean(),
            "positive_accuracy": (positive_scores > 0).float().mean(),
            "negative_accuracy": (negative_scores < 0).float().mean(),
            "balanced_binary_accuracy": torch.stack(
                [
                    0.5
                    * (
                        (group_scores[positive] > 0).float().mean()
                        + (group_scores[negative] < 0).float().mean()
                    )
                    for group_scores, positive, negative in zip(
                        metric_grouped_scores, metric_positives, metric_negatives
                    )
                ]
            ).mean(),
            "hard_negative_accuracy": (hardest_negative_scores < 0).float().mean(),
            "average_precision": average_precision,
            "rank5_accuracy": torch.stack(
                [
                    positive[
                        torch.argsort(group_scores, descending=True, stable=True)[:5]
                    ].any().float()
                    for group_scores, positive in zip(
                        metric_grouped_scores, metric_positives
                    )
                ]
            ).mean(),
            "margin": pair_margins.mean(),
            "inverse_pair_loss": inverse_pair_loss.detach(),
            "inverse_pair_accuracy": inverse_pair_accuracy.detach(),
            "inverse_pair_margin": inverse_pair_margin_mean.detach(),
            "weak_veto_full_loss": weak_veto_full_loss.detach(),
            "weak_veto_mil_loss": weak_veto_mil_loss.detach(),
            "weak_veto_full_margin": weak_veto_full_margin_mean.detach(),
            "weak_veto_smooth_witness": weak_veto_smooth_witness_mean.detach(),
            "weak_veto_any_field_accuracy": (
                weak_veto_any_field_accuracy.detach()
            ),
            "transition_pooled_loss": loss_transition_pooled.detach(),
            "transition_pooled_accuracy": transition_pooled_accuracy.detach(),
            # Keep the metric schema stable for non-policy objectives.  The
            # policy branch overwrites these values below with real telemetry.
            "policy_reward": scores.new_zeros(()),
            "policy_expected_ap": scores.new_zeros(()),
            "policy_expected_r1": scores.new_zeros(()),
            "policy_preserve_anchor": scores.new_zeros(()),
            "policy_first_entropy": scores.new_zeros(()),
            "policy_preserve_fraction": scores.new_zeros(()),
        }
        if policy_components is not None:
            metrics.update(
                {
                    name: values.mean()
                    for name, values in policy_components.items()
                }
            )
        if full_gallery_lambda_result is not None:
            metrics.update(
                {
                    "full_gallery_ap_delta": full_gallery_lambda_result.mean_ap_delta,
                    "full_gallery_oracle_ap_gain": (
                        full_gallery_lambda_result.mean_oracle_ap_gain
                    ),
                    "full_gallery_oracle_regret": (
                        full_gallery_lambda_result.mean_oracle_regret
                    ),
                    "full_gallery_head_recall": (
                        full_gallery_lambda_result.mean_head_recall
                    ),
                    "lambda_ap_weight_mass": (
                        full_gallery_lambda_result.lambda_weight_mass
                    ),
                    "lambda_ap_active_pair_fraction": (
                        full_gallery_lambda_result.active_pair_fraction
                    ),
                }
            )
        if r69_components is not None:
            if "smooth_ap" in r69_components:
                metrics.update(
                    {
                        "r69_smooth_ap": r69_components["smooth_ap"].mean(),
                        "r69_smooth_ap_loss": r69_components[
                            "smooth_ap_loss"
                        ].mean(),
                        "r69_soft_r1_loss": r69_components[
                            "soft_r1_loss"
                        ].mean(),
                    }
                )
            if "incumbent_guard_loss" in r69_components:
                metrics.update(
                    {
                        "incumbent_guard_loss": r69_components[
                            "incumbent_guard_loss"
                        ].mean(),
                        "incumbent_guard_pass_rate": r69_components[
                            "incumbent_guard_pass"
                        ].mean(),
                    }
                )
            if "safe_lambda_active_pair_fraction" in r69_components:
                metrics["safe_lambda_active_pair_fraction"] = r69_components[
                    "safe_lambda_active_pair_fraction"
                ].mean()
            if "misordered_lambda_ap_weight_mass" in r69_components:
                metrics["misordered_lambda_ap_weight_mass"] = r69_components[
                    "misordered_lambda_ap_weight_mass"
                ].mean()
            if "safe_residual_repair_loss" in r69_components:
                for name in (
                    "safe_residual_repair_pair_fraction",
                    "safe_residual_preserve_active_fraction",
                    "safe_residual_repair_loss",
                    "safe_residual_preservation_loss",
                ):
                    metrics[name] = r69_components[name].mean()
            if "robust_lambdaap" in r69_components:
                for name in (
                    "robust_lambdaap",
                    "robust_soft_r1",
                    "robust_pair_weight_mass",
                    "robust_positive_trim_fraction",
                    "robust_negative_trim_fraction",
                ):
                    metrics[name] = r69_components[name].mean()
            if "paired_view_consistency" in r69_components:
                for name in (
                    "paired_view_consistency",
                    "paired_view_effective_weight",
                    "paired_view_contribution_fraction",
                    "paired_view_centered_abs_gap",
                ):
                    metrics[name] = r69_components[name].mean()
            if "r535_clause_aux_loss" in r69_components:
                for name in (
                    "r535_natural_loss",
                    "r535_clause_aux_loss",
                    "r535_matched_delta",
                    "r535_veto_delta",
                    "r535_delta_contrast",
                    "r535_matched_sign_accuracy",
                    "r535_veto_sign_accuracy",
                ):
                    metrics[name] = r69_components[name].mean()
        if rank_mode in {"orthogonal_cycle", "deployment_weighted_robust_cycle"}:
            metrics.update(
                {
                    "oic_q_edge_accuracy": (edge_margins[:, 0] > 0).float().mean(),
                    "oic_qn_edge_accuracy": (edge_margins[:, 1] > 0).float().mean(),
                    "oic_both_edges_accuracy": (edge_margins > 0)
                    .all(dim=1)
                    .float()
                    .mean(),
                    "oic_cycle_gap": edge_margins.sum(dim=1).mean(),
                }
            )
            if rank_mode == "deployment_weighted_robust_cycle":
                metrics.update(
                    {
                        "oic_deployment_edge_loss": deployment_edge_losses.mean(),
                        "oic_robust_edge_loss": robust_edge_losses.mean(),
                        "oic_cycle_weight_mean": cycle_weights.mean(),
                        "oic_q_gradient_mass": q_gradient_mass.mean(),
                        "oic_qn_gradient_mass": qn_gradient_mass.mean(),
                    }
                )
        if rank_mode == "preservation_constrained_cycle_block":
            metrics.update(
                {
                    "r68_q_edge_accuracy": (
                        block_edge_margins[:, :, 0] > 0
                    ).float().mean(),
                    "r68_qn_edge_accuracy": (
                        block_edge_margins[:, :, 1] > 0
                    ).float().mean(),
                    "r68_both_edges_accuracy": (
                        block_edge_margins > 0
                    ).all(dim=2).float().mean(),
                    "r68_preservation_hinge": preservation_hinge_mean,
                    "r68_preservation_pass_rate": preservation_pass_rate,
                    "r68_repair_block_fraction": repair_block_fraction,
                    "r68_vulnerable_block_fraction": vulnerable_block_fraction,
                    "r68_diversity_block_fraction": diversity_block_fraction,
                }
            )
    return total, metrics


def cumulative_requirement_ordinal_loss(
    scores: torch.Tensor,
    satisfied_counts: torch.Tensor,
    total_counts: torch.Tensor,
    weights: torch.Tensor,
    *,
    threshold_gap: float = 1.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Make one deployed scalar encode a hard atomic-requirement count.

    For a row with ``M`` requirements and ``c`` satisfied requirements, the
    same yes-minus-no logit is trained against the cumulative hard events
    ``c >= 1, ..., c >= M``.  Fixed cutpoints are ``-(M-k)*gap`` so the final
    all-requirements cutpoint is exactly zero, aligned with the native binary
    yes/no boundary.  No extra prediction head or deployment-time output is
    introduced.
    """

    scores = scores.float().reshape(-1)
    satisfied = satisfied_counts.long().reshape(-1).to(scores.device)
    totals = total_counts.long().reshape(-1).to(scores.device)
    weights = weights.float().reshape(-1).to(scores.device)
    if not (scores.shape == satisfied.shape == totals.shape == weights.shape):
        raise ValueError("Requirement ordinal tensors must have identical shapes")
    if threshold_gap <= 0 or temperature <= 0:
        raise ValueError("Requirement ordinal gap and temperature must be positive")
    if bool((totals < 1).any()) or bool((satisfied < 0).any()) or bool((satisfied > totals).any()):
        raise ValueError("Requirement counts must satisfy 0 <= satisfied <= total and total >= 1")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("Requirement ordinal weights must be finite and nonnegative")
    selected = weights > 0
    count = int(selected.sum().item())
    if not count:
        zero = scores.sum() * 0.0
        return zero, {
            "requirement_ordinal_loss": zero.detach(),
            "requirement_threshold_accuracy": zero.detach(),
            "requirement_count_mae": zero.detach(),
        }, 0
    maximum = int(totals.max().item())
    thresholds = torch.arange(1, maximum + 1, device=scores.device).unsqueeze(0)
    active = thresholds <= totals.unsqueeze(1)
    targets = thresholds <= satisfied.unsqueeze(1)
    cutpoints = -(totals.unsqueeze(1) - thresholds).float() * float(threshold_gap)
    logits = (scores.unsqueeze(1) - cutpoints) / float(temperature)
    event_loss = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    per_row = (event_loss * active).sum(dim=1) / totals.float()
    normalizer = weights[selected].sum().clamp_min(1e-12)
    loss = (per_row[selected] * weights[selected]).sum() / normalizer
    event_weight = weights.unsqueeze(1) * active
    event_normalizer = event_weight.sum().clamp_min(1e-12)
    accuracy = (
        ((logits >= 0) == targets).float() * event_weight
    ).sum() / event_normalizer
    predicted_count = (torch.sigmoid(logits) * active).sum(dim=1)
    count_mae = (
        (predicted_count[selected] - satisfied[selected].float()).abs()
        * weights[selected]
    ).sum() / normalizer
    return loss, {
        "requirement_ordinal_loss": loss.detach(),
        "requirement_threshold_accuracy": accuracy.detach(),
        "requirement_count_mae": count_mae.detach(),
    }, count


class RankPointLoss:
    """Cosmos-RL loss callback with fixed distributed metric aggregation."""

    METRIC_NAMES = (
        "rank",
        "point",
        "teacher",
        "teacher_target_mean",
        "teacher_probability_mae",
        "pairwise_accuracy",
        "positive_score_mean",
        "negative_score_mean",
        "best_positive_score_mean",
        "hardest_negative_score_mean",
        "top1_margin",
        "bag_positive_energy",
        "bag_negative_energy",
        "bag_energy_margin",
        "bag_energy_accuracy",
        "top1_accuracy",
        "retriever_top1_accuracy",
        "retriever_success_preservation_accuracy",
        "retriever_error_repair_accuracy",
        "rank1_fix_rate",
        "rank1_break_rate",
        "rank1_net_gain",
        "binary_accuracy",
        "positive_accuracy",
        "negative_accuracy",
        "balanced_binary_accuracy",
        "hard_negative_accuracy",
        "average_precision",
        "rank5_accuracy",
        "margin",
        "transition_pooled_loss",
        "transition_pooled_accuracy",
        "inverse_pair_loss",
        "inverse_pair_accuracy",
        "inverse_pair_margin",
        "weak_veto_full_loss",
        "weak_veto_mil_loss",
        "weak_veto_full_margin",
        "weak_veto_smooth_witness",
        "weak_veto_any_field_accuracy",
        "policy_reward",
        "policy_expected_ap",
        "policy_expected_r1",
        "policy_preserve_anchor",
        "policy_first_entropy",
        "policy_preserve_fraction",
    )
    # Auxiliaries are sparse by configuration, unlike the legacy metrics above.
    # Keep them out of METRIC_NAMES so mean_metrics() retains its longstanding
    # default contract while still accumulating/reporting them when enabled.
    OPTIONAL_METRIC_NAMES = (
        "same_label_consistency",
        "requirement_ordinal_loss",
        "requirement_threshold_accuracy",
        "requirement_count_mae",
        "retriever_residual_alpha_mean",
        "retriever_residual_gate_regularization",
    )
    CYCLE_METRIC_NAMES = (
        "oic_q_edge_accuracy",
        "oic_qn_edge_accuracy",
        "oic_both_edges_accuracy",
        "oic_cycle_gap",
        "oic_deployment_edge_loss",
        "oic_robust_edge_loss",
        "oic_cycle_weight_mean",
        "oic_q_gradient_mass",
        "oic_qn_gradient_mass",
        "r68_q_edge_accuracy",
        "r68_qn_edge_accuracy",
        "r68_both_edges_accuracy",
        "r68_preservation_hinge",
        "r68_preservation_pass_rate",
        "r68_repair_block_fraction",
        "r68_vulnerable_block_fraction",
        "r68_diversity_block_fraction",
    )
    FULL_GALLERY_METRIC_NAMES = (
        "full_gallery_ap_delta",
        "full_gallery_oracle_ap_gain",
        "full_gallery_oracle_regret",
        "full_gallery_head_recall",
        "lambda_ap_weight_mass",
        "lambda_ap_active_pair_fraction",
    )
    R69_METRIC_NAMES = (
        "r69_smooth_ap",
        "r69_smooth_ap_loss",
        "r69_soft_r1_loss",
        "incumbent_guard_loss",
        "incumbent_guard_pass_rate",
        "safe_lambda_active_pair_fraction",
        "misordered_lambda_ap_weight_mass",
        "safe_residual_repair_pair_fraction",
        "safe_residual_preserve_active_fraction",
        "safe_residual_repair_loss",
        "safe_residual_preservation_loss",
        "robust_lambdaap",
        "robust_soft_r1",
        "robust_pair_weight_mass",
        "robust_positive_trim_fraction",
        "robust_negative_trim_fraction",
        "consensus_lambdaap",
        "consensus_soft_r1",
        "consensus_pair_weight_mass",
        "consensus_trim_triggered",
        "consensus_outlier_ratio",
        "paired_view_consistency",
        "paired_view_effective_weight",
        "paired_view_contribution_fraction",
        "paired_view_centered_abs_gap",
        "r535_natural_loss",
        "r535_clause_aux_loss",
        "r535_matched_delta",
        "r535_veto_delta",
        "r535_delta_contrast",
        "r535_matched_sign_accuracy",
        "r535_veto_sign_accuracy",
    )

    def __init__(
        self,
        response_spec: BinaryResponseSpec,
        *,
        rank_weight: float,
        point_weight: float,
        rank_temperature: float,
        point_temperature: float,
        rank_mode: Literal[
            "probability_mass",
            "bag_top1_logsumexp",
            "triplet_alignment_logsumexp",
            "worst_positive_hardest_negative_logmeanexp",
            "pu_certified_logsumexp",
            "pu_certified_logmeanexp",
            "pu_bag_top1_logsumexp",
            "nnpu_certified_hybrid",
            "orthogonal_cycle",
            "deployment_weighted_robust_cycle",
            "preservation_constrained_cycle_block",
            "smoothap_soft_r1",
            "plackett_luce_policy",
            "rb_plackett_luce_expected_ap",
            "counterfactual_set5",
            "paired_adjacent",
            "dynamic_top20_smoothap_soft_r1",
            "incumbent_guarded_smoothap_soft_r1",
            "incumbent_safe_lambda_ap",
            "misordered_lambda_ap",
            "asymmetric_safe_residual_lambda_ap",
            "robust_normalized_lambdaap_soft_r1",
            "paired_view_robust_normalized_lambdaap_soft_r1",
            "consensus_trimmed_lambdaap_soft_r1",
            "r535_semantic_and_aux",
            "all_pairs",
            "active_margin_all_pairs",
            "fixed_mass_active_all_pairs",
            "cardinality_compensated_fixed_mass",
            "trimmed_active_all_pairs",
            "natural20_inverse_pair",
            "weak_veto_mil",
            "active_lambda_ap",
            "head_all_pairs",
            "lambda_ndcg_top1",
            "constrained_retriever_lambda_ndcg_top1",
            "smooth_top1",
            "top1_competitor",
            "topm_all_pairs",
            "robust_topm_gce",
            "topm_logsumexp",
            "hybrid_all_pairs",
            "partial_negatives",
            "partial_negatives_logmeanexp",
            "retriever_top1_guarded",
            "retriever_top1_competitor",
            "constrained_retriever_top1_competitor",
            "parent_top1_corrective",
            "parent_top1_competitor",
            "dual_top1_competitor",
            "constrained_dual_top1_competitor",
            "dual_multi_positive_competitor",
            "query_selective_regret",
            "full_gallery_lambda_ap",
        ] = "probability_mass",
        bag_positive_temperature: float = 0.1,
        bag_negative_temperature: float = 0.1,
        pu_class_prior: float = 0.2,
        pu_nn_weight: float = 0.25,
        partial_negative_ratio: float = 0.2,
        rank_margin: float = 0.05,
        natural20_rank_weight: float = 1.0,
        inverse_pair_weight: float = 1.0,
        weak_veto_temperature: float = 0.2,
        weak_veto_full_weight: float = 1.0,
        weak_veto_mil_weight: float = 1.0,
        robust_pair_q: float = 0.3,
        all_pairs_aux_weight: float = 0.0,
        positive_tail_weight: float = 0.0,
        positive_tail_temperature: float = 0.2,
        retriever_success_weight: float = 1.0,
        retriever_error_weight: float = 1.0,
        near_miss_k: int | None = None,
        near_miss_weight: float = 1.0,
        parent_preservation_margin: float | None = None,
        parent_preservation_teacher_margin_slack: float | None = None,
        parent_preservation_weight: float = 1.0,
        topk_preservation_k: int | None = None,
        topk_preservation_margin: float = 0.0,
        topk_preservation_weight: float = 0.0,
        point_mode: Literal[
            "all",
            "positive_only",
            "class_balanced",
            "query_centered_class_balanced",
        ] = "all",
        negative_point_weight: float = 1.0,
        teacher_weight: float = 0.0,
        teacher_repair_group_weight: float = 1.0,
        teacher_temperature: float = 1.0,
        teacher_mode: Literal[
            "pointwise",
            "pairwise_margin",
            "pairwise_margin_clamped",
            "pairwise_logit_delta_clamped",
            "pairwise_logit_delta_correct_only",
        ] = "pointwise",
        group_size: int,
        ordinal_token_ids: Sequence[int] | None = None,
        ordinal_response_spec: OrdinalResponseSpec | None = None,
        ordinal_aux_response_spec: OrdinalResponseSpec | None = None,
        ordinal_ce_weight: float = 0.0,
        ordinal_aux_position: Literal["suffix", "predecision"] = "suffix",
        hidden_supcon_weight: float = 0.0,
        hidden_supcon_temperature: float = 0.1,
        hidden_supcon_center: bool = True,
        same_label_consistency_weight: float = 0.0,
        requirement_ordinal_weight: float = 0.0,
        requirement_ordinal_gap: float = 1.0,
        requirement_ordinal_temperature: float = 1.0,
        ordinal_score_readout: Literal["expected", "strict_logodds"] = "expected",
        query_consistency_weight: float = 0.0,
        query_cross_weight: float = 0.0,
        full_gallery_rank1_weight: float = 0.0,
        full_gallery_active_topk: int | None = None,
        transition_pooled_weight: float = 0.0,
        transition_pooled_topk: int = 4,
        transition_pooled_temperature: float = 0.2,
        transition_pooled_margin: float = 0.0,
        robust_cycle_rho: float = 0.25,
        robust_cycle_lambda: float = 1.0,
        preservation_cycle_margin: float = 2.0,
        preservation_cycle_weight: float = 4.0,
        repair_cycle_ap_weight: float = 0.25,
        preserve_cycle_ap_weight: float = 0.10,
        preserve_cycle_robust_scale: float = 0.25,
        r69_smoothap_temperature: float = 0.384007173733197,
        r69_soft_r1_temperature: float = 2.076483289679061,
        r69_smoothap_weight: float = 1.0,
        r69_soft_r1_weight: float = 0.25,
        policy_samples: int = 16,
        policy_exact_max_k: int = 7,
        policy_ap_reward_weight: float = 1.0,
        policy_r1_reward_weight: float = 1.0,
        policy_preserve_anchor_weight: float = 1.0,
        policy_preserve_anchor_margin: float = 0.0,
        binary_decision_position: Literal["prefix", "unique_suffix"] = "prefix",
        binary_readout: Literal["fp32", "bf16_ste"] = "fp32",
        retriever_residual_alpha: float | None = None,
        retriever_residual_epsilon: float = 1e-4,
    ) -> None:
        self.response_spec = response_spec
        self.rank_weight = float(rank_weight)
        self.point_weight = float(point_weight)
        self.rank_temperature = float(rank_temperature)
        self.bag_positive_temperature = float(bag_positive_temperature)
        self.bag_negative_temperature = float(bag_negative_temperature)
        self.pu_class_prior = float(pu_class_prior)
        self.pu_nn_weight = float(pu_nn_weight)
        self.rank_mode = rank_mode
        self.partial_negative_ratio = float(partial_negative_ratio)
        self.rank_margin = float(rank_margin)
        self.natural20_rank_weight = float(natural20_rank_weight)
        self.inverse_pair_weight = float(inverse_pair_weight)
        self.weak_veto_temperature = float(weak_veto_temperature)
        self.weak_veto_full_weight = float(weak_veto_full_weight)
        self.weak_veto_mil_weight = float(weak_veto_mil_weight)
        self.robust_pair_q = float(robust_pair_q)
        self.all_pairs_aux_weight = float(all_pairs_aux_weight)
        self.positive_tail_weight = float(positive_tail_weight)
        self.positive_tail_temperature = float(positive_tail_temperature)
        self.retriever_success_weight = float(retriever_success_weight)
        self.retriever_error_weight = float(retriever_error_weight)
        self.near_miss_k = near_miss_k
        self.near_miss_weight = float(near_miss_weight)
        self.parent_preservation_margin = (
            None
            if parent_preservation_margin is None
            else float(parent_preservation_margin)
        )
        self.parent_preservation_teacher_margin_slack = (
            None
            if parent_preservation_teacher_margin_slack is None
            else float(parent_preservation_teacher_margin_slack)
        )
        self.parent_preservation_weight = float(parent_preservation_weight)
        self.topk_preservation_k = topk_preservation_k
        self.topk_preservation_margin = float(topk_preservation_margin)
        self.topk_preservation_weight = float(topk_preservation_weight)
        self.point_temperature = float(point_temperature)
        self.point_mode = point_mode
        self.negative_point_weight = float(negative_point_weight)
        self.teacher_weight = float(teacher_weight)
        self.teacher_repair_group_weight = float(teacher_repair_group_weight)
        self.teacher_temperature = float(teacher_temperature)
        self.teacher_mode = teacher_mode
        self.group_size = int(group_size)
        self.ordinal_token_ids = (
            None if ordinal_token_ids is None else tuple(int(value) for value in ordinal_token_ids)
        )
        if self.ordinal_token_ids is not None and ordinal_response_spec is not None:
            raise ValueError(
                "legacy ordinal_token_ids and explicit ordinal_response_spec are exclusive"
            )
        if ordinal_response_spec is not None and ordinal_aux_response_spec is not None:
            raise ValueError("ordinal score and ordinal auxiliary specs are exclusive")
        if ordinal_ce_weight < 0:
            raise ValueError("ordinal_ce_weight cannot be negative")
        if hidden_supcon_weight < 0:
            raise ValueError("hidden_supcon_weight cannot be negative")
        if hidden_supcon_temperature <= 0:
            raise ValueError("hidden_supcon_temperature must be positive")
        self.ordinal_response_spec = ordinal_response_spec
        self.ordinal_aux_response_spec = ordinal_aux_response_spec
        self.ordinal_ce_weight = float(ordinal_ce_weight)
        if ordinal_aux_position not in {"suffix", "predecision"}:
            raise ValueError(f"Unsupported ordinal auxiliary position: {ordinal_aux_position!r}")
        self.ordinal_aux_position = ordinal_aux_position
        self.hidden_supcon_weight = float(hidden_supcon_weight)
        self.hidden_supcon_temperature = float(hidden_supcon_temperature)
        self.hidden_supcon_center = bool(hidden_supcon_center)
        if same_label_consistency_weight < 0:
            raise ValueError("same_label_consistency_weight cannot be negative")
        self.same_label_consistency_weight = float(same_label_consistency_weight)
        if requirement_ordinal_weight < 0:
            raise ValueError("requirement_ordinal_weight cannot be negative")
        if requirement_ordinal_gap <= 0 or requirement_ordinal_temperature <= 0:
            raise ValueError("requirement ordinal gap and temperature must be positive")
        self.requirement_ordinal_weight = float(requirement_ordinal_weight)
        self.requirement_ordinal_gap = float(requirement_ordinal_gap)
        self.requirement_ordinal_temperature = float(requirement_ordinal_temperature)
        self.ordinal_score_readout = ordinal_score_readout
        self.query_consistency_weight = float(query_consistency_weight)
        self.query_cross_weight = float(query_cross_weight)
        self.full_gallery_rank1_weight = float(full_gallery_rank1_weight)
        self.full_gallery_active_topk = full_gallery_active_topk
        self.transition_pooled_weight = float(transition_pooled_weight)
        self.transition_pooled_topk = int(transition_pooled_topk)
        self.transition_pooled_temperature = float(transition_pooled_temperature)
        self.transition_pooled_margin = float(transition_pooled_margin)
        self.robust_cycle_rho = float(robust_cycle_rho)
        self.robust_cycle_lambda = float(robust_cycle_lambda)
        self.preservation_cycle_margin = float(preservation_cycle_margin)
        self.preservation_cycle_weight = float(preservation_cycle_weight)
        self.repair_cycle_ap_weight = float(repair_cycle_ap_weight)
        self.preserve_cycle_ap_weight = float(preserve_cycle_ap_weight)
        self.preserve_cycle_robust_scale = float(preserve_cycle_robust_scale)
        self.r69_smoothap_temperature = float(r69_smoothap_temperature)
        self.r69_soft_r1_temperature = float(r69_soft_r1_temperature)
        self.r69_smoothap_weight = float(r69_smoothap_weight)
        self.r69_soft_r1_weight = float(r69_soft_r1_weight)
        self.policy_samples = int(policy_samples)
        self.policy_exact_max_k = int(policy_exact_max_k)
        self.policy_ap_reward_weight = float(policy_ap_reward_weight)
        self.policy_r1_reward_weight = float(policy_r1_reward_weight)
        self.policy_preserve_anchor_weight = float(policy_preserve_anchor_weight)
        self.policy_preserve_anchor_margin = float(policy_preserve_anchor_margin)
        self.binary_decision_position = binary_decision_position
        if binary_readout not in {"fp32", "bf16_ste"}:
            raise ValueError(f"Unsupported binary readout: {binary_readout!r}")
        if binary_readout != "fp32" and (
            self.ordinal_token_ids is not None
            or self.ordinal_response_spec is not None
            or self.ordinal_aux_response_spec is not None
        ):
            raise ValueError("bf16_ste requires the native yes/no score mode")
        self.binary_readout = binary_readout
        if retriever_residual_alpha is not None and (
            not math.isfinite(retriever_residual_alpha)
            or retriever_residual_alpha < 0
        ):
            raise ValueError(
                "retriever_residual_alpha must be finite and nonnegative"
            )
        if (
            not math.isfinite(retriever_residual_epsilon)
            or retriever_residual_epsilon <= 0
        ):
            raise ValueError(
                "retriever_residual_epsilon must be finite and positive"
            )
        self.retriever_residual_alpha = (
            None
            if retriever_residual_alpha is None
            else float(retriever_residual_alpha)
        )
        self.retriever_residual_epsilon = float(retriever_residual_epsilon)
        # The trainer may bind a tiny loss-side module here before the first
        # step and owns its explicit distributed optimizer/checkpoint lifecycle.
        self.retriever_residual_gate: QueryAdaptiveResidualGate | None = None
        self.retriever_residual_gate_log_alpha_l2 = 0.0
        self.canonical_validation = False
        self.projection_weight: torch.Tensor | None = None
        # RankPointLoss is an ordinary callable rather than nn.Module, so this
        # reference does not register the model head twice.
        self.projection_module: torch.nn.Module | None = None
        self.last_metrics: dict[str, torch.Tensor] = {}
        self.last_scores: torch.Tensor | None = None
        self.last_labels: torch.Tensor | None = None
        self._metric_sums: dict[str, torch.Tensor] = {}
        self._metric_counts: dict[str, int] = {}
        self._metric_groups = 0
        self._teacher_probabilities: list[float] | None = None
        self._candidate_loss_weights: list[float] | None = None
        self._total_relevant: list[float] | None = None
        self._cycle_weights: list[float] | None = None
        self._cycle_block_metadata: list[float] | None = None
        self._policy_preserve_flags: list[bool] | None = None
        self._inverse_pair_offsets: list[int] | None = None
        # One optional raw/edit pair offset per candidate group.  ``-1`` means
        # the group is ordinary replay.  The equality target is deliberately
        # label independent: labels are used only to audit that the selected
        # intervention is irrelevant to the query.
        self._same_label_consistency_offsets: list[int] | None = None
        self._requirement_ordinal_records: list[tuple[int, int, float]] | None = None
        self._retriever_residual_scores: list[float] | None = None
        # Optional r268 A-GEM split.  This is configured only by the dedicated
        # trainer; ordinary PAS objectives never consume these records.
        self._safe_update_num_strata = 0
        self._safe_update_cushion = 0.0
        self._safe_update_component = "combined"
        self._safe_update_records: list[tuple[float, bool, bool, int]] | None = None
        self.safe_update_last_count = 0

    def set_retriever_residual_scores(self, values: Sequence[float]) -> None:
        """Queue frozen SigLIP2 scores for the next train/validation batch."""

        if self.retriever_residual_alpha is None:
            raise ValueError("Retriever residual fusion is not configured")
        if self._retriever_residual_scores is not None:
            raise ValueError("Previous retriever residual scores were not consumed")
        scores = [float(value) for value in values]
        if not scores or any(not math.isfinite(value) for value in scores):
            raise ValueError("Retriever residual scores must be finite")
        self._retriever_residual_scores = scores

    def assert_retriever_residual_scores_consumed(self) -> None:
        if self._retriever_residual_scores:
            raise ValueError(
                f"{len(self._retriever_residual_scores)} retriever residual scores "
                "were not consumed"
            )
        self._retriever_residual_scores = None

    def _take_retriever_residual_scores(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if self.retriever_residual_alpha is None:
            return None
        if (
            self._retriever_residual_scores is None
            or len(self._retriever_residual_scores) < count
        ):
            available = (
                0
                if self._retriever_residual_scores is None
                else len(self._retriever_residual_scores)
            )
            raise ValueError(
                f"Need {count} retriever residual scores, found {available}"
            )
        selected = self._retriever_residual_scores[:count]
        del self._retriever_residual_scores[:count]
        return torch.tensor(selected, device=device, dtype=torch.float32)

    def set_candidate_loss_weights(self, values: Sequence[float]) -> None:
        """Queue per-candidate confidence weights for the next training batch."""

        if self._candidate_loss_weights is not None:
            raise ValueError("Previous candidate loss weights were not consumed")
        weights = [float(value) for value in values]
        if not weights or any(
            not math.isfinite(value) or value < 0 for value in weights
        ):
            raise ValueError("Candidate loss weights must be finite and nonnegative")
        self._candidate_loss_weights = weights

    def assert_candidate_loss_weights_consumed(self) -> None:
        if self._candidate_loss_weights:
            raise ValueError(
                f"{len(self._candidate_loss_weights)} candidate weights were not consumed"
            )
        self._candidate_loss_weights = None

    def _take_candidate_loss_weights(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if self.canonical_validation:
            return None
        if self._candidate_loss_weights is None:
            return None
        if len(self._candidate_loss_weights) < count:
            raise ValueError(
                f"Need {count} candidate weights, found "
                f"{len(self._candidate_loss_weights)}"
            )
        selected = self._candidate_loss_weights[:count]
        del self._candidate_loss_weights[:count]
        return torch.tensor(selected, device=device, dtype=torch.float32)

    def set_policy_preserve_flags(self, values: Sequence[bool]) -> None:
        if self._policy_preserve_flags is not None:
            raise ValueError("Previous PL preserve flags were not consumed")
        self._policy_preserve_flags = [bool(value) for value in values]

    def assert_policy_preserve_flags_consumed(self) -> None:
        if self._policy_preserve_flags:
            raise ValueError(
                f"{len(self._policy_preserve_flags)} PL preserve flags were not consumed"
            )
        self._policy_preserve_flags = None

    def _take_policy_preserve_flags(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if self.rank_mode not in {
            "plackett_luce_policy",
            "rb_plackett_luce_expected_ap",
        } or self.canonical_validation:
            return None
        if (
            self._policy_preserve_flags is None
            or len(self._policy_preserve_flags) < count
        ):
            available = (
                0
                if self._policy_preserve_flags is None
                else len(self._policy_preserve_flags)
            )
            raise ValueError(f"Need {count} PL preserve flags, found {available}")
        selected = self._policy_preserve_flags[:count]
        del self._policy_preserve_flags[:count]
        return torch.tensor(selected, device=device, dtype=torch.bool)

    def set_inverse_pair_offsets(self, values: Sequence[int]) -> None:
        """Queue one natural-negative offset for each K21 training group."""

        if self.rank_mode != "natural20_inverse_pair":
            raise ValueError(
                "inverse-pair offsets require rank_mode=natural20_inverse_pair"
            )
        if self._inverse_pair_offsets is not None:
            raise ValueError("Previous inverse-pair offsets were not consumed")
        offsets = [int(value) for value in values]
        if not offsets or any(value < 0 or value >= 20 for value in offsets):
            raise ValueError("inverse-pair offsets must lie in [0, 19]")
        self._inverse_pair_offsets = offsets

    def set_same_label_consistency_offsets(self, values: Sequence[int]) -> None:
        """Queue one optional consecutive raw/edit pair per K-way group."""

        if not self.same_label_consistency_weight:
            raise ValueError("same-label consistency is not configured")
        if self._same_label_consistency_offsets is not None:
            raise ValueError("Previous consistency offsets were not consumed")
        offsets = [int(value) for value in values]
        if not offsets or any(
            value < -1 or value + 1 >= self.group_size for value in offsets
        ):
            raise ValueError("Consistency offsets must be -1 or start a valid pair")
        self._same_label_consistency_offsets = offsets

    def set_requirement_ordinal_records(
        self, values: Sequence[tuple[int, int, float]]
    ) -> None:
        if not self.requirement_ordinal_weight:
            raise ValueError("requirement ordinal supervision is not configured")
        if self._requirement_ordinal_records is not None:
            raise ValueError("Previous requirement ordinal records were not consumed")
        records = [(int(c), int(m), float(w)) for c, m, w in values]
        if not records or any(
            m < 1 or c < 0 or c > m or not math.isfinite(w) or w < 0
            for c, m, w in records
        ):
            raise ValueError("Invalid requirement ordinal records")
        self._requirement_ordinal_records = records

    def assert_requirement_ordinal_records_consumed(self) -> None:
        if self._requirement_ordinal_records:
            raise ValueError(
                f"{len(self._requirement_ordinal_records)} requirement records were not consumed"
            )
        self._requirement_ordinal_records = None

    def _take_requirement_ordinal_records(
        self, count: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.requirement_ordinal_weight or self.canonical_validation:
            return None
        if self._requirement_ordinal_records is None or len(self._requirement_ordinal_records) < count:
            available = 0 if self._requirement_ordinal_records is None else len(self._requirement_ordinal_records)
            raise ValueError(f"Need {count} requirement records, found {available}")
        selected = self._requirement_ordinal_records[:count]
        del self._requirement_ordinal_records[:count]
        satisfied, totals, weights = zip(*selected, strict=True)
        return (
            torch.tensor(satisfied, device=device, dtype=torch.long),
            torch.tensor(totals, device=device, dtype=torch.long),
            torch.tensor(weights, device=device, dtype=torch.float32),
        )

    def assert_same_label_consistency_offsets_consumed(self) -> None:
        if self._same_label_consistency_offsets:
            raise ValueError(
                f"{len(self._same_label_consistency_offsets)} consistency offsets "
                "were not consumed"
            )
        self._same_label_consistency_offsets = None

    def _take_same_label_consistency_offsets(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if not self.same_label_consistency_weight or self.canonical_validation:
            return None
        if (
            self._same_label_consistency_offsets is None
            or len(self._same_label_consistency_offsets) < count
        ):
            available = (
                0
                if self._same_label_consistency_offsets is None
                else len(self._same_label_consistency_offsets)
            )
            raise ValueError(
                f"Need {count} consistency offsets, found {available}"
            )
        selected = self._same_label_consistency_offsets[:count]
        del self._same_label_consistency_offsets[:count]
        return torch.tensor(selected, device=device, dtype=torch.long)

    def assert_inverse_pair_offsets_consumed(self) -> None:
        if self._inverse_pair_offsets:
            raise ValueError(
                f"{len(self._inverse_pair_offsets)} inverse-pair offsets were not consumed"
            )
        self._inverse_pair_offsets = None

    def _take_inverse_pair_offsets(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if self.rank_mode != "natural20_inverse_pair":
            return None
        if self._inverse_pair_offsets is None or len(self._inverse_pair_offsets) < count:
            available = (
                0
                if self._inverse_pair_offsets is None
                else len(self._inverse_pair_offsets)
            )
            raise ValueError(
                f"Need {count} inverse-pair offsets, found {available}"
            )
        selected = self._inverse_pair_offsets[:count]
        del self._inverse_pair_offsets[:count]
        return torch.tensor(selected, device=device, dtype=torch.long)

    def configure_safe_update(self, *, num_strata: int, cushion: float) -> None:
        """Enable a hard parent-decision split without soft-label regression."""

        if self.rank_mode not in {
            "bag_top1_logsumexp",
            "asymmetric_safe_residual_lambda_ap",
            "misordered_lambda_ap",
        }:
            raise ValueError(
                "safe update requires bag_top1_logsumexp, "
                "asymmetric_safe_residual_lambda_ap, or misordered_lambda_ap"
            )
        if self.point_weight or self.teacher_weight:
            raise ValueError("safe update requires point_weight=teacher_weight=0")
        if num_strata < 1 or num_strata > 9:
            raise ValueError("safe update num_strata must lie in [1, 9]")
        if cushion <= 0.0:
            raise ValueError("safe update cushion must be positive")
        self._safe_update_num_strata = int(num_strata)
        self._safe_update_cushion = float(cushion)

    def set_safe_update_component(self, component: str) -> None:
        valid = {"combined", "repair", *(
            f"preserve_{index}" for index in range(self._safe_update_num_strata)
        )}
        if component not in valid:
            raise ValueError(f"Invalid safe-update component {component!r}")
        self._safe_update_component = component

    def set_safe_update_records(
        self, values: Sequence[tuple[float, bool, bool, int]]
    ) -> None:
        if not self._safe_update_num_strata:
            raise ValueError("safe update is not configured")
        if self._safe_update_records:
            raise ValueError("Previous safe-update metadata were not consumed")
        records = [
            (float(margin), bool(anchor), bool(correct), int(stratum))
            for margin, anchor, correct, stratum in values
        ]
        if any(
            not math.isfinite(margin)
            or stratum < 0
            or stratum >= self._safe_update_num_strata
            for margin, _, _, stratum in records
        ):
            raise ValueError("Invalid safe-update metadata")
        self._safe_update_records = records

    def assert_safe_update_records_consumed(self) -> None:
        if self._safe_update_records:
            raise ValueError(
                f"{len(self._safe_update_records)} safe-update records were not consumed"
            )
        self._safe_update_records = None

    def _take_safe_update_records(
        self, count: int
    ) -> list[tuple[float, bool, bool, int]] | None:
        if not self._safe_update_num_strata or self.canonical_validation:
            return None
        if self._safe_update_records is None or len(self._safe_update_records) < count:
            available = 0 if self._safe_update_records is None else len(self._safe_update_records)
            raise ValueError(
                f"Need {count} safe-update metadata rows, found {available}"
            )
        selected = self._safe_update_records[:count]
        del self._safe_update_records[:count]
        return selected

    @staticmethod
    def _dp_exact_selected_mean(
        local_losses: list[torch.Tensor], reference: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Mean over selected groups, invariant to their uneven DP placement."""

        local_count = torch.tensor(
            float(len(local_losses)), device=reference.device, dtype=torch.float32
        )
        global_count = local_count.clone()
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
        count = int(global_count.item())
        if not count:
            return reference.sum() * 0.0, 0
        local_sum = (
            torch.stack(local_losses).sum()
            if local_losses
            else reference.sum() * 0.0
        )
        # Cosmos averages parameter gradients over DP after backward.  Scaling
        # every local sum by W/N therefore gives the exact global selected mean.
        return local_sum * (float(world_size) / float(count)), count

    def set_cycle_block_metadata(self, values: Sequence[float]) -> None:
        """Queue ten numeric metadata values per 32-row r68 query block."""

        if self._cycle_block_metadata is not None:
            raise ValueError("Previous cycle-block metadata were not fully consumed")
        metadata = [float(value) for value in values]
        if not metadata or len(metadata) % 10 or any(
            not math.isfinite(value) for value in metadata
        ):
            raise ValueError("Cycle-block metadata must be finite 10-value records")
        self._cycle_block_metadata = metadata

    def assert_cycle_block_metadata_consumed(self) -> None:
        if self._cycle_block_metadata:
            raise ValueError(
                f"{len(self._cycle_block_metadata)} cycle-block metadata values "
                "were not consumed"
            )
        self._cycle_block_metadata = None

    def _take_cycle_block_metadata(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if self.rank_mode != "preservation_constrained_cycle_block":
            return None
        required = count * 10
        if (
            self._cycle_block_metadata is None
            or len(self._cycle_block_metadata) < required
        ):
            available = (
                0
                if self._cycle_block_metadata is None
                else len(self._cycle_block_metadata)
            )
            raise ValueError(
                f"Need {required} r68 metadata values, found {available}"
            )
        selected = self._cycle_block_metadata[:required]
        del self._cycle_block_metadata[:required]
        return torch.tensor(selected, device=device, dtype=torch.float32).reshape(
            count, 10
        )

    def set_cycle_weights(self, values: Sequence[float]) -> None:
        """Queue one hard-label AP weight per checkerboard group."""

        if self._cycle_weights is not None:
            raise ValueError("Previous cycle weights were not fully consumed")
        weights = [float(value) for value in values]
        if not weights or any(
            not math.isfinite(value) or value <= 0 for value in weights
        ):
            raise ValueError("Cycle weights must be finite and strictly positive")
        self._cycle_weights = weights

    def assert_cycle_weights_consumed(self) -> None:
        if self._cycle_weights:
            raise ValueError(
                f"{len(self._cycle_weights)} cycle weights were not consumed"
            )
        self._cycle_weights = None

    def _take_cycle_weights(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if self.rank_mode != "deployment_weighted_robust_cycle":
            return None
        if self._cycle_weights is None or len(self._cycle_weights) < count:
            available = 0 if self._cycle_weights is None else len(self._cycle_weights)
            raise ValueError(
                f"Need {count} cycle weights for deployment-weighted robust "
                f"checkerboards, found {available}"
            )
        selected = self._cycle_weights[:count]
        del self._cycle_weights[:count]
        return torch.tensor(selected, device=device, dtype=torch.float32)

    def set_teacher_probabilities(self, values: Sequence[float]) -> None:
        if self._teacher_probabilities:
            raise ValueError("Previous teacher probabilities were not fully consumed")
        probabilities = [float(value) for value in values]
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Teacher probabilities must lie in [0, 1]")
        self._teacher_probabilities = probabilities

    def assert_teacher_probabilities_consumed(self) -> None:
        if self._teacher_probabilities:
            raise ValueError(
                f"{len(self._teacher_probabilities)} teacher probabilities were not consumed"
            )
        self._teacher_probabilities = None

    def set_total_relevant(self, values: Sequence[int | float]) -> None:
        """Queue one full-gallery GT count per contiguous candidate group."""

        if self._total_relevant is not None:
            raise ValueError("Previous total-relevant values were not fully consumed")
        totals = [float(value) for value in values]
        if not totals or any(not math.isfinite(value) or value <= 0 for value in totals):
            raise ValueError("Full-gallery total-relevant values must be positive")
        self._total_relevant = totals

    def assert_total_relevant_consumed(self) -> None:
        if self._total_relevant:
            raise ValueError(
                f"{len(self._total_relevant)} total-relevant values were not consumed"
            )
        self._total_relevant = None

    def _take_total_relevant(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if self.rank_mode != "full_gallery_lambda_ap":
            return None
        if self._total_relevant is None or len(self._total_relevant) < count:
            available = 0 if self._total_relevant is None else len(self._total_relevant)
            raise ValueError(
                f"Need {count} total-relevant values for full_gallery_lambda_ap, "
                f"found {available}"
            )
        selected = self._total_relevant[:count]
        del self._total_relevant[:count]
        return torch.tensor(selected, device=device, dtype=torch.float32)

    def _take_teacher_targets(
        self, count: int, *, device: torch.device
    ) -> torch.Tensor | None:
        if not self.teacher_weight and self.rank_mode not in {
            "parent_top1_corrective",
            "parent_top1_competitor",
            "dual_top1_competitor",
            "constrained_dual_top1_competitor",
            "dual_multi_positive_competitor",
        }:
            return None
        if self._teacher_probabilities is None or len(self._teacher_probabilities) < count:
            available = (
                0
                if self._teacher_probabilities is None
                else len(self._teacher_probabilities)
            )
            raise ValueError(
                f"Need {count} teacher probabilities for this batch, found {available}"
            )
        selected = self._teacher_probabilities[:count]
        del self._teacher_probabilities[:count]
        return torch.tensor(selected, device=device, dtype=torch.float32)

    def reset_metrics(self) -> None:
        self.last_metrics = {}
        self._metric_sums = {}
        self._metric_counts = {}
        self._metric_groups = 0

    def mean_metrics(self) -> dict[str, torch.Tensor]:
        if not self._metric_groups:
            return {}
        return {
            name: value / self._metric_counts[name]
            for name, value in self._metric_sums.items()
        }

    def metric_totals(self) -> dict[str, tuple[torch.Tensor, int]]:
        if not self._metric_groups:
            return {}
        return {
            name: (self._metric_sums[name], self._metric_counts[name])
            for name in self._metric_sums
        }

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.LongTensor,
        *,
        ignore_index: int = -100,
        loss_scaling_factor: float = 1.0,
        output_packing_mask: torch.Tensor | None = None,
        target_packing_mask: torch.Tensor | None = None,
        lin_weight: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if output_packing_mask is not None or target_packing_mask is not None:
            raise ValueError("PAS rank/point loss does not support sequence packing")
        if lin_weight is not None:
            self.projection_weight = lin_weight
        projection_weight = (
            lin_weight if lin_weight is not None else self.projection_weight
        )
        ordinal_ce = None
        if self.ordinal_response_spec is None:
            scores, labels = extract_binary_scores(
                output,
                target,
                self.response_spec,
                ignore_index=ignore_index,
                projection_weight=projection_weight,
                projection_module=self.projection_module,
                ordinal_token_ids=self.ordinal_token_ids,
                decision_position=self.binary_decision_position,
                binary_readout=self.binary_readout,
            )
            if self.ordinal_aux_response_spec is not None:
                ordinal_ce = extract_ordinal_aux_loss(
                    output,
                    target,
                    self.ordinal_aux_response_spec,
                    ignore_index=ignore_index,
                    projection_weight=projection_weight,
                    binary_response_spec=(
                        self.response_spec
                        if self.ordinal_aux_position == "predecision"
                        else None
                    ),
                    binary_decision_position=self.binary_decision_position,
                )
        else:
            scores, labels, ordinal_ce = extract_ordinal_scores(
                output,
                target,
                self.ordinal_response_spec,
                ignore_index=ignore_index,
                projection_weight=projection_weight,
                score_readout=self.ordinal_score_readout,
            )
        rank_point = self.from_scores(
            scores,
            labels,
            loss_scaling_factor=loss_scaling_factor,
        )
        if self.hidden_supcon_weight and not self.canonical_validation:
            local_projection = projection_weight
            if local_projection is not None and hasattr(local_projection, "to_local"):
                local_projection = local_projection.to_local()
            if local_projection is None:
                raise ValueError(
                    "hidden_supcon_weight requires fused hidden-state output and "
                    "an LM-head projection weight"
                )
            if output.shape[-1] != local_projection.shape[1]:
                raise ValueError(
                    "hidden_supcon_weight requires fused hidden states, not "
                    f"full-vocabulary logits: output={output.shape}, "
                    f"projection={local_projection.shape}"
                )
            decision_states, hidden_labels = extract_binary_decision_states(
                output,
                target,
                self.response_spec,
                ignore_index=ignore_index,
                decision_position=self.binary_decision_position,
            )
            if not torch.equal(hidden_labels, labels):
                raise RuntimeError("Hidden-state and scalar binary labels disagree")
            hidden_supcon = within_query_supervised_contrastive_loss(
                decision_states,
                hidden_labels,
                group_size=self.group_size,
                temperature=self.hidden_supcon_temperature,
                center=self.hidden_supcon_center,
            )
            rank_point = rank_point + (
                self.hidden_supcon_weight
                * hidden_supcon
                * float(loss_scaling_factor)
            )
            # This optional metric is intentionally absent when the auxiliary
            # is disabled so the default metric contract remains unchanged.
            num_groups = labels.numel() // self.group_size
            detached = hidden_supcon.detach()
            self.last_metrics["hidden_supcon"] = detached
            weighted = detached * num_groups
            self._metric_sums["hidden_supcon"] = (
                weighted
                if "hidden_supcon" not in self._metric_sums
                else self._metric_sums["hidden_supcon"] + weighted
            )
            self._metric_counts["hidden_supcon"] = (
                self._metric_counts.get("hidden_supcon", 0) + num_groups
            )
        if (
            ordinal_ce is not None
            and self.ordinal_ce_weight
            and not self.canonical_validation
        ):
            rank_point = rank_point + (
                self.ordinal_ce_weight * ordinal_ce * float(loss_scaling_factor)
            )
        return rank_point

    def from_scores(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        *,
        loss_scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        """Apply the configured objective to an externally defined score.

        The ordinary binary reranker obtains ``scores`` from its first yes/no
        token.  Structured rerankers such as HCR instead aggregate several
        label-independent compatibility probes into the *deployed* score.
        Keeping the ranking/metric path here ensures training, validation and
        checkpoint selection all evaluate exactly the same candidate score.
        """

        scores = scores.float().reshape(-1)
        labels = labels.float().reshape(-1).to(device=scores.device)
        if scores.shape != labels.shape:
            raise ValueError(
                f"External PAS score/label shape mismatch: {scores.shape=} "
                f"{labels.shape=}"
            )
        retriever_scores = self._take_retriever_residual_scores(
            scores.numel(), device=scores.device
        )
        residual_alpha: torch.Tensor | None = None
        if retriever_scores is not None:
            alpha: float | torch.Tensor = float(self.retriever_residual_alpha)
            if self.retriever_residual_gate is not None:
                residual_alpha = self.retriever_residual_gate(
                    scores,
                    retriever_scores,
                    group_size=self.group_size,
                    epsilon=self.retriever_residual_epsilon,
                )
                alpha = residual_alpha
            scores = retriever_residual_fused_scores(
                scores,
                retriever_scores,
                group_size=self.group_size,
                alpha=alpha,
                epsilon=self.retriever_residual_epsilon,
            )
        self.last_scores = scores
        self.last_labels = labels
        candidate_weights = self._take_candidate_loss_weights(
            scores.numel(), device=scores.device
        )
        total_relevant = self._take_total_relevant(
            scores.numel() // self.group_size,
            device=scores.device,
        )
        cycle_weights = self._take_cycle_weights(
            scores.numel() // self.group_size,
            device=scores.device,
        )
        cycle_block_metadata = self._take_cycle_block_metadata(
            scores.numel() // self.group_size,
            device=scores.device,
        )
        policy_preserve = self._take_policy_preserve_flags(
            scores.numel() // self.group_size,
            device=scores.device,
        )
        inverse_pair_offsets = self._take_inverse_pair_offsets(
            scores.numel() // self.group_size,
            device=scores.device,
        )
        same_label_consistency_offsets = self._take_same_label_consistency_offsets(
            scores.numel() // self.group_size,
            device=scores.device,
        )
        requirement_records = self._take_requirement_ordinal_records(
            scores.numel(), device=scores.device
        )
        if self.canonical_validation:
            teacher_targets = self._take_teacher_targets(
                scores.numel(), device=scores.device
            )
            # Counterfactual training groups are adjacent cross-query pairs,
            # while checkpoint selection uses ordinary same-query K20 lists.
            # Evaluate those lists with the canonical metric surrogate rather
            # than pretending their arbitrary row adjacency is supervision.
            validation_rank_mode = (
                "smoothap_soft_r1"
                if self.rank_mode
                in {
                    "paired_adjacent",
                    "counterfactual_set5",
                    "plackett_luce_policy",
                    "rb_plackett_luce_expected_ap",
                }
                else self.rank_mode
            )
            loss, metrics = rank_point_loss_from_scores(
                scores,
                labels,
                retriever_reference_scores=retriever_scores,
                rank_weight=1.0,
                point_weight=0.0,
                rank_temperature=self.rank_temperature,
                bag_positive_temperature=self.bag_positive_temperature,
                bag_negative_temperature=self.bag_negative_temperature,
                pu_class_prior=self.pu_class_prior,
                pu_nn_weight=self.pu_nn_weight,
                rank_mode=validation_rank_mode,
                partial_negative_ratio=self.partial_negative_ratio,
                rank_margin=self.rank_margin,
                natural20_rank_weight=self.natural20_rank_weight,
                inverse_pair_weight=self.inverse_pair_weight,
                inverse_pair_offsets=inverse_pair_offsets,
                weak_veto_temperature=self.weak_veto_temperature,
                weak_veto_full_weight=self.weak_veto_full_weight,
                weak_veto_mil_weight=self.weak_veto_mil_weight,
                robust_pair_q=self.robust_pair_q,
                all_pairs_aux_weight=self.all_pairs_aux_weight,
                positive_tail_weight=self.positive_tail_weight,
                positive_tail_temperature=self.positive_tail_temperature,
                retriever_success_weight=self.retriever_success_weight,
                retriever_error_weight=self.retriever_error_weight,
                near_miss_k=self.near_miss_k,
                near_miss_weight=self.near_miss_weight,
                parent_preservation_margin=self.parent_preservation_margin,
                parent_preservation_teacher_margin_slack=self.parent_preservation_teacher_margin_slack,
                parent_preservation_weight=self.parent_preservation_weight,
                topk_preservation_k=self.topk_preservation_k,
                topk_preservation_margin=self.topk_preservation_margin,
                topk_preservation_weight=self.topk_preservation_weight,
                teacher_targets=teacher_targets,
                group_size=self.group_size,
                query_consistency_weight=self.query_consistency_weight,
                query_cross_weight=self.query_cross_weight,
                total_relevant=total_relevant,
                full_gallery_rank1_weight=self.full_gallery_rank1_weight,
                full_gallery_active_topk=self.full_gallery_active_topk,
                transition_pooled_weight=self.transition_pooled_weight,
                transition_pooled_topk=self.transition_pooled_topk,
                transition_pooled_temperature=self.transition_pooled_temperature,
                transition_pooled_margin=self.transition_pooled_margin,
                cycle_weights=cycle_weights,
                cycle_block_metadata=cycle_block_metadata,
                robust_cycle_rho=self.robust_cycle_rho,
                robust_cycle_lambda=self.robust_cycle_lambda,
                preservation_cycle_margin=self.preservation_cycle_margin,
                preservation_cycle_weight=self.preservation_cycle_weight,
                repair_cycle_ap_weight=self.repair_cycle_ap_weight,
                preserve_cycle_ap_weight=self.preserve_cycle_ap_weight,
                preserve_cycle_robust_scale=self.preserve_cycle_robust_scale,
                r69_smoothap_temperature=self.r69_smoothap_temperature,
                r69_soft_r1_temperature=self.r69_soft_r1_temperature,
                r69_smoothap_weight=self.r69_smoothap_weight,
                r69_soft_r1_weight=self.r69_soft_r1_weight,
                policy_samples=self.policy_samples,
                policy_exact_max_k=self.policy_exact_max_k,
                policy_ap_reward_weight=self.policy_ap_reward_weight,
                policy_r1_reward_weight=self.policy_r1_reward_weight,
                policy_preserve_anchor_weight=self.policy_preserve_anchor_weight,
                policy_preserve_anchor_margin=self.policy_preserve_anchor_margin,
            )
            # Canonical checkpoint selection intentionally remains rank-only,
            # but pointwise calibration is still useful validation telemetry.
            # Compute it separately so ``val/point_loss`` reports the actual
            # held-out BCE instead of the zero placeholder produced by the
            # disabled point objective above.  This does not contribute to
            # the returned validation loss or affect checkpoint selection.
            if self.rank_mode in {
                "orthogonal_cycle",
                "deployment_weighted_robust_cycle",
                "preservation_constrained_cycle_block",
            }:
                # Only the original deployment edge has ordinary point labels.
                # Slot 3 exists to complete the interaction square and must not
                # leak into canonical BCE telemetry.
                grouped = scores.reshape(-1, 4)
                anchor_scores = grouped[:, :2]
                anchor_targets = anchor_scores.new_tensor([1.0, 0.0]).expand_as(
                    anchor_scores
                )
                metrics["point"] = F.binary_cross_entropy_with_logits(
                    anchor_scores / self.point_temperature,
                    anchor_targets,
                ).detach()
            elif self.rank_mode == "weak_veto_mil":
                full_scores = scores.reshape(-1, self.group_size)[:, :2]
                full_targets = full_scores.new_tensor([1.0, 0.0]).expand_as(
                    full_scores
                )
                metrics["point"] = F.binary_cross_entropy_with_logits(
                    full_scores / self.point_temperature,
                    full_targets,
                ).detach()
            else:
                _, point_metrics = rank_point_loss_from_scores(
                    scores,
                    labels,
                    rank_weight=0.0,
                    point_weight=1.0,
                    rank_mode=(
                        "query_selective_regret"
                        if self.rank_mode == "query_selective_regret"
                        else "probability_mass"
                    ),
                    point_temperature=self.point_temperature,
                    point_mode=self.point_mode,
                    negative_point_weight=self.negative_point_weight,
                    group_size=self.group_size,
                    query_consistency_weight=self.query_consistency_weight,
                    query_cross_weight=self.query_cross_weight,
                )
                metrics["point"] = point_metrics["point"]
            # Keep the canonical checkpoint-selection objective rank-only, but
            # still measure deployment-parent retention on the same held-out
            # examples.  Previously these fields were silently reported as
            # zero because validation never consumed teacher targets.
            if teacher_targets is not None:
                _, teacher_metrics = rank_point_loss_from_scores(
                    scores,
                    labels,
                    rank_weight=0.0,
                    point_weight=0.0,
                    teacher_targets=teacher_targets,
                    teacher_weight=1.0,
                    teacher_repair_group_weight=self.teacher_repair_group_weight,
                    teacher_temperature=self.teacher_temperature,
                    teacher_mode=self.teacher_mode,
                    group_size=self.group_size,
                )
                for name in (
                    "teacher",
                    "teacher_target_mean",
                    "teacher_probability_mae",
                ):
                    metrics[name] = teacher_metrics[name]
        else:
            teacher_targets = self._take_teacher_targets(
                scores.numel(), device=scores.device
            )
            loss, metrics = rank_point_loss_from_scores(
                scores,
                labels,
                candidate_weights=candidate_weights,
                retriever_reference_scores=retriever_scores,
                rank_weight=self.rank_weight,
                point_weight=self.point_weight,
                rank_temperature=self.rank_temperature,
                bag_positive_temperature=self.bag_positive_temperature,
                bag_negative_temperature=self.bag_negative_temperature,
                pu_class_prior=self.pu_class_prior,
                pu_nn_weight=self.pu_nn_weight,
                rank_mode=self.rank_mode,
                partial_negative_ratio=self.partial_negative_ratio,
                rank_margin=self.rank_margin,
                natural20_rank_weight=self.natural20_rank_weight,
                inverse_pair_weight=self.inverse_pair_weight,
                inverse_pair_offsets=inverse_pair_offsets,
                weak_veto_temperature=self.weak_veto_temperature,
                weak_veto_full_weight=self.weak_veto_full_weight,
                weak_veto_mil_weight=self.weak_veto_mil_weight,
                robust_pair_q=self.robust_pair_q,
                all_pairs_aux_weight=self.all_pairs_aux_weight,
                positive_tail_weight=self.positive_tail_weight,
                positive_tail_temperature=self.positive_tail_temperature,
                retriever_success_weight=self.retriever_success_weight,
                retriever_error_weight=self.retriever_error_weight,
                near_miss_k=self.near_miss_k,
                near_miss_weight=self.near_miss_weight,
                parent_preservation_margin=self.parent_preservation_margin,
                parent_preservation_teacher_margin_slack=self.parent_preservation_teacher_margin_slack,
                parent_preservation_weight=self.parent_preservation_weight,
                topk_preservation_k=self.topk_preservation_k,
                topk_preservation_margin=self.topk_preservation_margin,
                topk_preservation_weight=self.topk_preservation_weight,
                point_temperature=self.point_temperature,
                point_mode=self.point_mode,
                negative_point_weight=self.negative_point_weight,
                teacher_targets=teacher_targets,
                teacher_weight=self.teacher_weight,
                teacher_repair_group_weight=self.teacher_repair_group_weight,
                teacher_temperature=self.teacher_temperature,
                teacher_mode=self.teacher_mode,
                group_size=self.group_size,
                query_consistency_weight=self.query_consistency_weight,
                query_cross_weight=self.query_cross_weight,
                total_relevant=total_relevant,
                full_gallery_rank1_weight=self.full_gallery_rank1_weight,
                full_gallery_active_topk=self.full_gallery_active_topk,
                transition_pooled_weight=self.transition_pooled_weight,
                transition_pooled_topk=self.transition_pooled_topk,
                transition_pooled_temperature=self.transition_pooled_temperature,
                transition_pooled_margin=self.transition_pooled_margin,
                cycle_weights=cycle_weights,
                cycle_block_metadata=cycle_block_metadata,
                robust_cycle_rho=self.robust_cycle_rho,
                robust_cycle_lambda=self.robust_cycle_lambda,
                preservation_cycle_margin=self.preservation_cycle_margin,
                preservation_cycle_weight=self.preservation_cycle_weight,
                repair_cycle_ap_weight=self.repair_cycle_ap_weight,
                preserve_cycle_ap_weight=self.preserve_cycle_ap_weight,
                preserve_cycle_robust_scale=self.preserve_cycle_robust_scale,
                r69_smoothap_temperature=self.r69_smoothap_temperature,
                r69_soft_r1_temperature=self.r69_soft_r1_temperature,
                r69_smoothap_weight=self.r69_smoothap_weight,
                r69_soft_r1_weight=self.r69_soft_r1_weight,
                policy_samples=self.policy_samples,
                policy_exact_max_k=self.policy_exact_max_k,
                policy_ap_reward_weight=self.policy_ap_reward_weight,
                policy_r1_reward_weight=self.policy_r1_reward_weight,
                policy_preserve_anchor_weight=self.policy_preserve_anchor_weight,
                policy_preserve_anchor_margin=self.policy_preserve_anchor_margin,
                policy_preserve=policy_preserve,
            )

        requirement_count = 0
        if requirement_records is not None:
            requirement_loss, requirement_metrics, requirement_count = (
                cumulative_requirement_ordinal_loss(
                    scores,
                    *requirement_records,
                    threshold_gap=self.requirement_ordinal_gap,
                    temperature=self.requirement_ordinal_temperature,
                )
            )
            loss = loss + self.requirement_ordinal_weight * requirement_loss
            metrics.update(requirement_metrics)

        consistency_count = 0
        if same_label_consistency_offsets is not None:
            grouped_scores = list(scores.split(self.group_size))
            grouped_labels = list(labels.split(self.group_size))
            pair_losses: list[torch.Tensor] = []
            pair_abs_gaps: list[torch.Tensor] = []
            for group_scores, group_labels, offset in zip(
                grouped_scores,
                grouped_labels,
                same_label_consistency_offsets.tolist(),
                strict=True,
            ):
                if offset < 0:
                    continue
                if bool(group_labels[offset] != group_labels[offset + 1]):
                    raise ValueError(
                        "Same-label consistency pair has different hard labels"
                    )
                gap = group_scores[offset] - group_scores[offset + 1]
                pair_losses.append(F.smooth_l1_loss(gap, gap.new_zeros(())))
                pair_abs_gaps.append(gap.abs())
            consistency_loss, consistency_count = self._dp_exact_selected_mean(
                pair_losses, scores
            )
            loss = loss + self.same_label_consistency_weight * consistency_loss
            # The optimized value and the interpretable score gap are both
            # available locally; publish mean absolute gap under the concise
            # telemetry name used by this experiment.
            consistency_gap, _ = self._dp_exact_selected_mean(pair_abs_gaps, scores)
            metrics["same_label_consistency"] = consistency_gap.detach()

        safe_records = self._take_safe_update_records(scores.numel())
        if safe_records is not None:
            if self._safe_update_component == "combined":
                raise RuntimeError(
                    "The dedicated safe-update trainer must select repair or "
                    "preservation before each training backward"
                )
            grouped_scores = list(scores.split(self.group_size))
            grouped_safe_labels = list(labels.split(self.group_size))
            grouped_records = [
                safe_records[begin : begin + self.group_size]
                for begin in range(0, len(safe_records), self.group_size)
            ]
            component_losses: list[torch.Tensor] = []
            for group_scores, group_labels, records in zip(
                grouped_scores, grouped_safe_labels, grouped_records, strict=True
            ):
                margins = {record[0] for record in records}
                correct = {record[2] for record in records}
                strata = {record[3] for record in records}
                anchor_indices = [
                    index for index, record in enumerate(records) if record[1]
                ]
                if len(margins) != 1 or len(correct) != 1 or len(strata) != 1:
                    raise ValueError("Safe-update metadata disagree inside a query group")
                parent_correct = next(iter(correct))
                stratum = next(iter(strata))
                positive = group_labels > 0.5
                negative = ~positive
                if self._safe_update_component == "repair":
                    if parent_correct:
                        continue
                    positive_energy = self.bag_positive_temperature * torch.logsumexp(
                        group_scores[positive] / self.bag_positive_temperature, dim=0
                    )
                    negative_energy = self.bag_negative_temperature * torch.logsumexp(
                        group_scores[negative] / self.bag_negative_temperature, dim=0
                    )
                    component_losses.append(
                        F.softplus(
                            (negative_energy - positive_energy + self.rank_margin)
                            / self.rank_temperature
                        )
                    )
                    continue
                selected_stratum = int(self._safe_update_component.rsplit("_", 1)[1])
                if not parent_correct or stratum != selected_stratum:
                    continue
                if len(anchor_indices) != 1 or not bool(positive[anchor_indices[0]]):
                    raise ValueError(
                        "A safe parent group must identify exactly one positive anchor"
                    )
                current_margin = (
                    group_scores[anchor_indices[0]] - group_scores[negative].amax()
                )
                inherited_margin = next(iter(margins))
                # The positive cushion makes the hinge active at the inherited
                # boundary, so its gradient is available to A-GEM before a
                # discrete rank-1 regression has already happened.
                component_losses.append(
                    F.relu(
                        current_margin.new_tensor(
                            inherited_margin + self._safe_update_cushion
                        )
                        - current_margin
                    )
                )
            loss, selected_count = self._dp_exact_selected_mean(
                component_losses, scores
            )
            self.safe_update_last_count = selected_count

            if self.rank_mode in {
                "asymmetric_safe_residual_lambda_ap",
                "misordered_lambda_ap",
            }:
                if (
                    self.rank_mode == "asymmetric_safe_residual_lambda_ap"
                    and retriever_scores is None
                ):
                    raise ValueError(
                        "cell-safe residual projection requires frozen retriever scores"
                    )
                grouped_retriever = (
                    list(retriever_scores.split(self.group_size))
                    if retriever_scores is not None
                    else [None] * len(grouped_scores)
                )
                cell_losses: list[torch.Tensor] = []
                repair_losses_by_cell: list[list[torch.Tensor]] = [
                    [] for _ in range(self._safe_update_num_strata)
                ]
                for group_scores, group_labels, group_retriever, records in zip(
                    grouped_scores,
                    grouped_safe_labels,
                    grouped_retriever,
                    grouped_records,
                    strict=True,
                ):
                    strata = {record[3] for record in records}
                    if len(strata) != 1:
                        raise ValueError(
                            "Cell-safe residual metadata disagree inside a query group"
                        )
                    stratum = next(iter(strata))
                    positive, negative, swap_weight = exact_ap_swap_weights(
                        group_scores,
                        group_labels,
                        group_labels.sum(),
                    )
                    fused_margin = (
                        group_scores[positive].unsqueeze(1)
                        - group_scores[negative].unsqueeze(0)
                    )
                    with torch.no_grad():
                        if group_retriever is None:
                            # Pure-CR3 branch: the dynamic pair role is based
                            # only on the current hard-label ordering. No
                            # retriever score enters either loss or selection.
                            partition_margin = fused_margin.detach()
                        else:
                            retriever_centered = group_retriever - group_retriever.mean()
                            retriever_z = retriever_centered / (
                                group_retriever.std(unbiased=False)
                                + self.retriever_residual_epsilon
                            )
                            partition_margin = (
                                retriever_z[positive].unsqueeze(1)
                                - retriever_z[negative].unsqueeze(0)
                            )
                    if self._safe_update_component == "repair":
                        mask = partition_margin <= self.rank_margin
                        weights = swap_weight * mask.to(swap_weight.dtype)
                        if bool(weights.sum() > 0):
                            pair_loss = self.rank_temperature * F.softplus(
                                (self.rank_margin - fused_margin)
                                / self.rank_temperature
                            )
                            group_repair_loss = (
                                (weights * pair_loss).sum()
                                / weights.sum().clamp_min(1e-12)
                            )
                        else:
                            group_repair_loss = group_scores.sum() * 0.0
                        # Normalize first within a query, then within a cell,
                        # then macro-average the nine cells. This prevents
                        # PA-hard volume, 3-vs-4 sampling, or positive-pair
                        # multiplicity from dominating the primary direction.
                        repair_losses_by_cell[stratum].append(group_repair_loss)
                        continue
                    selected_stratum = int(
                        self._safe_update_component.rsplit("_", 1)[1]
                    )
                    if stratum != selected_stratum:
                        continue
                    mask = partition_margin > self.rank_margin
                    weights = swap_weight * mask.to(swap_weight.dtype)
                    if bool(weights.sum() > 0):
                        # This is a directional constraint, not an auxiliary
                        # objective: its gradient points toward increasing the
                        # hard-label fused margin, and the trainer only uses it
                        # to forbid a primary update with negative dot product.
                        cell_losses.append(
                            -(weights * fused_margin).sum()
                            / weights.sum().clamp_min(1e-12)
                        )
                if self._safe_update_component == "repair":
                    macro_losses = []
                    selected_count = 0
                    for stratum_losses in repair_losses_by_cell:
                        stratum_loss, stratum_count = self._dp_exact_selected_mean(
                            stratum_losses, scores
                        )
                        if not stratum_count:
                            raise ValueError(
                                "Cell-macro repair requires every cell in every step"
                            )
                        macro_losses.append(stratum_loss)
                        selected_count += stratum_count
                    loss = torch.stack(macro_losses).mean()
                else:
                    loss, selected_count = self._dp_exact_selected_mean(
                        cell_losses, scores
                    )
                self.safe_update_last_count = selected_count

        gate_regularization = scores.new_zeros(())
        if residual_alpha is not None and self.retriever_residual_gate_log_alpha_l2:
            assert self.retriever_residual_gate is not None
            gate_regularization = (
                torch.log(residual_alpha.clamp_min(1e-6))
                - math.log(self.retriever_residual_gate.initial_alpha)
            ).square().mean()
            loss = loss + self.retriever_residual_gate_log_alpha_l2 * gate_regularization

        grouped_labels = list(labels.split(self.group_size))
        num_groups = len(grouped_labels)
        if self.rank_mode == "natural20_inverse_pair":
            metric_grouped_labels = [group[:20] for group in grouped_labels]
        elif self.rank_mode == "weak_veto_mil":
            metric_grouped_labels = [group[:2] for group in grouped_labels]
        else:
            metric_grouped_labels = grouped_labels
        metric_candidate_count = sum(group.numel() for group in metric_grouped_labels)
        if self.rank_mode == "query_selective_regret":
            # A query is repairable when any incumbent-first row says that its
            # challenger should replace it. The first challenger alone is not
            # sufficient to classify the query.
            retriever_correct_count = sum(
                int(bool((group[0::2] > 0.5).all())) for group in grouped_labels
            )
        elif self.rank_mode == "preservation_constrained_cycle_block":
            assert cycle_block_metadata is not None
            retriever_correct_count = int(
                (cycle_block_metadata[:, 0].round().long() != 0).sum().item()
            )
        else:
            retriever_correct_count = sum(
                int(group[0] > 0.5) for group in metric_grouped_labels
            )
        retriever_error_count = num_groups - retriever_correct_count
        positive_count = sum(
            int((group > 0.5).sum()) for group in metric_grouped_labels
        )
        negative_count = metric_candidate_count - positive_count
        pair_count = sum(
            int((group > 0.5).sum()) * int((group <= 0.5).sum())
            for group in metric_grouped_labels
        )
        metric_counts = {
            "rank": num_groups,
            "point": (
                positive_count
                if self.point_mode == "positive_only"
                else num_groups
                if self.point_mode == "class_balanced"
                else metric_candidate_count
            ),
            "teacher": pair_count if self.teacher_mode != "pointwise" else metric_candidate_count,
            "teacher_target_mean": pair_count if self.teacher_mode != "pointwise" else metric_candidate_count,
            "teacher_probability_mae": pair_count if self.teacher_mode != "pointwise" else metric_candidate_count,
            "pairwise_accuracy": pair_count,
            "positive_score_mean": positive_count,
            "negative_score_mean": negative_count,
            "best_positive_score_mean": num_groups,
            "hardest_negative_score_mean": num_groups,
            "top1_margin": num_groups,
            "bag_positive_energy": num_groups,
            "bag_negative_energy": num_groups,
            "bag_energy_margin": num_groups,
            "bag_energy_accuracy": num_groups,
            "top1_accuracy": num_groups,
            "retriever_top1_accuracy": num_groups,
            "retriever_success_preservation_accuracy": retriever_correct_count,
            "retriever_error_repair_accuracy": retriever_error_count,
            "rank1_fix_rate": num_groups,
            "rank1_break_rate": num_groups,
            "rank1_net_gain": num_groups,
            "binary_accuracy": metric_candidate_count,
            "positive_accuracy": positive_count,
            "negative_accuracy": negative_count,
            "balanced_binary_accuracy": num_groups,
            "hard_negative_accuracy": num_groups,
            "average_precision": num_groups,
            "rank5_accuracy": num_groups,
            "margin": pair_count,
            "transition_pooled_loss": num_groups,
            "transition_pooled_accuracy": num_groups,
            "inverse_pair_loss": num_groups,
            "inverse_pair_accuracy": num_groups,
            "inverse_pair_margin": num_groups,
            "weak_veto_full_loss": num_groups,
            "weak_veto_mil_loss": num_groups,
            "weak_veto_full_margin": num_groups,
            "weak_veto_smooth_witness": num_groups,
            "weak_veto_any_field_accuracy": num_groups,
            "oic_q_edge_accuracy": num_groups,
            "oic_qn_edge_accuracy": num_groups,
            "oic_both_edges_accuracy": num_groups,
            "oic_cycle_gap": num_groups,
            "oic_deployment_edge_loss": num_groups,
            "oic_robust_edge_loss": num_groups,
            "oic_cycle_weight_mean": num_groups,
            "oic_q_gradient_mass": num_groups,
            "oic_qn_gradient_mass": num_groups,
            "r68_q_edge_accuracy": 8 * num_groups,
            "r68_qn_edge_accuracy": 8 * num_groups,
            "r68_both_edges_accuracy": 8 * num_groups,
            "r68_preservation_hinge": max(
                1, retriever_correct_count
            ),
            "r68_preservation_pass_rate": max(
                1, retriever_correct_count
            ),
            "r68_repair_block_fraction": num_groups,
            "r68_vulnerable_block_fraction": num_groups,
            "r68_diversity_block_fraction": num_groups,
            "r69_smooth_ap": num_groups,
            "r69_smooth_ap_loss": num_groups,
            "r69_soft_r1_loss": num_groups,
            "incumbent_guard_loss": num_groups,
            "incumbent_guard_pass_rate": num_groups,
            "safe_lambda_active_pair_fraction": num_groups,
            "misordered_lambda_ap_weight_mass": num_groups,
            "safe_residual_repair_pair_fraction": num_groups,
            "safe_residual_preserve_active_fraction": num_groups,
            "safe_residual_repair_loss": num_groups,
            "safe_residual_preservation_loss": num_groups,
            "robust_lambdaap": num_groups,
            "robust_soft_r1": num_groups,
            "robust_pair_weight_mass": num_groups,
            "robust_positive_trim_fraction": num_groups,
            "robust_negative_trim_fraction": num_groups,
            "paired_view_consistency": num_groups,
            "paired_view_effective_weight": num_groups,
            "paired_view_contribution_fraction": num_groups,
            "paired_view_centered_abs_gap": num_groups,
            "r535_natural_loss": num_groups,
            "r535_clause_aux_loss": num_groups,
            "r535_matched_delta": num_groups,
            "r535_veto_delta": num_groups,
            "r535_delta_contrast": num_groups,
            "r535_matched_sign_accuracy": num_groups,
            "r535_veto_sign_accuracy": num_groups,
            "policy_reward": num_groups,
            "policy_expected_ap": num_groups,
            "policy_expected_r1": num_groups,
            "policy_preserve_anchor": num_groups,
            "policy_first_entropy": num_groups,
            "policy_preserve_fraction": num_groups,
            "same_label_consistency": consistency_count,
            "requirement_ordinal_loss": requirement_count,
            "requirement_threshold_accuracy": requirement_count,
            "requirement_count_mae": requirement_count,
        }
        if self.rank_mode == "full_gallery_lambda_ap":
            metric_counts.update(
                {
                    "full_gallery_ap_delta": num_groups,
                    "full_gallery_oracle_ap_gain": num_groups,
                    "full_gallery_oracle_regret": num_groups,
                    "full_gallery_head_recall": num_groups,
                    "lambda_ap_weight_mass": num_groups,
                    "lambda_ap_active_pair_fraction": num_groups,
                }
            )
        if residual_alpha is not None:
            metrics["retriever_residual_alpha_mean"] = residual_alpha.detach().mean()
            metrics["retriever_residual_gate_regularization"] = (
                gate_regularization.detach()
            )
            metric_counts["retriever_residual_alpha_mean"] = num_groups
            metric_counts["retriever_residual_gate_regularization"] = num_groups
        self.last_metrics = metrics
        self._metric_groups += num_groups
        for name in (
            *self.METRIC_NAMES,
            *self.OPTIONAL_METRIC_NAMES,
            *self.CYCLE_METRIC_NAMES,
            *self.FULL_GALLERY_METRIC_NAMES,
            *self.R69_METRIC_NAMES,
        ):
            if name not in metrics:
                continue
            count = metric_counts[name]
            weighted = metrics[name] * count
            self._metric_sums[name] = (
                weighted
                if name not in self._metric_sums
                else self._metric_sums[name] + weighted
            )
            self._metric_counts[name] = self._metric_counts.get(name, 0) + count
        return loss * float(loss_scaling_factor)
