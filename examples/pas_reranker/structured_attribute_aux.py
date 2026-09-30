"""Masked structured-attribute auxiliary objective for PAS reranking.

The ordinary CR3 decision remains the first assistant token span.  A fixed,
teacher-forced attribute-choice template follows it, and this module applies
cross entropy only at the isolated option-letter positions.  Template text,
field names, option lists, punctuation, and missing labels never enter the
loss.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F


# The order is part of the HCR data/model interface.  It deliberately follows
# the PAS ``*_attr_values`` column order and adds the set-valued accessory
# constraint last.  Do not derive this order from labels or from a candidate.
PAS_COMPATIBILITY_PROBE_FIELDS = (
    "top_outer_color",
    "top_outer_type",
    "bottom_color",
    "bottom_type",
    "shoe_color",
    "shoe_type",
    "viewpoint",
    "accessory",
)

_COMPATIBILITY_PROBE_PLACEHOLDER = "?"
_COMPATIBILITY_PROBE_MARKERS = tuple(
    f"[{field} compatibility: yes or no]\n"
    for field in PAS_COMPATIBILITY_PROBE_FIELDS
)
PAS_COMPATIBILITY_PROBE_RESPONSE = "\n".join(
    [
        "[constraint compatibility probes]",
        *(
            f"{marker}{_COMPATIBILITY_PROBE_PLACEHOLDER}"
            for marker in _COMPATIBILITY_PROBE_MARKERS
        ),
    ]
)

_ACTIVE_COMPATIBILITY_REQUIREMENT_MARKERS = tuple(
    f"[{field} required: yes or no]\n"
    for field in PAS_COMPATIBILITY_PROBE_FIELDS
)
_ACTIVE_COMPATIBILITY_SATISFACTION_MARKERS = tuple(
    f"[{field} satisfied: yes or no]\n"
    for field in PAS_COMPATIBILITY_PROBE_FIELDS
)
PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE = "\n".join(
    [
        "<answer>?</answer>",
        "[active-aware constraint compatibility probes]",
        *(
            "\n".join(
                [
                    f"{requirement_marker}{_COMPATIBILITY_PROBE_PLACEHOLDER}",
                    f"{satisfaction_marker}{_COMPATIBILITY_PROBE_PLACEHOLDER}",
                ]
            )
            for requirement_marker, satisfaction_marker in zip(
                _ACTIVE_COMPATIBILITY_REQUIREMENT_MARKERS,
                _ACTIVE_COMPATIBILITY_SATISFACTION_MARKERS,
                strict=True,
            )
        ),
    ]
)

# A compact, mutually-exclusive alternative to the two Bernoulli probes above.
# The response is deliberately label-independent: ``?`` is teacher-forced at
# every decision position, while supervision is carried only in side metadata.
# Field names retain enough semantics for the pretrained language model and the
# whole response is 40 tokens with the Cosmos3-Nano-VLM tokenizer (vs. 177 for
# ``PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE``).
PAS_TRISTATE_CLASS_NAMES = ("inactive", "active_satisfied", "active_violated")
PAS_TRISTATE_CLASS_TOKENS = ("A", "B", "C")
# These are single tokens in the Cosmos3 tokenizer.  They preserve the same
# three hard classes, but reuse pretrained semantic directions instead of
# forcing LoRA to invent an arbitrary A/B/C classifier from a small number of
# updates.  The strings are logits-only labels: the teacher-forced trajectory
# still contains ``?`` at every slot.
PAS_TRISTATE_SEMANTIC_CLASS_TOKENS = (
    " inactive",
    " satisfied",
    " violated",
)
_TRISTATE_COMPATIBILITY_MARKERS = (
    "top_color:",
    "top_type:",
    "bottom_color:",
    "bottom_type:",
    "shoe_color:",
    "shoe_type:",
    "viewpoint:",
    "accessory:",
)
PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE = "\n".join(
    [
        "<answer>?</answer>",
        *(
            f"{marker} {_COMPATIBILITY_PROBE_PLACEHOLDER}"
            for marker in _TRISTATE_COMPATIBILITY_MARKERS
        ),
    ]
)


def postdecision_tristate_probe_response(
    binary_label: int, targets: Sequence[int]
) -> str:
    """Return an answer-first semantic audit tail for train-only supervision.

    The deployed yes/no token is emitted before every field target.  Unlike
    the pre-decision trajectory, ground-truth semantic class tokens are safe
    here because causal attention prevents them from changing the earlier
    yes/no logits.
    """

    if binary_label not in {0, 1}:
        raise ValueError("binary_label must be zero or one")
    if len(targets) != len(PAS_COMPATIBILITY_PROBE_FIELDS) or any(
        int(target) not in range(3) for target in targets
    ):
        raise ValueError("post-decision response requires eight tri-state targets")
    answer = "yes" if binary_label else "no"
    return "\n".join(
        [
            f"<answer>{answer}</answer>",
            "[post-decision attribute audit]",
            *(
                f"{marker}{PAS_TRISTATE_SEMANTIC_CLASS_TOKENS[int(target)]}"
                for marker, target in zip(
                    _TRISTATE_COMPATIBILITY_MARKERS,
                    targets,
                    strict=True,
                )
            ),
        ]
    )

# Unlike the historical HCR templates above, this trajectory puts the neutral
# field probes *before* the deployed binary decision.  Every candidate still
# receives byte-identical teacher-forced text (only ``?`` placeholders), so the
# final answer cannot see a ground-truth A/B/C token.  It can, however, attend
# the earlier hidden states whose LM-head projections are trained to predict
# inactive / satisfied / violated.  This makes the checklist a causal latent
# scratchpad rather than an auxiliary task that occurs after the decision.
PAS_PREDECISION_TRISTATE_PROBE_RESPONSE = "\n".join(
    [
        "[latent requirement checklist]",
        "[state classes: inactive | satisfied | violated]",
        *(
            f"{marker} {_COMPATIBILITY_PROBE_PLACEHOLDER}"
            for marker in _TRISTATE_COMPATIBILITY_MARKERS
        ),
        "<answer>?</answer>",
    ]
)

PAS_PREDECISION_TRISTATE_PROMPT_MODES = (
    "semantic_legend",
    "strict_and",
    "violation_veto",
    "dual_rule",
    "violation_count",
    "all_satisfied",
)


def predecision_tristate_probe_response(prompt_mode: str) -> str:
    """Return a label-independent causal checklist prompt variant."""

    fields = [
        f"{marker} {_COMPATIBILITY_PROBE_PLACEHOLDER}"
        for marker in _TRISTATE_COMPATIBILITY_MARKERS
    ]
    if prompt_mode == "semantic_legend":
        return PAS_PREDECISION_TRISTATE_PROBE_RESPONSE
    if prompt_mode == "strict_and":
        return "\n".join(
            [
                "[compare every query requirement with the image independently]",
                "[inactive = not requested | satisfied = requested and matched | violated = requested and mismatched]",
                *fields,
                "[overall rule: answer yes only when every requested requirement is satisfied]",
                "<answer>?</answer>",
            ]
        )
    if prompt_mode == "violation_veto":
        return "\n".join(
            [
                "[search for any disqualifying mismatch]",
                "[state classes: inactive | satisfied | violated]",
                *fields,
                "[decision rule: any violated state means no; otherwise answer yes]",
                "<answer>?</answer>",
            ]
        )
    if prompt_mode == "dual_rule":
        return "\n".join(
            [
                "[strict all-requirements visual check]",
                "[inactive = absent from query | satisfied = matches image | violated = conflicts with image]",
                *fields,
                "[yes iff all active fields are satisfied; no iff at least one active field is violated]",
                "<answer>?</answer>",
            ]
        )
    if prompt_mode == "violation_count":
        return "\n".join(
            [
                "[check each query requirement against the image]",
                "[state classes: inactive | satisfied | violated]",
                *fields,
                "violation_count: ?",
                "[zero violations means yes; one or more violations means no]",
                "<answer>?</answer>",
            ]
        )
    if prompt_mode == "all_satisfied":
        return "\n".join(
            [
                "[check each query requirement against the image]",
                "[inactive = not requested | satisfied = matched | violated = mismatched]",
                *fields,
                "all_active_fields_satisfied: ?",
                "<answer>?</answer>",
            ]
        )
    raise ValueError(f"Unknown pre-decision prompt mode {prompt_mode!r}")


def build_compatibility_probe_targets(
    *,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Return exact PAS compatibility targets and active-constraint masks.

    PAS scalar query values below zero mean "unspecified" and are logically
    satisfied by every candidate.  Likewise an empty requested-accessory set
    is satisfied.  We supervise those wildcard slots as positive rather than
    masking them: the deployed scorer must infer applicability from query text
    and cannot consume hidden structured metadata.  Every slot is therefore
    valid, while active scalar constraints still match only by exact ID and
    accessories use the PAS subset relation.
    """

    scalar_width = len(PAS_COMPATIBILITY_PROBE_FIELDS) - 1
    if len(text_attr_values) != scalar_width:
        raise ValueError(
            f"Expected {scalar_width} PAS text attributes, got {len(text_attr_values)}"
        )
    if len(image_attr_values) != scalar_width:
        raise ValueError(
            f"Expected {scalar_width} PAS image attributes, got {len(image_attr_values)}"
        )

    required = tuple(int(value) for value in text_attr_values)
    observed = tuple(int(value) for value in image_attr_values)
    scalar_targets = tuple(
        wanted < 0 or wanted == actual
        for wanted, actual in zip(required, observed, strict=True)
    )
    requested_accessories = frozenset(int(value) for value in text_accessory_ids)
    candidate_accessories = frozenset(int(value) for value in image_accessory_ids)
    accessory_target = requested_accessories.issubset(candidate_accessories)
    return (
        (*scalar_targets, accessory_target),
        (True,) * len(PAS_COMPATIBILITY_PROBE_FIELDS),
    )


def build_compatibility_probe_response(
    *,
    sample_id: str,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
) -> tuple[str, dict[str, Any]]:
    """Build fixed teacher-forced probe text and separately stored targets.

    The returned response contains neutral ``?`` placeholders, never target
    yes/no tokens.  Consequently one probe cannot read an earlier probe's
    teacher-forced answer, and candidates with different labels receive
    byte-identical response text.
    """

    targets, valid = build_compatibility_probe_targets(
        text_attr_values=text_attr_values,
        image_attr_values=image_attr_values,
        text_accessory_ids=text_accessory_ids,
        image_accessory_ids=image_accessory_ids,
    )
    probe_targets = [
        {
            "field": field,
            "marker": marker,
            "target": target,
            "valid": is_valid,
        }
        for field, marker, target, is_valid in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _COMPATIBILITY_PROBE_MARKERS,
            targets,
            valid,
            strict=True,
        )
    ]
    return PAS_COMPATIBILITY_PROBE_RESPONSE, {
        "sample_id": str(sample_id),
        "response": PAS_COMPATIBILITY_PROBE_RESPONSE,
        "probe_targets": probe_targets,
    }


def build_active_compatibility_probe_targets(
    *,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
) -> tuple[
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, ...],
]:
    """Return requirement and satisfaction targets with their loss masks.

    Requirement is observable for all eight fields and is true exactly when
    the query specifies that constraint.  Satisfaction is supervised only for
    required constraints; its target is exact scalar equality or accessory-set
    inclusion, matching PAS relevance semantics.
    """

    legacy_targets, _ = build_compatibility_probe_targets(
        text_attr_values=text_attr_values,
        image_attr_values=image_attr_values,
        text_accessory_ids=text_accessory_ids,
        image_accessory_ids=image_accessory_ids,
    )
    requirement_targets = (
        *(int(value) >= 0 for value in text_attr_values),
        bool(frozenset(int(value) for value in text_accessory_ids)),
    )
    requirement_valid = (True,) * len(PAS_COMPATIBILITY_PROBE_FIELDS)
    satisfaction_valid = requirement_targets
    satisfaction_targets = tuple(
        target if is_required else False
        for target, is_required in zip(
            legacy_targets, requirement_targets, strict=True
        )
    )
    return (
        requirement_targets,
        requirement_valid,
        satisfaction_targets,
        satisfaction_valid,
    )


def build_active_compatibility_probe_response(
    *,
    sample_id: str,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
) -> tuple[str, dict[str, Any]]:
    """Build the fixed two-decision-per-field active-aware probe response."""

    (
        requirement_targets,
        requirement_valid,
        satisfaction_targets,
        satisfaction_valid,
    ) = build_active_compatibility_probe_targets(
        text_attr_values=text_attr_values,
        image_attr_values=image_attr_values,
        text_accessory_ids=text_accessory_ids,
        image_accessory_ids=image_accessory_ids,
    )
    probe_targets = [
        {
            "field": field,
            "requirement_marker": requirement_marker,
            "requirement_target": requirement_target,
            "requirement_valid": requirement_is_valid,
            "satisfaction_marker": satisfaction_marker,
            "satisfaction_target": satisfaction_target,
            "satisfaction_valid": satisfaction_is_valid,
        }
        for (
            field,
            requirement_marker,
            satisfaction_marker,
            requirement_target,
            requirement_is_valid,
            satisfaction_target,
            satisfaction_is_valid,
        ) in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _ACTIVE_COMPATIBILITY_REQUIREMENT_MARKERS,
            _ACTIVE_COMPATIBILITY_SATISFACTION_MARKERS,
            requirement_targets,
            requirement_valid,
            satisfaction_targets,
            satisfaction_valid,
            strict=True,
        )
    ]
    return PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE, {
        "sample_id": str(sample_id),
        "response": PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE,
        "active_probe_targets": probe_targets,
    }


def build_active_compatibility_probe_inference_record(
    *, sample_id: str, binary_label: int
) -> tuple[str, dict[str, Any]]:
    """Build an active-aware held-out record without structured field labels."""

    if binary_label not in {0, 1}:
        raise ValueError("binary_label must be zero or one")
    probe_targets = [
        {
            "field": field,
            "requirement_marker": requirement_marker,
            "requirement_target": False,
            "requirement_valid": False,
            "satisfaction_marker": satisfaction_marker,
            "satisfaction_target": False,
            "satisfaction_valid": False,
        }
        for field, requirement_marker, satisfaction_marker in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _ACTIVE_COMPATIBILITY_REQUIREMENT_MARKERS,
            _ACTIVE_COMPATIBILITY_SATISFACTION_MARKERS,
            strict=True,
        )
    ]
    return PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE, {
        "sample_id": str(sample_id),
        "response": PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE,
        "binary_label": int(binary_label),
        "active_probe_targets": probe_targets,
    }


def build_tristate_compatibility_probe_targets(
    *,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Return mutually-exclusive inactive/satisfied/violated field targets.

    The three classes remove the identifiability and gradient conflict in two
    independently normalized requirement/satisfaction heads.  PAS relevance is
    exactly the conjunction ``class != active_violated`` across all fields.
    """

    required, _, satisfied, _ = build_active_compatibility_probe_targets(
        text_attr_values=text_attr_values,
        image_attr_values=image_attr_values,
        text_accessory_ids=text_accessory_ids,
        image_accessory_ids=image_accessory_ids,
    )
    targets = tuple(
        0 if not is_required else (1 if is_satisfied else 2)
        for is_required, is_satisfied in zip(required, satisfied, strict=True)
    )
    return targets, (True,) * len(PAS_COMPATIBILITY_PROBE_FIELDS)


def build_tristate_compatibility_probe_response(
    *,
    sample_id: str,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
) -> tuple[str, dict[str, Any]]:
    """Build the fixed one-categorical-decision-per-field HCR response."""

    targets, valid = build_tristate_compatibility_probe_targets(
        text_attr_values=text_attr_values,
        image_attr_values=image_attr_values,
        text_accessory_ids=text_accessory_ids,
        image_accessory_ids=image_accessory_ids,
    )
    probe_targets = [
        {
            "field": field,
            "marker": marker,
            "target": int(target),
            "target_name": PAS_TRISTATE_CLASS_NAMES[int(target)],
            "valid": bool(is_valid),
        }
        for field, marker, target, is_valid in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _TRISTATE_COMPATIBILITY_MARKERS,
            targets,
            valid,
            strict=True,
        )
    ]
    return PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE, {
        "sample_id": str(sample_id),
        "response": PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE,
        "tristate_probe_targets": probe_targets,
    }


def build_postdecision_tristate_probe_response(
    *,
    sample_id: str,
    binary_label: int,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
) -> tuple[str, dict[str, Any]]:
    """Build an actual yes/no answer followed by eight semantic targets."""

    _, metadata = build_tristate_compatibility_probe_response(
        sample_id=sample_id,
        text_attr_values=text_attr_values,
        image_attr_values=image_attr_values,
        text_accessory_ids=text_accessory_ids,
        image_accessory_ids=image_accessory_ids,
    )
    targets = tuple(
        int(item["target"]) for item in metadata["tristate_probe_targets"]
    )
    response = postdecision_tristate_probe_response(binary_label, targets)
    metadata["response"] = response
    metadata["binary_label"] = int(binary_label)
    return response, metadata


def build_postdecision_tristate_probe_inference_record(
    *, sample_id: str, binary_label: int
) -> tuple[str, dict[str, Any]]:
    """Build a fully masked answer-first record when field labels are absent."""

    _, metadata = build_tristate_compatibility_probe_inference_record(
        sample_id=sample_id, binary_label=binary_label
    )
    response = postdecision_tristate_probe_response(
        binary_label, (0,) * len(PAS_COMPATIBILITY_PROBE_FIELDS)
    )
    metadata["response"] = response
    return response, metadata


def build_tristate_compatibility_probe_inference_record(
    *, sample_id: str, binary_label: int
) -> tuple[str, dict[str, Any]]:
    """Build a tri-state held-out record without structured field labels."""

    if binary_label not in {0, 1}:
        raise ValueError("binary_label must be zero or one")
    probe_targets = [
        {
            "field": field,
            "marker": marker,
            "target": 0,
            "target_name": PAS_TRISTATE_CLASS_NAMES[0],
            "valid": False,
        }
        for field, marker in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _TRISTATE_COMPATIBILITY_MARKERS,
            strict=True,
        )
    ]
    return PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE, {
        "sample_id": str(sample_id),
        "response": PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE,
        "binary_label": int(binary_label),
        "tristate_probe_targets": probe_targets,
    }


def build_predecision_tristate_probe_response(
    *,
    sample_id: str,
    text_attr_values: Sequence[int],
    image_attr_values: Sequence[int],
    text_accessory_ids: Sequence[int] = (),
    image_accessory_ids: Sequence[int] = (),
    prompt_mode: str = "semantic_legend",
) -> tuple[str, dict[str, Any]]:
    """Build a label-safe latent checklist followed by the binary score slot."""

    _, metadata = build_tristate_compatibility_probe_response(
        sample_id=sample_id,
        text_attr_values=text_attr_values,
        image_attr_values=image_attr_values,
        text_accessory_ids=text_accessory_ids,
        image_accessory_ids=image_accessory_ids,
    )
    response = predecision_tristate_probe_response(prompt_mode)
    metadata["response"] = response
    return response, metadata


def build_predecision_tristate_probe_inference_record(
    *, sample_id: str, binary_label: int, prompt_mode: str = "semantic_legend"
) -> tuple[str, dict[str, Any]]:
    """Build the identical held-out checklist without structured labels."""

    _, metadata = build_tristate_compatibility_probe_inference_record(
        sample_id=sample_id,
        binary_label=binary_label,
    )
    response = predecision_tristate_probe_response(prompt_mode)
    metadata["response"] = response
    return response, metadata


def build_compatibility_probe_inference_record(
    *, sample_id: str, binary_label: int
) -> tuple[str, dict[str, Any]]:
    """Return the fixed probe trajectory when field targets are unavailable.

    Held-out evaluation and deployment need only the eight score positions.
    Dummy field targets remain masked and never affect ranking; the ordinary
    relevance label is retained solely for metric computation.
    """

    if binary_label not in {0, 1}:
        raise ValueError("binary_label must be zero or one")
    probe_targets = [
        {
            "field": field,
            "marker": marker,
            "target": False,
            "valid": False,
        }
        for field, marker in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _COMPATIBILITY_PROBE_MARKERS,
            strict=True,
        )
    ]
    return PAS_COMPATIBILITY_PROBE_RESPONSE, {
        "sample_id": str(sample_id),
        "response": PAS_COMPATIBILITY_PROBE_RESPONSE,
        "binary_label": int(binary_label),
        "probe_targets": probe_targets,
    }


def yes_no_logit_difference(
    output: torch.Tensor,
    *,
    yes_token_id: int,
    no_token_id: int,
    projection_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project hidden states if needed and return ``logit(yes)-logit(no)``.

    ``output`` may have any leading dimensions.  Its final dimension must be
    either vocabulary size or hidden size when ``projection_weight`` is given.
    Only two LM-head rows are projected in the hidden-state case.
    """

    if output.ndim < 1:
        raise ValueError("yes/no output must have at least one dimension")
    if yes_token_id < 0 or no_token_id < 0:
        raise ValueError("yes/no token IDs must be non-negative")
    projection = projection_weight
    if projection is not None and hasattr(projection, "to_local"):
        projection = projection.to_local()
    if projection is None or output.shape[-1] == projection.shape[0]:
        required_width = max(int(yes_token_id), int(no_token_id)) + 1
        if output.shape[-1] < required_width:
            raise ValueError(
                "yes/no token IDs exceed the output vocabulary dimension: "
                f"output={output.shape}, yes={yes_token_id}, no={no_token_id}"
            )
        return output[..., int(yes_token_id)].float() - output[
            ..., int(no_token_id)
        ].float()
    if output.shape[-1] != projection.shape[1]:
        raise ValueError(
            "Yes/no projection shape mismatch: "
            f"output={output.shape}, weight={projection.shape}"
        )
    weights = projection[[int(yes_token_id), int(no_token_id)]].float()
    return F.linear(output.float(), weights[0] - weights[1])


def canonical_token_logits(
    output: torch.Tensor,
    *,
    token_ids: Sequence[int],
    projection_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact LM-head logits for a small canonical token vocabulary.

    ``output`` may already be full-vocabulary logits or may be the hidden
    states emitted by fused cross entropy.  In the latter case selecting the
    corresponding LM-head rows is algebraically identical to materializing the
    complete vocabulary projection, while avoiding its memory cost.
    """

    if output.ndim < 1:
        raise ValueError("categorical output must have at least one dimension")
    ids = tuple(int(token_id) for token_id in token_ids)
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError("Canonical token IDs must be non-negative and unique")
    projection = projection_weight
    if projection is not None and hasattr(projection, "to_local"):
        projection = projection.to_local()
    if projection is None or output.shape[-1] == projection.shape[0]:
        if output.shape[-1] <= max(ids):
            raise ValueError(
                "Canonical token IDs exceed the output vocabulary dimension: "
                f"output={output.shape}, token_ids={ids}"
            )
        return output.float()[..., list(ids)]
    if output.shape[-1] != projection.shape[1]:
        raise ValueError(
            "Canonical-token projection shape mismatch: "
            f"output={output.shape}, weight={projection.shape}"
        )
    return F.linear(output.float(), projection[list(ids)].float())


def tristate_class_token_ids(
    tokenizer: Any,
    *,
    class_tokens: Sequence[str] = PAS_TRISTATE_CLASS_TOKENS,
) -> tuple[int, int, int]:
    """Resolve and validate a three-way single-token class alphabet."""

    if len(class_tokens) != len(PAS_TRISTATE_CLASS_NAMES):
        raise ValueError("Tri-state HCR requires exactly three class tokens")

    token_ids: list[int] = []
    for name, token in zip(
        PAS_TRISTATE_CLASS_NAMES, class_tokens, strict=True
    ):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Tri-state HCR token {token!r} ({name}) is not one token: {ids}"
            )
        token_ids.append(int(ids[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Tri-state HCR class tokens are not distinct")
    return tuple(token_ids)  # type: ignore[return-value]


def extract_compatibility_probe_scores(
    output: torch.Tensor,
    target: torch.LongTensor,
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    yes_token_id: int,
    no_token_id: int,
    ignore_index: int = -100,
    projection_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
    """Extract fixed-slot yes/no scores, exact targets, and validity masks."""

    if output.ndim != 3 or target.ndim != 2 or output.shape[:2] != target.shape:
        raise ValueError(
            f"Expected output [B,L,D] and target [B,L], got {output.shape=} "
            f"{target.shape=}"
        )
    if len(records) != int(target.shape[0]):
        raise ValueError("Compatibility metadata and batch sizes disagree")

    batch_scores: list[torch.Tensor] = []
    batch_targets: list[list[bool]] = []
    batch_valid: list[list[bool]] = []
    for row_index, record in enumerate(records):
        probe_targets = record.get("probe_targets") or ()
        if len(probe_targets) != len(PAS_COMPATIBILITY_PROBE_FIELDS):
            raise ValueError(
                f"Sample {record.get('sample_id', row_index)!r} does not have "
                f"{len(PAS_COMPATIBILITY_PROBE_FIELDS)} compatibility probes"
            )
        if str(record.get("response")) != PAS_COMPATIBILITY_PROBE_RESPONSE:
            raise ValueError("Compatibility probe response is not the fixed template")
        supervised_positions = torch.nonzero(
            target[row_index].ne(ignore_index), as_tuple=False
        ).flatten()
        supervised_ids = target[row_index, supervised_positions].tolist()
        expected_ids = tokenizer.encode(
            PAS_COMPATIBILITY_PROBE_RESPONSE, add_special_tokens=False
        )
        if supervised_ids[: len(expected_ids)] != expected_ids:
            raise ValueError(
                f"Packed compatibility response differs for sample "
                f"{record.get('sample_id', row_index)!r}"
            )

        row_outputs: list[torch.Tensor] = []
        row_targets: list[bool] = []
        row_valid: list[bool] = []
        for expected_field, expected_marker, item in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _COMPATIBILITY_PROBE_MARKERS,
            probe_targets,
            strict=True,
        ):
            if str(item.get("field")) != expected_field:
                raise ValueError("Compatibility probe field order changed")
            if str(item.get("marker")) != expected_marker:
                raise ValueError(f"Compatibility marker changed for {expected_field}")
            marker_ids = tokenizer.encode(expected_marker, add_special_tokens=False)
            marker_offset = _find_unique_subsequence(expected_ids, marker_ids)
            placeholder_offset = marker_offset + len(marker_ids)
            prediction_position = int(
                supervised_positions[placeholder_offset].item()
            ) - 1
            if prediction_position < 0:
                raise ValueError(
                    f"Compatibility probe {expected_field} has no prediction token"
                )
            row_outputs.append(output[row_index, prediction_position])
            row_targets.append(bool(item["target"]))
            row_valid.append(bool(item["valid"]))
        batch_scores.append(
            yes_no_logit_difference(
                torch.stack(row_outputs),
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                projection_weight=projection_weight,
            )
        )
        batch_targets.append(row_targets)
        batch_valid.append(row_valid)

    return (
        torch.stack(batch_scores),
        torch.tensor(batch_targets, device=output.device, dtype=torch.bool),
        torch.tensor(batch_valid, device=output.device, dtype=torch.bool),
    )


def extract_active_compatibility_probe_scores(
    output: torch.Tensor,
    target: torch.LongTensor,
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    yes_token_id: int,
    no_token_id: int,
    ignore_index: int = -100,
    projection_weight: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.BoolTensor,
    torch.BoolTensor,
    torch.Tensor,
    torch.BoolTensor,
    torch.BoolTensor,
]:
    """Extract active-aware requirement and satisfaction tensors.

    Returns requirement logits, targets, and masks followed by satisfaction
    logits, targets, and masks.  All tensors have shape ``[batch, 8]``.
    """

    if output.ndim != 3 or target.ndim != 2 or output.shape[:2] != target.shape:
        raise ValueError(
            f"Expected output [B,L,D] and target [B,L], got {output.shape=} "
            f"{target.shape=}"
        )
    if len(records) != int(target.shape[0]):
        raise ValueError("Active compatibility metadata and batch sizes disagree")

    batch_requirement_outputs: list[torch.Tensor] = []
    batch_requirement_targets: list[list[bool]] = []
    batch_requirement_valid: list[list[bool]] = []
    batch_satisfaction_outputs: list[torch.Tensor] = []
    batch_satisfaction_targets: list[list[bool]] = []
    batch_satisfaction_valid: list[list[bool]] = []
    expected_ids = tokenizer.encode(
        PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE, add_special_tokens=False
    )
    for row_index, record in enumerate(records):
        probe_targets = record.get("active_probe_targets") or ()
        if len(probe_targets) != len(PAS_COMPATIBILITY_PROBE_FIELDS):
            raise ValueError(
                f"Sample {record.get('sample_id', row_index)!r} does not have "
                f"{len(PAS_COMPATIBILITY_PROBE_FIELDS)} active compatibility probes"
            )
        if str(record.get("response")) != PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE:
            raise ValueError(
                "Active compatibility probe response is not the fixed template"
            )
        supervised_positions = torch.nonzero(
            target[row_index].ne(ignore_index), as_tuple=False
        ).flatten()
        supervised_ids = target[row_index, supervised_positions].tolist()
        if supervised_ids[: len(expected_ids)] != expected_ids:
            raise ValueError(
                f"Packed active compatibility response differs for sample "
                f"{record.get('sample_id', row_index)!r}"
            )

        row_requirement_outputs: list[torch.Tensor] = []
        row_requirement_targets: list[bool] = []
        row_requirement_valid: list[bool] = []
        row_satisfaction_outputs: list[torch.Tensor] = []
        row_satisfaction_targets: list[bool] = []
        row_satisfaction_valid: list[bool] = []
        for (
            expected_field,
            expected_requirement_marker,
            expected_satisfaction_marker,
            item,
        ) in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _ACTIVE_COMPATIBILITY_REQUIREMENT_MARKERS,
            _ACTIVE_COMPATIBILITY_SATISFACTION_MARKERS,
            probe_targets,
            strict=True,
        ):
            if str(item.get("field")) != expected_field:
                raise ValueError("Active compatibility probe field order changed")
            for kind, expected_marker, outputs_for_kind in (
                (
                    "requirement",
                    expected_requirement_marker,
                    row_requirement_outputs,
                ),
                (
                    "satisfaction",
                    expected_satisfaction_marker,
                    row_satisfaction_outputs,
                ),
            ):
                if str(item.get(f"{kind}_marker")) != expected_marker:
                    raise ValueError(
                        f"Active compatibility {kind} marker changed for "
                        f"{expected_field}"
                    )
                marker_ids = tokenizer.encode(
                    expected_marker, add_special_tokens=False
                )
                marker_offset = _find_unique_subsequence(expected_ids, marker_ids)
                placeholder_offset = marker_offset + len(marker_ids)
                prediction_position = int(
                    supervised_positions[placeholder_offset].item()
                ) - 1
                if prediction_position < 0:
                    raise ValueError(
                        f"Active compatibility {kind} probe {expected_field} has "
                        "no prediction token"
                    )
                outputs_for_kind.append(output[row_index, prediction_position])
            row_requirement_targets.append(bool(item["requirement_target"]))
            row_requirement_valid.append(bool(item["requirement_valid"]))
            row_satisfaction_targets.append(bool(item["satisfaction_target"]))
            row_satisfaction_valid.append(bool(item["satisfaction_valid"]))

        batch_requirement_outputs.append(torch.stack(row_requirement_outputs))
        batch_requirement_targets.append(row_requirement_targets)
        batch_requirement_valid.append(row_requirement_valid)
        batch_satisfaction_outputs.append(torch.stack(row_satisfaction_outputs))
        batch_satisfaction_targets.append(row_satisfaction_targets)
        batch_satisfaction_valid.append(row_satisfaction_valid)

    requirement_logits = yes_no_logit_difference(
        torch.stack(batch_requirement_outputs),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        projection_weight=projection_weight,
    )
    satisfaction_logits = yes_no_logit_difference(
        torch.stack(batch_satisfaction_outputs),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        projection_weight=projection_weight,
    )
    return (
        requirement_logits,
        torch.tensor(
            batch_requirement_targets, device=output.device, dtype=torch.bool
        ),
        torch.tensor(
            batch_requirement_valid, device=output.device, dtype=torch.bool
        ),
        satisfaction_logits,
        torch.tensor(
            batch_satisfaction_targets, device=output.device, dtype=torch.bool
        ),
        torch.tensor(
            batch_satisfaction_valid, device=output.device, dtype=torch.bool
        ),
    )


def extract_tristate_compatibility_probe_logits(
    output: torch.Tensor,
    target: torch.LongTensor,
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    class_token_ids: Sequence[int] | None = None,
    ignore_index: int = -100,
    projection_weight: torch.Tensor | None = None,
    expected_response: str | None = PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE,
) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
    """Extract one three-class LM-head projection for each PAS field.

    Returns logits shaped ``[batch, 8, 3]``, integer class targets shaped
    ``[batch, 8]``, and the held-out supervision mask with the same shape as
    targets.  The response contains no A/B/C answer tokens.
    """

    if output.ndim != 3 or target.ndim != 2 or output.shape[:2] != target.shape:
        raise ValueError(
            f"Expected output [B,L,D] and target [B,L], got {output.shape=} "
            f"{target.shape=}"
        )
    if len(records) != int(target.shape[0]):
        raise ValueError("Tri-state compatibility metadata and batch sizes disagree")
    resolved_token_ids = (
        tuple(int(value) for value in class_token_ids)
        if class_token_ids is not None
        else tristate_class_token_ids(tokenizer)
    )
    if len(resolved_token_ids) != len(PAS_TRISTATE_CLASS_NAMES):
        raise ValueError("Tri-state HCR requires exactly three class token IDs")

    batch_outputs: list[torch.Tensor] = []
    batch_targets: list[list[int]] = []
    batch_valid: list[list[bool]] = []
    for row_index, record in enumerate(records):
        row_response = str(record.get("response") or "")
        row_expected_response = expected_response or row_response
        expected_ids = tokenizer.encode(
            row_expected_response, add_special_tokens=False
        )
        probe_targets = record.get("tristate_probe_targets") or ()
        if len(probe_targets) != len(PAS_COMPATIBILITY_PROBE_FIELDS):
            raise ValueError(
                f"Sample {record.get('sample_id', row_index)!r} does not have "
                f"{len(PAS_COMPATIBILITY_PROBE_FIELDS)} tri-state probes"
            )
        if row_response != row_expected_response:
            raise ValueError("Tri-state compatibility response is not fixed template")
        supervised_positions = torch.nonzero(
            target[row_index].ne(ignore_index), as_tuple=False
        ).flatten()
        supervised_ids = target[row_index, supervised_positions].tolist()
        if supervised_ids[: len(expected_ids)] != expected_ids:
            raise ValueError(
                f"Packed tri-state HCR response differs for sample "
                f"{record.get('sample_id', row_index)!r}"
            )

        row_outputs: list[torch.Tensor] = []
        row_targets: list[int] = []
        row_valid: list[bool] = []
        for expected_field, expected_marker, item in zip(
            PAS_COMPATIBILITY_PROBE_FIELDS,
            _TRISTATE_COMPATIBILITY_MARKERS,
            probe_targets,
            strict=True,
        ):
            if str(item.get("field")) != expected_field:
                raise ValueError("Tri-state compatibility field order changed")
            if str(item.get("marker")) != expected_marker:
                raise ValueError(f"Tri-state marker changed for {expected_field}")
            class_target = int(item["target"])
            if class_target not in range(len(PAS_TRISTATE_CLASS_NAMES)):
                raise ValueError(f"Invalid tri-state target {class_target}")
            marker_ids = tokenizer.encode(expected_marker, add_special_tokens=False)
            marker_offset = _find_unique_subsequence(expected_ids, marker_ids)
            placeholder_offset = marker_offset + len(marker_ids)
            prediction_position = int(
                supervised_positions[placeholder_offset].item()
            ) - 1
            if prediction_position < 0:
                raise ValueError(
                    f"Tri-state compatibility probe {expected_field} has no "
                    "prediction token"
                )
            row_outputs.append(output[row_index, prediction_position])
            row_targets.append(class_target)
            row_valid.append(bool(item["valid"]))
        batch_outputs.append(torch.stack(row_outputs))
        batch_targets.append(row_targets)
        batch_valid.append(row_valid)

    logits = canonical_token_logits(
        torch.stack(batch_outputs),
        token_ids=resolved_token_ids,
        projection_weight=projection_weight,
    )
    return (
        logits,
        torch.tensor(batch_targets, device=output.device, dtype=torch.long),
        torch.tensor(batch_valid, device=output.device, dtype=torch.bool),
    )


def active_aware_logical_score(
    requirement_logits: torch.Tensor,
    satisfaction_logits: torch.Tensor,
    *,
    dim: int = -1,
    reduction: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    """Aggregate differentiable conjunction log probability across fields.

    For each field, ``p_pass = (1 - p_required) +
    p_required * p_satisfied``.  The implementation works in log space and
    therefore remains stable even for confident probes.  Summing per-field log
    probabilities is the exact independent conjunction; ``mean`` provides a
    length-normalized variant with identical candidate ordering at fixed width.
    """

    if requirement_logits.shape != satisfaction_logits.shape:
        raise ValueError(
            "Requirement and satisfaction logits must have identical shapes: "
            f"{requirement_logits.shape} != {satisfaction_logits.shape}"
        )
    if not requirement_logits.is_floating_point() or not (
        satisfaction_logits.is_floating_point()
    ):
        raise TypeError("Active-aware logical logits must be floating point")
    log_not_required = F.logsigmoid(-requirement_logits.float())
    log_required_and_satisfied = F.logsigmoid(
        requirement_logits.float()
    ) + F.logsigmoid(satisfaction_logits.float())
    log_pass = torch.logaddexp(log_not_required, log_required_and_satisfied)
    if reduction == "sum":
        return log_pass.sum(dim=dim)
    if reduction == "mean":
        return log_pass.mean(dim=dim)
    raise ValueError(f"Unknown active-aware logical reduction {reduction!r}")


def tristate_logical_score(
    class_logits: torch.Tensor,
    *,
    dim: int = -1,
    reduction: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    """Aggregate ``log(1 - P(active_violated))`` across PAS fields."""

    if class_logits.ndim < 2 or class_logits.shape[-1] != 3:
        raise ValueError(
            "Tri-state logits must end in the three semantic classes, got "
            f"{class_logits.shape}"
        )
    if not class_logits.is_floating_point():
        raise TypeError("Tri-state logits must be floating point")
    log_normalizer = torch.logsumexp(class_logits.float(), dim=-1)
    log_pass = torch.logsumexp(class_logits.float()[..., :2], dim=-1) - log_normalizer
    if reduction == "sum":
        return log_pass.sum(dim=dim)
    if reduction == "mean":
        return log_pass.mean(dim=dim)
    raise ValueError(f"Unknown tri-state logical reduction {reduction!r}")


def predecision_smooth_and_score(
    class_logits: torch.Tensor,
    *,
    temperature: float = 0.5,
    dim: int = 1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a differentiable strict-AND score from checklist logits.

    Each field has three logits: ``inactive``, ``active_satisfied`` and
    ``active_violated``.  A field passes when it is either inactive or
    satisfied, so its pass evidence is ``log P(inactive or satisfied)``.
    The normalized soft minimum of those field evidences is a smooth AND:
    one violated requirement pulls the whole candidate score down, while an
    unspecified field contributes neutral pass evidence.  ``mask`` can be
    used by callers with a subset of fields; the normalizer keeps the score
    invariant when all valid fields have equal evidence.

    This intentionally does not use the binary label to construct the score:
    a partial negative (one or more violated fields) is therefore distinct
    from an all-negative/all-field target and receives gradient through the
    violating field(s).
    """

    if class_logits.ndim < 2 or class_logits.shape[-1] != 3:
        raise ValueError(
            "Pre-decision smooth-AND logits must end in three classes, got "
            f"{class_logits.shape}"
        )
    if not class_logits.is_floating_point():
        raise TypeError("Pre-decision smooth-AND logits must be floating point")
    pass_logprob = torch.log_softmax(class_logits.float(), dim=-1)[..., :2]
    pass_logprob = torch.logsumexp(pass_logprob, dim=-1)
    return normalized_softmin(
        pass_logprob,
        temperature=temperature,
        dim=dim,
        mask=mask,
    )


def normalized_softmin(
    values: torch.Tensor,
    *,
    temperature: float,
    dim: int = -1,
    mask: torch.Tensor | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Smooth minimum normalized so equal inputs aggregate to that input."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if values.ndim == 0:
        raise ValueError("normalized_softmin requires a reduction dimension")
    if not values.is_floating_point():
        raise TypeError("normalized_softmin requires floating-point values")
    if mask is None:
        count = values.shape[dim]
        if count == 0:
            raise ValueError("normalized_softmin cannot reduce an empty dimension")
        log_count = values.new_tensor(float(count)).log()
        result = -float(temperature) * (
            torch.logsumexp(-values / float(temperature), dim=dim, keepdim=keepdim)
            - log_count
        )
        return result

    valid = torch.broadcast_to(
        mask.to(device=values.device, dtype=torch.bool), values.shape
    )
    counts = valid.sum(dim=dim, keepdim=keepdim)
    if bool((counts == 0).any()):
        raise ValueError("normalized_softmin received a slice with no valid values")
    logits = (-values / float(temperature)).masked_fill(~valid, -torch.inf)
    return -float(temperature) * (
        torch.logsumexp(logits, dim=dim, keepdim=keepdim)
        - counts.to(values.dtype).log()
    )


def bottom_k_cvar(
    values: torch.Tensor,
    *,
    k: int,
    dim: int = -1,
    mask: torch.Tensor | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Mean the worst (smallest) ``k`` valid compatibility values per slice."""

    if k < 1:
        raise ValueError("k must be at least one")
    if values.ndim == 0:
        raise ValueError("bottom_k_cvar requires a reduction dimension")
    if not values.is_floating_point():
        raise TypeError("bottom_k_cvar requires floating-point values")
    width = values.shape[dim]
    if width == 0:
        raise ValueError("bottom_k_cvar cannot reduce an empty dimension")
    valid = (
        torch.ones_like(values, dtype=torch.bool)
        if mask is None
        else torch.broadcast_to(
            mask.to(device=values.device, dtype=torch.bool), values.shape
        )
    )
    counts = valid.sum(dim=dim, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("bottom_k_cvar received a slice with no valid values")

    ordered = values.masked_fill(~valid, torch.inf).sort(dim=dim).values
    positions_shape = [1] * values.ndim
    positions_shape[dim] = width
    positions = torch.arange(width, device=values.device).reshape(positions_shape)
    selected = (positions < k) & (positions < counts)
    numerator = ordered.masked_fill(~selected, 0.0).sum(dim=dim, keepdim=keepdim)
    denominator = counts.clamp(max=k).to(values.dtype)
    if not keepdim:
        denominator = denominator.squeeze(dim)
    return numerator / denominator


def normalized_field(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class StructuredAttributeAssets:
    fields: tuple[str, ...]
    choices_by_field: dict[str, tuple[str, ...]]
    labels_by_image: dict[str, tuple[int, ...]]
    query_labels_by_index: tuple[tuple[int, ...], ...] = field(default_factory=tuple)


def load_structured_attribute_assets(
    records_path: str | Path,
    vocab_path: str | Path,
    query_records_path: str | Path | None = None,
) -> StructuredAttributeAssets:
    records = json.loads(Path(records_path).read_text(encoding="utf-8"))
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    fields = tuple(str(value) for value in vocab["attributes"])
    choices = {
        field: tuple(str(value) for value in vocab["id_to_value"][field][1:])
        for field in fields
    }
    if any(not values or len(values) > 14 for values in choices.values()):
        raise ValueError("Structured PAS fields must contain between 1 and 14 choices")

    labels_by_image: dict[str, tuple[int, ...]] = {}
    for row in records:
        image = str(row["unique_name"])
        labels = tuple(int(value) for value in row["image_attr_values"])
        if len(labels) != len(fields):
            raise ValueError(f"Attribute width mismatch for {image}: {labels}")
        for field, label in zip(fields, labels, strict=True):
            if not 0 <= label <= len(choices[field]):
                raise ValueError(f"Invalid {field!r} label {label} for {image}")
        previous = labels_by_image.setdefault(image, labels)
        if previous != labels:
            raise ValueError(f"Conflicting structured labels for {image}")
    query_labels_by_index: tuple[tuple[int, ...], ...] = ()
    if query_records_path is not None:
        query_rows = json.loads(Path(query_records_path).read_text(encoding="utf-8"))
        labels_by_index = []
        for row_index, raw_labels in enumerate(query_rows):
            labels = tuple(int(value) for value in raw_labels)
            if len(labels) != len(fields):
                raise ValueError(f"Query attribute width mismatch at row {row_index}")
            labels_by_index.append(labels)
        query_labels_by_index = tuple(labels_by_index)
    return StructuredAttributeAssets(
        fields, choices, labels_by_image, query_labels_by_index
    )


def _stable_rng(seed: int, sample_id: str, field: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{sample_id}\0{field}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def append_structured_attribute_response(
    binary_response: str,
    *,
    sample_id: str,
    labels: Sequence[int],
    assets: StructuredAttributeAssets,
    seed: int,
    query_labels: Sequence[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Append full-vocabulary choices and isolated answers to yes/no output."""

    if len(labels) != len(assets.fields):
        raise ValueError("Structured attribute labels have the wrong width")
    option_lines: list[str] = []
    answer_lines: list[str] = []
    targets: list[dict[str, Any]] = []
    if query_labels is not None and len(query_labels) != len(assets.fields):
        raise ValueError("Structured query labels have the wrong width")
    scalar_match = True
    scalar_mismatch = False
    scalar_comparable = False
    for field_index, (field, raw_label) in enumerate(
        zip(assets.fields, labels, strict=True)
    ):
        field_key = normalized_field(field)
        canonical = list(assets.choices_by_field[field])
        shuffled = list(canonical)
        _stable_rng(seed, sample_id, field).shuffle(shuffled)
        rendered = "|".join(
            f"{chr(ord('A') + index)}={value}"
            for index, value in enumerate(shuffled)
        )
        option_lines.append(f"[{field_key} choices] {rendered}")
        marker = f"[{field_key} answer]\n"
        if int(raw_label) == 0:
            answer = "?"
        else:
            expected_value = canonical[int(raw_label) - 1]
            target_class = shuffled.index(expected_value)
            answer = chr(ord("A") + target_class)
            targets.append(
                {
                    "field": field_key,
                    "marker": marker,
                    "target_class": target_class,
                    "option_count": len(shuffled),
                    "target_value": expected_value,
                    "choices": shuffled,
                    **(
                        {
                            "query_target_class": shuffled.index(
                                canonical[int(query_labels[field_index]) - 1]
                            ),
                            "query_target_value": canonical[
                                int(query_labels[field_index]) - 1
                            ],
                        }
                        if query_labels is not None
                        and int(query_labels[field_index]) > 0
                        else {}
                    ),
                }
            )
        if (
            query_labels is not None
            and int(query_labels[field_index]) > 0
            and int(raw_label) > 0
        ):
            scalar_comparable = True
            differs = int(raw_label) != int(query_labels[field_index])
            scalar_mismatch = scalar_mismatch or differs
            scalar_match = scalar_match and not differs
        answer_lines.extend([marker.rstrip("\n"), answer])

    # Choice and field text are intentionally in the assistant prefix.  The
    # overall yes/no decision therefore has exactly the same user context and
    # prediction position as deployed binary reranking.
    response = "\n".join(
        [
            binary_response,
            "[masked attribute choice context]",
            *option_lines,
            "[masked attribute answers]",
            *answer_lines,
        ]
    )
    metadata = {
        "sample_id": str(sample_id),
        "response": response,
        "targets": targets,
        "scalar_match": scalar_match if query_labels is not None else None,
        "scalar_mismatch": scalar_mismatch if query_labels is not None else None,
        "scalar_comparable": scalar_comparable if query_labels is not None else None,
    }
    return response, metadata


def structured_attribute_mismatch_metadata(
    binary_response: str,
    *,
    sample_id: str,
    labels: Sequence[int],
    assets: StructuredAttributeAssets,
    query_labels: Sequence[int],
) -> dict[str, Any]:
    """Build scalar mismatch flags without appending choice-token targets."""

    if len(labels) != len(assets.fields) or len(query_labels) != len(assets.fields):
        raise ValueError("Structured image/query labels have the wrong width")
    comparable = [
        int(image_label) > 0 and int(query_label) > 0
        for image_label, query_label in zip(labels, query_labels, strict=True)
    ]
    differences = [
        is_comparable and int(image_label) != int(query_label)
        for image_label, query_label, is_comparable in zip(
            labels, query_labels, comparable, strict=True
        )
    ]
    scalar_comparable = any(comparable)
    scalar_mismatch = any(differences)
    return {
        "sample_id": str(sample_id),
        "response": binary_response,
        "targets": [],
        "scalar_match": scalar_comparable and not scalar_mismatch,
        "scalar_mismatch": scalar_mismatch,
        "scalar_comparable": scalar_comparable,
    }


def _find_unique_subsequence(values: list[int], needle: list[int]) -> int:
    matches = [
        index
        for index in range(len(values) - len(needle) + 1)
        if values[index : index + len(needle)] == needle
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one structured marker occurrence, found {len(matches)}"
        )
    return matches[0]


class StructuredAttributeChoiceLoss:
    """Constrained A-N CE at only the structured field answer positions."""

    def __init__(
        self,
        tokenizer,
        projection_weight,
        fields: Sequence[str],
        *,
        hard_example_weight: float = 1.0,
        hard_example_fields: Sequence[str] = (),
    ):
        self.tokenizer = tokenizer
        self.projection_weight = projection_weight
        self.fields = tuple(normalized_field(field) for field in fields)
        if hard_example_weight < 1.0:
            raise ValueError("hard_example_weight must be at least 1.0")
        self.hard_example_weight = float(hard_example_weight)
        self.hard_example_fields = frozenset(
            normalized_field(field) for field in hard_example_fields
        )
        unknown_hard_fields = self.hard_example_fields.difference(self.fields)
        if unknown_hard_fields:
            raise ValueError(
                f"Unknown hard-example fields: {sorted(unknown_hard_fields)}"
            )
        letter_ids = []
        for index in range(14):
            letter = chr(ord("A") + index)
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Structured label {letter!r} is not one token: {ids}")
            letter_ids.append(int(ids[0]))
        if len(set(letter_ids)) != len(letter_ids):
            raise ValueError("Structured option letters do not have unique token IDs")
        self.letter_ids = tuple(letter_ids)
        self._audit_enabled = False
        self._audit_records: list[dict[str, Any]] = []

    def set_audit_enabled(self, enabled: bool) -> None:
        self._audit_enabled = bool(enabled)
        if not self._audit_enabled:
            self._audit_records = []

    def pop_audit_records(self) -> list[dict[str, Any]]:
        records = self._audit_records
        self._audit_records = []
        return records

    def __call__(
        self,
        output: torch.Tensor,
        target: torch.LongTensor,
        records: Sequence[dict[str, Any] | None],
        *,
        ignore_index: int = -100,
        projection_weight: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, tuple[torch.Tensor, int]],
        torch.Tensor,
        torch.BoolTensor,
    ]:
        if len(records) != int(target.shape[0]):
            raise ValueError("Structured metadata and batch sizes disagree")
        projection = projection_weight if projection_weight is not None else self.projection_weight
        if hasattr(projection, "to_local"):
            projection = projection.to_local()
        letter_ids = torch.tensor(self.letter_ids, device=output.device)
        letter_weight = projection.index_select(0, letter_ids)

        losses: list[torch.Tensor] = []
        correct_by_field: dict[str, torch.Tensor] = {}
        count_by_field: Counter[str] = Counter()
        hard_by_field: Counter[str] = Counter()
        covered_samples = 0
        compatibility_scores: list[torch.Tensor] = []
        compatibility_valid: list[bool] = []
        for row_index, record in enumerate(records):
            if record is None or not record.get("targets"):
                compatibility_scores.append(output[row_index].sum() * 0.0)
                compatibility_valid.append(False)
                continue
            covered_samples += 1
            query_log_probabilities: list[torch.Tensor] = []
            supervised_positions = torch.nonzero(
                target[row_index].ne(ignore_index), as_tuple=False
            ).flatten()
            supervised_ids = target[row_index, supervised_positions].tolist()
            expected_ids = self.tokenizer.encode(
                str(record["response"]), add_special_tokens=False
            )
            if supervised_ids[: len(expected_ids)] != expected_ids:
                raise ValueError(
                    f"Packed structured response differs for sample {record['sample_id']}"
                )
            for item in record["targets"]:
                field = normalized_field(str(item["field"]))
                if field not in self.fields:
                    raise ValueError(f"Unknown structured field {field!r}")
                marker_ids = self.tokenizer.encode(
                    str(item["marker"]), add_special_tokens=False
                )
                marker_offset = _find_unique_subsequence(expected_ids, marker_ids)
                answer_offset = marker_offset + len(marker_ids)
                target_class = int(item["target_class"])
                option_count = int(item["option_count"])
                expected_letter_id = self.letter_ids[target_class]
                if expected_ids[answer_offset] != expected_letter_id:
                    raise ValueError(
                        f"Structured answer is not isolated for {field}: "
                        f"expected={expected_letter_id}, got={expected_ids[answer_offset]}"
                    )
                target_position = int(supervised_positions[answer_offset].item())
                if target_position <= 0:
                    raise ValueError("Structured answer has no prediction position")
                hidden = output[row_index, target_position - 1]
                logits = F.linear(hidden.to(letter_weight.dtype), letter_weight).float()
                logits = logits[:option_count]
                class_target = torch.tensor(
                    [target_class], device=logits.device, dtype=torch.long
                )
                predicted_class = int(logits.argmax().item())
                is_hard_example = (
                    predicted_class != target_class
                    and field in self.hard_example_fields
                    and self.hard_example_weight > 1.0
                )
                field_loss = F.cross_entropy(logits.unsqueeze(0), class_target)
                if is_hard_example:
                    field_loss = field_loss * self.hard_example_weight
                    hard_by_field[field] += 1
                losses.append(field_loss)
                if "query_target_class" in item:
                    query_log_probabilities.append(
                        logits.log_softmax(dim=0)[int(item["query_target_class"])]
                    )
                is_correct = logits.argmax().eq(target_class).float().detach()
                if self._audit_enabled:
                    probabilities = logits.softmax(dim=0).detach().cpu().tolist()
                    choices = [str(value) for value in item["choices"]]
                    self._audit_records.append(
                        {
                            **{
                                str(key): value
                                for key, value in record.get("audit", {}).items()
                            },
                            "sample_id": str(record["sample_id"]),
                            "field": field,
                            "choices": choices,
                            "label_value": str(item["target_value"]),
                            "predicted_value": choices[predicted_class],
                            "correct": bool(predicted_class == target_class),
                            "target_probability": float(probabilities[target_class]),
                            "probabilities_by_value": {
                                value: float(probabilities[index])
                                for index, value in enumerate(choices)
                            },
                        }
                    )
                correct_by_field[field] = (
                    is_correct
                    if field not in correct_by_field
                    else correct_by_field[field] + is_correct
                )
                count_by_field[field] += 1
            compatibility_scores.append(
                torch.stack(query_log_probabilities).mean()
                if query_log_probabilities
                else output[row_index].sum() * 0.0
            )
            compatibility_valid.append(bool(query_log_probabilities))

        zero = output.sum() * 0.0
        loss = torch.stack(losses).mean() if losses else zero
        loss_sum = (
            torch.stack([value.detach() for value in losses]).sum()
            if losses
            else zero.detach()
        )
        correct_sum = (
            torch.stack(list(correct_by_field.values())).sum()
            if correct_by_field
            else zero.detach()
        )
        metrics: dict[str, tuple[torch.Tensor, int]] = {
            "attribute_loss": (loss_sum, len(losses)),
            "attribute_accuracy": (correct_sum, len(losses)),
            "attribute_sample_coverage": (
                output.new_tensor(float(covered_samples)).detach(),
                int(target.shape[0]),
            ),
        }
        focused_count = sum(count_by_field[field] for field in self.hard_example_fields)
        hard_count = sum(hard_by_field.values())
        if focused_count:
            metrics["attribute_online_hard_example_rate"] = (
                output.new_tensor(float(hard_count)).detach(),
                focused_count,
            )
        for field in self.fields:
            count = count_by_field[field]
            if count:
                metrics[f"attribute_{field}_accuracy"] = (
                    correct_by_field[field],
                    count,
                )
                if field in self.hard_example_fields:
                    metrics[f"attribute_{field}_hard_example_rate"] = (
                        output.new_tensor(float(hard_by_field[field])).detach(),
                        count,
                    )
        return (
            loss,
            metrics,
            torch.stack(compatibility_scores),
            torch.tensor(compatibility_valid, device=output.device, dtype=torch.bool),
        )
