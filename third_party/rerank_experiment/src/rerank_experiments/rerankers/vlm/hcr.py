"""Active-aware HCR response layout and deterministic score composition.

This module is deliberately independent of vLLM so the token-position and
score contracts can be unit tested without loading a model.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

HCR_FIELDS = (
    "top_outer_color",
    "top_outer_type",
    "bottom_color",
    "bottom_type",
    "shoe_color",
    "shoe_type",
    "viewpoint",
    "accessory",
)

_REQUIREMENT_MARKERS = tuple(
    f"[{field} required: yes or no]\n" for field in HCR_FIELDS
)
_SATISFACTION_MARKERS = tuple(
    f"[{field} satisfied: yes or no]\n" for field in HCR_FIELDS
)
HCR_ACTIVE_RESPONSE = "\n".join(
    [
        "<answer>?</answer>",
        "[active-aware constraint compatibility probes]",
        *(
            "\n".join((f"{required}?", f"{satisfied}?"))
            for required, satisfied in zip(
                _REQUIREMENT_MARKERS, _SATISFACTION_MARKERS, strict=True
            )
        ),
    ]
)
HCR_ACTIVE_RESPONSE_SHA256 = hashlib.sha256(
    HCR_ACTIVE_RESPONSE.encode("utf-8")
).hexdigest()

HCR_TRISTATE_CLASS_NAMES = ("inactive", "active_satisfied", "active_violated")
HCR_TRISTATE_CLASS_TOKENS = ("A", "B", "C")
_TRISTATE_MARKERS = (
    "top_color:\n",
    "top_type:\n",
    "bottom_color:\n",
    "bottom_type:\n",
    "shoe_color:\n",
    "shoe_type:\n",
    "viewpoint:\n",
    "accessory:\n",
)
HCR_TRISTATE_RESPONSE = "\n".join(
    ["<answer>?</answer>", *(f"{marker}?" for marker in _TRISTATE_MARKERS)]
)
HCR_TRISTATE_RESPONSE_SHA256 = hashlib.sha256(
    HCR_TRISTATE_RESPONSE.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class HCRSlot:
    name: str
    field: str | None
    kind: Literal["global", "requirement", "satisfaction", "tristate"]
    token_offset: int
    character_offset: int


def tristate_class_token_ids(tokenizer: Any) -> tuple[int, int, int]:
    """Resolve the training-side canonical A/B/C class vocabulary."""

    values: list[int] = []
    for name, token in zip(
        HCR_TRISTATE_CLASS_NAMES, HCR_TRISTATE_CLASS_TOKENS, strict=True
    ):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Tri-state HCR token {token!r} ({name}) is not one token: {ids}"
            )
        values.append(int(ids[0]))
    if len(set(values)) != 3:
        raise ValueError("Tri-state HCR class tokens are not distinct")
    return tuple(values)  # type: ignore[return-value]


def _find_unique_subsequence(values: Sequence[int], needle: Sequence[int]) -> int:
    matches = [
        start
        for start in range(len(values) - len(needle) + 1)
        if list(values[start : start + len(needle)]) == list(needle)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one token-subsequence match, found {len(matches)}"
        )
    return matches[0]


def _unique_character_offset(text: str, marker: str) -> int:
    first = text.find(marker)
    if first < 0 or text.find(marker, first + 1) >= 0:
        raise ValueError(f"Expected one response marker {marker!r}")
    return first + len(marker)


def build_hcr_slots(tokenizer: Any, decision_offset: int) -> tuple[HCRSlot, ...]:
    """Resolve all 17 neutral decision positions in the canonical response."""

    response_ids = tokenizer.encode(HCR_ACTIVE_RESPONSE, add_special_tokens=False)
    if not response_ids:
        raise ValueError("HCR response tokenized to an empty sequence")
    global_character_offset = _unique_character_offset(HCR_ACTIVE_RESPONSE, "<answer>")
    global_prefix_ids = tokenizer.encode(
        HCR_ACTIVE_RESPONSE[:global_character_offset], add_special_tokens=False
    )
    if len(global_prefix_ids) != decision_offset:
        raise ValueError(
            "Binary decision offset disagrees with active HCR global slot: "
            f"{decision_offset} != {len(global_prefix_ids)}"
        )
    slots = [
        HCRSlot(
            name="global",
            field=None,
            kind="global",
            token_offset=decision_offset,
            character_offset=global_character_offset,
        )
    ]
    for field, requirement, satisfaction in zip(
        HCR_FIELDS, _REQUIREMENT_MARKERS, _SATISFACTION_MARKERS, strict=True
    ):
        for kind, marker in (
            ("requirement", requirement),
            ("satisfaction", satisfaction),
        ):
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            marker_start = _find_unique_subsequence(response_ids, marker_ids)
            token_offset = marker_start + len(marker_ids)
            character_offset = _unique_character_offset(HCR_ACTIVE_RESPONSE, marker)
            prefix_ids = tokenizer.encode(
                HCR_ACTIVE_RESPONSE[:character_offset], add_special_tokens=False
            )
            if prefix_ids != response_ids[:token_offset]:
                raise ValueError(
                    f"HCR {field}/{kind} prefix has a tokenizer boundary merge"
                )
            slots.append(
                HCRSlot(
                    name=f"{field}_{kind}",
                    field=field,
                    kind=kind,  # type: ignore[arg-type]
                    token_offset=token_offset,
                    character_offset=character_offset,
                )
            )
    if len(slots) != 17 or len({slot.token_offset for slot in slots}) != 17:
        raise ValueError("Active HCR must contain 17 distinct decision slots")
    return tuple(slots)


def build_tristate_hcr_slots(
    tokenizer: Any, decision_offset: int
) -> tuple[HCRSlot, ...]:
    """Resolve the global decision and eight compact categorical slots."""

    response_ids = tokenizer.encode(HCR_TRISTATE_RESPONSE, add_special_tokens=False)
    global_character_offset = _unique_character_offset(
        HCR_TRISTATE_RESPONSE, "<answer>"
    )
    global_prefix_ids = tokenizer.encode(
        HCR_TRISTATE_RESPONSE[:global_character_offset], add_special_tokens=False
    )
    if len(global_prefix_ids) != decision_offset:
        raise ValueError(
            "Binary decision offset disagrees with tri-state HCR global slot: "
            f"{decision_offset} != {len(global_prefix_ids)}"
        )
    slots = [
        HCRSlot(
            name="global",
            field=None,
            kind="global",
            token_offset=decision_offset,
            character_offset=global_character_offset,
        )
    ]
    for field, marker in zip(HCR_FIELDS, _TRISTATE_MARKERS, strict=True):
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        marker_start = _find_unique_subsequence(response_ids, marker_ids)
        token_offset = marker_start + len(marker_ids)
        character_offset = _unique_character_offset(HCR_TRISTATE_RESPONSE, marker)
        prefix_ids = tokenizer.encode(
            HCR_TRISTATE_RESPONSE[:character_offset], add_special_tokens=False
        )
        if prefix_ids != response_ids[:token_offset]:
            raise ValueError(f"Tri-state HCR {field} has a tokenizer boundary merge")
        slots.append(
            HCRSlot(
                name=field,
                field=field,
                kind="tristate",
                token_offset=token_offset,
                character_offset=character_offset,
            )
        )
    if len(slots) != 9 or len({slot.token_offset for slot in slots}) != 9:
        raise ValueError("Tri-state HCR must contain 9 distinct decision slots")
    return tuple(slots)


def compose_active_hcr_score(
    global_logit: float,
    requirement_logits: Sequence[float],
    satisfaction_logits: Sequence[float],
    *,
    native_weight: float,
    logical_weight: float,
    reduction: Literal["sum", "mean"],
) -> tuple[float, float, list[float]]:
    """Return final score, logical score, and per-field log-pass values."""

    if len(requirement_logits) != len(HCR_FIELDS) or len(satisfaction_logits) != len(
        HCR_FIELDS
    ):
        raise ValueError(f"Active HCR requires {len(HCR_FIELDS)} fields")
    values = [float(global_logit), *requirement_logits, *satisfaction_logits]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("HCR logits must all be finite")
    if native_weight < 0.0 or logical_weight < 0.0:
        raise ValueError("HCR score weights must be non-negative")
    if native_weight == 0.0 and logical_weight == 0.0:
        raise ValueError("At least one HCR score weight must be positive")
    # Match the trainer's explicit float32 composition. vLLM returns Python
    # floats, and silently retaining float64 here can perturb near ties.
    requirement = torch.tensor(requirement_logits, dtype=torch.float32)
    satisfaction = torch.tensor(satisfaction_logits, dtype=torch.float32)
    field_log_pass_tensor = torch.logaddexp(
        F.logsigmoid(-requirement),
        F.logsigmoid(requirement) + F.logsigmoid(satisfaction),
    )
    if reduction == "sum":
        logical_score_tensor = field_log_pass_tensor.sum()
    elif reduction == "mean":
        logical_score_tensor = field_log_pass_tensor.mean()
    else:
        raise ValueError(f"Unknown HCR logical reduction {reduction!r}")
    score_tensor = (
        torch.tensor(float(global_logit), dtype=torch.float32) * native_weight
        + logical_score_tensor * logical_weight
    )
    logical_score = float(logical_score_tensor.item())
    field_log_pass = [float(value) for value in field_log_pass_tensor.tolist()]
    score = float(score_tensor.item())
    if not math.isfinite(score):
        raise ValueError("Composite HCR score is non-finite")
    return score, logical_score, field_log_pass


def compose_tristate_hcr_score(
    global_logit: float,
    field_class_logits: Sequence[Sequence[float]],
    *,
    native_weight: float,
    logical_weight: float,
    reduction: Literal["sum", "mean"],
) -> tuple[float, float, list[float]]:
    """Compose the trainer-identical ``sum log(1-P(violated))`` score."""

    if len(field_class_logits) != len(HCR_FIELDS) or any(
        len(values) != 3 for values in field_class_logits
    ):
        raise ValueError(f"Tri-state HCR requires {len(HCR_FIELDS)}x3 logits")
    values = [float(global_logit)] + [
        float(value) for row in field_class_logits for value in row
    ]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("HCR logits must all be finite")
    if native_weight < 0.0 or logical_weight < 0.0:
        raise ValueError("HCR score weights must be non-negative")
    if native_weight == 0.0 and logical_weight == 0.0:
        raise ValueError("At least one HCR score weight must be positive")
    logits = torch.tensor(field_class_logits, dtype=torch.float32)
    field_log_pass_tensor = (
        torch.logsumexp(logits[:, :2], dim=-1)
        - torch.logsumexp(logits, dim=-1)
    )
    if reduction == "sum":
        logical_score_tensor = field_log_pass_tensor.sum()
    elif reduction == "mean":
        logical_score_tensor = field_log_pass_tensor.mean()
    else:
        raise ValueError(f"Unknown HCR logical reduction {reduction!r}")
    score_tensor = (
        torch.tensor(float(global_logit), dtype=torch.float32) * native_weight
        + logical_score_tensor * logical_weight
    )
    score = float(score_tensor.item())
    if not math.isfinite(score):
        raise ValueError("Composite HCR score is non-finite")
    return (
        score,
        float(logical_score_tensor.item()),
        [float(value) for value in field_log_pass_tensor.tolist()],
    )
