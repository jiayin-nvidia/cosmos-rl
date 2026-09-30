"""Cosmos-3 Reasoner (Nano) reranker."""

from __future__ import annotations

import base64
import hashlib
import math
from typing import Any, Mapping, Sequence

from PIL import Image

from ...config import RerankerConfig
from ..base import Candidate, CandidateScore
from ..frames import parallel_map, sampled_video_array
from .common import (VLMReranker, answer_tail, homogeneous_candidate_modality,
                     output_text, rationale_sampling_params,
                     read_output_logits, read_results, read_sampling_params)
from .hcr import (HCR_ACTIVE_RESPONSE, HCR_ACTIVE_RESPONSE_SHA256, HCR_FIELDS,
                  HCRSlot, build_hcr_slots, compose_active_hcr_score)

_NANO_REPO = "nvidia/Cosmos3-Nano"

ATOMIC_CONSTRAINT_FIELDS = (
    "top_outer_color",
    "top_outer_type",
    "bottom_color",
    "bottom_type",
    "shoe_color",
    "shoe_type",
    "viewpoint",
    "accessory_subset",
)

ATOMIC_CONSTRAINT_DESCRIPTIONS = {
    "top_outer_color": "upper-body clothing color",
    "top_outer_type": "upper-body clothing type",
    "bottom_color": "lower-body clothing color",
    "bottom_type": "lower-body clothing type",
    "shoe_color": "shoe color",
    "shoe_type": "shoe type",
    "viewpoint": "viewpoint",
    "accessory_subset": "accessories",
}


def atomic_constraint_prompt(query: str, field: str) -> str:
    """Match the exact single-field instruction used by atomic LoRA tuning."""

    try:
        readable = ATOMIC_CONSTRAINT_DESCRIPTIONS[field]
    except KeyError as exc:
        raise ValueError(f"unknown atomic constraint field: {field!r}") from exc
    return (
        f'Query: "{query}"\nConsidering only the query\'s {readable} '
        "requirement, does the person in the image match that requirement? "
        "Ignore every other detail. Answer with <answer>yes</answer> or "
        "<answer>no</answer>."
    )


def _log_sigmoid(value: float) -> float:
    return -math.log1p(math.exp(-abs(value))) + min(value, 0.0)


def compose_atomic_constraint_score(
    margins: Sequence[float], *, reduction: str, temperature: float
) -> float:
    """Turn independently measured requirement margins into a strict-AND score."""

    if not margins:
        raise ValueError("atomic constraint scoring requires at least one field")
    if temperature <= 0:
        raise ValueError("atomic constraint temperature must be positive")
    scaled = [float(value) / temperature for value in margins]
    if reduction == "sum_logprob":
        return sum(_log_sigmoid(value) for value in scaled)
    if reduction == "mean_logprob":
        return sum(_log_sigmoid(value) for value in scaled) / len(scaled)
    if reduction == "mean_margin":
        return sum(scaled) / len(scaled)
    if reduction == "min_margin":
        return min(scaled)
    if reduction == "softmin_margin":
        # A smooth minimum whose query-constant log(field-count) offset cannot
        # affect the ordering of candidates for the same query.
        lowest = min(scaled)
        return lowest - math.log(sum(math.exp(lowest - value) for value in scaled))
    raise ValueError(f"unknown atomic constraint reduction: {reduction!r}")


class CosmosReasonReranker(VLMReranker):
    """Pointwise CR3 reranker using offline vLLM chat."""

    name = "cosmos_reason"

    def __init__(
        self,
        cfg: RerankerConfig,
        *,
        default_model_id: str = _NANO_REPO,
        name: str | None = None,
    ) -> None:
        super().__init__(cfg, model_id=default_model_id, name=name)
        self.jpeg_quality = int(cfg.options.get("jpeg_quality", 95))
        self.lora_path = str(cfg.options.get("lora_path") or "").strip()
        image_min_pixels = cfg.options.get("image_min_pixels")
        image_max_pixels = cfg.options.get("image_max_pixels")
        image_processor_kwargs: dict[str, int] = {}
        if image_min_pixels is not None:
            image_processor_kwargs["min_pixels"] = int(image_min_pixels)
        if image_max_pixels is not None:
            image_processor_kwargs["max_pixels"] = int(image_max_pixels)
        self.image_mm_processor_kwargs = image_processor_kwargs or None
        self.video_mm_processor_kwargs = {"do_sample_frames": False}
        self.media_io_kwargs = {
            "video": {
                "num_frames": -1,
                "fps": self.fps,
                "do_sample_frames": False,
            }
        }

        from vllm import LLM

        self.llm = LLM(
            model=self.model_id,
            limit_mm_per_prompt={"image": 1, "video": 1},
            media_io_kwargs=self.media_io_kwargs,
            gpu_memory_utilization=float(cfg.options.get("gpu_memory_utilization", 0.85)),
            tensor_parallel_size=int(cfg.options.get("tensor_parallel_size", 1)),
            max_model_len=int(cfg.options.get("max_model_len", 32768)),
            enforce_eager=bool(cfg.options.get("enforce_eager", False)),
            hf_overrides=cfg.options.get("hf_overrides"),
            enable_lora=bool(self.lora_path),
            enable_tower_connector_lora=bool(
                cfg.options.get("enable_tower_connector_lora", False)
            ),
            max_lora_rank=int(cfg.options.get("max_lora_rank", 16)),
            # vLLM V1 computes returned top-k logprobs from raw vocabulary
            # logits by default, before ``allowed_token_ids`` is applied.  On
            # releases without targeted ``logprob_token_ids`` support that can
            # omit one of CR3's yes/no decision tokens entirely.  The
            # processed mode reports the constrained yes/no distribution; its
            # log-probability difference is the same raw yes-minus-no logit
            # margin because the shared normalization term cancels.
            logprobs_mode=str(cfg.options.get("logprobs_mode", "raw_logprobs")),
        )

        if self.lora_path:
            from vllm.lora.request import LoRARequest

            self.lora_request = LoRARequest("pas_cr3", 1, self.lora_path)
        else:
            self.lora_request = None

        self.set_score_token_ids(self.llm.get_tokenizer())
        self.atomic_constraint_and = bool(
            cfg.options.get("atomic_constraint_and", False)
        )
        self.atomic_constraint_reduction = str(
            cfg.options.get("atomic_constraint_reduction", "sum_logprob")
        )
        self.atomic_constraint_temperature = float(
            cfg.options.get("atomic_constraint_temperature", 1.0)
        )
        self.atomic_explicit_values = bool(
            cfg.options.get("atomic_explicit_values", False)
        )
        self.query_likelihood = bool(cfg.options.get("query_likelihood", False))
        # Label-independent teacher-forced prefix used by pre-decision HCR.
        # The final yes/no is still the only deployed scalar; this merely makes
        # inference expose the same causal hidden-state trajectory as training.
        self.predecision_response_prefix = str(
            cfg.options.get("predecision_response_prefix") or ""
        )
        self.predecision_response_full = str(
            cfg.options.get("predecision_response_full") or ""
        )
        expected_predecision_hash = str(
            cfg.options.get("predecision_response_sha256") or ""
        )
        if self.predecision_response_prefix:
            if self.cot or self.output_format != "logit_delta":
                raise ValueError(
                    "Predecision scoring requires reasoning=none and output_format=logit_delta"
                )
            if str(cfg.options.get("logprobs_mode", "raw_logprobs")) != "processed_logprobs":
                raise ValueError(
                    "Predecision diagnostics require processed_logprobs for the "
                    "audited constrained-continuation normalization"
                )
            if not self.predecision_response_prefix.endswith("<answer>"):
                raise ValueError("Predecision prefix must end immediately before yes/no")
            actual_hash = hashlib.sha256(
                self.predecision_response_prefix.encode("utf-8")
            ).hexdigest()
            if actual_hash != expected_predecision_hash:
                raise ValueError(
                    "Predecision response-prefix hash mismatch: "
                    f"expected={expected_predecision_hash!r} actual={actual_hash!r}"
                )
            if not self.predecision_response_full.startswith(
                self.predecision_response_prefix
            ) or self.predecision_response_full != self.predecision_response_prefix + "?</answer>":
                raise ValueError("Predecision full response/prefix mismatch")
            tokenizer = self.llm.get_tokenizer()
            self.predecision_response_ids = tuple(
                int(x) for x in tokenizer.encode(
                    self.predecision_response_full, add_special_tokens=False
                )
            )
            self.predecision_class_ids = tuple(
                int(tokenizer.encode(x, add_special_tokens=False)[0])
                for x in (" inactive", " satisfied", " violated")
            )
            if any(
                len(tokenizer.encode(x, add_special_tokens=False)) != 1
                for x in (" inactive", " satisfied", " violated")
            ):
                raise ValueError("Predecision semantic classes must be single tokens")
            self.predecision_field_offsets = []
            self.predecision_field_prefixes = []
            for marker in (
                "top_color:", "top_type:", "bottom_color:", "bottom_type:",
                "shoe_color:", "shoe_type:", "viewpoint:", "accessory:",
            ):
                marker_ids = tuple(
                    int(x) for x in tokenizer.encode(marker, add_special_tokens=False)
                )
                starts = [
                    i for i in range(len(self.predecision_response_ids) - len(marker_ids) + 1)
                    if self.predecision_response_ids[i:i + len(marker_ids)] == marker_ids
                ]
                if len(starts) != 1:
                    raise ValueError(f"Predecision marker mismatch: {marker}")
                token_offset = starts[0] + len(marker_ids)
                self.predecision_field_offsets.append(token_offset)
                text_needle = marker + " ?"
                if self.predecision_response_full.count(text_needle) != 1:
                    raise ValueError(f"Predecision text marker mismatch: {marker}")
                character_offset = (
                    self.predecision_response_full.index(text_needle) + len(marker)
                )
                field_prefix = self.predecision_response_full[:character_offset]
                if tuple(
                    int(x) for x in tokenizer.encode(field_prefix, add_special_tokens=False)
                ) != self.predecision_response_ids[:token_offset]:
                    raise ValueError(
                        f"Predecision fallback prefix tokenization changed: {marker}"
                    )
                self.predecision_field_prefixes.append(field_prefix)
            pos_ids = tokenizer.encode("<answer>yes</answer>", add_special_tokens=False)
            neg_ids = tokenizer.encode("<answer>no</answer>", add_special_tokens=False)
            decision_offset = next(i for i, (a, b) in enumerate(zip(pos_ids, neg_ids)) if a != b)
            answer_ids = tuple(
                int(x) for x in tokenizer.encode("<answer>?</answer>", add_special_tokens=False)
            )
            starts = [
                i for i in range(len(self.predecision_response_ids) - len(answer_ids) + 1)
                if self.predecision_response_ids[i:i + len(answer_ids)] == answer_ids
            ]
            if len(starts) != 1:
                raise ValueError("Predecision answer marker mismatch")
            self.predecision_answer_offset = starts[0] + decision_offset
            self.predecision_yes_no_ids = (int(pos_ids[decision_offset]), int(neg_ids[decision_offset]))
            answer_needle = "<answer>?"
            if self.predecision_response_full.count(answer_needle) != 1:
                raise ValueError("Predecision answer text marker mismatch")
            answer_character_offset = (
                self.predecision_response_full.index(answer_needle) + len("<answer>")
            )
            self.predecision_answer_prefix = self.predecision_response_full[
                :answer_character_offset
            ]
            if tuple(
                int(x)
                for x in tokenizer.encode(
                    self.predecision_answer_prefix, add_special_tokens=False
                )
            ) != self.predecision_response_ids[: self.predecision_answer_offset]:
                raise ValueError(
                    "Predecision answer fallback prefix tokenization changed"
                )
        self.query_likelihood_prompt = str(
            cfg.options.get(
                "query_likelihood_prompt",
                "Describe the person's visible clothing, shoes, accessories, and "
                "viewpoint precisely.",
            )
        ).strip()
        if self.query_likelihood:
            if self.cot or self.output_format != "logit_delta":
                raise ValueError(
                    "Query-likelihood scoring requires reasoning=none and "
                    "output_format=logit_delta"
                )
            if not self.query_likelihood_prompt:
                raise ValueError("Query-likelihood prompt cannot be empty")
        self.atomic_fields_by_query: dict[str, tuple[str, ...]] = {}
        self.atomic_values_by_query: dict[str, dict[str, str]] = {}
        if self.atomic_constraint_and:
            if self.cot or self.output_format != "logit_delta":
                raise ValueError(
                    "Atomic constraint AND requires reasoning=none and "
                    "output_format=logit_delta"
                )
            compose_atomic_constraint_score(
                [0.0],
                reduction=self.atomic_constraint_reduction,
                temperature=self.atomic_constraint_temperature,
            )
        self.hcr_active_logical = bool(cfg.options.get("hcr_active_logical", False))
        self.hcr_native_score_weight = float(
            cfg.options.get("hcr_native_score_weight", 1.0)
        )
        self.hcr_logical_score_weight = float(
            cfg.options.get("hcr_logical_score_weight", 1.0)
        )
        self.hcr_logical_reduction = str(
            cfg.options.get("hcr_logical_reduction", "sum")
        )
        self.hcr_prompt_logprobs = int(cfg.options.get("hcr_prompt_logprobs", 20))
        expected_template_hash = str(
            cfg.options.get("hcr_expected_template_sha256") or ""
        )
        if self.hcr_active_logical:
            if self.cot or self.output_format != "logit_delta":
                raise ValueError(
                    "Active HCR requires reasoning=none and output_format=logit_delta"
                )
            if self.hcr_logical_reduction not in {"sum", "mean"}:
                raise ValueError("HCR logical reduction must be 'sum' or 'mean'")
            if not 1 <= self.hcr_prompt_logprobs <= 20:
                raise ValueError("HCR prompt-logprobs count must be in [1, 20]")
            if expected_template_hash != HCR_ACTIVE_RESPONSE_SHA256:
                raise ValueError(
                    "Active HCR response hash disagrees with the training-side "
                    f"template: expected={expected_template_hash!r}, "
                    f"evaluator={HCR_ACTIVE_RESPONSE_SHA256!r}"
                )
            compose_active_hcr_score(
                0.0,
                [0.0] * len(HCR_FIELDS),
                [0.0] * len(HCR_FIELDS),
                native_weight=self.hcr_native_score_weight,
                logical_weight=self.hcr_logical_score_weight,
                reduction=self.hcr_logical_reduction,
            )
            tokenizer = self.llm.get_tokenizer()
            positive_ids = tokenizer.encode(
                "<answer>yes</answer>", add_special_tokens=False
            )
            negative_ids = tokenizer.encode(
                "<answer>no</answer>", add_special_tokens=False
            )
            shared = min(len(positive_ids), len(negative_ids))
            decision_offset = next(
                (
                    index
                    for index in range(shared)
                    if positive_ids[index] != negative_ids[index]
                ),
                shared,
            )
            if decision_offset >= shared:
                raise ValueError("Configured yes/no responses have no decision token")
            self.hcr_yes_token_id = int(positive_ids[decision_offset])
            self.hcr_no_token_id = int(negative_ids[decision_offset])
            self.hcr_response_token_ids = tuple(
                int(token_id)
                for token_id in tokenizer.encode(
                    HCR_ACTIVE_RESPONSE, add_special_tokens=False
                )
            )
            self.hcr_slots = build_hcr_slots(tokenizer, decision_offset)

    def _video_data_url(self, candidate: Candidate) -> str:
        sample = sampled_video_array(candidate, self.fps, self.max_frames)

        import cv2

        encoded_frames: list[str] = []
        for frame in sample.frames:
            ok, buffer = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                raise ValueError("Could not JPEG-encode CR3 sampled frame")
            encoded_frames.append(base64.b64encode(buffer).decode("ascii"))
        return "data:video/jpeg;base64," + ",".join(encoded_frames)

    def _image_data_url(self, candidate: Candidate) -> str:
        if candidate.image_path is None:
            raise ValueError("Candidate has no image_path for CR3 image scoring")
        with Image.open(candidate.image_path) as image:
            media_type = Image.MIME.get(image.format or "")
        if not media_type:
            raise ValueError(f"Could not determine image MIME type: {candidate.image_path}")
        encoded = base64.b64encode(candidate.image_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _messages(self, query: str, candidate: Candidate) -> list[dict[str, Any]]:
        if candidate.image_path is not None:
            image_url = self._image_data_url(candidate)
            prompt = self.image_prompt_text(query)
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        video_url = self._video_data_url(candidate)
        prompt = self.prompt_text(query)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def set_atomic_fields_by_query(
        self, fields_by_query: Mapping[str, Sequence[str]]
    ) -> None:
        """Install query-only constraint applicability for decomposed scoring."""

        normalized: dict[str, tuple[str, ...]] = {}
        allowed = set(ATOMIC_CONSTRAINT_FIELDS)
        for query, raw_fields in fields_by_query.items():
            fields = tuple(dict.fromkeys(str(field) for field in raw_fields))
            unknown = set(fields) - allowed
            if unknown:
                raise ValueError(f"unknown atomic fields for query {query!r}: {unknown}")
            if not fields:
                raise ValueError(f"query has no active atomic fields: {query!r}")
            normalized[str(query)] = fields
        self.atomic_fields_by_query = normalized

    def set_atomic_values_by_query(
        self, values_by_query: Mapping[str, Mapping[str, str]]
    ) -> None:
        """Install canonical explicit target values inferred from query text."""
        self.atomic_values_by_query = {
            str(query): {str(field): str(value) for field, value in values.items()}
            for query, values in values_by_query.items()
        }

    def _atomic_messages(
        self, query: str, candidate: Candidate, field: str
    ) -> list[dict[str, Any]]:
        messages = self._messages(query, candidate)
        content = messages[0]["content"]
        if not isinstance(content, list) or not content or content[-1].get("type") != "text":
            raise RuntimeError("unexpected CR3 multimodal message structure")
        if self.atomic_explicit_values and field != "accessory_subset":
            values = self.atomic_values_by_query.get(query, {})
            value = values.get(field)
            if value:
                try:
                    from examples.pas_reranker.atomic_query_fields import explicit_atomic_prompt
                except ModuleNotFoundError:
                    from atomic_query_fields import explicit_atomic_prompt  # type: ignore[no-redef]
                content[-1]["text"] = explicit_atomic_prompt(field, value)
            else:
                # Query-only parsing is intentionally conservative: some PAS
                # phrases identify an active field without mapping cleanly to
                # a canonical scalar value (for example ``coat`` or ``seen
                # from the back``).  Keep those requirements in the AND by
                # falling back to the original field-only prompt.  Dropping
                # them would weaken conjunction semantics, while raising here
                # would make a fully deployable query parser impossible.
                content[-1]["text"] = atomic_constraint_prompt(query, field)
        else:
            content[-1]["text"] = atomic_constraint_prompt(query, field)
        return messages

    def _query_likelihood_messages(
        self, query: str, candidate: Candidate
    ) -> list[dict[str, Any]]:
        """Render an image-only request followed by the query as its target.

        The query must not appear in the user message: otherwise a language
        model can score it by copying text without looking at the image.
        """

        messages = self._messages(query, candidate)
        content = messages[0]["content"]
        if not isinstance(content, list) or not content or content[-1].get("type") != "text":
            raise RuntimeError("unexpected CR3 multimodal message structure")
        content[-1]["text"] = self.query_likelihood_prompt
        messages.append({"role": "assistant", "content": query})
        return messages

    def _score_query_likelihood_pairs(
        self, pairs: list[tuple[str, Candidate]]
    ) -> list[CandidateScore]:
        """Score every image by teacher-forced mean log p(query | image)."""

        from vllm import SamplingParams

        messages_list = [
            self._query_likelihood_messages(query, candidate)
            for query, candidate in pairs
        ]
        outputs = self._chat(
            messages_list,
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                prompt_logprobs=1,
                flat_logprobs=True,
            ),
            mm_processor_kwargs=self.image_mm_processor_kwargs,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        tokenizer = self.llm.get_tokenizer()
        results: list[CandidateScore] = []
        for (query, _candidate), output in zip(pairs, outputs, strict=True):
            prompt_ids = [int(token_id) for token_id in (output.prompt_token_ids or [])]
            prompt_logprobs = output.prompt_logprobs
            target_ids = tuple(
                int(token_id)
                for token_id in tokenizer.encode(query, add_special_tokens=False)
            )
            if not target_ids:
                raise ValueError("Query-likelihood target cannot tokenize to empty")
            target_start = self._unique_subsequence(prompt_ids, target_ids)
            values: list[float] = []
            for position_index, token_id in enumerate(
                target_ids, start=target_start
            ):
                value = self._logprob_value(prompt_logprobs[position_index], token_id)
                if value is None:
                    raise RuntimeError(
                        "vLLM did not return the teacher-forced target token logprob"
                    )
                values.append(value)
            score = sum(values) / len(values)
            results.append(
                CandidateScore(
                    score,
                    {
                        "readout": "query_likelihood_v1",
                        "target_tokens": len(values),
                        "mean_query_logprob": score,
                    },
                )
            )
        return results

    def _score_atomic_pairs(
        self, pairs: list[tuple[str, Candidate]]
    ) -> list[CandidateScore]:
        if not self.atomic_fields_by_query:
            raise RuntimeError(
                "Atomic constraint AND needs query applicability metadata before scoring"
            )
        flat_messages: list[list[dict[str, Any]]] = []
        spans: list[tuple[tuple[str, ...], int, int]] = []
        for query, candidate in pairs:
            try:
                fields = self.atomic_fields_by_query[query]
            except KeyError as exc:
                raise KeyError(f"no atomic fields registered for query {query!r}") from exc
            begin = len(flat_messages)
            flat_messages.extend(
                self._atomic_messages(query, candidate, field) for field in fields
            )
            spans.append((fields, begin, len(flat_messages)))

        raw = self._score_batch(
            flat_messages, mm_processor_kwargs=self.image_mm_processor_kwargs
        )
        results: list[CandidateScore] = []
        for fields, begin, end in spans:
            field_results = raw[begin:end]
            if len(field_results) != len(fields):
                raise RuntimeError("CR3 returned the wrong number of atomic scores")
            if any(score is None for score, _meta in field_results):
                results.append(
                    CandidateScore(
                        None,
                        {
                            "readout": "atomic_constraint_and_v1",
                            "fields": list(fields),
                            "error": "one or more atomic field scores failed",
                        },
                    )
                )
                continue
            margins = [float(score) for score, _meta in field_results]
            composite = compose_atomic_constraint_score(
                margins,
                reduction=self.atomic_constraint_reduction,
                temperature=self.atomic_constraint_temperature,
            )
            results.append(
                CandidateScore(
                    composite,
                    {
                        "readout": "atomic_constraint_and_v1",
                        "field_margins": dict(zip(fields, margins, strict=True)),
                        "reduction": self.atomic_constraint_reduction,
                        "temperature": self.atomic_constraint_temperature,
                        "composite_score": composite,
                    },
                )
            )
        return results

    def score_pairs(self, pairs: list[tuple[str, Candidate]]) -> list[CandidateScore]:
        if not pairs:
            return []
        if self.query_likelihood:
            return self._score_query_likelihood_pairs(pairs)
        if self.atomic_constraint_and:
            return self._score_atomic_pairs(pairs)
        modality = homogeneous_candidate_modality([candidate for _query, candidate in pairs])
        messages = parallel_map(lambda item: self._messages(item[0], item[1]), pairs)
        mm_processor_kwargs = (
            self.video_mm_processor_kwargs
            if modality == "video"
            else self.image_mm_processor_kwargs
        )
        return self.candidate_scores(
            self._score_batch(messages, mm_processor_kwargs=mm_processor_kwargs)
        )

    def _score_batch(
        self,
        messages_list: list[list[dict[str, Any]]],
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[float | None, dict]]:
        if not messages_list:
            return []

        if self.hcr_active_logical:
            return self._score_hcr_batch(
                messages_list, mm_processor_kwargs=mm_processor_kwargs
            )

        if self.predecision_response_prefix:
            from vllm import SamplingParams

            # vLLM 0.23 returns only the teacher-forced token for V1 prompt
            # logprobs, regardless of requested top-k.  Read every checklist
            # position as an exact constrained one-token continuation instead.
            # Each prefix is byte/token-identical to the neutral training
            # trajectory before that position; batching all eight fields also
            # lets vLLM reuse the image/query prefix cache.
            field_messages = [
                [*messages, {"role": "assistant", "content": field_prefix}]
                for messages in messages_list
                for field_prefix in self.predecision_field_prefixes
            ]
            field_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                allowed_token_ids=list(self.predecision_class_ids),
                logprob_token_ids=list(self.predecision_class_ids),
            )
            field_outputs = self._chat(
                field_messages,
                field_params,
                mm_processor_kwargs=mm_processor_kwargs,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            answer_messages = [
                [
                    *messages,
                    {"role": "assistant", "content": self.predecision_answer_prefix},
                ]
                for messages in messages_list
            ]
            answer_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                allowed_token_ids=list(self.predecision_yes_no_ids),
                logprob_token_ids=list(self.predecision_yes_no_ids),
            )
            answer_outputs = self._chat(
                answer_messages,
                answer_params,
                mm_processor_kwargs=mm_processor_kwargs,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            expected_fields = len(messages_list) * len(self.predecision_field_prefixes)
            if len(field_outputs) != expected_fields or len(answer_outputs) != len(messages_list):
                raise RuntimeError("vLLM returned the wrong number of predecision fallbacks")

            def generated_values(output, ids):
                completions = getattr(output, "outputs", None)
                if not completions or not getattr(completions[0], "logprobs", None):
                    raise RuntimeError("vLLM omitted constrained continuation logprobs")
                position = completions[0].logprobs[0]
                values = [self._logprob_value(position, token_id) for token_id in ids]
                if any(value is None for value in values):
                    raise RuntimeError(
                        "vLLM omitted an explicitly requested continuation token"
                    )
                return [float(value) for value in values]

            prefix_hash = hashlib.sha256(
                self.predecision_response_prefix.encode("utf-8")
            ).hexdigest()
            results = []
            field_count = len(self.predecision_field_prefixes)
            for row_index, answer_output in enumerate(answer_outputs):
                native = generated_values(answer_output, self.predecision_yes_no_ids)
                begin = row_index * field_count
                classes = [
                    generated_values(output, self.predecision_class_ids)
                    for output in field_outputs[begin : begin + field_count]
                ]
                native_score = native[0] - native[1]
                results.append((native_score, {
                    "readout": "predecision_native_binary_with_tristate_diagnostics_v2",
                    "predecision_response_prefix_sha256": prefix_hash,
                    "predecision_class_logprobs": classes,
                    "global_logit": native_score,
                    "exact_adaptive_continuation_logprobs": True,
                    "fallback_prefixes_per_candidate": field_count + 1,
                }))
            return results

        if self.cot:
            rationale_params = rationale_sampling_params(self.max_think_tokens)
            rationale_outputs = self._chat(
                messages_list,
                rationale_params,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            rationales = [output_text(output) for output in rationale_outputs]
        else:
            rationales = ["" for _ in messages_list]

        read = self.spec["read"]
        prefix = self.spec["prefix"]
        read_messages_list = [
            [
                *messages,
                {"role": "assistant", "content": answer_tail(rationale, cot=self.cot) + prefix},
            ]
            for messages, rationale in zip(messages_list, rationales)
        ]

        params = read_sampling_params(read, self.score_ids)
        outputs = self._chat(
            read_messages_list,
            params,
            mm_processor_kwargs=mm_processor_kwargs,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        return read_results(read, outputs, self.score_ids, self.spec, rationales)

    @staticmethod
    def _unique_subsequence(values: list[int], needle: tuple[int, ...]) -> int:
        matches = [
            start
            for start in range(len(values) - len(needle) + 1)
            if tuple(values[start : start + len(needle)]) == needle
        ]
        if len(matches) != 1:
            raise ValueError(
                "Canonical active-HCR response must occur exactly once in the "
                f"rendered prompt; found {len(matches)} copies"
            )
        return matches[0]

    @staticmethod
    def _logprob_value(position: Any, token_id: int) -> float | None:
        if not isinstance(position, dict) or token_id not in position:
            return None
        item = position[token_id]
        value = item.get("logprob") if isinstance(item, dict) else getattr(item, "logprob", None)
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    def _prompt_slot_margins(self, output: Any) -> dict[str, float]:
        prompt_ids = [int(token_id) for token_id in (output.prompt_token_ids or [])]
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(prompt_ids):
            raise ValueError("vLLM returned missing or misaligned HCR prompt logprobs")
        response_start = self._unique_subsequence(
            prompt_ids, self.hcr_response_token_ids
        )
        margins: dict[str, float] = {}
        for slot in self.hcr_slots:
            position_index = response_start + slot.token_offset
            if not 0 <= position_index < len(prompt_logprobs):
                raise ValueError(f"HCR slot {slot.name!r} lies outside the prompt")
            position = prompt_logprobs[position_index]
            yes = self._logprob_value(position, self.hcr_yes_token_id)
            no = self._logprob_value(position, self.hcr_no_token_id)
            if yes is not None and no is not None:
                margins[slot.name] = yes - no
        return margins

    def _score_hcr_batch(
        self,
        messages_list: list[list[dict[str, Any]]],
        *,
        mm_processor_kwargs: dict[str, Any] | None,
    ) -> list[tuple[float | None, dict]]:
        """Teacher-force the neutral trajectory and read all 17 decisions."""

        from vllm import SamplingParams

        full_messages = [
            [*messages, {"role": "assistant", "content": HCR_ACTIVE_RESPONSE}]
            for messages in messages_list
        ]
        prompt_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=self.hcr_prompt_logprobs,
            flat_logprobs=True,
        )
        outputs = self._chat(
            full_messages,
            prompt_params,
            mm_processor_kwargs=mm_processor_kwargs,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        if len(outputs) != len(messages_list):
            raise RuntimeError("vLLM returned the wrong number of HCR outputs")
        margins = [self._prompt_slot_margins(output) for output in outputs]

        # A prompt target is always returned, but yes/no alternatives can fall
        # outside top-k. Re-read only those positions as exact one-token
        # continuations; the prefix is the same neutral training trajectory.
        missing: list[tuple[int, HCRSlot]] = [
            (row_index, slot)
            for row_index, row in enumerate(margins)
            for slot in self.hcr_slots
            if slot.name not in row
        ]
        fallback_counts = [0] * len(messages_list)
        if missing:
            fallback_messages = [
                [
                    *messages_list[row_index],
                    {
                        "role": "assistant",
                        "content": HCR_ACTIVE_RESPONSE[: slot.character_offset],
                    },
                ]
                for row_index, slot in missing
            ]
            fallback_params = read_sampling_params(
                "margin", [self.hcr_yes_token_id, self.hcr_no_token_id]
            )
            fallback_outputs = self._chat(
                fallback_messages,
                fallback_params,
                mm_processor_kwargs=mm_processor_kwargs,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            if len(fallback_outputs) != len(missing):
                raise RuntimeError("vLLM returned the wrong number of HCR fallbacks")
            fallback_spec = {"read": "margin", "pos": "yes", "neg": "no"}
            for (row_index, slot), fallback_output in zip(
                missing, fallback_outputs, strict=True
            ):
                fallback_prompt_ids = tuple(
                    int(token_id)
                    for token_id in (fallback_output.prompt_token_ids or [])
                )
                expected_prefix = self.hcr_response_token_ids[: slot.token_offset]
                if (
                    len(fallback_prompt_ids) < len(expected_prefix)
                    or fallback_prompt_ids[-len(expected_prefix) :] != expected_prefix
                ):
                    raise RuntimeError(
                        f"Rendered HCR fallback prefix changed for {slot.name!r}"
                    )
                score, meta = read_output_logits(
                    "margin",
                    fallback_output,
                    [self.hcr_yes_token_id, self.hcr_no_token_id],
                    fallback_spec,
                    "",
                )
                if score is None or not math.isfinite(float(score)):
                    raise RuntimeError(
                        f"Could not recover exact HCR slot {slot.name!r}: {meta}"
                    )
                margins[row_index][slot.name] = float(score)
                fallback_counts[row_index] += 1

        results: list[tuple[float | None, dict]] = []
        for row_index, row in enumerate(margins):
            if set(row) != {slot.name for slot in self.hcr_slots}:
                raise RuntimeError("Active HCR did not produce all 17 score slots")
            global_logit = row["global"]
            requirement_logits = [
                row[f"{field}_requirement"] for field in HCR_FIELDS
            ]
            satisfaction_logits = [
                row[f"{field}_satisfaction"] for field in HCR_FIELDS
            ]
            score, logical_score, field_log_pass = compose_active_hcr_score(
                global_logit,
                requirement_logits,
                satisfaction_logits,
                native_weight=self.hcr_native_score_weight,
                logical_weight=self.hcr_logical_score_weight,
                reduction=self.hcr_logical_reduction,
            )
            results.append(
                (
                    score,
                    {
                        "readout": "active_hcr_logical_v1",
                        "template_sha256": HCR_ACTIVE_RESPONSE_SHA256,
                        "global_logit": global_logit,
                        "requirement_logits": dict(zip(HCR_FIELDS, requirement_logits)),
                        "satisfaction_logits": dict(zip(HCR_FIELDS, satisfaction_logits)),
                        "field_log_pass": dict(zip(HCR_FIELDS, field_log_pass)),
                        "logical_score": logical_score,
                        "native_score_weight": self.hcr_native_score_weight,
                        "logical_score_weight": self.hcr_logical_score_weight,
                        "logical_reduction": self.hcr_logical_reduction,
                        "composite_score": score,
                        "prompt_logprobs": self.hcr_prompt_logprobs,
                        "fallback_slots": fallback_counts[row_index],
                    },
                )
            )
        return results

    def _chat(
        self,
        messages_list: list[list[dict[str, Any]]],
        sampling_params,
        *,
        mm_processor_kwargs: dict[str, Any] | None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ):
        kwargs: dict[str, Any] = {}
        if mm_processor_kwargs is not None:
            kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        return self.llm.chat(
            messages_list,
            sampling_params,
            use_tqdm=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            lora_request=self.lora_request,
            **kwargs,
        )
