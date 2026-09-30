"""Shared output-format and readout helpers for generative VLM rerankers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from ...config import RerankerConfig
from ..base import Candidate, CandidateScore, ScoringReranker

RELEVANCE_PROMPT = (
    'You are judging whether a short video clip matches a search query.\n'
    'Query: "{query}"\n'
    "The video is one candidate clip. Decide whether it matches the query."
)

IMAGE_RELEVANCE_PROMPT = (
    'Query: "{query}"\n'
    "Does the person in the image fully match the query?"
)

IMAGE_PROMPT_PREFIXES = {
    "default": "",
    "exact_attributes": (
        "Judge exact person-attribute agreement. Treat similar colors and "
        "different garment types as distinct (for example, white versus beige "
        "and leggings versus pants). A match must agree with every stated "
        "color, garment type, and accessory.\n"
    ),
    "strict_conjunction": (
        "Check each stated requirement independently: upper-body clothing, "
        "lower-body clothing, footwear, accessories, and viewpoint when they "
        "are mentioned. Both the item type and its color must match exactly. "
        "Do not compensate for one mismatch with matches on other attributes. "
        "This is a strict AND decision: answer yes only if every stated "
        "requirement matches. If even one requirement is different or absent, "
        "answer no.\n"
    ),
}

COT_SUFFIX = (
    " Reason step by step about whether the clip matches, then give the verdict "
    "as <answer>yes</answer> or <answer>no</answer>."
)

OUTPUT_FORMATS: dict[str, dict[str, Any]] = {
    "logit_delta": {
        "suffix": " Answer with <answer>yes</answer> or <answer>no</answer>.",
        "prefix": "<answer>",
        "read": "margin",
        "pos": "yes",
        "neg": "no",
    },
    "verbalized_yes_no": {
        "suffix": " Answer with <answer>yes</answer> or <answer>no</answer>.",
        "prefix": "<answer>",
        "read": "gen_binary",
    },
    "digit_expected": {
        "suffix": " Give a relevance score 0-9 as 'Relevance score (0-9): N'.",
        "prefix": "Relevance score (0-9): ",
        "read": "expected",
        "tokens": list("0123456789"),
        "values": [float(i) for i in range(10)],
    },
    "relevance_1to5_expected": {
        "suffix": (
            " Assign a relevance score from 1 (not a match) to 5 (clear, "
            "full exact match). Answer only as 'Relevance score (1-5): N'."
        ),
        "prefix": "Relevance score (1-5): ",
        "read": "expected",
        "tokens": list("12345"),
        "values": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    "credibility_1to5": {
        "suffix": (
            " Assign a credibility score from 1 (not a match) to 5 (clear, "
            "full match). Answer only as 'Credibility score (1-5): N'."
        ),
        "prefix": "Credibility score (1-5): ",
        "read": "gen_1to5",
    },
    "verbalized_numeric": {
        "suffix": " Give a relevance score 0-100 as 'Relevance (0-100): N'.",
        "prefix": "Relevance (0-100): ",
        "read": "gen_numeric",
    },
}

COT_OUTPUT_FORMATS = {"logit_delta", "verbalized_yes_no"}

CandidateModality = Literal["image", "video"]


def homogeneous_candidate_modality(candidates: Sequence[Candidate]) -> CandidateModality:
    """Return the batch modality, rejecting empty or mixed candidate batches."""

    if not candidates:
        raise ValueError("Candidate batch must contain at least one candidate")
    image_flags = {candidate.image_path is not None for candidate in candidates}
    if len(image_flags) != 1:
        raise ValueError("Candidate batch mixes image and video inputs")
    return "image" if image_flags.pop() else "video"


def output_spec(output_format: str, *, cot: bool) -> dict[str, Any]:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"unknown output_format {output_format!r}; supported: {sorted(OUTPUT_FORMATS)}"
        )
    if cot and output_format not in COT_OUTPUT_FORMATS:
        raise ValueError(
            f"CoT supports only logit_delta/verbalized_yes_no, got {output_format!r}"
        )
    return OUTPUT_FORMATS[output_format]


def prompt_suffix(spec: dict[str, Any], *, cot: bool) -> str:
    return COT_SUFFIX if cot else str(spec["suffix"])


def first_token_id(tokenizer: Any, text: str) -> int:
    return tokenizer(text, add_special_tokens=False).input_ids[0]


def score_token_ids(tokenizer: Any, spec: dict[str, Any]) -> list[int]:
    read = spec["read"]
    if read == "margin":
        return [first_token_id(tokenizer, spec["pos"]), first_token_id(tokenizer, spec["neg"])]
    if read == "expected":
        return [first_token_id(tokenizer, token) for token in spec["tokens"]]
    return []


def answer_tail(rationale: str, *, cot: bool, context: str = "") -> str:
    tail = rationale
    if cot:
        close = tail.rfind("</think>")
        if close != -1:
            tail = tail[: close + len("</think>")]
    if "<think>" in context + tail and "</think>" not in context + tail:
        tail += "</think>"
    return tail


def output_text(output: Any) -> str:
    outputs = getattr(output, "outputs", None) or []
    if not outputs:
        return ""
    return str(getattr(outputs[0], "text", "") or "")


def output_logprobs(output: Any) -> Any | None:
    outputs = getattr(output, "outputs", None) or []
    if not outputs:
        return None
    logprobs_by_pos = getattr(outputs[0], "logprobs", None)
    if not logprobs_by_pos:
        return None
    return logprobs_by_pos[0]


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract_logprobs(logprobs: Any, token_ids: list[int]) -> dict[int, float]:
    wanted = set(token_ids)
    out: dict[int, float] = {}
    if not isinstance(logprobs, dict):
        return out
    for raw_token_id, item in logprobs.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError):
            continue
        if token_id in wanted:
            out[token_id] = float(_field(item, "logprob"))
    return out


def read_logits(
    read: str,
    logprobs: Any,
    score_ids: list[int],
    spec: dict[str, Any],
    rationale: str,
) -> tuple[float | None, dict]:
    values_by_id = extract_logprobs(logprobs, score_ids)
    if any(token_id not in values_by_id for token_id in score_ids):
        return (
            None,
            {
                "rationale": rationale,
                "error": "missing token logprob",
                "seen_token_ids": sorted(values_by_id),
            },
        )
    values = [values_by_id[token_id] for token_id in score_ids]
    if read == "margin":
        return (values[0] - values[1], {"rationale": rationale, "logprobs": values})

    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    probs = [value / total for value in exps]
    score = sum(prob * target for prob, target in zip(probs, spec["values"]))
    return (score, {"rationale": rationale, "probs": probs})


def read_output_logits(
    read: str,
    output: Any,
    score_ids: list[int],
    spec: dict[str, Any],
    rationale: str,
) -> tuple[float | None, dict]:
    logprobs = output_logprobs(output)
    if logprobs is None:
        return (None, {"rationale": rationale, "error": "missing token logprobs"})
    return read_logits(read, logprobs, score_ids, spec, rationale)


def rationale_sampling_params(max_think_tokens: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=max_think_tokens,
        repetition_penalty=1.05,
        stop=["<answer>"],
    )


def read_sampling_params(read: str, score_ids: list[int]) -> Any:
    from vllm import SamplingParams

    if read in ("margin", "expected"):
        return SamplingParams(
            temperature=0.0,
            max_tokens=1,
            allowed_token_ids=score_ids,
            logprob_token_ids=score_ids,
        )
    return SamplingParams(
        temperature=0.0,
        max_tokens=3 if read == "gen_binary" else 8,
    )


def read_results(
    read: str,
    outputs: list[Any],
    score_ids: list[int],
    spec: dict[str, Any],
    rationales: list[str],
) -> list[tuple[float | None, dict]]:
    if read in ("gen_binary", "gen_numeric", "gen_1to5"):
        return [
            parse_generated(read, output_text(output), rationale)
            for output, rationale in zip(outputs, rationales)
        ]
    return [
        read_output_logits(read, output, score_ids, spec, rationale)
        for output, rationale in zip(outputs, rationales)
    ]


def parse_generated(read: str, text: str, rationale: str) -> tuple[float | None, dict]:
    if read == "gen_binary":
        value = text.strip().lower()
        if value.startswith("yes"):
            return (1.0, {"rationale": rationale, "text": text})
        if value.startswith("no"):
            return (0.0, {"rationale": rationale, "text": text})
        return (None, {"rationale": rationale, "text": text, "error": "unparsed"})

    match = re.search(r"\d+(\.\d+)?", text)
    if match is None:
        return (None, {"rationale": rationale, "text": text, "error": "unparsed"})
    score = float(match.group())
    if read == "gen_1to5" and score not in {1.0, 2.0, 3.0, 4.0, 5.0}:
        return (None, {"rationale": rationale, "text": text, "error": "out_of_range"})
    return (score, {"rationale": rationale, "text": text})


@dataclass(frozen=True)
class GenerateInput:
    """One vLLM offline generate request before sampling params are attached."""

    prompt: str
    multi_modal_data: dict[str, Any]
    mm_processor_kwargs: dict[str, Any] | None = None

    def request(self) -> dict[str, Any]:
        request = {
            "prompt": self.prompt,
            "multi_modal_data": self.multi_modal_data,
        }
        if self.mm_processor_kwargs is not None:
            request["mm_processor_kwargs"] = self.mm_processor_kwargs
        return request


class VLMReranker(ScoringReranker):
    """Shared config and readout state for pointwise generative VLM rerankers."""

    def __init__(
        self,
        cfg: RerankerConfig,
        *,
        model_id: str,
        name: str | None = None,
    ) -> None:
        super().__init__(
            rerank_depth=cfg.rerank_depth,
            score_chunk_size=int(cfg.options.get("score_chunk_size", 256)),
        )
        if name is not None:
            self.name = name
        self.model_id = str(cfg.options.get("model_id", model_id))
        self.fps = float(cfg.fps)
        self.max_frames = int(cfg.max_frames)
        self.cot = cfg.options.get("reasoning", "none") == "cot"
        self.max_think_tokens = int(cfg.options.get("max_think_tokens", 128))

        self.output_format = str(cfg.options.get("output_format", "logit_delta"))
        self.spec = output_spec(self.output_format, cot=self.cot)
        self.suffix = prompt_suffix(self.spec, cot=self.cot)
        image_prompt_mode = str(cfg.options.get("image_prompt_mode", "default"))
        if image_prompt_mode not in IMAGE_PROMPT_PREFIXES:
            raise ValueError(
                f"unknown image_prompt_mode {image_prompt_mode!r}; supported: "
                f"{sorted(IMAGE_PROMPT_PREFIXES)}"
            )
        self.image_prompt_prefix = IMAGE_PROMPT_PREFIXES[image_prompt_mode]
        self.score_ids: list[int] = []

    def prompt_text(self, query: str) -> str:
        return RELEVANCE_PROMPT.format(query=query) + self.suffix

    def image_prompt_text(self, query: str) -> str:
        return (
            self.image_prompt_prefix
            + IMAGE_RELEVANCE_PROMPT.format(query=query)
            + self.suffix.replace("clip", "image")
        )

    def set_score_token_ids(self, tokenizer: Any) -> None:
        self.score_ids = score_token_ids(tokenizer, self.spec)

    @staticmethod
    def candidate_scores(results: list[tuple[float | None, dict]]) -> list[CandidateScore]:
        return [CandidateScore(score, meta) for score, meta in results]


class OfflineGenerateVLMReranker(VLMReranker):
    """Shared two-pass vLLM ``generate`` scoring for offline VLM adapters.

    Subclasses must assign ``self.llm`` before calling ``_score_generate_batch``.
    """

    llm: Any

    def _score_generate_batch(
        self,
        built: list[GenerateInput],
    ) -> list[tuple[float | None, dict]]:
        if not built:
            return []

        if self.cot:
            rationale_params = rationale_sampling_params(self.max_think_tokens)
            rationale_outputs = self.llm.generate(
                [item.request() for item in built],
                rationale_params,
                use_tqdm=False,
            )
            rationales = [output_text(output) for output in rationale_outputs]
        else:
            rationales = ["" for _ in built]

        read = self.spec["read"]
        read_params = read_sampling_params(read, self.score_ids)
        prefix = self.spec["prefix"]
        read_requests = [
            GenerateInput(
                prompt=(
                    item.prompt
                    + answer_tail(rationale, cot=self.cot, context=item.prompt)
                    + prefix
                ),
                multi_modal_data=item.multi_modal_data,
                mm_processor_kwargs=item.mm_processor_kwargs,
            ).request()
            for item, rationale in zip(built, rationales)
        ]
        read_outputs = self.llm.generate(read_requests, read_params, use_tqdm=False)
        return read_results(read, read_outputs, self.score_ids, self.spec, rationales)
