"""Cosmos-RL entrypoint for PAS binary reranker fine-tuning.

Launch with::

    cosmos-rl --config examples/pas_reranker/train_standard_rankhead20_k8_full.toml \
      examples/pas_reranker/tao_pas_reranker.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import numpy as np
import pydantic
import toml
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from qwen_vl_utils import fetch_image
from safetensors.torch import load_file, save_file
from torch.utils.data import Sampler

import cosmos_rl.launcher.worker_entry
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer
from cosmos_rl.dispatcher.data.packer.qwen3_vl_data_packer import Qwen3_VL_DataPacker
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.report.wandb_logger import is_wandb_available, log_wandb

try:
    from examples.pas_reranker.loss import (
        RankPointLoss,
        QueryAdaptiveResidualGate,
        build_binary_response_spec,
        build_ordinal_response_spec,
        extract_binary_scores,
    )
    from examples.pas_reranker.structured_attribute_aux import (
        PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE,
        PAS_COMPATIBILITY_PROBE_FIELDS,
        PAS_COMPATIBILITY_PROBE_RESPONSE,
        PAS_PREDECISION_TRISTATE_PROBE_RESPONSE,
        PAS_TRISTATE_SEMANTIC_CLASS_TOKENS,
        PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE,
        StructuredAttributeAssets,
        StructuredAttributeChoiceLoss,
        append_structured_attribute_response,
        active_aware_logical_score,
        bottom_k_cvar,
        build_active_compatibility_probe_inference_record,
        build_active_compatibility_probe_response,
        build_compatibility_probe_inference_record,
        build_compatibility_probe_response,
        build_predecision_tristate_probe_inference_record,
        build_predecision_tristate_probe_response,
        build_postdecision_tristate_probe_inference_record,
        build_postdecision_tristate_probe_response,
        build_tristate_compatibility_probe_inference_record,
        build_tristate_compatibility_probe_response,
        extract_active_compatibility_probe_scores,
        extract_compatibility_probe_scores,
        extract_tristate_compatibility_probe_logits,
        load_structured_attribute_assets,
        normalized_softmin,
        predecision_smooth_and_score,
        predecision_tristate_probe_response,
        structured_attribute_mismatch_metadata,
        tristate_class_token_ids,
        tristate_logical_score,
        yes_no_logit_difference,
    )
    from examples.pas_reranker.build_hcr_constraint_assets import (
        DenseHCRAssets,
        load_dense_hcr_assets,
    )
    from examples.pas_reranker.build_trainonly_full_gallery_ap_metadata import (
        load_full_gallery_ap_metadata,
    )
    from examples.pas_reranker.atomic_query_fields import canonical_conjunction_prompt
    from examples.pas_reranker.visual_cache import VisualCacheReader, visual_cache_key
except ModuleNotFoundError:
    # Cosmos-RL launches a custom script by absolute path, in which case the
    # script directory rather than the repository root is on sys.path.
    from loss import (
        RankPointLoss,
        QueryAdaptiveResidualGate,
        build_binary_response_spec,
        build_ordinal_response_spec,
        extract_binary_scores,
    )
    from structured_attribute_aux import (
        PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE,
        PAS_COMPATIBILITY_PROBE_FIELDS,
        PAS_COMPATIBILITY_PROBE_RESPONSE,
        PAS_PREDECISION_TRISTATE_PROBE_RESPONSE,
        PAS_TRISTATE_SEMANTIC_CLASS_TOKENS,
        PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE,
        StructuredAttributeAssets,
        StructuredAttributeChoiceLoss,
        append_structured_attribute_response,
        active_aware_logical_score,
        bottom_k_cvar,
        build_active_compatibility_probe_inference_record,
        build_active_compatibility_probe_response,
        build_compatibility_probe_inference_record,
        build_compatibility_probe_response,
        build_predecision_tristate_probe_inference_record,
        build_predecision_tristate_probe_response,
        build_postdecision_tristate_probe_inference_record,
        build_postdecision_tristate_probe_response,
        build_tristate_compatibility_probe_inference_record,
        build_tristate_compatibility_probe_response,
        extract_active_compatibility_probe_scores,
        extract_compatibility_probe_scores,
        extract_tristate_compatibility_probe_logits,
        load_structured_attribute_assets,
        normalized_softmin,
        predecision_smooth_and_score,
        predecision_tristate_probe_response,
        structured_attribute_mismatch_metadata,
        tristate_class_token_ids,
        tristate_logical_score,
        yes_no_logit_difference,
    )
    from build_hcr_constraint_assets import DenseHCRAssets, load_dense_hcr_assets
    from build_trainonly_full_gallery_ap_metadata import load_full_gallery_ap_metadata
    from atomic_query_fields import canonical_conjunction_prompt
    from visual_cache import VisualCacheReader, visual_cache_key


_METRICS_PATH: Path | None = None
_OFFLINE_WANDB_RUN: Any | None = None


@torch.no_grad()
def load_lora_initialization(model, lora_path: str, *, allow_partial: bool) -> None:
    """Restore a PEFT LoRA export after Cosmos has materialized LoRA modules.

    Cosmos exposes ``policy.lora.lora_path`` in its config but its local LoRA
    plugin does not currently consume that field.  TAO needs exact adapter
    continuation without merging language weights into the BF16 base, so map
    the same local parameter names used by Cosmos' safetensors exporter back
    to the exported PEFT keys.  In partial mode, source tensors must still all
    match; only newly added target modules may retain their zero-delta LoRA
    initialization.
    """

    adapter_dir = Path(lora_path)
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete LoRA adapter directory: {adapter_dir}")

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source = load_file(str(weights_path), device="cpu")
    targets: dict[str, tuple[str, torch.nn.Parameter]] = {}
    for local_name, parameter in model.named_parameters():
        if not (
            local_name.endswith(".lora_A.weight")
            or local_name.endswith(".lora_B.weight")
        ):
            continue
        mapped = model.weight_mapper.policy_map_local_key_to_hf_key(local_name)
        if not mapped.startswith("base_model"):
            mapped = f"base_model.model.{mapped}"
        if mapped in targets:
            raise ValueError(f"Duplicate mapped LoRA target: {mapped}")
        targets[mapped] = (local_name, parameter)

    unexpected = sorted(set(source) - set(targets))
    missing = sorted(set(targets) - set(source))
    if unexpected:
        raise ValueError(
            f"LoRA initialization has {len(unexpected)} unexpected tensors: "
            f"{unexpected[:5]}"
        )
    if missing and not allow_partial:
        raise ValueError(
            f"LoRA initialization is missing {len(missing)} tensors: {missing[:5]}"
        )

    for key, tensor in source.items():
        local_name, parameter = targets[key]
        local_parameter = (
            parameter.to_local() if hasattr(parameter, "to_local") else parameter
        )
        if tuple(local_parameter.shape) != tuple(tensor.shape):
            raise ValueError(
                f"LoRA shape mismatch for {local_name}: "
                f"target={tuple(local_parameter.shape)}, source={tuple(tensor.shape)}"
            )
        local_parameter.copy_(
            tensor.to(device=local_parameter.device, dtype=local_parameter.dtype)
        )

    # A newly opened LoRA module is an exact no-op only when its B factor is
    # still zero.  Check this invariant before the first optimizer update.
    for key in missing:
        if not key.endswith(".lora_B.weight"):
            continue
        _, parameter = targets[key]
        local_parameter = (
            parameter.to_local() if hasattr(parameter, "to_local") else parameter
        )
        if bool(local_parameter.count_nonzero().item()):
            raise ValueError(f"New LoRA B factor is not zero initialized: {key}")

    logger.info(
        "Loaded LoRA initialization from %s: restored=%s newly_initialized=%s "
        "source_rank=%s source_alpha=%s",
        adapter_dir,
        len(source),
        len(missing),
        source_config.get("r"),
        source_config.get("lora_alpha"),
    )
    if os.environ.get("PAS_LOG_LORA_CHECKSUM", "0") == "1":
        # This is deliberately local-only: it must not introduce another
        # collective into transport smoke tests.  Hash the bytes actually
        # resident on each rank after continuation loading, including newly
        # opened deterministic LoRA modules, so logs can establish exact
        # initialization identity even when DDP init_sync is disabled.
        digest = hashlib.sha256()
        total_numel = 0
        for mapped_name, (local_name, parameter) in sorted(targets.items()):
            local_parameter = (
                parameter.to_local()
                if hasattr(parameter, "to_local")
                else parameter
            )
            value = local_parameter.detach().contiguous()
            digest.update(mapped_name.encode("utf-8"))
            digest.update(local_name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
            total_numel += value.numel()
        logger.info(
            "PAS local LoRA checksum rank=%s tensors=%s numel=%s sha256=%s",
            os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
            len(targets),
            total_numel,
            digest.hexdigest(),
        )


def configure_cuda_runtime() -> None:
    """Work around unsupported half-precision cuDNN Conv3d on this pod.

    CUDA kernels remain enabled. Only cuDNN convolution dispatch is disabled;
    PyTorch's native CUDA Conv3d handles the CR3 vision patch projection.
    """

    if os.environ.get("PAS_DISABLE_CUDNN_CONV", "0") == "1":
        torch.backends.cudnn.enabled = False
        logger.warning(
            "Disabled cuDNN convolution dispatch because the CUDA 13.2 "
            "forward-compatibility runtime cannot select a BF16 Conv3d engine"
        )


def _is_policy_master() -> bool:
    return (
        os.environ.get("COSMOS_ROLE") == "Policy"
        and int(os.environ.get("NODE_RANK", "0")) == 0
        and int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
    )


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def write_loss_metrics(report_data: dict[str, Any], step: int) -> None:
    """Write rank-zero training and validation reports as newline JSON."""

    if not _is_policy_master() or _METRICS_PATH is None:
        return
    record = {"step": int(step)}
    record.update({key: _json_value(value) for key, value in report_data.items()})
    with _METRICS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    if _OFFLINE_WANDB_RUN is not None:
        _OFFLINE_WANDB_RUN.log(
            {key: value for key, value in record.items() if key != "step"},
            step=int(step),
        )


def flush_terminal_validation_metrics(worker, report_data: dict[str, Any]) -> None:
    """Publish fixed diagnostics when validation occurs after the last update.

    Intermediate validation metrics are normally attached to the next training
    report. A terminal validation has no next update, so flush its accumulated
    components here. The hook is telemetry-only: it reads detached metric sums
    after validation and does not touch model, optimizer, or scheduler state.
    """

    del report_data
    # A preemption/diagnostic SIGUSR1 also makes the validation terminal even
    # when train_step < total_steps.  Previously those final rank metrics were
    # silently dropped because there was no following training report to flush
    # them, leaving only val/avg_loss in the log.
    signaled = bool(
        worker.signal_handler is not None
        and getattr(worker.signal_handler, "_signal_received", False)
    )
    if int(worker.train_step) != int(worker.total_steps) and not signaled:
        return
    trainer = worker.trainer
    if not hasattr(trainer, "_validation_metric_step") or not hasattr(
        trainer.loss_fn, "metric_totals"
    ):
        return
    metric_step = trainer._validation_metric_step
    if metric_step is None:
        return
    metrics: dict[str, Any] = {"val/metric_source_step": int(metric_step)}
    totals = trainer.loss_fn.metric_totals()
    if isinstance(totals, tuple):
        value = trainer._distributed_accuracy(*totals)
        if value is not None:
            metrics["val/mcq_accuracy"] = value
    else:
        for name, (metric_sum, metric_count) in totals.items():
            value = trainer._distributed_metric_mean(metric_sum, metric_count)
            if value is None:
                continue
            suffix = "_loss" if name in {"rank", "point"} else ""
            metrics[f"val/{name}{suffix}"] = value
    write_loss_metrics(metrics, int(metric_step))
    if (
        _is_policy_master()
        and "wandb" in trainer.config.logging.logger
        and is_wandb_available()
        and _OFFLINE_WANDB_RUN is None
    ):
        log_wandb(metrics, step=int(metric_step))


def begin_atomic_validation_score_audit(
    worker, report_data: dict[str, Any]
) -> None:
    """Start a per-step score audit that stays hidden until validation ends.

    A rank can be preempted while rewriting its JSONL. The old single append
    file allowed a concurrent summary to combine completed shards from the
    prior process with partial shards from a restarted process. Per-step
    partial files plus atomic rename make that mixed snapshot impossible.
    """

    del report_data
    trainer = worker.trainer
    # This hook is shared with the MCQ trainer, whose audit is maintained in
    # ``_audit_path`` instead.  It must remain a no-op for that trainer.
    if getattr(trainer, "_validation_score_audit_path", None) is None:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    final_path = (
        Path(trainer.config.train.output_dir)
        / f"validation_score_audit_step{int(worker.train_step):08d}_rank{rank:02d}.jsonl"
    )
    partial_path = final_path.with_suffix(final_path.suffix + ".partial")
    partial_path.unlink(missing_ok=True)
    trainer._validation_score_audit_path = partial_path
    trainer._validation_score_audit_final_path = final_path
    trainer._validation_score_audit_active_step = int(worker.train_step)


def finalize_atomic_validation_score_audit(
    worker, report_data: dict[str, Any]
) -> None:
    """Atomically publish this rank's score audit after full validation."""

    del report_data
    trainer = worker.trainer
    partial_path = getattr(trainer, "_validation_score_audit_path", None)
    final_path = getattr(trainer, "_validation_score_audit_final_path", None)
    active_step = getattr(trainer, "_validation_score_audit_active_step", None)
    if partial_path is None:
        return
    if final_path is None or active_step != int(worker.train_step):
        raise RuntimeError("validation score audit lifecycle is inconsistent")
    if not partial_path.exists() or partial_path.stat().st_size == 0:
        raise RuntimeError(f"validation score audit is empty: {partial_path}")
    os.replace(partial_path, final_path)
    trainer._validation_score_audit_path = final_path


def finalize_pas_validation(worker, report_data: dict[str, Any]) -> None:
    """Publish atomic audit artifacts, then flush terminal metrics."""

    finalize_atomic_validation_score_audit(worker, report_data)
    flush_terminal_validation_metrics(worker, report_data)


def init_offline_wandb(config) -> None:
    """Keep W&B logging usable without making credentials block a smoke run."""

    global _OFFLINE_WANDB_RUN
    if not _is_policy_master() or "wandb" not in config.logging.logger:
        return
    import wandb

    if wandb.api.api_key:
        return
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _OFFLINE_WANDB_RUN = wandb.init(
        project=config.logging.project_name,
        group=config.logging.group_name,
        name=config.logging.experiment_name,
        dir=str(output_dir),
        mode="offline",
        config={
            "policy.model_name_or_path": config.policy.model_name_or_path,
            "train.epoch": config.train.epoch,
            "train.optm_lr": config.train.optm_lr,
            "custom.loss": config.custom.get("loss", {}),
        },
    )
    logger.warning(
        "WANDB_API_KEY is not configured; recording an offline W&B run under %s",
        output_dir,
    )

try:
    from cosmos_reason1_utils.text import create_conversation
except ImportError:
    create_conversation = None


class VisionConfig(pydantic.BaseModel):
    """Stable PAS image-preprocessing schema across optional utility versions."""

    nframes: int = 8
    min_pixels: int | None = None
    max_pixels: int | None = None


class DatasetConfig(pydantic.BaseModel):
    annotation_path: str | list[str]
    media_path: str = ""
    response_mode: str = "binary"
    prompt_mode: Literal[
        "annotation", "strict_conjunction", "canonical_conjunction"
    ] = "annotation"
    score_audit_metadata: bool = False
    teacher_calibration_path: Optional[str] = None
    full_gallery_ap_metadata_path: Optional[str] = None
    teacher_positive_floor: Optional[float] = pydantic.Field(
        default=None, ge=0.0, le=1.0
    )
    teacher_negative_ceiling: Optional[float] = pydantic.Field(
        default=None, ge=0.0, le=1.0
    )

    @pydantic.field_validator("response_mode")
    @classmethod
    def validate_response_mode(cls, value: str) -> str:
        if value not in {"binary", "reasoning", "mcq"}:
            raise ValueError("response_mode must be 'binary', 'reasoning', or 'mcq'")
        return value

    @pydantic.model_validator(mode="after")
    def validate_teacher_label_intervals(self):
        floor = self.teacher_positive_floor
        ceiling = self.teacher_negative_ceiling
        if (floor is None) != (ceiling is None):
            raise ValueError(
                "teacher_positive_floor and teacher_negative_ceiling must be set together"
            )
        if floor is not None and not ceiling < floor:
            raise ValueError(
                "teacher_negative_ceiling must be below teacher_positive_floor"
            )
        return self


class AtomicAndConfig(pydantic.BaseModel):
    """End-to-end weakest-link aggregation over independent atomic prompts.

    The annotation contains ``num_candidates * num_fields`` binary rows per
    query in candidate-major order.  The same frozen CR3 yes/no readout scores
    every atomic requirement; no auxiliary head, teacher, or retriever score is
    introduced. Candidate scores use a differentiable normalized soft minimum
    by default, with an active-field arithmetic mean available as an ablation,
    and are the only scores sent to the listwise loss.
    """

    num_candidates: int = pydantic.Field(default=5, gt=1)
    num_fields: int = pydantic.Field(default=8, gt=0)
    aggregation: Literal["softmin", "mean", "log_product"] = "softmin"
    temperature: float = pydantic.Field(default=0.5, gt=0.0)
    field_point_weight: float = pydantic.Field(default=0.25, ge=0.0)
    normalization: Literal["raw", "query_center", "query_zscore"] = "raw"
    normalization_epsilon: float = pydantic.Field(default=1e-4, gt=0.0)


def aggregate_atomic_candidate_scores(
    score_grid: torch.Tensor,
    active: torch.Tensor,
    *,
    aggregation: Literal["softmin", "mean", "log_product"] = "softmin",
    temperature: float = 0.5,
) -> torch.Tensor:
    """Reduce active atomic margins to one differentiable candidate score."""

    if score_grid.shape != active.shape or score_grid.ndim != 3:
        raise ValueError("Atomic score/activity grids must be same-shape rank-3 tensors")
    if active.dtype is not torch.bool:
        raise TypeError("Atomic activity grid must be boolean")
    active_count = active.sum(dim=2)
    if not bool(active_count.gt(0).all()):
        raise ValueError("Every candidate requires at least one active atomic field")
    if aggregation == "mean":
        return score_grid.masked_fill(~active, 0.0).sum(dim=2) / active_count
    if aggregation == "log_product":
        if temperature <= 0.0:
            raise ValueError("Atomic log-product temperature must be positive")
        # A CR3 yes/no margin is the logit of an atomic pass probability.
        # Summing log probabilities implements a differentiable strict AND
        # (Product of Experts) while inactive padded fields contribute zero.
        return F.logsigmoid(score_grid / temperature).masked_fill(
            ~active, 0.0
        ).sum(dim=2)
    if aggregation == "softmin":
        if temperature <= 0.0:
            raise ValueError("Atomic softmin temperature must be positive")
        masked = score_grid.masked_fill(~active, float("inf"))
        count = active_count.to(dtype=score_grid.dtype)
        return -temperature * torch.logsumexp(
            -masked / temperature, dim=2
        ) + temperature * count.log()
    raise ValueError(f"Unsupported atomic aggregation: {aggregation}")


STRICT_CONJUNCTION_INSTRUCTION = (
    "Check each stated requirement independently: upper-body clothing, "
    "lower-body clothing, footwear, accessories, and viewpoint when they "
    "are mentioned. Both the item type and its color must match exactly. "
    "Do not compensate for one mismatch with matches on other attributes. "
    "This is a strict AND decision: answer yes only if every stated "
    "requirement matches. If even one requirement is different or absent, "
    "answer no.\n"
)


def strict_conjunction_prompt(query: str) -> str:
    """Make the Boolean-AND decision explicit without changing the model."""

    query = query.strip()
    if not query:
        raise ValueError("strict-conjunction prompting requires a nonempty query")
    return (
        f'<image>\n{STRICT_CONJUNCTION_INSTRUCTION}Query: "{query}"\n'
        "Does the person in the image fully match the query? Answer with "
        "<answer>yes</answer> or "
        "<answer>no</answer>."
    )


class LossConfig(pydantic.BaseModel):
    rank_weight: float = pydantic.Field(default=1.0, ge=0.0)
    point_weight: float = pydantic.Field(default=1.0, ge=0.0)
    rank_temperature: float = pydantic.Field(default=1.0, gt=0.0)
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
    ] = "probability_mass"
    bag_positive_temperature: float = pydantic.Field(default=0.1, gt=0.0)
    bag_negative_temperature: float = pydantic.Field(default=0.1, gt=0.0)
    pu_class_prior: float = pydantic.Field(default=0.2, gt=0.0, lt=1.0)
    pu_nn_weight: float = pydantic.Field(default=0.25, ge=0.0)
    partial_negative_ratio: float = pydantic.Field(default=0.2, gt=0.0, le=1.0)
    rank_margin: float = pydantic.Field(default=0.05, ge=0.0)
    natural20_rank_weight: float = pydantic.Field(default=1.0, ge=0.0)
    inverse_pair_weight: float = pydantic.Field(default=1.0, ge=0.0)
    weak_veto_temperature: float = pydantic.Field(default=0.2, gt=0.0)
    weak_veto_full_weight: float = pydantic.Field(default=1.0, ge=0.0)
    weak_veto_mil_weight: float = pydantic.Field(default=1.0, ge=0.0)
    robust_pair_q: float = pydantic.Field(default=0.3, gt=0.0, le=1.0)
    all_pairs_aux_weight: float = pydantic.Field(default=0.0, ge=0.0)
    positive_tail_weight: float = pydantic.Field(default=0.0, ge=0.0)
    positive_tail_temperature: float = pydantic.Field(default=0.2, gt=0.0)
    retriever_success_weight: float = pydantic.Field(default=1.0, ge=1.0)
    retriever_error_weight: float = pydantic.Field(default=1.0, ge=0.0)
    near_miss_k: Optional[int] = pydantic.Field(default=None, ge=2)
    near_miss_weight: float = pydantic.Field(default=1.0, ge=1.0)
    parent_preservation_margin: Optional[float] = pydantic.Field(default=None, ge=0.0)
    parent_preservation_teacher_margin_slack: Optional[float] = pydantic.Field(
        default=None, ge=0.0
    )
    parent_preservation_weight: float = pydantic.Field(default=1.0, ge=1.0)
    topk_preservation_k: Optional[int] = pydantic.Field(default=None, gt=0)
    topk_preservation_margin: float = pydantic.Field(default=0.0, ge=0.0)
    topk_preservation_weight: float = pydantic.Field(default=0.0, ge=0.0)
    point_temperature: float = pydantic.Field(default=1.0, gt=0.0)
    point_mode: Literal[
        "all",
        "positive_only",
        "class_balanced",
        "query_centered_class_balanced",
    ] = "all"
    negative_point_weight: float = pydantic.Field(default=1.0, gt=0.0)
    teacher_weight: float = pydantic.Field(default=0.0, ge=0.0)
    teacher_repair_group_weight: float = pydantic.Field(default=1.0, ge=0.0, le=1.0)
    teacher_weight_final: Optional[float] = pydantic.Field(default=None, ge=0.0)
    teacher_decay_steps: Optional[int] = pydantic.Field(default=None, gt=0)
    teacher_temperature: float = pydantic.Field(default=1.0, gt=0.0)
    teacher_mode: Literal[
        "pointwise",
        "pairwise_margin",
        "pairwise_margin_clamped",
        "pairwise_logit_delta_clamped",
        "pairwise_logit_delta_correct_only",
    ] = "pointwise"
    candidate_group_size: int = pydantic.Field(default=4, gt=1)
    positive_response: str = "<answer>yes</answer>"
    negative_response: str = "<answer>no</answer>"
    # Ordinary datasets put the binary response first.  A fixed,
    # label-independent causal scratchpad may instead precede one unique final
    # yes/no response; the deployed score is still the native yes-no LM logit.
    binary_decision_position: Literal["prefix", "unique_suffix"] = "prefix"
    binary_readout: Literal["fp32", "bf16_ste"] = "fp32"
    # Optional residual-learning path. ``retriever_score`` is a frozen input
    # feature, not a teacher target: the deployed z-score fusion itself is sent
    # through the listwise loss so gradients train CR3 to repair the retriever.
    retriever_residual_alpha: Optional[float] = pydantic.Field(default=None, ge=0.0)
    retriever_residual_epsilon: float = pydantic.Field(default=1e-4, gt=0.0)
    retriever_residual_gate: Literal["fixed", "query_score"] = "fixed"
    retriever_residual_gate_feature_version: Literal[
        "r539_linear_v1", "r542_interactions_v1"
    ] = "r539_linear_v1"
    retriever_residual_alpha_minimum: float = pydantic.Field(default=0.0, ge=0.0)
    retriever_residual_alpha_maximum: float = pydantic.Field(default=8.0, gt=0.0)
    retriever_residual_gate_lr: float = pydantic.Field(default=1e-3, gt=0.0)
    retriever_residual_gate_log_alpha_l2: float = pydantic.Field(
        default=1e-2, ge=0.0
    )
    score_mode: Literal[
        "binary_delta",
        "relevance_1to5_expected",
        "relevance_1to5_supervised_expected",
        "relevance_1to5_supervised_strict_logodds",
        "binary_delta_with_ordinal_aux",
    ] = "binary_delta"
    ordinal_ce_weight: float = pydantic.Field(default=0.0, ge=0.0)
    ordinal_aux_position: Literal["suffix", "predecision"] = "suffix"
    # Representation auxiliary at the same final hidden state used to predict
    # yes/no. It is off by default and therefore preserves existing runs.
    hidden_supcon_weight: float = pydantic.Field(default=0.0, ge=0.0)
    hidden_supcon_temperature: float = pydantic.Field(default=0.1, gt=0.0)
    hidden_supcon_center: bool = True
    # Enforce augmentation invariance only on explicitly marked raw/edit pairs
    # whose query omits the changed field and whose hard labels agree.
    same_label_consistency_weight: float = pydantic.Field(default=0.0, ge=0.0)
    # Cumulative hard requirement-count supervision applied directly to the
    # deployed yes-minus-no scalar.  This adds no prediction head.
    requirement_ordinal_weight: float = pydantic.Field(default=0.0, ge=0.0)
    requirement_ordinal_gap: float = pydantic.Field(default=1.0, gt=0.0)
    requirement_ordinal_temperature: float = pydantic.Field(default=1.0, gt=0.0)
    query_consistency_weight: float = pydantic.Field(default=0.0, ge=0.0)
    query_cross_weight: float = pydantic.Field(default=0.0, ge=0.0)
    full_gallery_rank1_weight: float = pydantic.Field(default=0.0, ge=0.0)
    full_gallery_active_topk: Optional[int] = pydantic.Field(default=None, gt=0)
    transition_pooled_weight: float = pydantic.Field(default=0.0, ge=0.0)
    transition_pooled_topk: int = pydantic.Field(default=4, gt=0)
    transition_pooled_temperature: float = pydantic.Field(default=0.2, gt=0.0)
    transition_pooled_margin: float = pydantic.Field(default=0.0, ge=0.0)
    robust_cycle_rho: float = pydantic.Field(default=0.25, gt=0.0)
    robust_cycle_lambda_start: float = pydantic.Field(default=0.25, ge=0.0)
    robust_cycle_lambda_final: float = pydantic.Field(default=1.0, ge=0.0)
    robust_cycle_lambda_ramp_steps: int = pydantic.Field(default=500, gt=0)
    preservation_cycle_margin: float = pydantic.Field(default=2.0, ge=0.0)
    preservation_cycle_weight: float = pydantic.Field(default=4.0, ge=0.0)
    repair_cycle_ap_weight: float = pydantic.Field(default=0.25, ge=0.0)
    preserve_cycle_ap_weight: float = pydantic.Field(default=0.10, ge=0.0)
    preserve_cycle_robust_scale: float = pydantic.Field(default=0.25, ge=0.0)
    r69_smoothap_temperature: float = pydantic.Field(
        default=0.384007173733197, gt=0.0
    )
    r69_soft_r1_temperature: float = pydantic.Field(
        default=2.076483289679061, gt=0.0
    )
    r69_smoothap_weight: float = pydantic.Field(default=1.0, ge=0.0)
    r69_soft_r1_weight: float = pydantic.Field(default=0.25, ge=0.0)
    policy_samples: int = pydantic.Field(default=16, ge=2)
    policy_exact_max_k: int = pydantic.Field(default=7, ge=2)
    policy_ap_reward_weight: float = pydantic.Field(default=1.0, ge=0.0)
    policy_r1_reward_weight: float = pydantic.Field(default=1.0, ge=0.0)
    policy_preserve_anchor_weight: float = pydantic.Field(default=1.0, ge=0.0)
    policy_preserve_anchor_margin: float = pydantic.Field(default=0.0, ge=0.0)

    @pydantic.model_validator(mode="after")
    def check_nonzero_weight(self):
        if self.rank_weight + self.point_weight + self.teacher_weight <= 0:
            raise ValueError(
                "rank_weight, point_weight, and teacher_weight cannot all be zero"
            )
        if (
            self.binary_decision_position == "unique_suffix"
            and self.score_mode != "binary_delta"
        ):
            raise ValueError(
                "unique_suffix binary decisions require score_mode=binary_delta"
            )
        if self.hidden_supcon_weight and self.score_mode != "binary_delta":
            raise ValueError("hidden_supcon_weight requires score_mode=binary_delta")
        if self.retriever_residual_alpha is not None:
            if self.score_mode != "binary_delta":
                raise ValueError("retriever residual fusion requires binary_delta")
            if self.point_weight or self.teacher_weight:
                raise ValueError(
                    "retriever residual fusion requires point_weight=teacher_weight=0"
                )
        if self.requirement_ordinal_weight and self.score_mode != "binary_delta":
            raise ValueError("requirement_ordinal_weight requires binary_delta")
        if self.topk_preservation_weight and self.topk_preservation_k is None:
            raise ValueError(
                "topk_preservation_k is required when topk_preservation_weight is nonzero"
            )
        if self.all_pairs_aux_weight and not self.rank_weight:
            raise ValueError("all_pairs_aux_weight requires nonzero rank_weight")
        if self.near_miss_weight > 1.0 and self.near_miss_k is None:
            raise ValueError("near_miss_k is required when near_miss_weight exceeds 1")
        if self.rank_mode in {
            "nested_tail_risk_selective_override",
            "global_nested_tail_risk_selective_override",
            "global_action_faithful_topm_ap",
            "topm_all_pairs",
            "robust_topm_gce",
            "topm_logsumexp",
        } and self.near_miss_k is None:
            raise ValueError(f"{self.rank_mode} requires near_miss_k")
        if self.teacher_mode == "pointwise" and self.teacher_repair_group_weight != 1.0:
            raise ValueError(
                "teacher_repair_group_weight currently requires a pairwise teacher mode"
            )
        if (
            self.full_gallery_rank1_weight
            and self.rank_mode != "full_gallery_lambda_ap"
        ):
            raise ValueError(
                "full_gallery_rank1_weight requires rank_mode=full_gallery_lambda_ap"
            )
        if (
            self.full_gallery_active_topk is not None
            and self.rank_mode != "full_gallery_lambda_ap"
        ):
            raise ValueError(
                "full_gallery_active_topk requires rank_mode=full_gallery_lambda_ap"
            )
        if self.transition_pooled_weight and self.rank_mode != "full_gallery_lambda_ap":
            raise ValueError(
                "transition_pooled_weight requires rank_mode=full_gallery_lambda_ap"
            )
        if self.rank_mode == "deployment_weighted_robust_cycle":
            if self.candidate_group_size != 4:
                raise ValueError(
                    "deployment_weighted_robust_cycle requires candidate_group_size=4"
                )
            if self.point_weight or self.teacher_weight:
                raise ValueError(
                    "deployment_weighted_robust_cycle requires point_weight=0 and "
                    "teacher_weight=0"
                )
        if self.rank_mode == "preservation_constrained_cycle_block":
            if self.candidate_group_size != 32:
                raise ValueError(
                    "preservation_constrained_cycle_block requires "
                    "candidate_group_size=32"
                )
            if self.point_weight or self.teacher_weight:
                raise ValueError(
                    "preservation_constrained_cycle_block requires point_weight=0 "
                    "and teacher_weight=0"
                )
        if self.rank_mode == "worst_positive_hardest_negative_logmeanexp":
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "worst_positive_hardest_negative_logmeanexp requires "
                    "rank_weight>0, point_weight=0, and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    "worst_positive_hardest_negative_logmeanexp requires native "
                    "binary_delta"
                )
        if self.rank_mode in {
            "smoothap_soft_r1",
            "counterfactual_set5",
            "dynamic_top20_smoothap_soft_r1",
            "incumbent_guarded_smoothap_soft_r1",
            "incumbent_safe_lambda_ap",
        }:
            if (
                self.rank_mode != "smoothap_soft_r1"
                and self.candidate_group_size != 20
            ):
                raise ValueError(
                    f"{self.rank_mode} requires candidate_group_size=20"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    f"{self.rank_mode} requires rank_weight>0, point_weight=0, "
                    "and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    f"{self.rank_mode} requires native binary_delta"
                )
            if self.r69_smoothap_weight + self.r69_soft_r1_weight <= 0:
                raise ValueError("r69 objective weights cannot both be zero")
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                )
            ):
                raise ValueError(
                    f"{self.rank_mode} does not accept auxiliary ranking objectives"
                )
        if self.rank_mode == "misordered_lambda_ap":
            if self.candidate_group_size != 20:
                raise ValueError("misordered_lambda_ap requires candidate_group_size=20")
            if not self.rank_weight or self.teacher_weight:
                raise ValueError(
                    "misordered_lambda_ap requires rank_weight>0 and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError("misordered_lambda_ap requires native binary_delta")
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                )
            ):
                raise ValueError(
                    "misordered_lambda_ap does not accept auxiliary ranking objectives"
                )
        if self.rank_mode == "asymmetric_safe_residual_lambda_ap":
            if self.candidate_group_size != 20:
                raise ValueError(
                    "asymmetric_safe_residual_lambda_ap requires candidate_group_size=20"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "asymmetric_safe_residual_lambda_ap requires rank_weight>0, "
                    "point_weight=0, and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    "asymmetric_safe_residual_lambda_ap requires native binary_delta"
                )
            if self.retriever_residual_alpha is None:
                raise ValueError(
                    "asymmetric_safe_residual_lambda_ap requires retriever_residual_alpha"
                )
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                )
            ):
                raise ValueError(
                    "asymmetric_safe_residual_lambda_ap does not accept auxiliary ranking objectives"
                )
        if self.rank_mode == "robust_normalized_lambdaap_soft_r1":
            if self.candidate_group_size != 20:
                raise ValueError(
                    "robust_normalized_lambdaap_soft_r1 requires candidate_group_size=20"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "robust_normalized_lambdaap_soft_r1 requires rank_weight>0, "
                    "point_weight=0, and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    "robust_normalized_lambdaap_soft_r1 requires native binary_delta"
                )
            if self.r69_smoothap_weight + self.r69_soft_r1_weight <= 0:
                raise ValueError("R522 LambdaAP and soft-R1 weights cannot both be zero")
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                )
            ):
                raise ValueError(
                    "robust_normalized_lambdaap_soft_r1 does not accept auxiliary objectives"
                )
        if self.rank_mode == "paired_view_robust_normalized_lambdaap_soft_r1":
            if self.candidate_group_size != 40:
                raise ValueError(
                    "paired_view_robust_normalized_lambdaap_soft_r1 requires "
                    "candidate_group_size=40"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "paired-view robust LambdaAP requires rank_weight>0, "
                    "point_weight=0, and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    "paired-view robust LambdaAP requires native binary_delta"
                )
            if self.r69_smoothap_weight + self.r69_soft_r1_weight <= 0:
                raise ValueError("paired-view objective weights cannot both be zero")
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                    self.same_label_consistency_weight,
                )
            ):
                raise ValueError(
                    "paired-view robust LambdaAP does not accept auxiliary objectives"
                )
        if self.rank_mode == "consensus_trimmed_lambdaap_soft_r1":
            if self.candidate_group_size != 20:
                raise ValueError(
                    "consensus_trimmed_lambdaap_soft_r1 requires candidate_group_size=20"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "consensus_trimmed_lambdaap_soft_r1 requires rank_weight>0, "
                    "point_weight=0, and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    "consensus_trimmed_lambdaap_soft_r1 requires native binary_delta"
                )
            if self.r69_smoothap_weight + self.r69_soft_r1_weight <= 0:
                raise ValueError("R534 LambdaAP and soft-R1 weights cannot both be zero")
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                )
            ):
                raise ValueError(
                    "consensus_trimmed_lambdaap_soft_r1 does not accept auxiliary objectives"
                )
        if self.rank_mode == "r535_semantic_and_aux":
            if self.candidate_group_size != 24:
                raise ValueError("r535_semantic_and_aux requires candidate_group_size=24")
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "r535_semantic_and_aux requires rank_weight>0, "
                    "point_weight=0, and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError("r535_semantic_and_aux requires native binary_delta")
            if self.r69_smoothap_weight + self.r69_soft_r1_weight <= 0:
                raise ValueError("R535 natural LambdaAP and soft-R1 weights cannot both be zero")
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                )
            ):
                raise ValueError("r535_semantic_and_aux does not accept external auxiliaries")
        if self.rank_mode == "natural20_inverse_pair":
            if self.candidate_group_size != 21:
                raise ValueError(
                    "natural20_inverse_pair requires candidate_group_size=21"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "natural20_inverse_pair requires rank_weight>0, "
                    "point_weight=0, and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    "natural20_inverse_pair requires native binary_delta"
                )
            if self.natural20_rank_weight + self.inverse_pair_weight <= 0:
                raise ValueError(
                    "natural20/inverse-pair weights cannot both be zero"
                )
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.positive_tail_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                    self.hidden_supcon_weight,
                )
            ):
                raise ValueError(
                    "natural20_inverse_pair does not accept auxiliary objectives"
                )
        if self.rank_mode == "weak_veto_mil":
            if self.candidate_group_size < 6 or self.candidate_group_size % 2:
                raise ValueError(
                    "weak_veto_mil requires one full pair and at least two "
                    "attribute pairs"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "weak_veto_mil requires rank_weight>0, point_weight=0, "
                    "and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError("weak_veto_mil requires native binary_delta")
            if self.weak_veto_full_weight + self.weak_veto_mil_weight <= 0:
                raise ValueError("weak-veto full/MIL weights cannot both be zero")
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.positive_tail_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                    self.hidden_supcon_weight,
                )
            ):
                raise ValueError("weak_veto_mil does not accept auxiliary objectives")
            if self.near_miss_k is not None:
                raise ValueError("weak_veto_mil does not use near_miss_k")
        if self.rank_mode in {
            "plackett_luce_policy",
            "rb_plackett_luce_expected_ap",
        }:
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    f"{self.rank_mode} requires rank_weight>0 and no "
                    "point/teacher objective"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(f"{self.rank_mode} requires native binary_delta")
            if self.policy_ap_reward_weight + self.policy_r1_reward_weight <= 0:
                raise ValueError("policy AP/R1 reward weights cannot both be zero")
            if self.rank_mode == "rb_plackett_luce_expected_ap":
                if self.candidate_group_size != 20:
                    raise ValueError(
                        "rb_plackett_luce_expected_ap requires K20 groups"
                    )
                if self.policy_exact_max_k < 10:
                    raise ValueError(
                        "RB-PL K20 requires at least 10 inner quadrature points"
                    )
        if self.rank_mode == "paired_adjacent":
            if self.candidate_group_size % 2:
                raise ValueError("paired_adjacent requires an even candidate_group_size")
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    "paired_adjacent requires rank_weight>0, point_weight=0, "
                    "and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError("paired_adjacent requires native binary_delta")
        if self.rank_mode in {
            "bag_top1_logsumexp",
            "pu_bag_top1_logsumexp",
            "nnpu_certified_hybrid",
        }:
            if self.candidate_group_size != 20:
                raise ValueError(
                    f"{self.rank_mode} requires candidate_group_size=20"
                )
            if not self.rank_weight or self.point_weight or self.teacher_weight:
                raise ValueError(
                    f"{self.rank_mode} requires rank_weight>0, point_weight=0, "
                    "and teacher_weight=0"
                )
            if self.score_mode != "binary_delta" or self.ordinal_ce_weight:
                raise ValueError(
                    f"{self.rank_mode} requires native binary_delta"
                )
            if any(
                (
                    self.all_pairs_aux_weight,
                    self.topk_preservation_weight,
                    self.transition_pooled_weight,
                    self.query_consistency_weight,
                    self.query_cross_weight,
                )
            ):
                raise ValueError(
                    f"{self.rank_mode} does not accept auxiliary objectives"
                )
        if (
            self.ordinal_ce_weight
            and self.score_mode not in {
                "relevance_1to5_supervised_expected",
                "relevance_1to5_supervised_strict_logodds",
                "binary_delta_with_ordinal_aux",
            }
        ):
            raise ValueError(
                "ordinal_ce_weight requires supervised 1-to-5 relevance scoring"
            )
        return self


def linear_teacher_weight(
    initial: float,
    final: float,
    *,
    train_step: int,
    decay_steps: int,
) -> float:
    """Linearly decay the teacher coefficient over optimizer updates."""

    progress = min(max(float(train_step) / float(decay_steps), 0.0), 1.0)
    return initial + (final - initial) * progress


def rank_mode_requires_parent_scores(rank_mode: str) -> bool:
    """Return whether a ranking objective consumes frozen-parent scores."""

    return rank_mode in {
        "parent_top1_corrective",
        "parent_top1_competitor",
        "dual_top1_competitor",
        "constrained_dual_top1_competitor",
        "dual_multi_positive_competitor",
    }


class VisualCacheConfig(pydantic.BaseModel):
    manifests: list[str]
    resident_shards: int = pydantic.Field(default=2, gt=0)
    slice_reads: bool = False

    @pydantic.field_validator("manifests")
    @classmethod
    def require_manifests(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("visual_cache.manifests cannot be empty")
        return value


class AttributeReplayConfig(pydantic.BaseModel):
    records_path: str
    attribute_vocab_path: str
    attribute_heads_path: str
    sample_fraction: float = pydantic.Field(default=0.15, gt=0.0, le=1.0)
    loss_weight: float = pydantic.Field(default=1.0, gt=0.0)
    label_smoothing: float = pydantic.Field(default=0.05, ge=0.0, lt=1.0)
    seed: int = 260826


class StructuredAttributeConfig(pydantic.BaseModel):
    train_records_path: str
    validation_records_path: str
    attribute_vocab_path: str
    query_records_path: Optional[str] = None
    # Zero disables the attribute-choice CE while retaining the structured
    # compatibility scores needed by mismatch-aware ranking.
    loss_weight: float = pydantic.Field(default=0.25, ge=0.0)
    mismatch_rank_weight: float = pydantic.Field(default=0.0, ge=0.0)
    mismatch_rank_mode: Literal["all_pairs", "top1_hinge"] = "all_pairs"
    mismatch_rank_margin: float = pydantic.Field(default=0.0, ge=0.0)
    compatibility_score_weight: float = pydantic.Field(default=0.1, gt=0.0)
    # Online hard-example selection: misclassified fields in the selected
    # families receive this multiplier. A value of 1.0 disables selection.
    online_hard_example_weight: float = pydantic.Field(default=1.0, ge=1.0)
    online_hard_example_fields: list[str] = pydantic.Field(default_factory=list)
    mismatch_only: bool = False
    seed: int = 110826

    @pydantic.model_validator(mode="after")
    def validate_mismatch_only(self):
        if self.mismatch_only and (
            self.mismatch_rank_weight <= 0.0 or self.loss_weight != 0.0
        ):
            raise ValueError(
                "structured_attribute.mismatch_only requires "
                "mismatch_rank_weight > 0 and loss_weight = 0"
            )
        return self


class HCRConfig(pydantic.BaseModel):
    """Hierarchical Constraint-Risk score used as the deployed reranker score."""

    asset_manifest_path: str
    mode: Literal[
        "legacy_compatibility",
        "active_logical",
        "tristate_logical",
        "predecision_tristate",
        "postdecision_tristate",
        "decision_state_tristate",
    ] = (
        "legacy_compatibility"
    )
    field_loss_weight: float = pydantic.Field(default=0.25, ge=0.0)
    aggregation: Literal["normalized_softmin", "bottom_k_cvar"] = (
        "normalized_softmin"
    )
    constraint_temperature: float = pydantic.Field(default=0.5, gt=0.0)
    bottom_k: int = pydantic.Field(default=2, gt=0)
    logical_reduction: Literal["sum", "mean"] = "sum"
    native_score_weight: float = pydantic.Field(default=1.0, ge=0.0)
    logical_score_weight: float = pydantic.Field(default=1.0, ge=0.0)
    # ``None`` preserves the historical behavior: the configured final weight
    # is active from step zero.  Setting an explicit start supports a stable
    # curriculum in which the parent-compatible native score is retained while
    # the new constraint probes first learn their semantic tasks.
    logical_score_weight_start: Optional[float] = pydantic.Field(default=None, ge=0.0)
    logical_score_warmup_steps: int = pydantic.Field(default=0, ge=0)
    logical_score_ramp_steps: int = pydantic.Field(default=0, ge=0)
    native_point_loss_weight: float = pydantic.Field(default=0.25, ge=0.0)
    requirement_loss_weight: float = pydantic.Field(default=1.0, ge=0.0)
    satisfaction_loss_weight: float = pydantic.Field(default=1.0, ge=0.0)
    tristate_loss_weight: float = pydantic.Field(default=1.0, ge=0.0)
    # ``conditional`` respects the two different statistical grains encoded by
    # the three classes: applicability is a query-level target, while
    # satisfied-vs-violated is candidate-level conditional on applicability.
    # The historical categorical CE treats K repeated copies of a query as K
    # independent applicability labels and balances whichever classes happen
    # to occur in each small per-device batch.
    tristate_loss_mode: Literal["categorical", "conditional"] = "categorical"
    # Semantic class tokens are a label-safe optimization prior: they are used
    # only as LM-head projection rows, while every input slot remains ``?``.
    tristate_token_mode: Literal["letters", "semantic"] = "letters"
    predecision_prompt_mode: Literal[
        "semantic_legend",
        "strict_and",
        "violation_veto",
        "dual_rule",
        "violation_count",
        "all_satisfied",
    ] = "semantic_legend"
    # Tri-state class probabilities entangle query applicability with image
    # satisfaction.  The shared mode pools applicability across all candidates
    # of a query before forming the per-candidate conjunction.
    predecision_logical_mode: Literal[
        "tristate", "active_aware_shared"
    ] = "tristate"
    tristate_activity_consistency_weight: float = pydantic.Field(
        default=0.0, ge=0.0
    )
    requirement_group_consistency_weight: float = pydantic.Field(
        default=0.0, ge=0.0
    )
    # Counterfactual K-group supervision. A negative tagged
    # ``sole_violation:<field>`` is compared only with the positive candidate
    # on that field's satisfaction margin. This removes the 4--7 already
    # satisfied fields from the gradient for a one-field near miss instead of
    # letting their much more frequent easy labels dilute the veto signal.
    counterfactual_veto_rank_weight: float = pydantic.Field(default=0.0, ge=0.0)
    counterfactual_veto_rank_margin: float = pydantic.Field(default=0.0, ge=0.0)
    counterfactual_veto_rank_temperature: float = pydantic.Field(default=1.0, gt=0.0)
    # K5 construction has one positive and one sole-violation row per field,
    # so a flat pair mean also weights every query and field equally.  Real
    # K20 packets have variable positive counts and duplicate sole violations;
    # group_field_mean preserves the K5 semantics by averaging pairs within a
    # field, fields within a query, then queries within a batch.
    counterfactual_veto_rank_reduction: Literal[
        "pair_mean", "group_field_mean"
    ] = "pair_mean"
    # Requirement applicability is a property of the query, not of an image.
    # Pool it across the complete candidate group before logical conjunction.
    group_shared_requirement: bool = False
    # Evaluate several native + logical readouts from one forward pass.  These
    # are telemetry only: they never change the optimized/deployed score.
    diagnostic_logical_weights: list[float] = pydantic.Field(default_factory=list)
    diagnostic_logical_only: bool = False
    # Keep the canonical full-query objective authoritative while using HCR as
    # an auxiliary representation-learning signal. ``output_pcgrad`` applies
    # PCGrad in the model-output tangent space (the hidden states passed
    # to this loss), with the dot product and primary norm summed over the full
    # distributed data-parallel batch. It is intentionally not described as
    # parameter-space PCGrad: projecting two 93M-coordinate LoRA gradients
    # before the FSDP/DP reduction would require two extra full-gradient
    # collectives per microbatch.
    auxiliary_gradient_mode: Literal[
        "none", "output_pcgrad", "parameter_pcgrad"
    ] = "none"
    auxiliary_gradient_epsilon: float = pydantic.Field(default=1e-12, gt=0.0)

    @pydantic.model_validator(mode="after")
    def validate_active_score(self):
        if (
            self.mode in {"active_logical", "tristate_logical"}
            and self.native_score_weight == 0.0
            and self.logical_score_weight == 0.0
        ):
            raise ValueError(
                f"{self.mode} requires a positive native or logical score weight"
            )
        if any(weight < 0.0 for weight in self.diagnostic_logical_weights):
            raise ValueError("diagnostic_logical_weights must be non-negative")
        if self.counterfactual_veto_rank_weight and self.mode != "active_logical":
            raise ValueError(
                "counterfactual_veto_rank_weight currently requires "
                "active_logical HCR"
            )
        if self.auxiliary_gradient_mode != "none":
            if self.mode != "predecision_tristate":
                raise ValueError(
                    "HCR auxiliary gradient surgery is restricted to the "
                    "label-independent predecision_tristate path"
                )
            if self.native_score_weight != 1.0 or self.logical_score_weight != 0.0:
                raise ValueError(
                    "HCR auxiliary gradient surgery requires canonical native "
                    "yes/no ranking as the sole primary score"
                )
            if self.logical_score_weight_start not in {None, 0.0}:
                raise ValueError(
                    "HCR auxiliary gradient surgery forbids a logical-score ramp"
                )
            if self.native_point_loss_weight != 0.0:
                raise ValueError(
                    "HCR auxiliary gradient surgery requires "
                    "native_point_loss_weight=0 to keep HCR purely auxiliary"
                )
        if self.mode == "decision_state_tristate":
            if self.native_score_weight != 1.0 or self.logical_score_weight != 0.0:
                raise ValueError(
                    "decision_state_tristate preserves plain yes/no as the sole score"
                )
            if self.logical_score_weight_start not in {None, 0.0}:
                raise ValueError(
                    "decision_state_tristate forbids a logical-score schedule"
                )
            if self.native_point_loss_weight != 0.0:
                raise ValueError(
                    "decision_state_tristate uses the ordinary base point loss exactly once"
                )
            if self.diagnostic_logical_weights or self.diagnostic_logical_only:
                raise ValueError(
                    "decision_state_tristate has no inference-time logical readout"
                )
        if self.mode == "postdecision_tristate":
            if self.native_score_weight != 1.0 or self.logical_score_weight != 0.0:
                raise ValueError(
                    "postdecision_tristate preserves plain yes/no as the sole score"
                )
            if self.logical_score_weight_start not in {None, 0.0}:
                raise ValueError(
                    "postdecision_tristate forbids a logical-score schedule"
                )
            if self.native_point_loss_weight != 0.0:
                raise ValueError(
                    "postdecision_tristate uses the base point loss exactly once"
                )
            if self.diagnostic_logical_weights or self.diagnostic_logical_only:
                raise ValueError(
                    "postdecision_tristate has no inference-time logical readout"
                )
        return self


class VisualDistillationConfig(pydantic.BaseModel):
    embedding_metadata_path: str
    image_index_path: str
    loss_weight: float = pydantic.Field(default=1.0, gt=0.0)


@dataclass(frozen=True)
class VisualDistillationAssets:
    """Memory-mapped SigLIP2 image embeddings and their stable image-key index."""

    embeddings: np.ndarray
    index_by_image_key: dict[str, int]
    embedding_path: Path

    def teacher_batch(self, indices: list[int], *, device: torch.device) -> torch.Tensor:
        # Advanced indexing materializes only the requested rows from the mmap.
        rows = np.asarray(self.embeddings[np.asarray(indices, dtype=np.int64)]).copy()
        return torch.from_numpy(rows).to(device=device, dtype=torch.float32)


def load_visual_distillation_assets(
    embedding_metadata_path: str, image_index_path: str
) -> VisualDistillationAssets:
    metadata_path = Path(embedding_metadata_path)
    index_path = Path(image_index_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    embedding_path = Path(metadata["array"])
    embeddings = np.load(embedding_path, mmap_mode="r")
    expected_shape = tuple(int(value) for value in metadata["shape"])
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"SigLIP2 embedding shape {embeddings.shape} != metadata {expected_shape}"
        )
    image_keys = index.get("image_keys")
    if not isinstance(image_keys, list) or len(image_keys) != embeddings.shape[0]:
        raise ValueError(
            "SigLIP2 image index length does not match the embedding row count"
        )
    if len(set(image_keys)) != len(image_keys):
        raise ValueError("SigLIP2 image index contains duplicate image keys")
    return VisualDistillationAssets(
        embeddings=embeddings,
        index_by_image_key={str(key): row for row, key in enumerate(image_keys)},
        embedding_path=embedding_path,
    )


def pool_visual_tokens(
    visual_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Mean-pool Qwen visual tokens independently for every input image."""

    token_counts = (
        grid_thw.to(device="cpu", dtype=torch.long).prod(dim=1)
        // int(spatial_merge_size) ** 2
    ).tolist()
    if sum(token_counts) != int(visual_embeds.shape[0]):
        raise ValueError(
            f"Visual token count {visual_embeds.shape[0]} != grid-derived {sum(token_counts)}"
        )
    return torch.stack(
        [chunk.float().mean(dim=0) for chunk in visual_embeds.split(token_counts)], dim=0
    )


def relational_visual_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, int]]]:
    """Match within-query image geometry, without copying SigLIP2 rank scores."""

    if student.shape[0] != teacher.shape[0] or student.shape[0] % group_size:
        raise ValueError("Visual teacher/student rows must form complete candidate groups")
    student = F.normalize(student.float(), dim=-1).view(-1, group_size, student.shape[-1])
    teacher = F.normalize(teacher.float(), dim=-1).view(-1, group_size, teacher.shape[-1])
    student_similarity = student @ student.transpose(1, 2)
    teacher_similarity = teacher @ teacher.transpose(1, 2)
    mask = ~torch.eye(group_size, device=student.device, dtype=torch.bool)
    student_pairs = student_similarity[:, mask]
    teacher_pairs = teacher_similarity[:, mask]
    difference = student_pairs - teacher_pairs
    per_group_loss = difference.square().mean(dim=1)
    per_group_mae = difference.abs().mean(dim=1)
    student_centered = student_pairs - student_pairs.mean(dim=1, keepdim=True)
    teacher_centered = teacher_pairs - teacher_pairs.mean(dim=1, keepdim=True)
    correlation = (student_centered * teacher_centered).sum(dim=1) / (
        student_centered.square().sum(dim=1).sqrt()
        * teacher_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    metrics = {
        "visual_distill": (per_group_loss.detach().sum(), per_group_loss.numel()),
        "visual_similarity_mae": (per_group_mae.detach().sum(), per_group_mae.numel()),
        "visual_similarity_correlation": (
            correlation.detach().sum(),
            correlation.numel(),
        ),
    }
    return per_group_loss.mean(), metrics


class SafeUpdateConfig(pydantic.BaseModel):
    """Exact LoRA-gradient constraints from hard parent decisions only."""

    num_strata: int = pydantic.Field(default=4, ge=1, le=9)
    cushion: float = pydantic.Field(default=0.05, gt=0.0)
    projection_sweeps: int = pydantic.Field(default=2, ge=1, le=8)
    final_grad_norm_clip: float = pydantic.Field(default=1.0, gt=0.0)
    epsilon: float = pydantic.Field(default=1e-30, gt=0.0)


class CustomConfig(pydantic.BaseModel):
    train_dataset: DatasetConfig
    val_dataset: Optional[DatasetConfig] = None
    system_prompt: str = ""
    vision: VisionConfig = pydantic.Field(default_factory=VisionConfig)
    loss: LossConfig = pydantic.Field(default_factory=LossConfig)
    visual_cache: Optional[VisualCacheConfig] = None
    attribute_replay: Optional[AttributeReplayConfig] = None
    structured_attribute: Optional[StructuredAttributeConfig] = None
    hcr: Optional[HCRConfig] = None
    atomic_and: Optional[AtomicAndConfig] = None
    # Optional train-only audit telemetry.  Canonical validation still returns
    # rank loss only; these labels merely expose held-out probe accuracies.
    hcr_validation_targets_for_audit: bool = False
    visual_distillation: Optional[VisualDistillationConfig] = None
    safe_update: Optional[SafeUpdateConfig] = None
    fp32_trainable_master_only: bool = False
    mcq_audit_predictions: bool = False
    mcq_partial_order_weight: float = pydantic.Field(default=0.0, ge=0.0)
    # ``mass`` is the historical set likelihood -log(sum_{p in P} prob(p)).
    # It is satisfied by assigning all probability to only one positive.  The
    # listwise reranker can instead use a uniform target over every positive,
    # which gives every relevant candidate a direct learning signal.
    mcq_positive_set_mode: Literal["mass", "uniform_ce"] = "mass"
    # Factor a four-state pair target into two relevance marginals. Defaults
    # retain the ordinary singleton MCQ cross entropy exactly.
    mcq_factorized_marginal_weight: float = pydantic.Field(default=0.0, ge=0.0)
    mcq_factorized_joint_weight: float = pydantic.Field(default=1.0, ge=0.0)
    # Defense-in-depth for epoch=0 evaluation configs.  The MCQ trainer refuses
    # any optimizer-bearing step if a future runner/config regression enters the
    # training loop unexpectedly.
    validation_only: bool = False

    @pydantic.model_validator(mode="after")
    def disallow_two_attribute_objectives(self):
        attribute_objectives = sum(
            value is not None
            for value in (
                self.attribute_replay,
                self.structured_attribute,
                self.hcr,
                self.atomic_and,
            )
        )
        if attribute_objectives > 1:
            raise ValueError(
                "attribute_replay, structured_attribute, and hcr are mutually exclusive"
            )
        if self.visual_distillation is not None and (
            attribute_objectives
        ):
            raise ValueError(
                "visual_distillation cannot be combined with attribute auxiliary losses"
            )
        if self.loss.retriever_residual_alpha is not None:
            if self.loss.candidate_group_size != 20:
                raise ValueError("retriever residual fusion requires exact K20 groups")
            if self.loss.rank_mode not in {
                "smoothap_soft_r1",
                "incumbent_safe_lambda_ap",
                "misordered_lambda_ap",
                "asymmetric_safe_residual_lambda_ap",
                "robust_normalized_lambdaap_soft_r1",
                "paired_view_robust_normalized_lambdaap_soft_r1",
                "consensus_trimmed_lambdaap_soft_r1",
                "full_gallery_lambda_ap",
            }:
                raise ValueError(
                    "retriever residual fusion requires a K20 LambdaAP/soft-R1 mode"
                )
            if attribute_objectives or self.visual_distillation is not None:
                raise ValueError(
                    "retriever residual fusion is an isolated one-LoRA path"
                )
        if self.safe_update is not None:
            if attribute_objectives or self.visual_distillation is not None:
                raise ValueError("safe_update is an isolated LoRA-only training path")
            if self.loss.rank_mode not in {
                "bag_top1_logsumexp",
                "asymmetric_safe_residual_lambda_ap",
                "misordered_lambda_ap",
            }:
                raise ValueError(
                    "safe_update requires bag_top1_logsumexp, "
                    "asymmetric_safe_residual_lambda_ap, or misordered_lambda_ap"
                )
            if (
                self.loss.rank_mode in {
                    "asymmetric_safe_residual_lambda_ap",
                    "misordered_lambda_ap",
                }
                and self.safe_update.num_strata != 9
            ):
                raise ValueError(
                    "cell-safe residual projection requires exactly nine constraints"
                )
            if self.loss.point_weight or self.loss.teacher_weight:
                raise ValueError("safe_update forbids point and teacher objectives")
        if self.loss.rank_mode == "full_gallery_lambda_ap":
            if self.train_dataset.full_gallery_ap_metadata_path is None:
                raise ValueError(
                    "full_gallery_lambda_ap requires train_dataset."
                    "full_gallery_ap_metadata_path"
                )
            if (
                self.val_dataset is not None
                and self.val_dataset.full_gallery_ap_metadata_path is None
            ):
                raise ValueError(
                    "full_gallery_lambda_ap requires val_dataset."
                    "full_gallery_ap_metadata_path when validation is configured"
                )
        if self.atomic_and is not None:
            expected_rows = (
                self.atomic_and.num_candidates * self.atomic_and.num_fields
            )
            if self.loss.candidate_group_size != expected_rows:
                raise ValueError(
                    "atomic_and requires candidate_group_size="
                    f"num_candidates*num_fields={expected_rows}"
                )
            if self.loss.teacher_weight:
                raise ValueError("atomic_and forbids teacher supervision")
            if self.loss.rank_mode in {
                "orthogonal_cycle",
                "deployment_weighted_robust_cycle",
                "preservation_constrained_cycle_block",
                "dynamic_top20_smoothap_soft_r1",
                "incumbent_guarded_smoothap_soft_r1",
                "incumbent_safe_lambda_ap",
                "misordered_lambda_ap",
                "asymmetric_safe_residual_lambda_ap",
                "robust_normalized_lambdaap_soft_r1",
                "paired_view_robust_normalized_lambdaap_soft_r1",
                "consensus_trimmed_lambdaap_soft_r1",
                "r535_semantic_and_aux",
                "bag_top1_logsumexp",
                "full_gallery_lambda_ap",
            }:
                raise ValueError(
                    f"atomic_and does not support rank_mode={self.loss.rank_mode}"
                )
            if self.loss.point_weight:
                raise ValueError(
                    "atomic_and aggregate point_weight must be zero; use "
                    "atomic_and.field_point_weight for hard atomic labels"
                )
        if self.loss.rank_mode in {
            "deployment_weighted_robust_cycle",
            "preservation_constrained_cycle_block",
            "smoothap_soft_r1",
            "counterfactual_set5",
            "dynamic_top20_smoothap_soft_r1",
            "incumbent_guarded_smoothap_soft_r1",
            "incumbent_safe_lambda_ap",
            "misordered_lambda_ap",
            "asymmetric_safe_residual_lambda_ap",
            "robust_normalized_lambdaap_soft_r1",
            "paired_view_robust_normalized_lambdaap_soft_r1",
            "consensus_trimmed_lambdaap_soft_r1",
            "r535_semantic_and_aux",
        } and any(
            value is not None
            for value in (
                self.attribute_replay,
                self.structured_attribute,
                self.hcr,
                self.visual_distillation,
            )
        ):
            raise ValueError(
                f"{self.loss.rank_mode} is an isolated one-LoRA path; "
                "attribute and visual-distillation wrappers are not supported"
            )
        if self.loss.rank_mode in {
            "smoothap_soft_r1",
            "counterfactual_set5",
            "dynamic_top20_smoothap_soft_r1",
            "incumbent_guarded_smoothap_soft_r1",
            "incumbent_safe_lambda_ap",
            "misordered_lambda_ap",
            "asymmetric_safe_residual_lambda_ap",
            "robust_normalized_lambdaap_soft_r1",
            "paired_view_robust_normalized_lambdaap_soft_r1",
            "consensus_trimmed_lambdaap_soft_r1",
            "r535_semantic_and_aux",
        }:
            datasets = [self.train_dataset]
            if self.val_dataset is not None:
                datasets.append(self.val_dataset)
            if any(dataset.response_mode != "binary" for dataset in datasets):
                raise ValueError(
                    f"{self.loss.rank_mode} requires binary datasets"
                )
            if any(dataset.teacher_calibration_path for dataset in datasets):
                raise ValueError(
                    f"{self.loss.rank_mode} forbids teacher calibration"
                )
        return self


class McqClassificationLoss:
    """One-token A-N classification, including set-valued relevant options.

    Ordinary MCQ rows retain their singleton cross entropy.  PAS listwise rows
    may instead mark every relevant identifier as positive; their primary loss
    is ``-log(sum(P(identifier) for identifier in positives))``.  An optional
    partial-order term asks every positive logit to exceed every negative logit
    without imposing an arbitrary order inside either set.
    """

    def __init__(
        self,
        tokenizer,
        projection_weight,
        max_options: int = 14,
        partial_order_weight: float = 0.0,
        positive_set_mode: Literal["mass", "uniform_ce"] = "mass",
        factorized_marginal_weight: float = 0.0,
        factorized_joint_weight: float = 1.0,
    ):
        if partial_order_weight < 0:
            raise ValueError("MCQ partial_order_weight must be non-negative")
        if positive_set_mode not in {"mass", "uniform_ce"}:
            raise ValueError(f"Unknown MCQ positive-set mode: {positive_set_mode}")
        if factorized_marginal_weight < 0 or factorized_joint_weight < 0:
            raise ValueError("MCQ factorized loss weights must be non-negative")
        if factorized_marginal_weight > 0 and partial_order_weight > 0:
            raise ValueError("Factorized marginal and partial-order MCQ losses are isolated")
        if factorized_marginal_weight > 0 and factorized_joint_weight == 0:
            logger.info("MCQ factorized objective disables joint bit-code CE")
        self.projection_weight = projection_weight
        self.partial_order_weight = float(partial_order_weight)
        self.positive_set_mode = positive_set_mode
        self.factorized_marginal_weight = float(factorized_marginal_weight)
        self.factorized_joint_weight = float(factorized_joint_weight)
        letter_ids = []
        for index in range(max_options):
            letter = chr(ord("A") + index)
            tokens = tokenizer.encode(letter, add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(f"MCQ label {letter!r} is not one token: {tokens}")
            letter_ids.append(tokens[0])
        if len(set(letter_ids)) != len(letter_ids):
            raise ValueError("MCQ option letters do not have unique token IDs")
        self.letter_ids = tuple(letter_ids)
        self.metric_sum: torch.Tensor | None = None
        self.metric_count = 0
        self.audit_metadata: list[dict[str, Any]] | None = None
        self.option_counts: list[int] | None = None
        self.positive_option_indices: list[list[int]] | None = None
        self.factorized_choices: list[list[str]] | None = None
        self.sample_loss_weights: list[float] | None = None
        self.factorized_metric_totals_by_name: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.last_audit_records: list[dict[str, Any]] = []
        logger.info(
            "MCQ objective is per-sample variable-choice option-letter cross "
            "entropy with at most %s options",
            len(self.letter_ids),
        )

    def reset_metrics(self) -> None:
        self.metric_sum = None
        self.metric_count = 0
        self.factorized_metric_totals_by_name = {}

    def set_audit_metadata(self, metadata: list[dict[str, Any]] | None) -> None:
        self.audit_metadata = metadata

    def set_option_counts(self, option_counts: list[int]) -> None:
        self.option_counts = [int(count) for count in option_counts]

    def set_positive_option_indices(
        self, positive_option_indices: list[list[int]] | None
    ) -> None:
        self.positive_option_indices = (
            None
            if positive_option_indices is None
            else [list(map(int, indices)) for indices in positive_option_indices]
        )

    def set_factorized_metadata(
        self,
        choices: list[list[str]],
        sample_loss_weights: list[float],
    ) -> None:
        if self.factorized_choices is not None or self.sample_loss_weights is not None:
            raise ValueError("Previous MCQ factorized metadata were not consumed")
        if len(choices) != len(sample_loss_weights):
            raise ValueError("MCQ factorized choices/weights length mismatch")
        weights = [float(value) for value in sample_loss_weights]
        if any(not math.isfinite(value) or value <= 0 for value in weights):
            raise ValueError("sample_loss_weight must be finite and strictly positive")
        self.factorized_choices = [list(map(str, row)) for row in choices]
        self.sample_loss_weights = weights

    def factorized_metric_totals(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return self.factorized_metric_totals_by_name

    @staticmethod
    def _weighted_mean(per_row: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """DP-exact weighted mean under ordinary averaged DDP gradients."""

        denominator = weights.detach().sum().float()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise ValueError("MCQ factorized sample weights have no positive mass")
        return (per_row * weights).sum() * world_size / denominator

    def pop_audit_records(self) -> list[dict[str, Any]]:
        records = self.last_audit_records
        self.last_audit_records = []
        return records

    def metric_totals(self) -> tuple[torch.Tensor | None, int]:
        return self.metric_sum, self.metric_count

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        if kwargs.get("output_packing_mask") is not None:
            raise ValueError("MCQ classification does not support sequence packing")
        labelled = target.ne(-100)
        if not bool(labelled.any(dim=-1).all()):
            raise ValueError("MCQ batch contains a response without supervised tokens")
        answer_positions = labelled.to(torch.int64).argmax(dim=-1)
        batch_indices = torch.arange(target.shape[0], device=target.device)
        target_tokens = target[batch_indices, answer_positions]
        letter_ids = torch.tensor(self.letter_ids, device=target.device)
        target_matches = target_tokens[:, None].eq(letter_ids[None, :])
        if not bool(target_matches.sum(dim=-1).eq(1).all()):
            raise ValueError("MCQ assistant response is not exactly one A-N token")
        target_classes = target_matches.to(torch.int64).argmax(dim=-1)
        if bool(answer_positions.eq(0).any()):
            raise ValueError("MCQ answer token cannot be the first sequence token")
        hidden = output[batch_indices, answer_positions - 1]
        # Under FSDP2 the frozen LM-head parameter is a replicated DTensor,
        # while the token IDs and hidden states passed to the loss are local
        # tensors.  Project with the local replica, just as the PAS yes/no
        # objective does, to avoid mixing Tensor and DTensor operations.
        local_projection = self.projection_weight
        if hasattr(local_projection, "to_local"):
            local_projection = local_projection.to_local()
        letter_weight = local_projection.index_select(0, letter_ids)
        logits = F.linear(hidden.to(letter_weight.dtype), letter_weight).float()
        if self.option_counts is None:
            raise ValueError("MCQ option counts must be supplied for every batch")
        if len(self.option_counts) != int(logits.shape[0]):
            raise ValueError(
                "MCQ option counts and classifier batch sizes disagree: "
                f"counts={len(self.option_counts)}, logits={logits.shape[0]}"
            )
        option_counts = torch.tensor(self.option_counts, device=logits.device)
        if bool(option_counts.lt(2).any()) or bool(option_counts.gt(len(self.letter_ids)).any()):
            raise ValueError(
                f"MCQ option counts must be in [2, {len(self.letter_ids)}]: "
                f"{self.option_counts}"
            )
        valid_options = torch.arange(
            len(self.letter_ids), device=logits.device
        )[None, :].lt(option_counts[:, None])
        if not bool(valid_options[batch_indices, target_classes].all()):
            raise ValueError("MCQ target letter lies outside the sample's choices")
        logits = logits.masked_fill(~valid_options, float("-inf"))
        self.option_counts = None
        if self.positive_option_indices is None:
            positive_indices = [[int(target)] for target in target_classes.tolist()]
        else:
            if len(self.positive_option_indices) != int(logits.shape[0]):
                raise ValueError(
                    "MCQ positive sets and classifier batch sizes disagree: "
                    f"sets={len(self.positive_option_indices)}, logits={logits.shape[0]}"
                )
            positive_indices = self.positive_option_indices
        self.positive_option_indices = None
        positive_mask = torch.zeros_like(valid_options)
        for row_index, indices in enumerate(positive_indices):
            unique = sorted(set(indices))
            if not unique:
                raise ValueError("MCQ positive option set cannot be empty")
            if unique[0] < 0 or unique[-1] >= int(option_counts[row_index]):
                raise ValueError(
                    f"MCQ positive option lies outside row {row_index}'s choices: {unique}"
                )
            if int(target_classes[row_index]) not in unique:
                raise ValueError(
                    "MCQ placeholder answer token must belong to its positive set"
                )
            positive_mask[row_index, unique] = True
        if self.factorized_marginal_weight:
            if self.factorized_choices is None or self.sample_loss_weights is None:
                raise ValueError("Factorized MCQ loss requires per-row state choices/weights")
            if len(self.factorized_choices) != int(logits.shape[0]):
                raise ValueError("MCQ factorized choices and classifier batch disagree")
            if any(len(indices) != 1 for indices in positive_indices):
                raise ValueError("Factorized bit-code MCQ requires singleton hard targets")
            marginal_losses = []
            per_bit_losses: list[list[torch.Tensor]] = []
            bit_widths = []
            for row_index, (choices, target_class) in enumerate(
                zip(self.factorized_choices, target_classes.tolist(), strict=True)
            ):
                if len(choices) != int(option_counts[row_index]):
                    raise ValueError("Factorized bit-code choices disagree with option count")
                widths = {len(state) for state in choices}
                if len(widths) != 1:
                    raise ValueError("Factorized bit-code choices have mixed widths")
                bit_width = next(iter(widths))
                if bit_width < 1:
                    raise ValueError("Factorized bit-code choices cannot be empty")
                expected_states = {
                    format(index, f"0{bit_width}b") for index in range(1 << bit_width)
                }
                if set(choices) != expected_states:
                    raise ValueError(
                        "Factorized choices must be one complete permutation of all "
                        f"{bit_width}-bit codes"
                    )
                target_state = choices[target_class]
                row_logits = logits[row_index, : len(choices)]
                row_bit_losses = []
                for bit_index in range(bit_width):
                    bit_one = torch.tensor(
                        [state[bit_index] == "1" for state in choices],
                        device=logits.device,
                        dtype=torch.bool,
                    )
                    # Positive BCE logit is log P(bit=1) - log P(bit=0).
                    bit_logit = torch.logsumexp(row_logits[bit_one], dim=0) - torch.logsumexp(
                        row_logits[~bit_one], dim=0
                    )
                    row_bit_losses.append(
                        F.binary_cross_entropy_with_logits(
                            bit_logit,
                            row_logits.new_tensor(float(target_state[bit_index] == "1")),
                        )
                    )
                per_bit_losses.append(row_bit_losses)
                marginal_losses.append(torch.stack(row_bit_losses).mean())
                bit_widths.append(bit_width)
            marginal_losses = torch.stack(marginal_losses)
            joint_losses = F.cross_entropy(logits, target_classes, reduction="none")
            weights = logits.new_tensor(self.sample_loss_weights)
            per_row = (
                self.factorized_marginal_weight * marginal_losses
                + self.factorized_joint_weight * joint_losses
            )
            loss = self._weighted_mean(per_row, weights)
            # Keep one identical metric schema on every DP rank. A mixed
            # 2-bit/3-bit batch otherwise makes ranks enter different numbers
            # of all-reduces and deadlocks after the first optimizer step.
            detached_weights = weights.detach()
            metric_values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "marginal": (marginal_losses, detached_weights),
                "joint": (joint_losses, detached_weights),
            }
            for bit_index in range(3):  # 2**4 exceeds the A-N option budget.
                indices = [
                    row_index
                    for row_index, width in enumerate(bit_widths)
                    if width > bit_index
                ]
                metric_values[f"bit_{bit_index}_marginal"] = (
                    torch.stack([per_bit_losses[index][bit_index] for index in indices])
                    if indices
                    else logits.new_empty((0,)),
                    detached_weights[indices]
                    if indices
                    else detached_weights.new_empty((0,)),
                )
            pair_indices = [
                row_index for row_index, width in enumerate(bit_widths) if width == 2
            ]
            for name, bit_index in (("left_marginal", 0), ("right_marginal", 1)):
                metric_values[name] = (
                    torch.stack([per_bit_losses[index][bit_index] for index in pair_indices])
                    if pair_indices
                    else logits.new_empty((0,)),
                    detached_weights[pair_indices]
                    if pair_indices
                    else detached_weights.new_empty((0,)),
                )
            for name, (values, metric_weights) in metric_values.items():
                metric_sum = (values.detach() * metric_weights).sum()
                metric_weight = metric_weights.sum()
                if name in self.factorized_metric_totals_by_name:
                    previous_sum, previous_weight = self.factorized_metric_totals_by_name[name]
                    metric_sum = previous_sum + metric_sum
                    metric_weight = previous_weight + metric_weight
                self.factorized_metric_totals_by_name[name] = (
                    metric_sum,
                    metric_weight,
                )
            self.factorized_choices = None
            self.sample_loss_weights = None
        else:
            # Clear metadata even for the default 0/1 setting, whose singleton
            # joint CE is exactly the historical MCQ probability-mass loss.
            self.factorized_choices = None
            sample_weights = self.sample_loss_weights
            self.sample_loss_weights = None
            log_probabilities = F.log_softmax(logits, dim=-1)
            if self.positive_set_mode == "uniform_ce":
                per_row_primary = -(
                    log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=-1)
                    / positive_mask.sum(dim=-1)
                )
            else:
                positive_log_mass = torch.logsumexp(
                    log_probabilities.masked_fill(~positive_mask, float("-inf")), dim=-1
                )
                per_row_primary = -positive_log_mass
            if sample_weights is None or all(value == 1.0 for value in sample_weights):
                loss = per_row_primary.mean()
            else:
                weights = logits.new_tensor(sample_weights)
                loss = self._weighted_mean(per_row_primary, weights)
        if self.partial_order_weight:
            pair_losses = []
            for row_logits, row_positive, row_valid in zip(
                logits, positive_mask, valid_options, strict=True
            ):
                positive_logits = row_logits[row_positive]
                negative_logits = row_logits[row_valid & ~row_positive]
                if positive_logits.numel() and negative_logits.numel():
                    pair_losses.append(
                        F.softplus(
                            negative_logits[:, None] - positive_logits[None, :]
                        ).mean()
                    )
            if pair_losses:
                loss = loss + self.partial_order_weight * torch.stack(pair_losses).mean()
        loss = loss * float(kwargs.get("loss_scaling_factor", 1.0))
        predicted = logits.argmax(dim=-1)
        correct = positive_mask[batch_indices, predicted]
        self.last_audit_records = []
        if self.audit_metadata is not None:
            if len(self.audit_metadata) != int(logits.shape[0]):
                raise ValueError(
                    "MCQ audit metadata and classifier batch sizes disagree: "
                    f"metadata={len(self.audit_metadata)}, logits={logits.shape[0]}"
                )
            probabilities = logits.softmax(dim=-1)
            for row, target_class, predicted_class, row_logits, row_probabilities, row_positive in zip(
                self.audit_metadata,
                target_classes.tolist(),
                predicted.tolist(),
                logits.tolist(),
                probabilities.tolist(),
                positive_mask.tolist(),
            ):
                choices = list(row["choices"])
                predicted_value = (
                    choices[predicted_class]
                    if predicted_class < len(choices)
                    else f"__invalid_option_{predicted_class}"
                )
                positive_classes = [
                    index for index, is_positive in enumerate(row_positive[: len(choices)])
                    if is_positive
                ]
                wrong_classes = [
                    index for index in range(len(choices)) if index not in positive_classes
                ]
                record = {
                        **row,
                        "target_class": int(target_class),
                        "positive_classes": positive_classes,
                        "predicted_class": int(predicted_class),
                        "predicted_value": predicted_value,
                        "correct": bool(predicted_class in positive_classes),
                        "target_probability": float(row_probabilities[target_class]),
                        "positive_set_probability": float(
                            sum(row_probabilities[index] for index in positive_classes)
                        ),
                        "probabilities_by_value": {
                            str(value): float(row_probabilities[index])
                            for index, value in enumerate(choices)
                        },
                    }
                if wrong_classes:
                    hardest_wrong = max(wrong_classes, key=row_logits.__getitem__)
                    record.update(
                        {
                            "hardest_wrong_class": int(hardest_wrong),
                            "hardest_wrong_value": choices[hardest_wrong],
                            "hardest_wrong_probability": float(
                                row_probabilities[hardest_wrong]
                            ),
                            "best_positive_minus_hardest_wrong_logit": float(
                                max(row_logits[index] for index in positive_classes)
                                - row_logits[hardest_wrong]
                            ),
                            **(
                                {
                                    "target_minus_hardest_wrong_logit": float(
                                        row_logits[target_class]
                                        - row_logits[hardest_wrong]
                                    )
                                }
                                if len(positive_classes) == 1
                                else {}
                            ),
                        }
                    )
                self.last_audit_records.append(record)
        self.audit_metadata = None
        batch_sum = correct.float().sum().detach()
        self.metric_sum = batch_sum if self.metric_sum is None else self.metric_sum + batch_sum
        self.metric_count += int(correct.numel())
        return loss


def query_index_from_sample(sample: dict[str, Any]) -> int:
    """Resolve the source query index after targeted datasets rename groups."""

    query_group_id = str(
        sample.get("original_group_id") or sample.get("group_id") or ""
    )
    match = re.fullmatch(r"query_(\d+)(?:_view_\d+)?", query_group_id)
    if match is None:
        raise ValueError(
            "Mismatch-aware attributes require a mined query_<index> group_id "
            "or original_group_id, got "
            f"group_id={sample.get('group_id')!r}, "
            f"original_group_id={sample.get('original_group_id')!r}"
        )
    return int(match.group(1))


class PasConversationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        custom_config: CustomConfig,
        dataset_config: DatasetConfig,
        *,
        attribute_replay_assets: AttributeReplayAssets | None = None,
        structured_attribute_assets: StructuredAttributeAssets | None = None,
        hcr_assets: DenseHCRAssets | None = None,
        visual_distillation_assets: VisualDistillationAssets | None = None,
    ):
        annotation_paths = (
            [dataset_config.annotation_path]
            if isinstance(dataset_config.annotation_path, str)
            else dataset_config.annotation_path
        )
        if not annotation_paths:
            raise ValueError("PAS annotation_path list cannot be empty")
        self.annotations = []
        for annotation_path in annotation_paths:
            with open(annotation_path, encoding="utf-8") as handle:
                shard = json.load(handle)
            if not isinstance(shard, list):
                raise TypeError(
                    f"PAS annotations in {annotation_path} must be a JSON list"
                )
            self.annotations.extend(shard)
        logger.info(
            "Loaded %s PAS annotation shard(s), total rows=%s",
            len(annotation_paths),
            len(self.annotations),
        )
        self.media_path = dataset_config.media_path
        self.system_prompt = custom_config.system_prompt
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)
        self.group_size = custom_config.loss.candidate_group_size
        self.rank_mode = custom_config.loss.rank_mode
        self.positive_response = custom_config.loss.positive_response
        self.negative_response = custom_config.loss.negative_response
        self.binary_decision_position = custom_config.loss.binary_decision_position
        self.score_mode = custom_config.loss.score_mode
        self.requirement_ordinal_weight = custom_config.loss.requirement_ordinal_weight
        self.response_mode = dataset_config.response_mode
        self.prompt_mode = dataset_config.prompt_mode
        self.score_audit_metadata = dataset_config.score_audit_metadata
        self.total_relevant_by_group: dict[str, int] | None = None
        if dataset_config.full_gallery_ap_metadata_path is not None:
            self.total_relevant_by_group = load_full_gallery_ap_metadata(
                dataset_config.full_gallery_ap_metadata_path
            ).as_lookup()
        self.teacher_positive_floor = dataset_config.teacher_positive_floor
        self.teacher_negative_ceiling = dataset_config.teacher_negative_ceiling
        self.teacher_calibrations: dict[str, tuple[float, float]] | None = None
        if dataset_config.teacher_calibration_path:
            calibration_path = Path(dataset_config.teacher_calibration_path)
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            if payload.get("score_field") != "retriever_score":
                raise ValueError(
                    "PAS teacher calibration must declare score_field=retriever_score"
                )
            calibrations = payload.get("calibrations")
            if not isinstance(calibrations, dict):
                raise ValueError("PAS teacher calibration has no calibrations mapping")
            self.teacher_calibrations = {
                str(query_type): (
                    float(values["coefficient"]),
                    float(values["intercept"]),
                )
                for query_type, values in calibrations.items()
            }
        self.visual_cache_enabled = custom_config.visual_cache is not None
        self.attribute_replay = (
            custom_config.attribute_replay
            if attribute_replay_assets is not None
            else None
        )
        self.attribute_labels_by_image = (
            attribute_replay_assets.labels_by_image
            if attribute_replay_assets is not None
            else None
        )
        self.structured_attribute_assets = structured_attribute_assets
        self.hcr_enabled = custom_config.hcr is not None
        self.hcr_mode = (
            custom_config.hcr.mode
            if custom_config.hcr is not None
            else "legacy_compatibility"
        )
        self.hcr_predecision_prompt_mode = (
            custom_config.hcr.predecision_prompt_mode
            if custom_config.hcr is not None
            else "semantic_legend"
        )
        self.hcr_assets = hcr_assets
        self.structured_attribute_seed = (
            int(custom_config.structured_attribute.seed)
            if structured_attribute_assets is not None
            and custom_config.structured_attribute is not None
            else None
        )
        self.structured_attribute_mismatch_only = bool(
            structured_attribute_assets is not None
            and custom_config.structured_attribute is not None
            and custom_config.structured_attribute.mismatch_only
        )
        self.visual_teacher_index_by_image_key = (
            visual_distillation_assets.index_by_image_key
            if visual_distillation_assets is not None
            else None
        )
        self.group_ranges = self._candidate_group_ranges()
        if self.total_relevant_by_group is not None:
            annotation_group_ids = {
                str(self.annotations[begin].get("group_id") or "")
                for begin, _ in self.group_ranges
            }
            metadata_group_ids = set(self.total_relevant_by_group)
            if annotation_group_ids != metadata_group_ids:
                raise ValueError(
                    "Full-gallery AP metadata does not exactly match annotations: "
                    f"missing={len(annotation_group_ids - metadata_group_ids)}, "
                    f"extra={len(metadata_group_ids - annotation_group_ids)}"
                )
        self._validate_candidate_groups()

    def _teacher_probability(self, sample: dict) -> float | None:
        explicit = sample.get("teacher_probability")
        if explicit is not None:
            probability = float(explicit)
        elif self.teacher_calibrations is not None:
            query_type = str(sample.get("query_type") or "")
            if query_type not in self.teacher_calibrations:
                raise ValueError(
                    f"No teacher calibration for query_type={query_type!r}"
                )
            if "retriever_score" not in sample:
                raise ValueError("Teacher-calibrated row has no retriever_score")
            coefficient, intercept = self.teacher_calibrations[query_type]
            logit = coefficient * float(sample["retriever_score"]) + intercept
            probability = (
                1.0 / (1.0 + math.exp(-logit))
                if logit >= 0
                else math.exp(logit) / (1.0 + math.exp(logit))
            )
        else:
            return None
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Teacher probability lies outside [0, 1]: {probability}")
        if self.teacher_positive_floor is not None:
            if "label" not in sample:
                raise ValueError("Label-safe teacher row has no PAS label")
            probability = label_safe_teacher_probability(
                probability,
                int(sample["label"]),
                positive_floor=self.teacher_positive_floor,
                negative_ceiling=self.teacher_negative_ceiling,
            )
        return probability

    def _candidate_group_ranges(self) -> list[tuple[int, int]]:
        if self.response_mode == "mcq":
            return [(index, index + 1) for index in range(len(self.annotations))]
        if len(self.annotations) % self.group_size:
            raise ValueError(
                f"Annotation count {len(self.annotations)} is not divisible by "
                f"candidate_group_size={self.group_size}"
            )
        return [
            (begin, begin + self.group_size)
            for begin in range(0, len(self.annotations), self.group_size)
        ]

    def _validate_candidate_groups(self) -> None:
        if self.response_mode == "mcq":
            for index, row in enumerate(self.annotations):
                answer = row["conversations"][1]["value"]
                expected = row["label_letter"]
                if answer != expected:
                    raise ValueError(
                        f"MCQ row {index} response {answer!r} != {expected!r}"
                    )
                if row["label_value"] not in row["choices"]:
                    raise ValueError(
                        f"MCQ row {index} answer is absent from its choices"
                    )
                positive_letters = row.get("positive_label_letters")
                if positive_letters is not None:
                    if not isinstance(positive_letters, list) or not positive_letters:
                        raise ValueError(
                            f"MCQ row {index} positive_label_letters must be nonempty"
                        )
                    option_letters = {
                        chr(ord("A") + option) for option in range(len(row["choices"]))
                    }
                    if len(set(map(str, positive_letters))) != len(positive_letters):
                        raise ValueError(f"MCQ row {index} has duplicate positive letters")
                    if not set(map(str, positive_letters)) <= option_letters:
                        raise ValueError(
                            f"MCQ row {index} has a positive letter outside its choices"
                        )
                    if str(expected) not in set(map(str, positive_letters)):
                        raise ValueError(
                            f"MCQ row {index} placeholder answer is not positive"
                        )
                    positive_values = row.get("positive_label_values")
                    if positive_values is not None:
                        expected_values = {
                            str(row["choices"][ord(str(letter)) - ord("A")])
                            for letter in positive_letters
                        }
                        if set(map(str, positive_values)) != expected_values:
                            raise ValueError(
                                f"MCQ row {index} positive values disagree with letters"
                            )
            return
        previous_group_id = None
        for begin, end in self.group_ranges:
            rows = self.annotations[begin:end]
            group_ids = {row.get("group_id") for row in rows}
            if None in group_ids or len(group_ids) != 1:
                raise ValueError(
                    f"Rows {begin}:{end} must have one shared group_id"
                )
            group_id = next(iter(group_ids))
            if group_id == previous_group_id:
                raise ValueError(f"group_id={group_id!r} spans multiple candidate blocks")
            previous_group_id = group_id
            labels = []
            retriever_ranks = []
            for row in rows:
                response = row["conversations"][1]["value"]
                if (
                    self.response_mode == "binary"
                    and self.score_mode in {
                        "relevance_1to5_supervised_expected",
                        "relevance_1to5_supervised_strict_logodds",
                        "binary_delta_with_ordinal_aux",
                    }
                ):
                    grade = int(row.get("relevance_grade", 0))
                    if grade not in {1, 2, 3, 4, 5}:
                        raise ValueError(
                            f"group_id={group_id!r} has invalid relevance_grade={grade}"
                        )
                    expected_response = f"<score>{grade}</score>"
                    if response != expected_response:
                        raise ValueError(
                            f"group_id={group_id!r} response {response!r} != "
                            f"ordinal target {expected_response!r}"
                        )
                    response_label = int(grade == 5)
                elif self.response_mode == "binary" and self.score_mode == "binary_delta":
                    if self.binary_decision_position == "prefix":
                        supported_responses = {
                            self.positive_response,
                            self.negative_response,
                        }
                        if response not in supported_responses:
                            raise ValueError(
                                f"group_id={group_id!r} has unsupported response {response!r}"
                            )
                        response_label = int(response.startswith(self.positive_response))
                    else:
                        positive = response.endswith(self.positive_response)
                        negative = response.endswith(self.negative_response)
                        if positive == negative:
                            raise ValueError(
                                f"group_id={group_id!r} has no unique final yes/no response"
                            )
                        if response.count("<answer>") != 1:
                            raise ValueError(
                                f"group_id={group_id!r} has an ambiguous answer marker"
                            )
                        response_label = int(positive)
                elif self.response_mode != "binary":
                    if not response.startswith("<think>\n") or "\n</think>\n" not in response:
                        raise ValueError(
                            f"group_id={group_id!r} has malformed reasoning response"
                        )
                    positive = response.endswith(self.positive_response)
                    negative = response.endswith(self.negative_response)
                    if positive == negative:
                        raise ValueError(
                            f"group_id={group_id!r} reasoning response has no unique answer"
                        )
                    response_label = int(positive)
                else:
                    # Ordinal scoring rewrites the legacy yes/no annotation at
                    # access time.  PAS's explicit label remains authoritative.
                    response_label = int(row["label"])
                if "label" in row and int(row["label"]) != response_label:
                    raise ValueError(
                        f"group_id={group_id!r} label disagrees with assistant response"
                    )
                labels.append(response_label)
                if self.requirement_ordinal_weight:
                    satisfied = int(row.get("requirement_satisfied_count", -1))
                    total = int(row.get("requirement_total_count", -1))
                    weight = float(row.get("requirement_ordinal_weight", -1))
                    if total < 1 or satisfied < 0 or satisfied > total or weight < 0:
                        raise ValueError(
                            f"group_id={group_id!r} has invalid requirement coverage "
                            f"satisfied={satisfied}, total={total}, weight={weight}"
                        )
                    if int(satisfied == total) != int(row["label"]):
                        raise ValueError(
                            f"group_id={group_id!r} requirement coverage disagrees "
                            "with exact conjunction label"
                        )
                if "retriever_rank" in row:
                    retriever_ranks.append(int(row["retriever_rank"]))
            if not 0 < sum(labels) < len(labels):
                raise ValueError(
                    f"group_id={group_id!r} must contain at least one positive and "
                    f"one negative; got {labels}"
                )
            if self.rank_mode == "natural20_inverse_pair":
                if len(rows) != 21:
                    raise ValueError(
                        "natural20_inverse_pair requires exactly 21 rows per group"
                    )
                offsets = {
                    row.get("inverse_pair_original_offset") for row in rows
                }
                if len(offsets) != 1 or None in offsets:
                    raise ValueError(
                        f"group_id={group_id!r} has inconsistent inverse-pair offsets"
                    )
                original_offset = int(next(iter(offsets)))
                if original_offset < 0 or original_offset >= 20:
                    raise ValueError(
                        f"group_id={group_id!r} inverse-pair offset is outside [0, 19]"
                    )
                natural_queries = {
                    " ".join(str(row.get("query") or "").casefold().split())
                    for row in rows[:20]
                }
                corrected_query = " ".join(
                    str(rows[20].get("query") or "").casefold().split()
                )
                if len(natural_queries) != 1 or corrected_query in natural_queries:
                    raise ValueError(
                        f"group_id={group_id!r} does not contain natural K20 plus corrected Q+"
                    )
                if (
                    labels[original_offset] != 0
                    or labels[20] != 1
                    or rows[20].get("image") != rows[original_offset].get("image")
                ):
                    raise ValueError(
                        f"group_id={group_id!r} has a broken same-image inverse pair"
                    )
            if self.rank_mode == "weak_veto_mil":
                expected_labels = [1, 0] * (len(rows) // 2)
                expected_roles = ["full_positive", "full_negative"] + [
                    role
                    for _ in range(len(rows) // 2 - 1)
                    for role in ("field_positive", "field_negative")
                ]
                roles = [row.get("weak_veto_role") for row in rows]
                fields = [row.get("weak_veto_field") for row in rows]
                if labels != expected_labels or roles != expected_roles:
                    raise ValueError(
                        f"group_id={group_id!r} has invalid weak-veto pair roles"
                    )
                if fields[:2] != ["full", "full"]:
                    raise ValueError(
                        f"group_id={group_id!r} must mark its first pair as full"
                    )
                atomic_fields = []
                for offset in range(2, len(rows), 2):
                    if (
                        not fields[offset]
                        or fields[offset] != fields[offset + 1]
                        or str(rows[offset].get("query") or "")
                        != str(rows[offset + 1].get("query") or "")
                    ):
                        raise ValueError(
                            f"group_id={group_id!r} has a malformed weak-veto field pair"
                        )
                    atomic_fields.append(str(fields[offset]))
                if len(set(atomic_fields)) != len(atomic_fields):
                    raise ValueError(
                        f"group_id={group_id!r} repeats a weak-veto field hypothesis"
                    )
                positive_images = {
                    str(rows[offset].get("image"))
                    for offset in range(0, len(rows), 2)
                }
                negative_images = {
                    str(rows[offset].get("image"))
                    for offset in range(1, len(rows), 2)
                }
                if (
                    len(positive_images) != 1
                    or len(negative_images) != 1
                    or positive_images == negative_images
                    or str(rows[0].get("query") or "")
                    != str(rows[1].get("query") or "")
                ):
                    raise ValueError(
                        f"group_id={group_id!r} does not reuse one positive/negative image pair"
                    )
            if self.rank_mode in {
                "plackett_luce_policy",
                "rb_plackett_luce_expected_ap",
            }:
                preserve_flags = {
                    row.get("policy_preserve_group") for row in rows
                }
                if preserve_flags not in ({True}, {False}):
                    raise ValueError(
                        f"group_id={group_id!r} requires one shared hard "
                        f"policy_preserve_group flag; got {preserve_flags}"
                    )
            if retriever_ranks:
                if len(retriever_ranks) != len(rows):
                    raise ValueError(
                        f"group_id={group_id!r} must provide retriever_rank for "
                        "every candidate"
                    )
                explicit_parent_ranks = [
                    int(row["parent_order_rank"])
                    for row in rows
                    if "parent_order_rank" in row
                ]
                if explicit_parent_ranks and (
                    len(explicit_parent_ranks) != len(rows)
                    or explicit_parent_ranks != list(range(1, len(rows) + 1))
                ):
                    raise ValueError(
                        f"group_id={group_id!r} must retain parent_order_rank 1..K; "
                        f"got {explicit_parent_ranks}"
                    )
                parent_ordered = bool(explicit_parent_ranks) or (
                    self.rank_mode
                    in {
                        "incumbent_guarded_smoothap_soft_r1",
                        "incumbent_safe_lambda_ap",
                    }
                )
                if self.rank_mode == "paired_view_robust_normalized_lambdaap_soft_r1":
                    valid_retriever_ranks = (
                        len(rows) == 40
                        and retriever_ranks[:20] == list(range(1, 21))
                        and retriever_ranks[20:] == list(range(1, 21))
                    )
                else:
                    valid_retriever_ranks = (
                        sorted(retriever_ranks) == list(range(1, len(rows) + 1))
                        if parent_ordered
                        else retriever_ranks[0] == 1
                        and all(
                            right > left
                            for left, right in zip(
                                retriever_ranks, retriever_ranks[1:]
                            )
                        )
                    )
                if not valid_retriever_ranks:
                    required_order = (
                        "a complete retriever-rank permutation"
                        if parent_ordered
                        else "strictly increasing retriever ranks beginning at 1"
                    )
                    raise ValueError(
                        f"group_id={group_id!r} must retain {required_order}; "
                        f"got {retriever_ranks}"
                    )

    def setup(self, config, tokenizer=None):
        return None

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> list[dict]:
        sample = self.annotations[index]
        conversations = sample["conversations"]
        user_prompt = conversations[0]["value"]
        response = conversations[1]["value"]
        if self.prompt_mode == "strict_conjunction":
            if self.response_mode != "binary":
                raise ValueError(
                    "strict_conjunction prompt mode requires binary responses"
                )
            user_prompt = strict_conjunction_prompt(str(sample.get("query") or ""))
        elif self.prompt_mode == "canonical_conjunction":
            if self.response_mode != "binary":
                raise ValueError(
                    "canonical_conjunction prompt mode requires binary responses"
                )
            user_prompt = canonical_conjunction_prompt(
                str(sample.get("query") or "")
            )
        if self.score_mode in {
            "relevance_1to5_expected",
            "relevance_1to5_supervised_expected",
            "relevance_1to5_supervised_strict_logodds",
        }:
            if self.response_mode != "binary":
                raise ValueError("Ordinal relevance scoring requires grouped PAS data")
            if self.score_mode in {
                "relevance_1to5_supervised_expected",
                "relevance_1to5_supervised_strict_logodds",
            }:
                grade = int(sample.get("relevance_grade", 0))
                if grade not in {1, 2, 3, 4, 5}:
                    raise ValueError("Supervised ordinal row has no valid relevance_grade")
                response = f"<score>{grade}</score>"
            else:
                response = (
                    self.positive_response
                    if int(sample["label"]) == 1
                    else self.negative_response
                )
            query = str(sample.get("query") or "").strip()
            if not query:
                raise ValueError("Ordinal relevance row is missing its query text")
            user_prompt = (
                f'<image>\nQuery: "{query}"\nCheck every stated clothing type, '
                "color, footwear, accessory, and viewpoint requirement independently. "
                "Give 5 only when every requirement matches; give 4 for exactly one "
                "mismatch, 3 for two, 2 for three, and 1 for four or more mismatches. "
                "Do not compensate for a mismatch with other matches. Answer only as "
                "<score>N</score>, where N is 1, 2, 3, 4, or 5."
            )
        elif self.score_mode == "binary_delta_with_ordinal_aux":
            grade = int(sample.get("relevance_grade", 0))
            if grade not in {1, 2, 3, 4, 5}:
                raise ValueError("Ordinal auxiliary row has no valid relevance_grade")
            binary_response = (
                self.positive_response
                if int(sample["label"]) == 1
                else self.negative_response
            )
            # The newline is a tokenizer boundary: without it Qwen merges the
            # closing ``>`` of </answer> with the opening ``<`` of <score>, so
            # the standalone ordinal response spec cannot locate the suffix.
            response = f"{binary_response}\n<score>{grade}</score>"
        mcq_metadata = None
        teacher_probability = self._teacher_probability(sample)
        visual_teacher_index = None
        if self.visual_teacher_index_by_image_key is not None:
            source_image_key = str(sample.get("source_image_key") or "")
            if not source_image_key:
                raise ValueError(
                    "Visual-distillation row is missing source_image_key"
                )
            try:
                visual_teacher_index = self.visual_teacher_index_by_image_key[
                    source_image_key
                ]
            except KeyError as error:
                raise KeyError(
                    f"No SigLIP2 teacher embedding for {source_image_key!r}"
                ) from error
        runtime_group_metadata = (
            {
                "_pas_group_id": str(sample["group_id"]),
                "_pas_binary_label": int(sample["label"]),
                "_pas_query_identity": str(
                    sample.get("original_group_id") or sample["group_id"]
                ),
                "_pas_query_text": str(sample.get("query") or ""),
                "_pas_dataset": str(sample.get("dataset") or ""),
                "_pas_query_type": str(sample.get("query_type") or ""),
                "_pas_retriever_rank": int(sample.get("retriever_rank") or 0),
                **(
                    {"_pas_full_list_view_role": str(sample["full_list_view_role"])}
                    if sample.get("full_list_view_role") is not None
                    else {}
                ),
                **(
                    {"_pas_view_candidate_id": str(sample["view_candidate_id"])}
                    if sample.get("view_candidate_id") is not None
                    else {}
                ),
                **(
                    {"_pas_retriever_score": float(sample["retriever_score"])}
                    if sample.get("retriever_score") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_inverse_pair_original_offset": int(
                            sample["inverse_pair_original_offset"]
                        )
                    }
                    if sample.get("inverse_pair_original_offset") is not None
                    else {}
                ),
                **(
                    {"_pas_weak_veto_role": str(sample["weak_veto_role"])}
                    if sample.get("weak_veto_role") is not None
                    else {}
                ),
                **(
                    {"_pas_weak_veto_field": str(sample["weak_veto_field"])}
                    if sample.get("weak_veto_field") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_rank_loss_weight": float(
                            sample["rank_loss_weight"]
                        )
                    }
                    if sample.get("rank_loss_weight") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_same_label_consistency_role": str(
                            sample["same_label_consistency_role"]
                        )
                    }
                    if sample.get("same_label_consistency_role") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_requirement_coverage": (
                            int(sample["requirement_satisfied_count"]),
                            int(sample["requirement_total_count"]),
                            float(sample["requirement_ordinal_weight"]),
                        )
                    }
                    if sample.get("requirement_total_count") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_parent_margin": float(sample["parent_margin"]),
                        "_pas_parent_anchor": bool(sample["parent_anchor"]),
                        "_pas_parent_top1_correct": bool(
                            sample["parent_top1_correct"]
                        ),
                        "_pas_parent_margin_stratum": int(
                            sample["parent_margin_stratum"]
                        ),
                    }
                    if sample.get("parent_margin") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_score_audit": {
                            "id": str(sample.get("id", index)),
                            "group_id": str(sample["group_id"]),
                            "image": str(
                                sample.get("image")
                                or (sample.get("images") or [""])[0]
                            ),
                            "source_image_key": str(
                                sample.get("source_image_key") or ""
                            ),
                            "dataset": str(sample.get("dataset") or ""),
                            "query_type": str(sample.get("query_type") or ""),
                            "query": str(sample.get("query") or ""),
                            "label": int(sample["label"]),
                            "retriever_rank": int(
                                sample.get("retriever_rank") or 0
                            ),
                            **(
                                {"cycle_role": str(sample["cycle_role"])}
                                if sample.get("cycle_role") is not None
                                else {}
                            ),
                        }
                    }
                    if self.score_audit_metadata
                    else {}
                ),
                **(
                    {
                        "_pas_total_relevant": self.total_relevant_by_group[
                            str(sample["group_id"])
                        ]
                    }
                    if self.total_relevant_by_group is not None
                    else {}
                ),
                **(
                    {"_pas_interaction_cycle_role": str(sample["cycle_role"])}
                    if sample.get("cycle_role") is not None
                    else {}
                ),
                **(
                    {"_pas_cycle_ap_weight": float(sample["cycle_ap_weight"])}
                    if sample.get("cycle_ap_weight") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_r68_block_role": str(sample["r68_block_role"]),
                        "_pas_r68_cycle_index": int(sample["r68_cycle_index"]),
                        "_pas_r68_support_cycle": bool(
                            sample["r68_support_cycle"]
                        ),
                        "_pas_r68_cycle_weight": float(
                            sample["r68_cycle_weight"]
                        ),
                    }
                    if sample.get("r68_block_role") is not None
                    else {}
                ),
                **(
                    {"_pas_atomic_and": dict(sample["_pas_atomic_and"])}
                    if sample.get("_pas_atomic_and") is not None
                    else {}
                ),
                **(
                    {
                        "_pas_policy_preserve": bool(
                            sample["policy_preserve_group"]
                        )
                    }
                    if sample.get("policy_preserve_group") is not None
                    else {}
                ),
            }
            if self.response_mode != "mcq"
            else {}
        )
        if self.response_mode == "mcq":
            mcq_metadata = {
                "id": str(sample["id"]),
                "field": str(sample["field"]),
                "label_value": str(sample["label_value"]),
                "label_letter": str(sample["label_letter"]),
                "choices": list(sample["choices"]),
                "image": str(
                    sample.get("image")
                    or (sample.get("images") or [""])[0]
                ),
                "images": [
                    str(image)
                    for image in (
                        sample.get("images")
                        or ([sample["image"]] if sample.get("image") else [])
                    )
                ],
                "dataset": str(sample.get("dataset") or ""),
                "person_group": str(sample.get("person_group") or ""),
                # Joint listwise validation emits several balanced permutations
                # of the same candidate set.  Keep the source identity, query
                # slice, and parent-order mapping in the score sidecar so a
                # CPU audit can aggregate rotations and reconstruct top-5
                # Rank-1/AP without joining on fragile row-number encodings.
                "query": str(sample.get("query") or ""),
                "query_type": str(sample.get("query_type") or ""),
                "sample_loss_weight": float(sample.get("sample_loss_weight", 1.0)),
                "original_group_id": str(sample.get("original_group_id") or ""),
                "target_original_rank": int(sample.get("target_original_rank") or 0),
                "candidate_original_ranks": [
                    int(rank) for rank in sample.get("candidate_original_ranks", [])
                ],
                **(
                    {
                        "incumbent_outcome": str(sample["incumbent_outcome"]),
                        "preservation_target": bool(sample["preservation_target"]),
                    }
                    if sample.get("incumbent_outcome") is not None
                    else {}
                ),
                **(
                    {
                        "positive_label_letters": [
                            str(letter)
                            for letter in sample["positive_label_letters"]
                        ],
                        "positive_label_values": [
                            str(value)
                            for value in sample.get("positive_label_values", [])
                        ],
                    }
                    if sample.get("positive_label_letters") is not None
                    else {}
                ),
            }
        images = sample.get("image") or sample.get("images")
        if isinstance(images, str):
            images = [images]
        attribute_labels = None
        if self.attribute_replay is not None:
            if not images or len(images) != 1:
                raise ValueError("Attribute replay requires exactly one image")
            image_alias = str(images[0])
            if deterministic_replay_sample(
                str(sample.get("id", index)),
                fraction=self.attribute_replay.sample_fraction,
                seed=self.attribute_replay.seed,
            ):
                assert self.attribute_labels_by_image is not None
                attribute_labels = self.attribute_labels_by_image.get(image_alias)
        structured_attribute_metadata = None
        if self.structured_attribute_assets is not None:
            if self.response_mode != "binary":
                raise ValueError(
                    "Structured attributes currently require binary PAS responses"
                )
            if not images or len(images) != 1:
                raise ValueError("Structured attributes require exactly one image")
            labels = self.structured_attribute_assets.labels_by_image.get(str(images[0]))
            if labels is not None:
                assert self.structured_attribute_seed is not None
                query_labels = None
                if self.structured_attribute_assets.query_labels_by_index:
                    query_index = query_index_from_sample(sample)
                    try:
                        query_labels = self.structured_attribute_assets.query_labels_by_index[
                            query_index
                        ]
                    except IndexError as error:
                        raise ValueError(
                            f"No train-pair query attributes at index {query_index}"
                        ) from error
                if self.structured_attribute_mismatch_only:
                    if query_labels is None:
                        raise ValueError(
                            "Mismatch-only structured attributes require query labels"
                        )
                    structured_attribute_metadata = (
                        structured_attribute_mismatch_metadata(
                            response,
                            sample_id=str(sample.get("id", index)),
                            labels=labels,
                            assets=self.structured_attribute_assets,
                            query_labels=query_labels,
                        )
                    )
                else:
                    response, structured_attribute_metadata = (
                        append_structured_attribute_response(
                            response,
                            sample_id=str(sample.get("id", index)),
                            labels=labels,
                            assets=self.structured_attribute_assets,
                            seed=self.structured_attribute_seed,
                            query_labels=query_labels,
                        )
                    )
                structured_attribute_metadata["audit"] = {
                    "id": str(sample.get("id", index)),
                    "group_id": str(sample.get("group_id") or ""),
                    "image": str(images[0]),
                    "dataset": str(sample.get("dataset") or ""),
                    "query_type": str(sample.get("query_type") or ""),
                    "query": str(sample.get("query") or ""),
                    "binary_label": int(sample.get("label", 0)),
                    "audit_padding": bool(sample.get("audit_padding", False)),
                }
        hcr_metadata = None
        if self.hcr_enabled:
            if self.response_mode != "binary":
                raise ValueError("HCR currently requires binary PAS annotations")
            if not images or len(images) != 1:
                raise ValueError("HCR requires exactly one candidate image")
            binary_label = int(sample["label"])
            binary_response = response
            if self.hcr_assets is not None:
                query_alias = str(
                    sample.get("original_group_id") or sample.get("group_id") or ""
                )
                (
                    text_attr_values,
                    image_attr_values,
                    text_accessory_ids,
                    image_accessory_ids,
                ) = self.hcr_assets.constraint_values(
                    query_alias=query_alias,
                    candidate_alias=str(images[0]),
                )
                builder = {
                    "legacy_compatibility": build_compatibility_probe_response,
                    "active_logical": build_active_compatibility_probe_response,
                    "tristate_logical": build_tristate_compatibility_probe_response,
                    "predecision_tristate": build_predecision_tristate_probe_response,
                    "postdecision_tristate": build_postdecision_tristate_probe_response,
                    "decision_state_tristate": build_tristate_compatibility_probe_response,
                }[self.hcr_mode]
                builder_kwargs = dict(
                    sample_id=str(sample.get("id", index)),
                    text_attr_values=text_attr_values,
                    image_attr_values=image_attr_values,
                    text_accessory_ids=text_accessory_ids,
                    image_accessory_ids=image_accessory_ids,
                )
                if self.hcr_mode == "predecision_tristate":
                    builder_kwargs["prompt_mode"] = self.hcr_predecision_prompt_mode
                elif self.hcr_mode == "postdecision_tristate":
                    builder_kwargs["binary_label"] = binary_label
                built_response, hcr_metadata = builder(**builder_kwargs)
                if self.hcr_mode == "decision_state_tristate":
                    # The eight hard attribute targets supervise alternative
                    # frozen LM-head rows at the ordinary binary decision
                    # hidden state. They must never alter the assistant text.
                    response = binary_response
                    hcr_metadata["response"] = binary_response
                else:
                    response = built_response
                hcr_metadata["binary_label"] = binary_label
                # Selection roles are hard train-only construction metadata,
                # never a score or a soft target. Preserve the sole-violation
                # field so the loss can contrast exactly the affected probe.
                hcr_metadata["selection_role"] = str(
                    sample.get("r247_selection_role") or ""
                )
                if self.hcr_mode == "active_logical":
                    target_label = int(
                        all(
                            not bool(item["requirement_target"])
                            or bool(item["satisfaction_target"])
                            for item in hcr_metadata["active_probe_targets"]
                        )
                    )
                elif self.hcr_mode in {
                    "tristate_logical",
                    "predecision_tristate",
                    "postdecision_tristate",
                    "decision_state_tristate",
                }:
                    target_label = int(
                        all(
                            int(item["target"]) != 2
                            for item in hcr_metadata["tristate_probe_targets"]
                        )
                    )
                else:
                    target_label = int(
                        all(
                            bool(item["target"])
                            for item in hcr_metadata["probe_targets"]
                        )
                    )
                if target_label != binary_label:
                    raise ValueError(
                        "HCR constraint conjunction disagrees with PAS annotation: "
                        f"sample={sample.get('id', index)!r}, "
                        f"constraints={target_label}, label={binary_label}"
                    )
            else:
                inference_builder = {
                    "legacy_compatibility": build_compatibility_probe_inference_record,
                    "active_logical": build_active_compatibility_probe_inference_record,
                    "tristate_logical": build_tristate_compatibility_probe_inference_record,
                    "predecision_tristate": build_predecision_tristate_probe_inference_record,
                    "postdecision_tristate": build_postdecision_tristate_probe_inference_record,
                    "decision_state_tristate": build_tristate_compatibility_probe_inference_record,
                }[self.hcr_mode]
                inference_kwargs = dict(
                    sample_id=str(sample.get("id", index)),
                    binary_label=binary_label,
                )
                if self.hcr_mode == "predecision_tristate":
                    inference_kwargs["prompt_mode"] = self.hcr_predecision_prompt_mode
                built_response, hcr_metadata = inference_builder(**inference_kwargs)
                if self.hcr_mode == "decision_state_tristate":
                    response = binary_response
                    hcr_metadata["response"] = binary_response
                else:
                    response = built_response
        if images and self.media_path:
            images = [os.path.join(self.media_path, image) for image in images]
        user_prompt = re.sub(r"(\n)?</?image>(\n)?", "", user_prompt)

        if create_conversation is not None:
            messages = create_conversation(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                response=response,
                images=images,
                videos=None,
                vision_kwargs=self.vision_kwargs,
            )
        else:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            content = [
                *(
                    {"type": "image", "image": image, **self.vision_kwargs}
                    for image in images or []
                ),
                {"type": "text", "text": user_prompt},
            ]
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": response})
        if not self.visual_cache_enabled:
            if not images or not 1 <= len(images) <= 5:
                raise ValueError(
                    "PAS full-image training requires one to five images; "
                    f"got {len(images or [])}"
                )
            # Qwen3_VL_DataPacker's local-path shortcut drops per-image vision
            # kwargs such as max_pixels before calling the HF processor.  Resize
            # explicitly with qwen_vl_utils, then tell the PAS packer below not
            # to resize a second time.  This makes uncached training use the
            # exact same pixels and grid as the cache builder and evaluator.
            fetched = [
                fetch_image({"image": path, **self.vision_kwargs}, image_patch_size=16)
                for path in images
            ]
            return {
                "messages": messages,
                "images": fetched,
                **runtime_group_metadata,
                **({"_pas_mcq_metadata": mcq_metadata} if mcq_metadata else {}),
                **(
                    {"_pas_teacher_probability": teacher_probability}
                    if teacher_probability is not None
                    else {}
                ),
                **(
                    {"_pas_visual_teacher_index": visual_teacher_index}
                    if visual_teacher_index is not None
                    else {}
                ),
                **(
                    {"_pas_attribute_labels": attribute_labels}
                    if attribute_labels is not None
                    else {}
                ),
                **(
                    {"_pas_structured_attributes": structured_attribute_metadata}
                    if structured_attribute_metadata is not None
                    else {}
                ),
                **({"_pas_hcr": hcr_metadata} if hcr_metadata is not None else {}),
            }
        if not images or not 1 <= len(images) <= 5:
            raise ValueError(
                "PAS visual cache requires one to five images per sample; "
                f"got {len(images or [])}"
            )
        cache_key_override = sample.get("visual_cache_key_override")
        if cache_key_override is not None:
            if len(images) != 1:
                raise ValueError("A visual-cache key override requires exactly one image")
            cache_key = str(cache_key_override)
            if len(cache_key) != 40 or any(
                char not in "0123456789abcdef" for char in cache_key
            ):
                raise ValueError("visual_cache_key_override must be a lowercase SHA1")
            cache_keys = [cache_key]
        else:
            cache_keys = [visual_cache_key(image) for image in images]
        return {
            "messages": messages,
            "_pas_visual_cache_keys": cache_keys,
            **runtime_group_metadata,
            **({"_pas_mcq_metadata": mcq_metadata} if mcq_metadata else {}),
            **(
                {"_pas_teacher_probability": teacher_probability}
                if teacher_probability is not None
                else {}
            ),
            **(
                {"_pas_visual_teacher_index": visual_teacher_index}
                if visual_teacher_index is not None
                else {}
            ),
            **(
                {"_pas_attribute_labels": attribute_labels}
                if attribute_labels is not None
                else {}
            ),
            **(
                {"_pas_structured_attributes": structured_attribute_metadata}
                if structured_attribute_metadata is not None
                else {}
            ),
            **({"_pas_hcr": hcr_metadata} if hcr_metadata is not None else {}),
        }


_PAS_PACKER_SIDECAR_FIELDS = (
    "_pas_mcq_metadata",
    "_pas_group_id",
    "_pas_binary_label",
    "_pas_query_identity",
    "_pas_query_text",
    "_pas_dataset",
    "_pas_query_type",
    "_pas_retriever_rank",
    "_pas_retriever_score",
    "_pas_full_list_view_role",
    "_pas_view_candidate_id",
    # Training-only PU masks must survive the Qwen data packer.  Without this
    # sidecar the dataset correctly parsed ``rank_loss_weight`` but silently
    # dropped it before the trainer queued candidate weights, causing every
    # zero-weight unknown to be optimized as a hard negative.
    "_pas_rank_loss_weight",
    "_pas_inverse_pair_original_offset",
    "_pas_weak_veto_role",
    "_pas_weak_veto_field",
    "_pas_same_label_consistency_role",
    "_pas_requirement_coverage",
    "_pas_parent_margin",
    "_pas_parent_anchor",
    "_pas_parent_top1_correct",
    "_pas_parent_margin_stratum",
    "_pas_score_audit",
    "_pas_teacher_probability",
    "_pas_total_relevant",
    "_pas_visual_teacher_index",
    "_pas_attribute_labels",
    "_pas_structured_attributes",
    "_pas_hcr",
    "_pas_atomic_and",
    "_pas_policy_preserve",
    "_pas_interaction_cycle_role",
    "_pas_cycle_ap_weight",
    "_pas_r68_block_role",
    "_pas_r68_cycle_index",
    "_pas_r68_support_cycle",
    "_pas_r68_cycle_weight",
)


def _copy_pas_packer_sidecars(processed, sample):
    """Preserve supervision that the upstream Qwen packer does not know about."""
    for name in _PAS_PACKER_SIDECAR_FIELDS:
        if sample.get(name) is not None:
            processed[name] = sample[name]
    return processed


class PasFullImageQwen3VLDataPacker(Qwen3_VL_DataPacker):
    """Pack explicitly resized PAS images without a second HF resize."""

    class _CopyableString(str):
        def copy(self):
            return self

    def setup(self, config, *args, **kwargs):
        super().setup(config, *args, **kwargs)
        self.hf_processor.image_processor.do_resize = False

    def sft_process_sample(self, sample):
        messages = copy.deepcopy(sample["messages"])
        for message in messages:
            if message.get("role") == "assistant" and isinstance(
                message.get("content"), str
            ):
                message["content"] = self._CopyableString(message["content"])
        processed = super().sft_process_sample(
            {"messages": messages, "images": sample["images"]}
        )
        return _copy_pas_packer_sidecars(processed, sample)


class PasCachedQwen3VLDataPacker(Qwen3_VL_DataPacker):
    """Substitute cached frozen visual tensors after ordinary token packing."""

    class _CopyableString(str):
        """Work around the upstream packer's unconditional content.copy()."""

        def copy(self):
            return self

    def __init__(self, cache_config: VisualCacheConfig, vision_config: VisionConfig):
        super().__init__()
        self.cache_config = cache_config
        self.vision_config = vision_config
        self.cache: VisualCacheReader | None = None

    def setup(self, config, *args, **kwargs):
        super().setup(config, *args, **kwargs)
        self.cache = VisualCacheReader(
            self.cache_config.manifests,
            resident_shards=self.cache_config.resident_shards,
            slice_reads=self.cache_config.slice_reads,
        )
        expected_preprocessing = {
            "image_patch_size": 16,
            "spatial_merge_size": 2,
            # Older cosmos_reason1_utils VisionConfig releases do not expose
            # min_pixels.  The cache builder records an unset minimum as None,
            # so preserve that meaning instead of failing during worker setup.
            "min_pixels": getattr(self.vision_config, "min_pixels", None),
            "max_pixels": self.vision_config.max_pixels,
        }
        if self.cache.preprocessing != expected_preprocessing:
            raise ValueError(
                "Visual cache preprocessing disagrees with [custom.vision]: "
                f"cache={self.cache.preprocessing}, expected={expected_preprocessing}"
            )
        # Cached grids already describe the final resize. The processor only
        # needs to create matching image placeholder tokens and mRoPE positions.
        self.hf_processor.image_processor.do_resize = False
        logger.info("Loaded PAS visual-cache index with %s images", len(self.cache.entries))

    def sft_process_sample(self, sample):
        cache_keys = sample.get("_pas_visual_cache_keys")
        if cache_keys is None:
            # Compatibility with samples produced before multi-image cache
            # support was added.
            cache_keys = [sample["_pas_visual_cache_key"]]
        assert self.cache is not None
        grids = [self.cache.grid(cache_key) for cache_key in cache_keys]
        if any(grid_t != 1 for grid_t, _, _ in grids):
            raise ValueError(f"PAS image cache requires grid_t=1, got {grids}")

        # The upstream packer needs an image solely to determine how many image
        # placeholder tokens and mRoPE positions to emit. A black image with
        # the cached pre-merge grid dimensions produces the exact same layout;
        # its cheap CPU pixels are discarded in _collate_fn below.
        dummy_images = [
            Image.new("RGB", (grid_w * 16, grid_h * 16))
            for _, grid_h, grid_w in grids
        ]
        messages = copy.deepcopy(sample["messages"])
        for message in messages:
            if message.get("role") == "assistant" and isinstance(
                message.get("content"), str
            ):
                message["content"] = self._CopyableString(message["content"])
        processed = super().sft_process_sample(
            {"messages": messages, "images": dummy_images}
        )
        processed["_pas_visual_cache_keys"] = list(cache_keys)
        return _copy_pas_packer_sidecars(processed, sample)

    def _collate_fn(self, processed_samples, computed_max_len):
        batch = super()._collate_fn(processed_samples, computed_max_len)
        assert self.cache is not None
        cache_keys = [
            cache_key
            for sample in processed_samples
            for cache_key in sample.get(
                "_pas_visual_cache_keys", [sample.get("_pas_visual_cache_key")]
            )
        ]
        if any(cache_key is None for cache_key in cache_keys):
            raise ValueError("Processed PAS sample is missing its visual-cache key")
        records = [
            self.cache.get(cache_key)
            for cache_key in cache_keys
        ]
        cached_grids = torch.tensor([grid for _, grid in records], dtype=torch.long)
        if not torch.equal(cached_grids, batch["image_grid_thw"]):
            raise ValueError("Cached image grids disagree with Qwen preprocessing")
        if self.cache.prefix_block is None:
            raise ValueError("Visual prefix cache is missing prefix_block metadata")
        # Reuse the standard image/video tensor fields so existing dataloader,
        # transfer, and FSDP signatures stay unchanged. The Qwen3-VL patch
        # recognizes this shape pair as a cached visual prefix.
        batch["pixel_values"] = torch.cat(
            [value[0] for value, _ in records], dim=0
        )
        batch["pixel_values_videos"] = torch.cat(
            [value[1] for value, _ in records], dim=0
        )
        batch["video_grid_thw"] = torch.tensor(
            [self.cache.prefix_block], dtype=torch.long
        )
        return batch


class DistributedGroupBatchSampler(Sampler[list[int]]):
    """Keep every positive/negative candidate group on one DP rank.

    PyTorch's ordinary DistributedSampler strides through individual examples.
    Since PAS stores one positive followed by three negatives, that would send
    positives and negatives to different ranks and make a local ranking loss
    undefined. This sampler distributes whole, contiguous groups instead.
    """

    def __init__(
        self,
        dataset,
        *,
        group_size: int,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        pad: bool = True,
        seed: int = 0,
    ) -> None:
        if group_size <= 1:
            raise ValueError(f"group_size must exceed one, got {group_size}")
        if len(dataset) % group_size:
            raise ValueError(
                f"Dataset length {len(dataset)} must be divisible by group_size={group_size}"
            )
        self.group_size = int(group_size)
        if batch_size < group_size or batch_size % group_size:
            raise ValueError(
                f"batch_size={batch_size} must be a multiple of group_size={group_size}"
            )
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.pad = bool(pad)
        self.seed = int(seed)
        self.epoch = 0
        self.groups_per_batch = self.batch_size // self.group_size
        self.num_groups = len(dataset) // self.group_size
        self.batches_per_rank = math.ceil(
            self.num_groups / (self.num_replicas * self.groups_per_batch)
        )
        self.padded_num_groups = (
            self.batches_per_rank * self.num_replicas * self.groups_per_batch
        )
        if not self.pad and self.padded_num_groups != self.num_groups:
            multiple = self.num_replicas * self.groups_per_batch
            raise ValueError(
                f"Validation has {self.num_groups} groups, but exact distributed "
                f"validation requires a multiple of {multiple}. Choose a validation "
                "subset without sampler padding."
            )

    def __iter__(self) -> Iterator[list[int]]:
        groups = list(range(self.num_groups))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        padded = self.padded_num_groups
        if padded > len(groups):
            groups.extend(groups[index % len(groups)] for index in range(padded - len(groups)))
        rank_groups = groups[self.rank:padded:self.num_replicas]
        for offset in range(0, len(rank_groups), self.groups_per_batch):
            indices = []
            for group in rank_groups[offset : offset + self.groups_per_batch]:
                begin = group * self.group_size
                indices.extend(range(begin, begin + self.group_size))
            yield indices

    def __len__(self) -> int:
        return self.batches_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _pas_hidden_state_lm_head_forward(
    _module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Skip full-vocabulary logits; the loss projects only yes/no head rows."""

    return hidden_states


def _pas_group_batch_sampler(
    dataset,
    *,
    batch_size: int,
    sampler=None,
    num_replicas: int | None = None,
    rank: int | None = None,
    config=None,
    pad: bool,
    **_,
) -> DistributedGroupBatchSampler:
    """Cosmos-RL batch-sampler factory for train and validation."""

    if num_replicas is None:
        num_replicas = sampler.num_replicas
    if rank is None:
        rank = sampler.rank
    shuffle = bool(getattr(sampler, "shuffle", False))
    loss_config = LossConfig.model_validate(
        config.custom.get("loss", {}) if config is not None else {}
    )
    return DistributedGroupBatchSampler(
        dataset,
        group_size=loss_config.candidate_group_size,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        pad=pad,
        seed=(config.train.train_policy.dataloader_seed if config is not None else 0),
    )


def pas_train_group_batch_sampler(
    dataset,
    *,
    batch_size: int,
    sampler=None,
    num_replicas: int | None = None,
    rank: int | None = None,
    config=None,
    **kwargs,
) -> DistributedGroupBatchSampler:
    return _pas_group_batch_sampler(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_replicas=num_replicas,
        rank=rank,
        config=config,
        pad=True,
        **kwargs,
    )


def pas_val_group_batch_sampler(
    dataset,
    *,
    batch_size: int,
    sampler=None,
    num_replicas: int | None = None,
    rank: int | None = None,
    config=None,
    **kwargs,
) -> DistributedGroupBatchSampler:
    return _pas_group_batch_sampler(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_replicas=num_replicas,
        rank=rank,
        config=config,
        pad=False,
        **kwargs,
    )


@TrainerRegistry.register(trainer_type="pas_pairwise_action_sft")
class PasPairwiseActionSFTTrainer(SFTTrainer):
    """Ordinary token SFT initialized from a joint language+vision LoRA.

    The stock SFT trainer creates the configured LoRA modules but does not load
    continuation weights.  Pairwise action learning must start from the exact
    deployed pointwise parent, including its tower adapters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lora_path = self.config.policy.lora.lora_path
        if lora_path:
            load_lora_initialization(
                self.model,
                lora_path,
                allow_partial=bool(
                    self.config.custom.get("allow_partial_lora_initialization", False)
                ),
            )
        init_offline_wandb(self.config)


@TrainerRegistry.register(trainer_type="mcq_sft")
class McqSFTTrainer(SFTTrainer):
    """Standard SFT with cheap, classification-aligned PAS MCQ telemetry."""

    def __init__(self, *args, **kwargs):
        global _METRICS_PATH

        super().__init__(*args, **kwargs)
        if self.parallel_dims.pp_enabled or self.parallel_dims.cp_enabled:
            raise ValueError("MCQ telemetry currently requires pp_size=cp_size=1")
        if not self.config.policy.enable_liger_fused_cross_entropy:
            raise ValueError("MCQ telemetry expects fused CE hidden states")
        # Fused-CE mode must hand the loss the final hidden states rather than
        # materializing full-vocabulary logits.  The PAS rank/point trainer
        # installs the same LM-head bypass below; MCQ validation needs it too.
        hf_model = getattr(self.model, "model", None)
        if hf_model is None or not hasattr(hf_model, "lm_head"):
            raise ValueError("Could not locate the HF LM head for MCQ projection")
        original_lm_head = hf_model.lm_head
        if not getattr(original_lm_head, "_pas_hidden_state_forward", False):
            original_lm_head.forward = types.MethodType(
                _pas_hidden_state_lm_head_forward, original_lm_head
            )
            original_lm_head._pas_hidden_state_forward = True
        lora_path = self.config.policy.lora.lora_path
        if lora_path:
            load_lora_initialization(
                self.model,
                lora_path,
                allow_partial=bool(
                    self.config.custom.get(
                        "allow_partial_lora_initialization", False
                    )
                ),
            )
        self.loss_fn = McqClassificationLoss(
            self.data_packer.tokenizer,
            self.model.lm_head.weight,
            partial_order_weight=float(
                self.config.custom.get("mcq_partial_order_weight", 0.0)
            ),
            positive_set_mode=str(
                self.config.custom.get("mcq_positive_set_mode", "mass")
            ),
            factorized_marginal_weight=float(
                self.config.custom.get("mcq_factorized_marginal_weight", 0.0)
            ),
            factorized_joint_weight=float(
                self.config.custom.get("mcq_factorized_joint_weight", 1.0)
            ),
        )
        logger.info(
            "MCQ positive-set mode=%s partial-order weight=%s",
            self.loss_fn.positive_set_mode,
            self.loss_fn.partial_order_weight,
        )
        logger.info(
            "MCQ factorized bit-code weights: marginal=%s joint=%s",
            self.loss_fn.factorized_marginal_weight,
            self.loss_fn.factorized_joint_weight,
        )
        self._validation_metric_step: int | None = None
        self._audit_predictions = bool(
            self.config.custom.get("mcq_audit_predictions", False)
        )
        self._audit_path: Path | None = None
        if self._audit_predictions:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            self._audit_path = (
                Path(self.config.train.output_dir)
                / f"mcq_audit_rank{rank:02d}.jsonl"
            )
            self._audit_path.unlink(missing_ok=True)
        _METRICS_PATH = Path(self.config.train.output_dir) / "loss_metrics.jsonl"
        _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        init_offline_wandb(self.config)

    def _distributed_accuracy(
        self, metric_sum: torch.Tensor | None, metric_count: int
    ) -> float | None:
        if metric_sum is None or metric_count <= 0:
            return None
        totals = torch.stack(
            [metric_sum.float(), metric_sum.new_tensor(float(metric_count)).float()]
        )
        if self.parallel_dims.dp_replicate_enabled or self.parallel_dims.dp_shard_enabled:
            dist.all_reduce(
                totals,
                op=dist.ReduceOp.SUM,
                group=self.parallel_dims.mesh["dp"].get_group(),
            )
        if totals[1].item() == 0:
            return None
        return (totals[0] / totals[1]).item()

    def _queue_mcq_targets(self, metadata: list[dict[str, Any]]) -> None:
        self.loss_fn.set_option_counts([len(row["choices"]) for row in metadata])
        positive_sets = []
        has_set_target = False
        for row in metadata:
            letters = row.get("positive_label_letters")
            if letters is None:
                letters = [row["label_letter"]]
            else:
                has_set_target = True
            positive_sets.append(
                [ord(str(letter)) - ord("A") for letter in letters]
            )
        self.loss_fn.set_positive_option_indices(
            positive_sets if has_set_target else None
        )
        self.loss_fn.set_factorized_metadata(
            [list(map(str, row["choices"])) for row in metadata],
            [float(row.get("sample_loss_weight", 1.0)) for row in metadata],
        )

    def _distributed_factorized_metrics(self) -> dict[str, float]:
        metric_totals = self.loss_fn.factorized_metric_totals()
        if not metric_totals:
            return {}
        names = sorted(metric_totals)
        # The loss always creates the same seven names on every rank, including
        # zero-count bit-2/pair aliases, so this collective order is fixed.
        expected_names = [
            "bit_0_marginal",
            "bit_1_marginal",
            "bit_2_marginal",
            "joint",
            "left_marginal",
            "marginal",
            "right_marginal",
        ]
        if names != expected_names:
            raise ValueError(f"Unexpected MCQ factorized metric schema: {names}")
        totals = torch.stack(
            [value.float() for name in names for value in metric_totals[name]]
        )
        if self.parallel_dims.dp_replicate_enabled or self.parallel_dims.dp_shard_enabled:
            dist.all_reduce(
                totals,
                op=dist.ReduceOp.SUM,
                group=self.parallel_dims.mesh["dp"].get_group(),
            )
        metrics = {}
        for index, name in enumerate(names):
            metric_sum = float(totals[2 * index])
            denominator = float(totals[2 * index + 1])
            if not math.isfinite(denominator) or denominator < 0:
                raise ValueError("MCQ factorized metric weight is invalid")
            if denominator:
                metrics[f"train/mcq_factorized_{name}_loss"] = (
                    metric_sum / denominator
                )
        return metrics

    def step_training(self, *args, **kwargs):
        if bool(self.config.custom.get("validation_only", False)):
            raise RuntimeError(
                "validation_only MCQ configuration refused a training step"
            )
        validation_totals = (
            self.loss_fn.metric_totals()
            if self._validation_metric_step is not None
            else (None, 0)
        )
        validation_step = self._validation_metric_step
        self._validation_metric_step = None
        self.loss_fn.reset_metrics()
        global_batch = args[0] if args else kwargs.get("global_batch")
        if global_batch is None:
            raise ValueError("MCQ training batch is unavailable")
        metadata = [sample.get("_pas_mcq_metadata") for sample in global_batch]
        if any(row is None for row in metadata):
            raise ValueError("MCQ training sample is missing metadata")
        self._queue_mcq_targets(metadata)
        report_data = super().step_training(*args, **kwargs)
        train_accuracy = self._distributed_accuracy(*self.loss_fn.metric_totals())
        if train_accuracy is not None:
            report_data["train/mcq_accuracy"] = train_accuracy
        report_data.update(self._distributed_factorized_metrics())
        val_accuracy = self._distributed_accuracy(*validation_totals)
        if val_accuracy is not None:
            report_data["val/mcq_accuracy"] = val_accuracy
            report_data["val/mcq_metric_source_step"] = int(validation_step)
        return report_data

    def step_validation(self, val_global_batch, train_step: int, total_steps: int):
        if self._validation_metric_step != train_step:
            self.loss_fn.reset_metrics()
            self._validation_metric_step = train_step
        metadata = [sample.get("_pas_mcq_metadata") for sample in val_global_batch]
        if any(row is None for row in metadata):
            raise ValueError("MCQ validation sample is missing metadata")
        self._queue_mcq_targets(metadata)
        if self._audit_predictions:
            self.loss_fn.set_audit_metadata(metadata)
        score = super().step_validation(val_global_batch, train_step, total_steps)
        if self._audit_predictions:
            assert self._audit_path is not None
            records = self.loss_fn.pop_audit_records()
            with self._audit_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    record["validation_step"] = int(train_step)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        return score


class RankPointWithAttributeReplayLoss:
    """Add frozen-head attribute CE to the ordinary PAS ranking callback."""

    def __init__(
        self,
        base: RankPointLoss,
        visual_model,
        assets: AttributeReplayAssets,
        config: AttributeReplayConfig,
    ) -> None:
        self.base = base
        self.visual_model = visual_model
        self.assets = assets
        self.config = config
        self._attribute_labels: list[tuple[int, ...] | None] | None = None
        self._attribute_metric_sums: dict[str, torch.Tensor] = {}
        self._attribute_metric_counts: dict[str, int] = {}

    @property
    def rank_mode(self) -> str:
        return self.base.rank_mode

    @property
    def teacher_weight(self) -> float:
        return self.base.teacher_weight

    @teacher_weight.setter
    def teacher_weight(self, value: float) -> None:
        self.base.teacher_weight = float(value)

    @property
    def topk_preservation_weight(self) -> float:
        return self.base.topk_preservation_weight

    @property
    def last_scores(self):
        """Expose the exact deployed scores recorded by the wrapped loss."""

        return self.base.last_scores

    @property
    def last_labels(self):
        return self.base.last_labels

    @property
    def canonical_validation(self) -> bool:
        return self.base.canonical_validation

    @canonical_validation.setter
    def canonical_validation(self, value: bool) -> None:
        self.base.canonical_validation = bool(value)

    @property
    def projection_weight(self):
        return self.base.projection_weight

    @projection_weight.setter
    def projection_weight(self, value) -> None:
        self.base.projection_weight = value

    def set_teacher_probabilities(self, values) -> None:
        self.base.set_teacher_probabilities(values)

    def assert_teacher_probabilities_consumed(self) -> None:
        self.base.assert_teacher_probabilities_consumed()

    def set_total_relevant(self, values) -> None:
        self.base.set_total_relevant(values)

    def assert_total_relevant_consumed(self) -> None:
        self.base.assert_total_relevant_consumed()

    def assert_cycle_weights_consumed(self) -> None:
        self.base.assert_cycle_weights_consumed()

    def assert_cycle_block_metadata_consumed(self) -> None:
        self.base.assert_cycle_block_metadata_consumed()

    def set_attribute_labels(
        self, values: list[tuple[int, ...] | list[int] | None]
    ) -> None:
        if self._attribute_labels:
            raise ValueError("Previous attribute replay labels were not consumed")
        self._attribute_labels = [
            None if value is None else tuple(int(item) for item in value)
            for value in values
        ]

    def assert_attribute_labels_consumed(self) -> None:
        if self._attribute_labels:
            raise ValueError(
                f"{len(self._attribute_labels)} attribute replay labels were not consumed"
            )
        self._attribute_labels = None

    def _take_attribute_labels(
        self, count: int, *, device: torch.device
    ) -> torch.LongTensor:
        if self._attribute_labels is None or len(self._attribute_labels) < count:
            available = 0 if self._attribute_labels is None else len(self._attribute_labels)
            raise ValueError(
                f"Need {count} attribute replay rows for this batch, found {available}"
            )
        selected = self._attribute_labels[:count]
        del self._attribute_labels[:count]
        width = len(self.assets.fields)
        return torch.tensor(
            [([0] * width if value is None else value) for value in selected],
            device=device,
            dtype=torch.long,
        )

    def reset_metrics(self) -> None:
        self.base.reset_metrics()
        self._attribute_metric_sums = {}
        self._attribute_metric_counts = {}

    def metric_totals(self) -> dict[str, tuple[torch.Tensor, int]]:
        totals = dict(self.base.metric_totals())
        totals.update(
            {
                name: (value, self._attribute_metric_counts[name])
                for name, value in self._attribute_metric_sums.items()
            }
        )
        return totals

    def mean_metrics(self) -> dict[str, torch.Tensor]:
        result = dict(self.base.mean_metrics())
        result.update(
            {
                name: value / max(self._attribute_metric_counts[name], 1)
                for name, value in self._attribute_metric_sums.items()
            }
        )
        return result

    def __call__(self, output, target, **kwargs):
        rank_point = self.base(output, target, **kwargs)
        visual_embeds = getattr(
            self.visual_model, "_pas_last_visual_embeds", None
        )
        grid_thw = getattr(
            self.visual_model, "_pas_last_visual_grid_thw", None
        )
        if hasattr(self.visual_model, "_pas_last_visual_embeds"):
            del self.visual_model._pas_last_visual_embeds
        if hasattr(self.visual_model, "_pas_last_visual_grid_thw"):
            del self.visual_model._pas_last_visual_grid_thw
        if self.canonical_validation:
            return rank_point
        if visual_embeds is None or grid_thw is None:
            raise ValueError(
                "Attribute replay requires the differentiable cached-prefix visual output"
            )
        labels = self._take_attribute_labels(
            int(target.shape[0]), device=visual_embeds.device
        )
        contribution, metrics = attribute_replay_loss(
            visual_embeds,
            grid_thw,
            labels,
            self.assets,
            spatial_merge_size=int(self.visual_model.visual.config.spatial_merge_size),
            label_smoothing=self.config.label_smoothing,
        )
        for name, (metric_sum, metric_count) in metrics.items():
            self._attribute_metric_sums[name] = (
                metric_sum
                if name not in self._attribute_metric_sums
                else self._attribute_metric_sums[name] + metric_sum
            )
            self._attribute_metric_counts[name] = (
                self._attribute_metric_counts.get(name, 0) + metric_count
            )
        return rank_point + (
            contribution
            * self.config.loss_weight
            * float(kwargs.get("loss_scaling_factor", 1.0))
        )


class RankPointWithVisualDistillationLoss:
    """Add train-only SigLIP2 relational supervision to CR3 visual features."""

    def __init__(
        self,
        base: RankPointLoss,
        visual_model,
        assets: VisualDistillationAssets,
        config: VisualDistillationConfig,
        *,
        group_size: int,
        device: torch.device,
    ) -> None:
        self.base = base
        self.visual_model = visual_model
        self.assets = assets
        self.config = config
        self.group_size = int(group_size)
        self.device = device
        self._teacher_embeddings: torch.Tensor | None = None
        self._visual_metric_sums: dict[str, torch.Tensor] = {}
        self._visual_metric_counts: dict[str, int] = {}

    @property
    def rank_mode(self) -> str:
        return self.base.rank_mode

    @property
    def teacher_weight(self) -> float:
        return self.base.teacher_weight

    @teacher_weight.setter
    def teacher_weight(self, value: float) -> None:
        self.base.teacher_weight = float(value)

    @property
    def topk_preservation_weight(self) -> float:
        return self.base.topk_preservation_weight

    @property
    def canonical_validation(self) -> bool:
        return self.base.canonical_validation

    @canonical_validation.setter
    def canonical_validation(self, value: bool) -> None:
        self.base.canonical_validation = bool(value)

    @property
    def projection_weight(self):
        return self.base.projection_weight

    @projection_weight.setter
    def projection_weight(self, value) -> None:
        self.base.projection_weight = value

    def set_teacher_probabilities(self, values) -> None:
        self.base.set_teacher_probabilities(values)

    def assert_teacher_probabilities_consumed(self) -> None:
        self.base.assert_teacher_probabilities_consumed()

    def set_total_relevant(self, values) -> None:
        self.base.set_total_relevant(values)

    def assert_total_relevant_consumed(self) -> None:
        self.base.assert_total_relevant_consumed()

    def set_visual_teacher_indices(self, indices: list[int]) -> None:
        if self._teacher_embeddings is not None:
            raise ValueError("Previous visual teacher embeddings were not consumed")
        self._teacher_embeddings = self.assets.teacher_batch(
            indices, device=self.device
        )

    def assert_visual_teacher_embeddings_consumed(self) -> None:
        if self._teacher_embeddings is not None:
            raise ValueError(
                f"{self._teacher_embeddings.shape[0]} visual teacher rows were not consumed"
            )

    def _take_teacher_embeddings(self, count: int) -> torch.Tensor:
        if self._teacher_embeddings is None or self._teacher_embeddings.shape[0] < count:
            available = (
                0
                if self._teacher_embeddings is None
                else int(self._teacher_embeddings.shape[0])
            )
            raise ValueError(
                f"Need {count} SigLIP2 teacher rows for this batch, found {available}"
            )
        selected = self._teacher_embeddings[:count]
        remaining = self._teacher_embeddings[count:]
        self._teacher_embeddings = remaining if remaining.shape[0] else None
        return selected

    def reset_metrics(self) -> None:
        self.base.reset_metrics()
        self._visual_metric_sums = {}
        self._visual_metric_counts = {}

    def metric_totals(self) -> dict[str, tuple[torch.Tensor, int]]:
        totals = dict(self.base.metric_totals())
        totals.update(
            {
                name: (value, self._visual_metric_counts[name])
                for name, value in self._visual_metric_sums.items()
            }
        )
        return totals

    def mean_metrics(self) -> dict[str, torch.Tensor]:
        result = dict(self.base.mean_metrics())
        result.update(
            {
                name: value / max(self._visual_metric_counts[name], 1)
                for name, value in self._visual_metric_sums.items()
            }
        )
        return result

    def __call__(self, output, target, **kwargs):
        rank_point = self.base(output, target, **kwargs)
        visual_embeds = getattr(self.visual_model, "_pas_last_visual_embeds", None)
        grid_thw = getattr(self.visual_model, "_pas_last_visual_grid_thw", None)
        if hasattr(self.visual_model, "_pas_last_visual_embeds"):
            del self.visual_model._pas_last_visual_embeds
        if hasattr(self.visual_model, "_pas_last_visual_grid_thw"):
            del self.visual_model._pas_last_visual_grid_thw
        # A validation batch without teacher rows is still legal for backwards
        # compatibility.  When rows are supplied, measure representation
        # alignment on the fixed diagnostic set but keep the returned
        # canonical validation objective rank-only.
        if self.canonical_validation and self._teacher_embeddings is None:
            return rank_point
        if visual_embeds is None or grid_thw is None:
            raise ValueError(
                "Visual distillation requires the differentiable cached-prefix visual output"
            )
        student = pool_visual_tokens(
            visual_embeds,
            grid_thw,
            spatial_merge_size=int(self.visual_model.visual.config.spatial_merge_size),
        )
        teacher = self._take_teacher_embeddings(int(target.shape[0]))
        contribution, metrics = relational_visual_distillation_loss(
            student,
            teacher,
            group_size=self.group_size,
        )
        for name, (metric_sum, metric_count) in metrics.items():
            self._visual_metric_sums[name] = (
                metric_sum
                if name not in self._visual_metric_sums
                else self._visual_metric_sums[name] + metric_sum
            )
            self._visual_metric_counts[name] = (
                self._visual_metric_counts.get(name, 0) + metric_count
            )
        if self.canonical_validation:
            return rank_point
        return rank_point + (
            contribution
            * self.config.loss_weight
            * float(kwargs.get("loss_scaling_factor", 1.0))
        )


def binary_scalar_mismatch_rank_loss(
    binary_scores: torch.Tensor,
    binary_labels: torch.Tensor,
    scalar_mismatches: torch.BoolTensor,
    scalar_comparable: torch.BoolTensor,
    *,
    group_size: int,
    mode: Literal["all_pairs", "top1_hinge"] = "all_pairs",
    margin: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Rank PAS scalar-mismatch negatives below positives using deployed scores.

    The deployed reranker orders candidates with ``binary_scores``.  An older
    implementation added raw attribute-choice log-probabilities to those
    scores.  Those uncalibrated log-probabilities can be hundreds of nats
    negative, which made ``softplus(negative - positive)`` underflow to zero
    and silently disabled this objective.  Attribute CE already supervises
    the choice logits; this focused ranking term must train the score that is
    actually used at inference.
    """
    losses: list[torch.Tensor] = []
    for group_scores, group_labels, group_mismatches, group_comparable in zip(
        binary_scores.split(group_size),
        binary_labels.split(group_size),
        scalar_mismatches.split(group_size),
        scalar_comparable.split(group_size),
        strict=True,
    ):
        positive = (group_labels > 0.5) & group_comparable
        scalar_mismatch = group_mismatches & group_comparable
        if positive.any() and scalar_mismatch.any():
            if mode == "all_pairs":
                losses.append(
                    torch.nn.functional.softplus(
                        group_scores[scalar_mismatch].unsqueeze(0)
                        - group_scores[positive].unsqueeze(1)
                    ).mean()
                )
            elif mode == "top1_hinge":
                # Rank-1 changes only at the strongest positive/negative
                # boundary.  Focus scalar supervision on the mismatch that
                # can currently win, and make the term a true constraint so
                # already-safe groups receive exactly zero drift.
                losses.append(
                    torch.nn.functional.relu(
                        group_scores[scalar_mismatch].amax()
                        + margin
                        - group_scores[positive].amax()
                    )
                )
            else:
                raise ValueError(f"Unsupported mismatch rank mode: {mode!r}")
    return (
        torch.stack(losses).mean() if losses else binary_scores.sum() * 0.0,
        len(losses),
    )


class RankPointWithStructuredAttributeLoss:
    """Add masked multi-field choice CE after the ordinary yes/no decision."""

    def __init__(
        self,
        base: RankPointLoss,
        choice_loss: StructuredAttributeChoiceLoss,
        config: StructuredAttributeConfig,
    ) -> None:
        self.base = base
        self.choice_loss = choice_loss
        self.config = config
        self._records: list[dict[str, Any] | None] | None = None
        self._attribute_metric_sums: dict[str, torch.Tensor] = {}
        self._attribute_metric_counts: dict[str, int] = {}

    def __getattr__(self, name: str):
        """Delegate the ordinary rank-loss contract to the wrapped loss.

        The trainer's metadata queues evolve independently of this optional
        auxiliary wrapper (for example, same-label consistency was added to
        ``RankPointLoss`` after this class).  Forwarding unknown attributes
        keeps a structured-attribute run behaviorally identical to its base
        ranking control except for the explicitly added auxiliary term.
        """

        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def rank_mode(self) -> str:
        return self.base.rank_mode

    @property
    def teacher_weight(self) -> float:
        return self.base.teacher_weight

    @teacher_weight.setter
    def teacher_weight(self, value: float) -> None:
        self.base.teacher_weight = float(value)

    @property
    def topk_preservation_weight(self) -> float:
        return self.base.topk_preservation_weight

    @property
    def canonical_validation(self) -> bool:
        return self.base.canonical_validation

    @canonical_validation.setter
    def canonical_validation(self, value: bool) -> None:
        self.base.canonical_validation = bool(value)

    @property
    def projection_weight(self):
        return self.base.projection_weight

    @projection_weight.setter
    def projection_weight(self, value) -> None:
        self.base.projection_weight = value

    def set_teacher_probabilities(self, values) -> None:
        self.base.set_teacher_probabilities(values)

    def assert_teacher_probabilities_consumed(self) -> None:
        self.base.assert_teacher_probabilities_consumed()

    def set_total_relevant(self, values) -> None:
        self.base.set_total_relevant(values)

    def assert_total_relevant_consumed(self) -> None:
        self.base.assert_total_relevant_consumed()

    def set_structured_records(
        self, records: Sequence[dict[str, Any] | None]
    ) -> None:
        if self._records:
            raise ValueError("Previous structured metadata was not consumed")
        self._records = list(records)

    def _take_records(self, count: int) -> list[dict[str, Any] | None]:
        if self._records is None or len(self._records) < count:
            available = 0 if self._records is None else len(self._records)
            raise ValueError(
                f"Need {count} structured metadata rows, found {available}"
            )
        selected = self._records[:count]
        del self._records[:count]
        return selected

    def assert_structured_records_consumed(self) -> None:
        if self._records:
            raise ValueError(
                f"{len(self._records)} structured metadata rows were not consumed"
            )
        self._records = None

    def set_attribute_audit_enabled(self, enabled: bool) -> None:
        self.choice_loss.set_audit_enabled(enabled)

    def pop_attribute_audit_records(self) -> list[dict[str, Any]]:
        return self.choice_loss.pop_audit_records()

    def reset_metrics(self) -> None:
        self.base.reset_metrics()
        self._attribute_metric_sums = {}
        self._attribute_metric_counts = {}

    def metric_totals(self) -> dict[str, tuple[torch.Tensor, int]]:
        totals = dict(self.base.metric_totals())
        totals.update(
            {
                name: (value, self._attribute_metric_counts[name])
                for name, value in self._attribute_metric_sums.items()
            }
        )
        return totals

    def mean_metrics(self) -> dict[str, torch.Tensor]:
        result = dict(self.base.mean_metrics())
        result.update(
            {
                name: value / max(self._attribute_metric_counts[name], 1)
                for name, value in self._attribute_metric_sums.items()
            }
        )
        return result

    def __call__(self, output, target, **kwargs):
        rank_point = self.base(output, target, **kwargs)
        records = self._take_records(int(target.shape[0]))
        contribution, metrics, _compatibility_scores, _compatibility_valid = self.choice_loss(
            output,
            target,
            records,
            projection_weight=(
                kwargs.get("lin_weight")
                if kwargs.get("lin_weight") is not None
                else self.projection_weight
            ),
        )
        if self.config.mismatch_rank_weight:
            binary_scores = self.base.last_scores
            binary_labels = self.base.last_labels
            if binary_scores is None or binary_labels is None:
                raise ValueError("Base PAS scores were not retained for mismatch ranking")
            group_size = self.base.group_size
            scalar_mismatches = output.new_tensor(
                [
                    bool(record and record.get("scalar_mismatch"))
                    for record in records
                ],
                dtype=torch.bool,
            )
            scalar_comparable = output.new_tensor(
                [
                    bool(record and record.get("scalar_comparable"))
                    for record in records
                ],
                dtype=torch.bool,
            )
            mismatch_loss, mismatch_group_count = binary_scalar_mismatch_rank_loss(
                binary_scores,
                binary_labels,
                scalar_mismatches,
                scalar_comparable,
                group_size=group_size,
                mode=self.config.mismatch_rank_mode,
                margin=self.config.mismatch_rank_margin,
            )
            metrics["attribute_mismatch_rank_loss"] = (
                mismatch_loss.detach() * max(mismatch_group_count, 1),
                max(mismatch_group_count, 1),
            )
            total_groups = max(int(binary_scores.numel()) // group_size, 1)
            metrics["attribute_mismatch_group_coverage"] = (
                output.new_tensor(float(mismatch_group_count)).detach(),
                total_groups,
            )
        for name, (metric_sum, metric_count) in metrics.items():
            self._attribute_metric_sums[name] = (
                metric_sum
                if name not in self._attribute_metric_sums
                else self._attribute_metric_sums[name] + metric_sum
            )
            self._attribute_metric_counts[name] = (
                self._attribute_metric_counts.get(name, 0) + metric_count
            )
        if self.canonical_validation:
            return rank_point
        mismatch_contribution = (
            mismatch_loss * self.config.mismatch_rank_weight
            if self.config.mismatch_rank_weight
            else output.sum() * 0.0
        )
        return rank_point + (
            contribution
            * self.config.loss_weight
            * float(kwargs.get("loss_scaling_factor", 1.0))
        ) + mismatch_contribution * float(kwargs.get("loss_scaling_factor", 1.0))


class RankPointWithAtomicAndLoss:
    """Train a configurable aggregation of atomic CR3 calls end to end.

    Each forward row is still an ordinary CR3 yes/no conversation.  Rows are
    arranged as ``[query, candidate, field]``; this wrapper reshapes their
    yes-minus-no margins, reduces only active fields with a normalized soft
    minimum or arithmetic mean, and sends the scores through the listwise
    objective.  Thus gradients from full-query AP flow back through the exact
    multi-call score used at inference.
    """

    def __init__(self, base: RankPointLoss, config: AtomicAndConfig) -> None:
        self.base = base
        self.config = config
        # The trainer/sampler groups atomic rows, whereas RankPointLoss groups
        # the reduced candidate scores.  Keep those two cardinalities explicit.
        self.atomic_group_size = config.num_candidates * config.num_fields
        self.base.group_size = int(config.num_candidates)
        self._records: list[dict[str, Any] | None] | None = None
        self._metric_sums: dict[str, torch.Tensor] = {}
        self._metric_counts: dict[str, int] = {}

    @property
    def rank_mode(self) -> str:
        return self.base.rank_mode

    @property
    def teacher_weight(self) -> float:
        return self.base.teacher_weight

    @teacher_weight.setter
    def teacher_weight(self, value: float) -> None:
        self.base.teacher_weight = float(value)

    @property
    def topk_preservation_weight(self) -> float:
        return self.base.topk_preservation_weight

    @property
    def canonical_validation(self) -> bool:
        return self.base.canonical_validation

    @canonical_validation.setter
    def canonical_validation(self, value: bool) -> None:
        self.base.canonical_validation = bool(value)

    @property
    def projection_weight(self):
        return self.base.projection_weight

    @projection_weight.setter
    def projection_weight(self, value) -> None:
        self.base.projection_weight = value

    @property
    def last_scores(self):
        return self.base.last_scores

    @property
    def last_labels(self):
        return self.base.last_labels

    def __getattr__(self, name: str):
        # Keep the standard trainer contract (teacher/cycle/safe-update queues)
        # without duplicating bookkeeping that this isolated path never alters.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def set_atomic_and_records(
        self, records: Sequence[dict[str, Any] | None]
    ) -> None:
        if self._records:
            raise ValueError("Previous atomic-AND metadata were not consumed")
        self._records = list(records)

    def _take_records(self, count: int) -> list[dict[str, Any]]:
        if self._records is None or len(self._records) < count:
            available = 0 if self._records is None else len(self._records)
            raise ValueError(
                f"Need {count} atomic-AND metadata rows, found {available}"
            )
        selected = self._records[:count]
        del self._records[:count]
        if any(record is None for record in selected):
            raise ValueError("Every atomic-AND row requires metadata")
        return [record for record in selected if record is not None]

    def assert_atomic_and_records_consumed(self) -> None:
        if self._records:
            raise ValueError(
                f"{len(self._records)} atomic-AND metadata rows were not consumed"
            )
        self._records = None

    def reset_metrics(self) -> None:
        self.base.reset_metrics()
        self._metric_sums = {}
        self._metric_counts = {}

    def metric_totals(self) -> dict[str, tuple[torch.Tensor, int]]:
        result = dict(self.base.metric_totals())
        result.update(
            {
                name: (value, self._metric_counts[name])
                for name, value in self._metric_sums.items()
            }
        )
        return result

    def mean_metrics(self) -> dict[str, torch.Tensor]:
        result = dict(self.base.mean_metrics())
        result.update(
            {
                name: value / max(self._metric_counts[name], 1)
                for name, value in self._metric_sums.items()
            }
        )
        return result

    def _record_metric(self, name: str, value: torch.Tensor, count: int) -> None:
        detached = value.detach()
        self._metric_sums[name] = (
            detached
            if name not in self._metric_sums
            else self._metric_sums[name] + detached
        )
        self._metric_counts[name] = self._metric_counts.get(name, 0) + int(count)

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
            raise ValueError("Atomic-AND loss does not support sequence packing")
        if lin_weight is not None:
            self.projection_weight = lin_weight
        atomic_scores, response_labels = extract_binary_scores(
            output,
            target,
            self.base.response_spec,
            ignore_index=ignore_index,
            projection_weight=(
                lin_weight if lin_weight is not None else self.projection_weight
            ),
        )
        row_count = int(atomic_scores.numel())
        if row_count % self.atomic_group_size:
            raise ValueError(
                f"Atomic-AND batch has {row_count} rows, not a multiple of "
                f"{self.atomic_group_size}"
            )
        records = self._take_records(row_count)
        groups = row_count // self.atomic_group_size
        candidates = self.config.num_candidates
        fields = self.config.num_fields

        expected_positions = [
            (candidate_index, field_index)
            for _ in range(groups)
            for candidate_index in range(candidates)
            for field_index in range(fields)
        ]
        observed_positions = [
            (int(record["candidate_index"]), int(record["field_index"]))
            for record in records
        ]
        if observed_positions != expected_positions:
            raise ValueError("Atomic-AND rows are not in candidate-major field order")

        score_grid = atomic_scores.float().reshape(groups, candidates, fields)
        response_grid = response_labels.reshape(groups, candidates, fields)
        active = torch.tensor(
            [bool(record["active"]) for record in records],
            device=score_grid.device,
            dtype=torch.bool,
        ).reshape_as(score_grid)
        metadata_atomic_labels = torch.tensor(
            [float(record["atomic_label"]) for record in records],
            device=score_grid.device,
            dtype=torch.float32,
        ).reshape_as(score_grid)
        if not torch.equal(response_grid, metadata_atomic_labels):
            raise ValueError("Atomic assistant responses disagree with hard metadata")
        if not bool(active.any(dim=2).all()):
            raise ValueError("Every candidate requires at least one active atomic field")
        # Query requirements must be identical for every candidate in a group.
        if not torch.equal(active, active[:, :1, :].expand_as(active)):
            raise ValueError("Atomic active-field masks differ across candidates")

        full_labels = torch.tensor(
            [
                float(records[(group * candidates + candidate) * fields]["full_label"])
                for group in range(groups)
                for candidate in range(candidates)
            ],
            device=score_grid.device,
            dtype=torch.float32,
        ).reshape(groups, candidates)
        hard_conjunction = torch.where(
            active,
            metadata_atomic_labels.bool(),
            torch.ones_like(active),
        ).all(dim=2)
        if not torch.equal(hard_conjunction, full_labels.bool()):
            raise ValueError("Atomic hard-label conjunction disagrees with full label")

        normalized = score_grid
        if self.config.normalization in {"query_center", "query_zscore"}:
            mean = score_grid.mean(dim=1, keepdim=True)
            normalized = score_grid - mean
            if self.config.normalization == "query_zscore":
                scale = score_grid.std(dim=1, keepdim=True, unbiased=False).clamp_min(
                    self.config.normalization_epsilon
                )
                normalized = normalized / scale

        candidate_scores = aggregate_atomic_candidate_scores(
            normalized,
            active,
            aggregation=self.config.aggregation,
            temperature=float(self.config.temperature),
        )
        masked = normalized.masked_fill(~active, float("inf"))
        rank_loss = self.base.from_scores(
            candidate_scores.reshape(-1),
            full_labels.reshape(-1),
            loss_scaling_factor=loss_scaling_factor,
        )

        valid_scores = score_grid[active]
        valid_labels = metadata_atomic_labels[active]
        field_loss = F.binary_cross_entropy_with_logits(
            valid_scores / self.base.point_temperature,
            valid_labels,
        )
        valid_count = int(valid_labels.numel())
        self._record_metric(
            "atomic_field_point_loss", field_loss * valid_count, valid_count
        )
        self._record_metric(
            "atomic_field_accuracy",
            valid_scores.gt(0).eq(valid_labels.bool()).float().sum(),
            valid_count,
        )
        weakest_index = masked.argmin(dim=2)
        weakest_is_failed = torch.gather(
            metadata_atomic_labels, 2, weakest_index.unsqueeze(2)
        ).squeeze(2).eq(0)
        negative = ~full_labels.bool()
        negative_count = int(negative.sum().item())
        if negative_count:
            self._record_metric(
                "atomic_weakest_field_identifies_failure",
                weakest_is_failed[negative].float().sum(),
                negative_count,
            )
        if self.canonical_validation or not self.config.field_point_weight:
            return rank_loss
        return rank_loss + (
            float(self.config.field_point_weight)
            * field_loss
            * float(loss_scaling_factor)
        )


class RankPointWithHCRLoss:
    """Use fixed compatibility probes as part of the actual reranker score.

    Legacy mode deploys a direct probe conjunction. Active-logical mode uses
    separate required/satisfied Bernoulli probes; tri-state-logical replaces
    each pair with one mutually-exclusive inactive/satisfied/violated head.
    Both logical modes retain the parent's native score. The exact composite
    sent to the ranking loss is retained in ``base.last_scores`` and therefore
    drives validation metrics and deployment ordering as well.
    """

    def __init__(
        self,
        base: RankPointLoss,
        tokenizer: Any,
        config: HCRConfig,
        *,
        requirement_balance_group=None,
    ) -> None:
        self.base = base
        self.tokenizer = tokenizer
        self.config = config
        self.requirement_balance_group = requirement_balance_group
        self._current_logical_score_weight = self._scheduled_logical_score_weight(0)
        self._records: list[dict[str, Any] | None] | None = None
        self._metric_sums: dict[str, torch.Tensor] = {}
        self._metric_counts: dict[str, int] = {}
        self._tristate_token_ids = (
            tristate_class_token_ids(
                tokenizer,
                class_tokens=(
                    PAS_TRISTATE_SEMANTIC_CLASS_TOKENS
                    if config.tristate_token_mode == "semantic"
                    else ("A", "B", "C")
                ),
            )
            if config.mode in {
                "tristate_logical",
                "predecision_tristate",
                "postdecision_tristate",
            }
            else None
        )
        self._decision_state_token_strings = (
            tuple("BCDEFGHIJKLMNOPQRSTUVWXY")
            if config.mode == "decision_state_tristate"
            else ()
        )
        self._decision_state_token_ids: tuple[tuple[int, int, int], ...] | None = None
        if self._decision_state_token_strings:
            encoded = [
                tokenizer.encode(token, add_special_tokens=False)
                for token in self._decision_state_token_strings
            ]
            if any(len(ids) != 1 for ids in encoded):
                raise ValueError(
                    "Every decision-state auxiliary class token must be one token"
                )
            flat_ids = tuple(int(ids[0]) for ids in encoded)
            if len(set(flat_ids)) != 24:
                raise ValueError(
                    "Decision-state auxiliary requires 24 unique token IDs"
                )
            binary_template_ids = set(base.response_spec.positive_ids) | set(
                base.response_spec.negative_ids
            )
            overlap = sorted(set(flat_ids) & binary_template_ids)
            if overlap:
                raise ValueError(
                    "Decision-state auxiliary token conflicts with yes/no template: "
                    f"{overlap}"
                )
            special_overlap = sorted(set(flat_ids) & set(tokenizer.all_special_ids))
            if special_overlap:
                raise ValueError(
                    "Decision-state auxiliary token must be an ordinary vocab row: "
                    f"{special_overlap}"
                )
            self._decision_state_token_ids = tuple(
                tuple(flat_ids[index : index + 3])
                for index in range(0, 24, 3)
            )
            logger.info(
                "Decision-state HCR token triplets (inactive,satisfied,violated): %s",
                dict(
                    zip(
                        PAS_COMPATIBILITY_PROBE_FIELDS,
                        self._decision_state_token_ids,
                        strict=True,
                    )
                ),
            )
        self._predecision_response = predecision_tristate_probe_response(
            config.predecision_prompt_mode
        )
        self._gradient_component_mode: Literal[
            "combined", "primary", "auxiliary"
        ] = "combined"

    @property
    def rank_mode(self) -> str:
        return self.base.rank_mode

    @property
    def teacher_weight(self) -> float:
        return self.base.teacher_weight

    @teacher_weight.setter
    def teacher_weight(self, value: float) -> None:
        self.base.teacher_weight = float(value)

    @property
    def topk_preservation_weight(self) -> float:
        return self.base.topk_preservation_weight

    @property
    def last_scores(self):
        """Expose the exact deployed scores recorded by the wrapped loss."""

        return self.base.last_scores

    @property
    def last_labels(self):
        return self.base.last_labels

    @property
    def canonical_validation(self) -> bool:
        return self.base.canonical_validation

    @canonical_validation.setter
    def canonical_validation(self, value: bool) -> None:
        self.base.canonical_validation = bool(value)

    @property
    def projection_weight(self):
        return self.base.projection_weight

    @projection_weight.setter
    def projection_weight(self, value) -> None:
        self.base.projection_weight = value

    def _scheduled_logical_score_weight(self, train_step: int) -> float:
        """Return the active logical-residual weight for an optimizer step."""

        if train_step < 0:
            raise ValueError("HCR train_step must be non-negative")
        final = float(self.config.logical_score_weight)
        start = (
            final
            if self.config.logical_score_weight_start is None
            else float(self.config.logical_score_weight_start)
        )
        if start == final:
            return final
        warmup = int(self.config.logical_score_warmup_steps)
        ramp = int(self.config.logical_score_ramp_steps)
        if train_step <= warmup:
            return start
        if ramp == 0:
            return final
        progress = min(max((train_step - warmup) / ramp, 0.0), 1.0)
        return start + (final - start) * progress

    def set_train_step(self, train_step: int) -> None:
        """Update all step-scheduled HCR state used by train and validation."""

        self._current_logical_score_weight = self._scheduled_logical_score_weight(
            int(train_step)
        )

    @property
    def current_logical_score_weight(self) -> float:
        """Logical-residual weight currently used in the deployed score."""

        return self._current_logical_score_weight

    def set_teacher_probabilities(self, values) -> None:
        self.base.set_teacher_probabilities(values)

    def assert_teacher_probabilities_consumed(self) -> None:
        self.base.assert_teacher_probabilities_consumed()

    def set_total_relevant(self, values) -> None:
        self.base.set_total_relevant(values)

    def assert_total_relevant_consumed(self) -> None:
        self.base.assert_total_relevant_consumed()

    def assert_cycle_weights_consumed(self) -> None:
        self.base.assert_cycle_weights_consumed()

    def assert_cycle_block_metadata_consumed(self) -> None:
        self.base.assert_cycle_block_metadata_consumed()

    def set_hcr_records(self, records: Sequence[dict[str, Any] | None]) -> None:
        if self._records:
            raise ValueError("Previous HCR metadata was not consumed")
        self._records = list(records)

    def _take_records(self, count: int) -> list[dict[str, Any]]:
        if self._records is None or len(self._records) < count:
            available = 0 if self._records is None else len(self._records)
            raise ValueError(f"Need {count} HCR metadata rows, found {available}")
        selected = self._records[:count]
        del self._records[:count]
        if any(record is None for record in selected):
            raise ValueError("Every HCR candidate must have constraint metadata")
        return [record for record in selected if record is not None]

    def assert_hcr_records_consumed(self) -> None:
        if self._records:
            raise ValueError(f"{len(self._records)} HCR metadata rows were not consumed")
        self._records = None

    def reset_metrics(self) -> None:
        self.base.reset_metrics()
        self._metric_sums = {}
        self._metric_counts = {}

    def metric_totals(self) -> dict[str, tuple[torch.Tensor, int]]:
        result = dict(self.base.metric_totals())
        result.update(
            {
                name: (value, self._metric_counts[name])
                for name, value in self._metric_sums.items()
            }
        )
        return result

    def mean_metrics(self) -> dict[str, torch.Tensor]:
        result = dict(self.base.mean_metrics())
        result.update(
            {
                name: value / max(self._metric_counts[name], 1)
                for name, value in self._metric_sums.items()
            }
        )
        return result

    def _record_metric(self, name: str, total: torch.Tensor, count: int) -> None:
        detached = total.detach()
        self._metric_sums[name] = (
            detached
            if name not in self._metric_sums
            else self._metric_sums[name] + detached
        )
        self._metric_counts[name] = self._metric_counts.get(name, 0) + int(count)

    def set_gradient_component_mode(
        self, mode: Literal["combined", "primary", "auxiliary"]
    ) -> None:
        """Select the scalar component consumed by an external gradient surgeon."""

        if mode not in {"combined", "primary", "auxiliary"}:
            raise ValueError(f"Invalid HCR gradient component mode: {mode}")
        self._gradient_component_mode = mode

    def _combine_primary_and_auxiliary(
        self,
        output: torch.Tensor,
        primary: torch.Tensor,
        auxiliary: torch.Tensor,
    ) -> torch.Tensor:
        """Return primary plus a non-conflicting HCR output-space update.

        For ``output_pcgrad``, let ``p=d(primary)/d(output)`` and
        ``a=d(auxiliary)/d(output)`` over the concatenated DP batch. If their
        dot product is negative, replace ``a`` by its projection onto the
        half-space ``<p, a'> >= 0``. The detached projected tangent is then
        backpropagated through the ordinary model graph with a value-preserving
        surrogate, so the reported scalar remains ``primary + auxiliary``.

        This is PCGrad in output space, but only an approximation after mapping
        through the output-to-parameter Jacobian: output orthogonality does not
        imply parameter-gradient orthogonality unless ``J J^T`` is isotropic.
        It is deliberately not parameter-space PCGrad; see the config comment.
        """

        if self.config.auxiliary_gradient_mode == "none":
            return primary + auxiliary
        if self.config.auxiliary_gradient_mode == "parameter_pcgrad":
            if self._gradient_component_mode == "primary":
                return primary
            if self._gradient_component_mode == "auxiliary":
                return auxiliary
            raise RuntimeError(
                "parameter_pcgrad requires the dedicated PAS parameter-PCGrad "
                "trainer to request primary and auxiliary backwards separately"
            )
        if self.config.auxiliary_gradient_mode != "output_pcgrad":
            raise ValueError(
                f"Unsupported HCR auxiliary gradient mode: "
                f"{self.config.auxiliary_gradient_mode}"
            )
        if not torch.is_grad_enabled() or not output.requires_grad:
            raise RuntimeError(
                "output_pcgrad requires a differentiable training forward pass"
            )

        primary_gradient = torch.autograd.grad(
            primary,
            output,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        auxiliary_gradient = torch.autograd.grad(
            auxiliary,
            output,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        primary_f32 = primary_gradient.float()
        auxiliary_f32 = auxiliary_gradient.float()
        dot = (primary_f32 * auxiliary_f32).sum()
        primary_norm_sq = primary_f32.square().sum()
        auxiliary_norm_sq = auxiliary_f32.square().sum()
        if self.requirement_balance_group is not None:
            # Output rows on different DP ranks are disjoint coordinates of the
            # global batch, so these scalar sums are exact global output-space
            # dot products/norms without exchanging the full tensors.
            torch.distributed.all_reduce(dot, group=self.requirement_balance_group)
            torch.distributed.all_reduce(
                primary_norm_sq, group=self.requirement_balance_group
            )
            torch.distributed.all_reduce(
                auxiliary_norm_sq, group=self.requirement_balance_group
            )

        conflict = dot < 0
        coefficient = torch.where(
            conflict,
            -dot
            / primary_norm_sq.clamp_min(
                float(self.config.auxiliary_gradient_epsilon)
            ),
            dot.new_zeros(()),
        )
        projected_auxiliary = auxiliary_gradient + coefficient.to(
            dtype=auxiliary_gradient.dtype
        ) * primary_gradient
        projected_f32 = projected_auxiliary.float()
        projected_dot = (primary_f32 * projected_f32).sum()
        projected_norm_sq = projected_f32.square().sum()
        if self.requirement_balance_group is not None:
            torch.distributed.all_reduce(
                projected_dot, group=self.requirement_balance_group
            )
            torch.distributed.all_reduce(
                projected_norm_sq, group=self.requirement_balance_group
            )

        cosine = dot / (
            primary_norm_sq.clamp_min(
                float(self.config.auxiliary_gradient_epsilon)
            ).sqrt()
            * auxiliary_norm_sq.clamp_min(
                float(self.config.auxiliary_gradient_epsilon)
            ).sqrt()
        )
        retained_norm_ratio = (
            projected_norm_sq.clamp_min(0.0).sqrt()
            / auxiliary_norm_sq.clamp_min(
                float(self.config.auxiliary_gradient_epsilon)
            ).sqrt()
        )
        self._record_metric("hcr_aux_output_cosine", cosine, 1)
        self._record_metric("hcr_aux_conflict", conflict.float(), 1)
        self._record_metric("hcr_aux_projection_coefficient", coefficient, 1)
        self._record_metric("hcr_aux_projected_dot", projected_dot, 1)
        self._record_metric("hcr_aux_retained_norm_ratio", retained_norm_ratio, 1)

        desired_output_gradient = primary_gradient + projected_auxiliary
        surrogate = torch.dot(
            output.reshape(-1), desired_output_gradient.detach().reshape(-1)
        )
        reported_value = (primary + auxiliary).detach()
        return reported_value + surrogate - surrogate.detach()

    def _record_score_grid_metrics(
        self,
        native_scores: torch.Tensor,
        logical_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Record grouped AP/R1/R5 for cheap HCR readout calibration.

        All variants reuse the same CR3 forward pass and are detached
        telemetry.  This makes score-selection experiments cheap without
        silently changing the loss or selecting a different checkpoint.
        """

        variants: list[tuple[str, torch.Tensor]] = []
        for weight in self.config.diagnostic_logical_weights:
            suffix = format(float(weight), ".6g").replace("-", "m").replace(".", "p")
            variants.append(
                (f"hcr_grid_w{suffix}", native_scores + float(weight) * logical_scores)
            )
        if self.config.diagnostic_logical_only:
            variants.append(("hcr_grid_logical_only", logical_scores))
        if not variants:
            return

        group_size = self.base.group_size
        if scores_count := int(labels.numel()):
            if scores_count % group_size:
                raise ValueError("HCR score-grid metrics require complete groups")
        grouped_labels = labels.reshape(-1, group_size).bool()
        if not bool(grouped_labels.any(dim=1).all()):
            raise ValueError("Every HCR score-grid group requires a positive")
        if not bool((~grouped_labels).any(dim=1).all()):
            raise ValueError("Every HCR score-grid group requires a negative")
        num_groups = int(grouped_labels.shape[0])

        with torch.no_grad():
            for prefix, flat_scores in variants:
                grouped_scores = flat_scores.reshape(-1, group_size)
                ap_values: list[torch.Tensor] = []
                r1_values: list[torch.Tensor] = []
                r5_values: list[torch.Tensor] = []
                for group_scores, positive in zip(
                    grouped_scores, grouped_labels, strict=True
                ):
                    order = torch.argsort(group_scores, descending=True, stable=True)
                    ordered_positive = positive[order].float()
                    precision = ordered_positive.cumsum(0) / torch.arange(
                        1,
                        group_size + 1,
                        device=group_scores.device,
                        dtype=torch.float32,
                    )
                    ap_values.append(
                        (precision * ordered_positive).sum()
                        / ordered_positive.sum()
                    )
                    r1_values.append(ordered_positive[0])
                    r5_values.append(ordered_positive[:5].any().float())
                self._record_metric(
                    f"{prefix}_average_precision", torch.stack(ap_values).sum(), num_groups
                )
                self._record_metric(
                    f"{prefix}_top1_accuracy", torch.stack(r1_values).sum(), num_groups
                )
                self._record_metric(
                    f"{prefix}_rank5_accuracy", torch.stack(r5_values).sum(), num_groups
                )

    @staticmethod
    def _share_query_logits(logits: torch.Tensor, group_size: int) -> torch.Tensor:
        """Replace candidate-dependent query logits by their group mean."""

        if group_size <= 0 or logits.shape[0] % group_size:
            raise ValueError(
                "Group-shared HCR requirements require complete candidate groups"
            )
        return (
            logits.reshape(-1, group_size, *logits.shape[1:])
            .mean(dim=1, keepdim=True)
            .expand(-1, group_size, *([-1] * (logits.ndim - 1)))
            .reshape_as(logits)
        )

    def __call__(self, output, target, **kwargs):
        records = self._take_records(int(target.shape[0]))
        projection = (
            kwargs.get("lin_weight")
            if kwargs.get("lin_weight") is not None
            else self.projection_weight
        )
        if self.config.mode == "active_logical":
            return self._active_logical_call(
                output,
                target,
                records=records,
                projection=projection,
                **kwargs,
            )
        if self.config.mode == "tristate_logical":
            return self._tristate_logical_call(
                output,
                target,
                records=records,
                projection=projection,
                **kwargs,
            )
        if self.config.mode == "postdecision_tristate":
            return self._tristate_logical_call(
                output,
                target,
                records=records,
                projection=projection,
                postdecision=True,
                **kwargs,
            )
        if self.config.mode == "predecision_tristate":
            return self._predecision_tristate_call(
                output,
                target,
                records=records,
                projection=projection,
                **kwargs,
            )
        if self.config.mode == "decision_state_tristate":
            return self._decision_state_tristate_call(
                output,
                target,
                records=records,
                projection=projection,
                **kwargs,
            )
        return self._legacy_compatibility_call(
            output,
            target,
            records=records,
            projection=projection,
            **kwargs,
        )

    def _decision_state_tristate_call(
        self,
        output,
        target,
        *,
        records: Sequence[dict[str, Any]],
        projection,
        **kwargs,
    ):
        """Supervise eight hard-label probes at the deployed yes/no state."""

        if self._decision_state_token_ids is None:
            raise RuntimeError("Decision-state token triplets were not initialized")
        # This is the unmodified production objective and the sole deployed
        # score. It reads yes-minus-no from the same hidden state used below.
        rank_point = self.base(output, target, **kwargs)
        if self.canonical_validation:
            return rank_point

        ignore_index = int(kwargs.get("ignore_index", -100))
        decision_outputs: list[torch.Tensor] = []
        for row_index, record in enumerate(records):
            supervised_positions = torch.nonzero(
                target[row_index].ne(ignore_index), as_tuple=False
            ).flatten()
            if supervised_positions.numel() <= self.base.response_spec.decision_offset:
                raise ValueError("Decision-state sample has no binary decision token")
            supervised_ids = target[row_index, supervised_positions].tolist()
            positive_prefix = self.base.response_spec.positive_ids[
                : self.base.response_spec.decision_offset + 1
            ]
            negative_prefix = self.base.response_spec.negative_ids[
                : self.base.response_spec.decision_offset + 1
            ]
            if not (
                tuple(supervised_ids[: len(positive_prefix)]) == positive_prefix
                or tuple(supervised_ids[: len(negative_prefix)]) == negative_prefix
            ):
                raise ValueError(
                    "Decision-state HCR response is not the plain binary template: "
                    f"sample={record.get('sample_id', row_index)!r}"
                )
            target_position = int(
                supervised_positions[self.base.response_spec.decision_offset].item()
            )
            if target_position <= 0:
                raise ValueError("Decision-state binary token has no prediction state")
            decision_outputs.append(output[row_index, target_position - 1])
        decision_hidden = torch.stack(decision_outputs)

        flat_token_ids = tuple(
            token_id
            for triplet in self._decision_state_token_ids
            for token_id in triplet
        )
        local_projection = projection
        if hasattr(local_projection, "to_local"):
            local_projection = local_projection.to_local()
        if local_projection is None or decision_hidden.shape[-1] == local_projection.shape[0]:
            class_logits = decision_hidden[..., list(flat_token_ids)].float()
        elif decision_hidden.shape[-1] == local_projection.shape[1]:
            token_tensor = torch.tensor(
                flat_token_ids, device=local_projection.device, dtype=torch.long
            )
            weights = local_projection.index_select(0, token_tensor).float()
            class_logits = F.linear(decision_hidden.float(), weights)
        else:
            raise ValueError(
                "Decision-state projection shape mismatch: "
                f"hidden={decision_hidden.shape}, weight={local_projection.shape}"
            )
        class_logits = class_logits.reshape(
            decision_hidden.shape[0], len(PAS_COMPATIBILITY_PROBE_FIELDS), 3
        )

        targets: list[list[int]] = []
        valid: list[list[bool]] = []
        for row_index, record in enumerate(records):
            probes = record.get("tristate_probe_targets")
            if not isinstance(probes, list) or len(probes) != len(
                PAS_COMPATIBILITY_PROBE_FIELDS
            ):
                raise ValueError(
                    f"Decision-state sample {row_index} lacks eight hard targets"
                )
            fields = tuple(str(probe.get("field")) for probe in probes)
            if fields != tuple(PAS_COMPATIBILITY_PROBE_FIELDS):
                raise ValueError(f"Decision-state field order mismatch: {fields}")
            row_targets = [int(probe["target"]) for probe in probes]
            if any(value not in {0, 1, 2} for value in row_targets):
                raise ValueError("Decision-state target lies outside {0,1,2}")
            targets.append(row_targets)
            valid.append([bool(probe.get("valid", True)) for probe in probes])
        class_targets = torch.tensor(
            targets, device=class_logits.device, dtype=torch.long
        )
        class_valid = torch.tensor(valid, device=class_logits.device, dtype=torch.bool)
        if not bool(class_valid.all()):
            raise ValueError(
                "Decision-state training requires hard labels for all eight fields"
            )
        tristate_loss = self._masked_categorical_loss(
            class_logits, class_targets, class_valid
        )

        with torch.no_grad():
            predictions = class_logits.argmax(dim=-1)
            self._record_metric(
                "decision_state_tristate_accuracy",
                predictions.eq(class_targets).float().sum(),
                class_targets.numel(),
            )
            self._record_metric(
                "decision_state_tristate_loss",
                tristate_loss * class_logits.shape[0],
                class_logits.shape[0],
            )
            conjunction = predictions.ne(2).all(dim=1)
            labels = torch.tensor(
                [bool(record["binary_label"]) for record in records],
                device=class_logits.device,
                dtype=torch.bool,
            )
            self._record_metric(
                "decision_state_conjunction_accuracy",
                conjunction.eq(labels).float().sum(),
                labels.numel(),
            )

        auxiliary = (
            self.config.field_loss_weight
            * self.config.tristate_loss_weight
            * tristate_loss
            * float(kwargs.get("loss_scaling_factor", 1.0))
        )
        return self._combine_primary_and_auxiliary(
            output, rank_point, auxiliary
        )

    def _legacy_compatibility_call(
        self,
        output,
        target,
        *,
        records: Sequence[dict[str, Any]],
        projection,
        **kwargs,
    ):
        field_scores, field_targets, field_valid = extract_compatibility_probe_scores(
            output,
            target,
            records,
            self.tokenizer,
            yes_token_id=self.base.response_spec.positive_token_id,
            no_token_id=self.base.response_spec.negative_token_id,
            projection_weight=projection,
        )
        # Every slot is part of the deployed score. Training assets supervise
        # wildcards as satisfied constraints; held-out inference may omit
        # field targets entirely because aggregation never consumes them.
        if self.config.aggregation == "normalized_softmin":
            scores = normalized_softmin(
                field_scores,
                temperature=self.config.constraint_temperature,
                dim=1,
            )
        else:
            scores = bottom_k_cvar(
                field_scores,
                k=self.config.bottom_k,
                dim=1,
            )
        labels = output.new_tensor(
            [int(record["binary_label"]) for record in records],
            dtype=torch.float32,
        )
        scale = float(kwargs.get("loss_scaling_factor", 1.0))
        rank_point = self.base.from_scores(
            scores,
            labels,
            loss_scaling_factor=scale,
        )

        field_losses: list[torch.Tensor] = []
        for field_index in range(field_scores.shape[1]):
            valid = field_valid[:, field_index]
            if not bool(valid.any()):
                continue
            logits = field_scores[valid, field_index]
            targets = field_targets[valid, field_index].float()
            elementwise = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            positive = targets > 0.5
            negative = ~positive
            if bool(positive.any()) and bool(negative.any()):
                field_losses.append(
                    0.5 * elementwise[positive].mean()
                    + 0.5 * elementwise[negative].mean()
                )
            else:
                field_losses.append(elementwise.mean())
        field_loss = (
            torch.stack(field_losses).mean()
            if field_losses
            else field_scores.sum() * 0.0
        )

        with torch.no_grad():
            constraint_correct = ((field_scores > 0) == field_targets) & field_valid
            conjunction_prediction = (field_scores > 0).all(dim=1)
            valid_count = int(field_valid.sum().item())
            if valid_count:
                self._record_metric(
                    "hcr_field_loss",
                    field_loss * field_scores.shape[0],
                    field_scores.shape[0],
                )
                self._record_metric(
                    "hcr_constraint_accuracy",
                    constraint_correct.float().sum(),
                    valid_count,
                )
            self._record_metric(
                "hcr_conjunction_accuracy",
                (conjunction_prediction == labels.bool()).float().sum(),
                labels.numel(),
            )
            self._record_metric("hcr_score", scores.sum(), scores.numel())

        if self.canonical_validation:
            return rank_point
        return self._combine_primary_and_auxiliary(
            output,
            rank_point,
            self.config.field_loss_weight * field_loss * scale,
        )

    def _fixed_response_native_scores(
        self,
        output: torch.Tensor,
        target: torch.LongTensor,
        records: Sequence[dict[str, Any]],
        projection,
        *,
        expected_response: str,
        mode_name: str,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Read the parent-compatible score before the neutral ``?`` token."""

        decision_outputs: list[torch.Tensor] = []
        decision_offset = self.base.response_spec.decision_offset
        expected_ids = self.tokenizer.encode(expected_response, add_special_tokens=False)
        for row_index, record in enumerate(records):
            if str(record.get("response")) != expected_response:
                raise ValueError(f"{mode_name} HCR response is not the fixed template")
            supervised_positions = torch.nonzero(
                target[row_index].ne(ignore_index), as_tuple=False
            ).flatten()
            supervised_ids = target[row_index, supervised_positions].tolist()
            if supervised_ids[: len(expected_ids)] != expected_ids:
                raise ValueError(
                    f"Packed {mode_name} HCR response differs for sample "
                    f"{record.get('sample_id', row_index)!r}"
                )
            if decision_offset >= supervised_positions.numel():
                raise ValueError(f"{mode_name} HCR has no neutral native score slot")
            prediction_position = int(
                supervised_positions[decision_offset].item()
            ) - 1
            if prediction_position < 0:
                raise ValueError(f"{mode_name} HCR native score has no prediction token")
            decision_outputs.append(output[row_index, prediction_position])
        return yes_no_logit_difference(
            torch.stack(decision_outputs),
            yes_token_id=self.base.response_spec.positive_token_id,
            no_token_id=self.base.response_spec.negative_token_id,
            projection_weight=projection,
        )

    def _fixed_response_marker_scores(
        self,
        output: torch.Tensor,
        target: torch.LongTensor,
        records: Sequence[dict[str, Any]],
        projection,
        *,
        expected_response: str,
        marker_response: str,
        mode_name: str,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Read yes/no logits at a neutral marker anywhere in a fixed response."""

        expected_ids = self.tokenizer.encode(expected_response, add_special_tokens=False)
        marker_ids = self.tokenizer.encode(marker_response, add_special_tokens=False)
        decision_offset = self.base.response_spec.decision_offset
        if marker_ids[:decision_offset] != list(
            self.base.response_spec.positive_ids[:decision_offset]
        ):
            raise ValueError(f"{mode_name} marker does not share the binary prefix")
        offsets = [
            index
            for index in range(len(expected_ids) - len(marker_ids) + 1)
            if expected_ids[index : index + len(marker_ids)] == marker_ids
        ]
        if len(offsets) != 1:
            raise ValueError(f"{mode_name} response does not contain one answer marker")
        decision_index = offsets[0] + decision_offset

        decision_outputs: list[torch.Tensor] = []
        for row_index, record in enumerate(records):
            if str(record.get("response")) != expected_response:
                raise ValueError(f"{mode_name} HCR response is not the fixed template")
            supervised_positions = torch.nonzero(
                target[row_index].ne(ignore_index), as_tuple=False
            ).flatten()
            supervised_ids = target[row_index, supervised_positions].tolist()
            if supervised_ids[: len(expected_ids)] != expected_ids:
                raise ValueError(
                    f"Packed {mode_name} HCR response differs for sample "
                    f"{record.get('sample_id', row_index)!r}"
                )
            prediction_position = int(supervised_positions[decision_index].item()) - 1
            if prediction_position < 0:
                raise ValueError(f"{mode_name} score has no prediction token")
            decision_outputs.append(output[row_index, prediction_position])
        return yes_no_logit_difference(
            torch.stack(decision_outputs),
            yes_token_id=self.base.response_spec.positive_token_id,
            no_token_id=self.base.response_spec.negative_token_id,
            projection_weight=projection,
        )

    @staticmethod
    def _masked_binary_loss(
        logits: torch.Tensor,
        targets: torch.BoolTensor,
        valid: torch.BoolTensor,
    ) -> torch.Tensor:
        """Average class-balanced BCE independently across semantic fields.

        Requirement prevalence differs sharply by field (for example, garment
        type is almost always specified while viewpoint often is not).  A
        single flattened BCE would therefore let frequent fields/classes
        dominate the very activity errors that can disable a logical veto.
        """

        field_losses: list[torch.Tensor] = []
        for field_index in range(logits.shape[1]):
            field_valid = valid[:, field_index]
            if not bool(field_valid.any()):
                continue
            field_logits = logits[field_valid, field_index]
            field_targets = targets[field_valid, field_index].float()
            elementwise = F.binary_cross_entropy_with_logits(
                field_logits, field_targets, reduction="none"
            )
            positive = field_targets > 0.5
            negative = ~positive
            if bool(positive.any()) and bool(negative.any()):
                field_losses.append(
                    0.5 * elementwise[positive].mean()
                    + 0.5 * elementwise[negative].mean()
                )
            else:
                field_losses.append(elementwise.mean())
        return (
            torch.stack(field_losses).mean()
            if field_losses
            else logits.sum() * 0.0
        )

    def _query_level_requirement_loss(
        self,
        logits: torch.Tensor,
        targets: torch.BoolTensor,
        valid: torch.BoolTensor,
    ) -> torch.Tensor:
        """Class-balance requirement supervision over independent queries.

        A requirement belongs to the query, not to any one candidate.  PAS
        repeats each query K times, so balancing the candidate rows makes six
        queries look like 120 independent observations and lets image-dependent
        noise affect the semantic head.  Supervise the group-mean logit instead.
        When data parallelism is active, all-reduce detached class counts so
        every replica uses the same global class prior without gathering an
        autograd graph or duplicating the global loss on every rank.
        """

        group_size = self.base.group_size
        if logits.shape[0] % group_size:
            raise ValueError(
                "Active HCR requirement loss requires complete candidate groups"
            )
        grouped_logits = logits.reshape(-1, group_size, logits.shape[1])
        grouped_targets = targets.reshape(-1, group_size, targets.shape[1])
        grouped_valid = valid.reshape(-1, group_size, valid.shape[1])
        if not bool((grouped_targets == grouped_targets[:, :1]).all()):
            raise ValueError("HCR requirement targets changed within a query group")
        if not bool((grouped_valid == grouped_valid[:, :1]).all()):
            raise ValueError("HCR requirement masks changed within a query group")

        query_logits = grouped_logits.mean(dim=1)
        query_targets = grouped_targets[:, 0]
        query_valid = grouped_valid[:, 0]

        # Only counts cross the DP boundary.  Gathering differentiable logits
        # would make every replica evaluate the identical global loss and rely
        # on subtle all-gather-backward/FSDP reduction cancellation for scale.
        # Shared detached counts provide stable class weights while each rank
        # backpropagates only through its own independent query summaries.
        positive_counts = (query_targets & query_valid).sum(dim=0).float()
        negative_counts = ((~query_targets) & query_valid).sum(dim=0).float()
        class_counts = torch.stack((positive_counts, negative_counts))
        if self.requirement_balance_group is not None:
            torch.distributed.all_reduce(
                class_counts,
                op=torch.distributed.ReduceOp.SUM,
                group=self.requirement_balance_group,
            )

        field_losses: list[torch.Tensor] = []
        for field_index in range(query_logits.shape[1]):
            field_valid = query_valid[:, field_index]
            if not bool(field_valid.any()):
                # Requirement masks are uniform across PAS DP ranks (all valid
                # for train, all masked for held-out inference). Keeping this
                # branch local also avoids a mean over an empty tensor.
                continue
            field_logits = query_logits[field_valid, field_index]
            field_targets = query_targets[field_valid, field_index].float()
            elementwise = F.binary_cross_entropy_with_logits(
                field_logits, field_targets, reduction="none"
            )
            global_positive = class_counts[0, field_index]
            global_negative = class_counts[1, field_index]
            global_total = global_positive + global_negative
            if bool(global_positive > 0) and bool(global_negative > 0):
                positive_weight = global_total / (2.0 * global_positive)
                negative_weight = global_total / (2.0 * global_negative)
                weights = torch.where(
                    field_targets > 0.5, positive_weight, negative_weight
                )
                field_losses.append((elementwise * weights).mean())
            else:
                field_losses.append(elementwise.mean())
        return (
            torch.stack(field_losses).mean()
            if field_losses
            else query_logits.sum() * 0.0
        )

    def _counterfactual_veto_rank_loss(
        self,
        satisfaction_logits: torch.Tensor,
        labels: torch.Tensor,
        records: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, int, int]:
        """Rank each sole-violation negative on its causal field only.

        All targets are deterministic hard construction labels. The loss is
        query-relative, so it does not assume that satisfaction margins are
        globally calibrated across different captions.
        """

        zero = satisfaction_logits.sum() * 0.0
        if not self.config.counterfactual_veto_rank_weight:
            return zero, 0, 0
        group_size = self.base.group_size
        if satisfaction_logits.shape[0] % group_size:
            raise ValueError("Counterfactual veto ranking requires complete groups")
        if len(records) != int(satisfaction_logits.shape[0]):
            raise ValueError("Counterfactual veto records/logits disagree")
        field_indices = {
            field: index
            for index, field in enumerate(PAS_COMPATIBILITY_PROBE_FIELDS)
        }
        # The dataset calls the set-valued accessory field accessory_subset;
        # the fixed model response template calls the same field accessory.
        field_indices["accessory_subset"] = field_indices["accessory"]
        pair_losses: list[torch.Tensor] = []
        group_losses: list[torch.Tensor] = []
        tau = float(self.config.counterfactual_veto_rank_temperature)
        margin = float(self.config.counterfactual_veto_rank_margin)
        for begin in range(0, len(records), group_size):
            group_labels = labels[begin : begin + group_size] > 0.5
            positive_offsets = torch.nonzero(
                group_labels, as_tuple=False
            ).flatten().tolist()
            if not positive_offsets:
                raise ValueError("Counterfactual veto group has no positive")
            field_pair_losses: dict[str, list[torch.Tensor]] = {}
            for offset, record in enumerate(records[begin : begin + group_size]):
                role = str(record.get("selection_role") or "")
                if not role.startswith("sole_violation:"):
                    continue
                if bool(group_labels[offset]):
                    raise ValueError("sole_violation row cannot be positive")
                field = role.split(":", 1)[1]
                if field not in field_indices:
                    raise ValueError(f"Unknown sole-violation field {field!r}")
                field_index = field_indices[field]
                negative_logit = satisfaction_logits[begin + offset, field_index]
                for positive_offset in positive_offsets:
                    positive_logit = satisfaction_logits[
                        begin + positive_offset, field_index
                    ]
                    pair_loss = F.softplus(
                        (negative_logit - positive_logit + margin) / tau
                    )
                    pair_losses.append(pair_loss)
                    field_pair_losses.setdefault(field, []).append(pair_loss)
            if field_pair_losses:
                group_losses.append(
                    torch.stack(
                        [
                            torch.stack(field_pair_losses[field]).mean()
                            for field in sorted(field_pair_losses)
                        ]
                    ).mean()
                )
        if not pair_losses:
            if not self.canonical_validation:
                raise ValueError(
                    "counterfactual_veto_rank_weight is enabled but this "
                    "training batch has no sole_violation metadata"
                )
            return zero, 0, 0
        if self.config.counterfactual_veto_rank_reduction == "group_field_mean":
            return torch.stack(group_losses).mean(), len(pair_losses), len(group_losses)
        return torch.stack(pair_losses).mean(), len(pair_losses), len(pair_losses)

    def _active_logical_call(
        self,
        output,
        target,
        *,
        records: Sequence[dict[str, Any]],
        projection,
        **kwargs,
    ):
        (
            requirement_logits,
            requirement_targets,
            requirement_valid,
            satisfaction_logits,
            satisfaction_targets,
            satisfaction_valid,
        ) = extract_active_compatibility_probe_scores(
            output,
            target,
            records,
            self.tokenizer,
            yes_token_id=self.base.response_spec.positive_token_id,
            no_token_id=self.base.response_spec.negative_token_id,
            projection_weight=projection,
        )
        native_scores = self._fixed_response_native_scores(
            output,
            target,
            records,
            projection,
            expected_response=PAS_ACTIVE_COMPATIBILITY_PROBE_RESPONSE,
            mode_name="active-logical",
        )
        score_requirement_logits = requirement_logits
        if self.config.group_shared_requirement:
            score_requirement_logits = self._share_query_logits(
                requirement_logits, self.base.group_size
            )
        logical_scores = active_aware_logical_score(
            score_requirement_logits,
            satisfaction_logits,
            dim=1,
            reduction=self.config.logical_reduction,
        )
        scores = (
            self.config.native_score_weight * native_scores
            + self.current_logical_score_weight * logical_scores
        )
        labels = output.new_tensor(
            [int(record["binary_label"]) for record in records],
            dtype=torch.float32,
        )
        self._record_score_grid_metrics(native_scores, logical_scores, labels)
        scale = float(kwargs.get("loss_scaling_factor", 1.0))
        rank_point = self.base.from_scores(
            scores,
            labels,
            loss_scaling_factor=scale,
        )

        native_elementwise = F.binary_cross_entropy_with_logits(
            native_scores, labels, reduction="none"
        )
        native_positive = labels > 0.5
        native_negative = ~native_positive
        if bool(native_positive.any()) and bool(native_negative.any()):
            native_point_loss = (
                0.5 * native_elementwise[native_positive].mean()
                + 0.5 * native_elementwise[native_negative].mean()
            )
        else:
            native_point_loss = native_elementwise.mean()

        requirement_loss = self._query_level_requirement_loss(
            requirement_logits, requirement_targets, requirement_valid
        )
        satisfaction_loss = self._masked_binary_loss(
            satisfaction_logits, satisfaction_targets, satisfaction_valid
        )
        veto_rank_loss, veto_pair_count, veto_loss_weight = self._counterfactual_veto_rank_loss(
            satisfaction_logits, labels, records
        )
        group_size = self.base.group_size
        if requirement_logits.shape[0] % group_size:
            raise ValueError(
                "Active HCR requirement consistency requires complete candidate groups"
            )
        grouped_requirements = requirement_logits.reshape(
            -1, group_size, requirement_logits.shape[1]
        )
        group_centers = grouped_requirements.mean(dim=1, keepdim=True)
        requirement_consistency = (
            grouped_requirements - group_centers
        ).square().mean()

        with torch.no_grad():
            requirement_correct = (
                (requirement_logits > 0) == requirement_targets
            ) & requirement_valid
            satisfaction_correct = (
                (satisfaction_logits > 0) == satisfaction_targets
            ) & satisfaction_valid
            requirement_count = int(requirement_valid.sum().item())
            satisfaction_count = int(satisfaction_valid.sum().item())
            if requirement_count:
                self._record_metric(
                    "hcr_requirement_accuracy",
                    requirement_correct.float().sum(),
                    requirement_count,
                )
                self._record_metric(
                    "hcr_requirement_loss",
                    requirement_loss * requirement_logits.shape[0],
                    requirement_logits.shape[0],
                )
            if satisfaction_count:
                self._record_metric(
                    "hcr_satisfaction_accuracy",
                    satisfaction_correct.float().sum(),
                    satisfaction_count,
                )
                self._record_metric(
                    "hcr_satisfaction_loss",
                    satisfaction_loss * satisfaction_logits.shape[0],
                    satisfaction_logits.shape[0],
                )
            self._record_metric(
                "hcr_requirement_group_consistency",
                requirement_consistency,
                1,
            )
            if veto_loss_weight:
                self._record_metric(
                    "hcr_counterfactual_veto_rank_loss",
                    veto_rank_loss * veto_loss_weight,
                    veto_loss_weight,
                )
                self._record_metric(
                    "hcr_counterfactual_veto_pairs",
                    scores.new_tensor(float(veto_pair_count)),
                    1,
                )
            self._record_metric("hcr_native_score", native_scores.sum(), scores.numel())
            self._record_metric(
                "hcr_native_point_loss",
                native_point_loss * native_scores.numel(),
                native_scores.numel(),
            )
            self._record_metric(
                "hcr_logical_score", logical_scores.sum(), scores.numel()
            )
            self._record_metric(
                "hcr_logical_score_weight",
                scores.new_tensor(self.current_logical_score_weight),
                1,
            )
            self._record_metric("hcr_score", scores.sum(), scores.numel())

        if self.canonical_validation:
            return rank_point
        auxiliary = (
            self.config.native_point_loss_weight * native_point_loss
            + self.config.requirement_loss_weight * requirement_loss
            + self.config.satisfaction_loss_weight * satisfaction_loss
            + self.config.requirement_group_consistency_weight
            * requirement_consistency
            + self.config.counterfactual_veto_rank_weight * veto_rank_loss
        )
        return self._combine_primary_and_auxiliary(
            output,
            rank_point,
            self.config.field_loss_weight * auxiliary * scale,
        )

    @staticmethod
    def _masked_categorical_loss(
        logits: torch.Tensor,
        targets: torch.LongTensor,
        valid: torch.BoolTensor,
    ) -> torch.Tensor:
        """Average class-balanced CE independently across semantic fields."""

        field_losses: list[torch.Tensor] = []
        for field_index in range(logits.shape[1]):
            field_valid = valid[:, field_index]
            if not bool(field_valid.any()):
                continue
            field_logits = logits[field_valid, field_index]
            field_targets = targets[field_valid, field_index]
            elementwise = F.cross_entropy(
                field_logits, field_targets, reduction="none"
            )
            class_losses = [
                elementwise[field_targets == class_index].mean()
                for class_index in range(logits.shape[-1])
                if bool((field_targets == class_index).any())
            ]
            field_losses.append(torch.stack(class_losses).mean())
        return (
            torch.stack(field_losses).mean()
            if field_losses
            else logits.sum() * 0.0
        )

    def _predecision_tristate_call(
        self,
        output,
        target,
        *,
        records: Sequence[dict[str, Any]],
        projection,
        **kwargs,
    ):
        """Train latent field decisions before the final answer.

        The fixed ``?`` inputs make this label-safe.  Field supervision shapes
        earlier hidden states, while the final answer hidden state can causally
        attend those states.  A configured logical residual is computed only
        from this same CR3 pass; a zero weight preserves the native-only
        deployed scalar.
        """

        if self._tristate_token_ids is None:
            raise RuntimeError("Pre-decision tri-state token IDs were not initialized")
        class_logits, class_targets, class_valid = (
            extract_tristate_compatibility_probe_logits(
                output,
                target,
                records,
                self.tokenizer,
                class_token_ids=self._tristate_token_ids,
                projection_weight=projection,
                expected_response=self._predecision_response,
            )
        )
        native_scores = self._fixed_response_marker_scores(
            output,
            target,
            records,
            projection,
            expected_response=self._predecision_response,
            marker_response="<answer>?</answer>",
            mode_name="predecision-tristate",
        )
        activity_logits = torch.logsumexp(class_logits[..., 1:], dim=-1) - (
            class_logits[..., 0]
        )
        satisfaction_logits = class_logits[..., 1] - class_logits[..., 2]
        if self.config.predecision_logical_mode == "active_aware_shared":
            shared_activity_logits = self._share_query_logits(
                activity_logits, self.base.group_size
            )
            logical_scores = active_aware_logical_score(
                shared_activity_logits,
                satisfaction_logits,
                dim=1,
                reduction=self.config.logical_reduction,
            )
        else:
            logical_scores = predecision_smooth_and_score(
                class_logits,
                temperature=self.config.constraint_temperature,
                dim=1,
            )
        scores = (
            self.config.native_score_weight * native_scores
            + self.current_logical_score_weight * logical_scores
        )
        labels = output.new_tensor(
            [int(record["binary_label"]) for record in records],
            dtype=torch.float32,
        )
        self._record_score_grid_metrics(native_scores, logical_scores, labels)
        scale = float(kwargs.get("loss_scaling_factor", 1.0))
        rank_point = self.base.from_scores(
            scores,
            labels,
            loss_scaling_factor=scale,
        )

        activity_targets = class_targets.ne(0)
        satisfaction_targets = class_targets.eq(1)
        satisfaction_valid = class_valid & activity_targets
        if self.config.tristate_loss_mode == "conditional":
            activity_loss = self._query_level_requirement_loss(
                activity_logits, activity_targets, class_valid
            )
            satisfaction_loss = self._masked_binary_loss(
                satisfaction_logits, satisfaction_targets, satisfaction_valid
            )
            tristate_loss = activity_loss + satisfaction_loss
        else:
            activity_loss = class_logits.sum() * 0.0
            satisfaction_loss = class_logits.sum() * 0.0
            tristate_loss = self._masked_categorical_loss(
                class_logits, class_targets, class_valid
            )

        group_size = self.base.group_size
        if activity_logits.shape[0] % group_size:
            raise ValueError(
                "Pre-decision HCR activity consistency requires complete groups"
            )
        grouped_activity = activity_logits.reshape(
            -1, group_size, activity_logits.shape[1]
        )
        activity_consistency = (
            grouped_activity - grouped_activity.mean(dim=1, keepdim=True)
        ).square().mean()

        with torch.no_grad():
            predictions = class_logits.argmax(dim=-1)
            exact = (predictions == class_targets) & class_valid
            active_valid = class_valid & class_targets.ne(0)
            active_correct = (
                predictions.ne(0) == class_targets.ne(0)
            ) & class_valid
            satisfaction_correct = (predictions == class_targets) & active_valid
            valid_count = int(class_valid.sum().item())
            active_count = int(active_valid.sum().item())
            if valid_count:
                self._record_metric(
                    "hcr_tristate_loss",
                    tristate_loss * class_logits.shape[0],
                    class_logits.shape[0],
                )
                self._record_metric(
                    "hcr_tristate_accuracy", exact.float().sum(), valid_count
                )
                self._record_metric(
                    "hcr_activity_accuracy", active_correct.float().sum(), valid_count
                )
                if self.config.tristate_loss_mode == "conditional":
                    query_count = class_logits.shape[0] // group_size
                    self._record_metric(
                        "hcr_activity_loss", activity_loss * query_count, query_count
                    )
            if active_count:
                self._record_metric(
                    "hcr_satisfaction_accuracy",
                    satisfaction_correct.float().sum(),
                    active_count,
                )
                if self.config.tristate_loss_mode == "conditional":
                    self._record_metric(
                        "hcr_satisfaction_loss",
                        satisfaction_loss * class_logits.shape[0],
                        class_logits.shape[0],
                    )
            conjunction_prediction = predictions.ne(2).all(dim=1)
            self._record_metric(
                "hcr_conjunction_accuracy",
                (conjunction_prediction == labels.bool()).float().sum(),
                labels.numel(),
            )
            self._record_metric(
                "hcr_activity_group_consistency", activity_consistency, 1
            )
            self._record_metric(
                "hcr_native_score", native_scores.sum(), native_scores.numel()
            )
            self._record_metric(
                "hcr_logical_score", logical_scores.sum(), logical_scores.numel()
            )
            self._record_metric(
                "hcr_logical_score_weight",
                scores.new_tensor(self.current_logical_score_weight),
                1,
            )
            self._record_metric("hcr_score", scores.sum(), scores.numel())

        if self.canonical_validation:
            return rank_point
        auxiliary = (
            self.config.tristate_loss_weight * tristate_loss
            + self.config.tristate_activity_consistency_weight
            * activity_consistency
        )
        return self._combine_primary_and_auxiliary(
            output,
            rank_point,
            self.config.field_loss_weight * auxiliary * scale,
        )

    def _tristate_logical_call(
        self,
        output,
        target,
        *,
        records: Sequence[dict[str, Any]],
        projection,
        postdecision: bool = False,
        **kwargs,
    ):
        if self._tristate_token_ids is None:
            raise RuntimeError("Tri-state HCR class tokens were not initialized")
        class_logits, class_targets, class_valid = (
            extract_tristate_compatibility_probe_logits(
                output,
                target,
                records,
                self.tokenizer,
                class_token_ids=self._tristate_token_ids,
                projection_weight=projection,
                expected_response=(
                    None
                    if postdecision
                    else PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE
                ),
            )
        )
        if postdecision:
            native_scores, response_labels = extract_binary_scores(
                output,
                target,
                self.base.response_spec,
                projection_weight=projection,
            )
            metadata_labels = output.new_tensor(
                [int(record["binary_label"]) for record in records],
                dtype=torch.float32,
            )
            if not bool(response_labels.eq(metadata_labels).all()):
                raise ValueError("post-decision response label disagrees with metadata")
        else:
            native_scores = self._fixed_response_native_scores(
                output,
                target,
                records,
                projection,
                expected_response=PAS_TRISTATE_COMPATIBILITY_PROBE_RESPONSE,
                mode_name="tri-state-logical",
            )
        logical_scores = tristate_logical_score(
            class_logits,
            dim=1,
            reduction=self.config.logical_reduction,
        )
        scores = (
            self.config.native_score_weight * native_scores
            + self.current_logical_score_weight * logical_scores
        )
        labels = output.new_tensor(
            [int(record["binary_label"]) for record in records],
            dtype=torch.float32,
        )
        self._record_score_grid_metrics(native_scores, logical_scores, labels)
        scale = float(kwargs.get("loss_scaling_factor", 1.0))
        rank_point = self.base.from_scores(
            scores,
            labels,
            loss_scaling_factor=scale,
        )

        native_elementwise = F.binary_cross_entropy_with_logits(
            native_scores, labels, reduction="none"
        )
        native_positive = labels > 0.5
        native_negative = ~native_positive
        if bool(native_positive.any()) and bool(native_negative.any()):
            native_point_loss = (
                0.5 * native_elementwise[native_positive].mean()
                + 0.5 * native_elementwise[native_negative].mean()
            )
        else:
            native_point_loss = native_elementwise.mean()
        # Activity is query-only. Its categorical log-odds are
        # logsumexp(satisfied, violated) - inactive; only satisfaction remains
        # candidate-dependent.
        activity_logits = torch.logsumexp(class_logits[..., 1:], dim=-1) - (
            class_logits[..., 0]
        )
        activity_targets = class_targets.ne(0)
        satisfaction_logits = class_logits[..., 1] - class_logits[..., 2]
        satisfaction_targets = class_targets.eq(1)
        satisfaction_valid = class_valid & activity_targets
        if self.config.tristate_loss_mode == "conditional":
            activity_loss = self._query_level_requirement_loss(
                activity_logits, activity_targets, class_valid
            )
            satisfaction_loss = self._masked_binary_loss(
                satisfaction_logits, satisfaction_targets, satisfaction_valid
            )
            # These two conditional Bernoulli decisions identify the same
            # three-class distribution while supervising each at its natural
            # query or candidate grain.
            tristate_loss = activity_loss + satisfaction_loss
        else:
            tristate_loss = self._masked_categorical_loss(
                class_logits, class_targets, class_valid
            )
            activity_loss = class_logits.sum() * 0.0
            satisfaction_loss = class_logits.sum() * 0.0
        group_size = self.base.group_size
        if activity_logits.shape[0] % group_size:
            raise ValueError(
                "Tri-state HCR activity consistency requires complete groups"
            )
        grouped_activity = activity_logits.reshape(
            -1, group_size, activity_logits.shape[1]
        )
        activity_centers = grouped_activity.mean(dim=1, keepdim=True)
        activity_consistency = (
            grouped_activity - activity_centers
        ).square().mean()

        with torch.no_grad():
            predictions = class_logits.argmax(dim=-1)
            exact = (predictions == class_targets) & class_valid
            active_valid = class_valid & class_targets.ne(0)
            active_correct = (
                predictions.ne(0) == class_targets.ne(0)
            ) & class_valid
            satisfaction_correct = (
                predictions == class_targets
            ) & active_valid
            valid_count = int(class_valid.sum().item())
            active_count = int(active_valid.sum().item())
            if valid_count:
                self._record_metric(
                    "hcr_tristate_loss",
                    tristate_loss * class_logits.shape[0],
                    class_logits.shape[0],
                )
                self._record_metric(
                    "hcr_tristate_accuracy", exact.float().sum(), valid_count
                )
                self._record_metric(
                    "hcr_activity_accuracy", active_correct.float().sum(), valid_count
                )
                if self.config.tristate_loss_mode == "conditional":
                    group_count = class_logits.shape[0] // self.base.group_size
                    self._record_metric(
                        "hcr_activity_loss", activity_loss * group_count, group_count
                    )
            if active_count:
                self._record_metric(
                    "hcr_satisfaction_accuracy",
                    satisfaction_correct.float().sum(),
                    active_count,
                )
                if self.config.tristate_loss_mode == "conditional":
                    self._record_metric(
                        "hcr_satisfaction_loss",
                        satisfaction_loss * class_logits.shape[0],
                        class_logits.shape[0],
                    )
            conjunction_prediction = predictions.ne(2).all(dim=1)
            self._record_metric(
                "hcr_conjunction_accuracy",
                (conjunction_prediction == labels.bool()).float().sum(),
                labels.numel(),
            )
            self._record_metric(
                "hcr_activity_group_consistency", activity_consistency, 1
            )
            self._record_metric("hcr_native_score", native_scores.sum(), scores.numel())
            self._record_metric(
                "hcr_native_point_loss",
                native_point_loss * native_scores.numel(),
                native_scores.numel(),
            )
            self._record_metric(
                "hcr_logical_score", logical_scores.sum(), scores.numel()
            )
            self._record_metric(
                "hcr_logical_score_weight",
                scores.new_tensor(self.current_logical_score_weight),
                1,
            )
            self._record_metric("hcr_score", scores.sum(), scores.numel())

        if self.canonical_validation:
            return rank_point
        auxiliary = (
            self.config.native_point_loss_weight * native_point_loss
            + self.config.tristate_loss_weight * tristate_loss
            + self.config.tristate_activity_consistency_weight
            * activity_consistency
        )
        return self._combine_primary_and_auxiliary(
            output,
            rank_point,
            self.config.field_loss_weight * auxiliary * scale,
        )


@TrainerRegistry.register(trainer_type="pas_rank_point_sft")
class PasRankPointTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        global _METRICS_PATH

        config = kwargs.get("config", args[0] if args else None)
        if config is None:
            raise ValueError("PAS trainer requires a Cosmos config")
        early_loss_config = LossConfig.model_validate(config.custom.get("loss", {}))
        use_query_gate = early_loss_config.retriever_residual_gate == "query_score"
        if use_query_gate and early_loss_config.retriever_residual_alpha is None:
            raise ValueError("query_score residual gate requires retriever_residual_alpha")
        if use_query_gate and not (
            early_loss_config.retriever_residual_alpha_minimum
            < float(early_loss_config.retriever_residual_alpha)
            < early_loss_config.retriever_residual_alpha_maximum
        ):
            raise ValueError("query_score residual alpha must lie strictly inside its bounds")

        # A loss-side module needs its own explicitly synchronized optimizer:
        # LLMTrainer has already built the policy optimizer by the time this
        # subclass creates RankPointLoss.  Keeping it outside the policy also
        # avoids relying on DDP semantics for a parameter used outside forward.
        super().__init__(*args, **kwargs)
        lora_path = self.config.policy.lora.lora_path
        if lora_path:
            load_lora_initialization(
                self.model,
                lora_path,
                allow_partial=bool(
                    self.config.custom.get(
                        "allow_partial_lora_initialization", False
                    )
                ),
            )
        if self.parallel_dims.pp_enabled:
            raise ValueError("PAS rank/point loss requires policy.parallelism.pp_size=1")
        if self.config.train.sequence_packing:
            raise ValueError("PAS rank/point loss requires train.sequence_packing=false")
        if self.parallel_dims.cp_enabled:
            raise ValueError("PAS rank/point loss requires policy.parallelism.cp_size=1")

        loss_config = LossConfig.model_validate(self.config.custom.get("loss", {}))
        response_spec = build_binary_response_spec(
            self.data_packer.tokenizer,
            positive_response=loss_config.positive_response,
            negative_response=loss_config.negative_response,
        )
        ordinal_token_ids = None
        ordinal_response_spec = None
        ordinal_aux_response_spec = None
        if loss_config.score_mode == "relevance_1to5_expected":
            ordinal_token_ids = []
            for value in "12345":
                ids = self.data_packer.tokenizer.encode(value, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Ordinal score {value!r} is not one token: {ids}")
                ordinal_token_ids.append(int(ids[0]))
            if len(set(ordinal_token_ids)) != 5:
                raise ValueError("Ordinal relevance tokens are not unique")
        elif loss_config.score_mode in {
            "relevance_1to5_supervised_expected",
            "relevance_1to5_supervised_strict_logodds",
        }:
            ordinal_response_spec = build_ordinal_response_spec(
                self.data_packer.tokenizer
            )
        elif loss_config.score_mode == "binary_delta_with_ordinal_aux":
            ordinal_aux_response_spec = build_ordinal_response_spec(
                self.data_packer.tokenizer
            )
        self.loss_fn = RankPointLoss(
            response_spec,
            rank_weight=loss_config.rank_weight,
            point_weight=loss_config.point_weight,
            rank_temperature=loss_config.rank_temperature,
            bag_positive_temperature=loss_config.bag_positive_temperature,
            bag_negative_temperature=loss_config.bag_negative_temperature,
            pu_class_prior=loss_config.pu_class_prior,
            pu_nn_weight=loss_config.pu_nn_weight,
            rank_mode=loss_config.rank_mode,
            partial_negative_ratio=loss_config.partial_negative_ratio,
            rank_margin=loss_config.rank_margin,
            natural20_rank_weight=loss_config.natural20_rank_weight,
            inverse_pair_weight=loss_config.inverse_pair_weight,
            weak_veto_temperature=loss_config.weak_veto_temperature,
            weak_veto_full_weight=loss_config.weak_veto_full_weight,
            weak_veto_mil_weight=loss_config.weak_veto_mil_weight,
            robust_pair_q=loss_config.robust_pair_q,
            all_pairs_aux_weight=loss_config.all_pairs_aux_weight,
            positive_tail_weight=loss_config.positive_tail_weight,
            positive_tail_temperature=loss_config.positive_tail_temperature,
            retriever_success_weight=loss_config.retriever_success_weight,
            retriever_error_weight=loss_config.retriever_error_weight,
            near_miss_k=loss_config.near_miss_k,
            near_miss_weight=loss_config.near_miss_weight,
            parent_preservation_margin=loss_config.parent_preservation_margin,
            parent_preservation_teacher_margin_slack=loss_config.parent_preservation_teacher_margin_slack,
            parent_preservation_weight=loss_config.parent_preservation_weight,
            topk_preservation_k=loss_config.topk_preservation_k,
            topk_preservation_margin=loss_config.topk_preservation_margin,
            topk_preservation_weight=loss_config.topk_preservation_weight,
            point_temperature=loss_config.point_temperature,
            point_mode=loss_config.point_mode,
            negative_point_weight=loss_config.negative_point_weight,
            teacher_weight=loss_config.teacher_weight,
            teacher_repair_group_weight=loss_config.teacher_repair_group_weight,
            teacher_temperature=loss_config.teacher_temperature,
            teacher_mode=loss_config.teacher_mode,
            group_size=loss_config.candidate_group_size,
            ordinal_token_ids=ordinal_token_ids,
            ordinal_response_spec=ordinal_response_spec,
            ordinal_aux_response_spec=ordinal_aux_response_spec,
            ordinal_ce_weight=loss_config.ordinal_ce_weight,
            ordinal_aux_position=loss_config.ordinal_aux_position,
            hidden_supcon_weight=loss_config.hidden_supcon_weight,
            hidden_supcon_temperature=loss_config.hidden_supcon_temperature,
            hidden_supcon_center=loss_config.hidden_supcon_center,
            same_label_consistency_weight=loss_config.same_label_consistency_weight,
            requirement_ordinal_weight=loss_config.requirement_ordinal_weight,
            requirement_ordinal_gap=loss_config.requirement_ordinal_gap,
            requirement_ordinal_temperature=loss_config.requirement_ordinal_temperature,
            ordinal_score_readout=(
                "strict_logodds"
                if loss_config.score_mode
                == "relevance_1to5_supervised_strict_logodds"
                else "expected"
            ),
            query_consistency_weight=loss_config.query_consistency_weight,
            query_cross_weight=loss_config.query_cross_weight,
            full_gallery_rank1_weight=loss_config.full_gallery_rank1_weight,
            full_gallery_active_topk=loss_config.full_gallery_active_topk,
            transition_pooled_weight=loss_config.transition_pooled_weight,
            transition_pooled_topk=loss_config.transition_pooled_topk,
            transition_pooled_temperature=loss_config.transition_pooled_temperature,
            transition_pooled_margin=loss_config.transition_pooled_margin,
            robust_cycle_rho=loss_config.robust_cycle_rho,
            robust_cycle_lambda=loss_config.robust_cycle_lambda_start,
            preservation_cycle_margin=loss_config.preservation_cycle_margin,
            preservation_cycle_weight=loss_config.preservation_cycle_weight,
            repair_cycle_ap_weight=loss_config.repair_cycle_ap_weight,
            preserve_cycle_ap_weight=loss_config.preserve_cycle_ap_weight,
            preserve_cycle_robust_scale=loss_config.preserve_cycle_robust_scale,
            r69_smoothap_temperature=loss_config.r69_smoothap_temperature,
            r69_soft_r1_temperature=loss_config.r69_soft_r1_temperature,
            r69_smoothap_weight=loss_config.r69_smoothap_weight,
            r69_soft_r1_weight=loss_config.r69_soft_r1_weight,
            policy_samples=loss_config.policy_samples,
            policy_exact_max_k=loss_config.policy_exact_max_k,
            policy_ap_reward_weight=loss_config.policy_ap_reward_weight,
            policy_r1_reward_weight=loss_config.policy_r1_reward_weight,
            policy_preserve_anchor_weight=loss_config.policy_preserve_anchor_weight,
            policy_preserve_anchor_margin=loss_config.policy_preserve_anchor_margin,
            binary_decision_position=loss_config.binary_decision_position,
            binary_readout=loss_config.binary_readout,
            retriever_residual_alpha=loss_config.retriever_residual_alpha,
            retriever_residual_epsilon=loss_config.retriever_residual_epsilon,
        )
        if use_query_gate:
            if (
                self.parallel_dims.tp != 1
                or self.parallel_dims.cp != 1
                or self.parallel_dims.pp != 1
                or self.parallel_dims.dp_shard != 1
            ):
                raise ValueError(
                    "query_score residual gate initially supports pure DP replication only"
                )
            if self.config.train.resume:
                raise ValueError(
                    "query_score residual gate resume is disabled until composite "
                    "optimizer restore is exercised"
                )
            gate = QueryAdaptiveResidualGate(
                initial_alpha=float(loss_config.retriever_residual_alpha),
                minimum=loss_config.retriever_residual_alpha_minimum,
                maximum=loss_config.retriever_residual_alpha_maximum,
                feature_version=loss_config.retriever_residual_gate_feature_version,
            ).to(device=self.device, dtype=torch.float32)
            self.loss_fn.retriever_residual_gate = gate
            self.loss_fn.retriever_residual_gate_log_alpha_l2 = (
                loss_config.retriever_residual_gate_log_alpha_l2
            )
            self._residual_gate_optimizer = torch.optim.AdamW(
                gate.parameters(),
                lr=loss_config.retriever_residual_gate_lr,
                betas=(0.9, 0.99),
                weight_decay=0.0,
            )
            if self.parallel_dims.dp_replicate_enabled:
                for parameter in gate.parameters():
                    dist.broadcast(
                        parameter.data,
                        group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                        group_src=0,
                    )
            logger.info(
                "PAS residual fusion uses a jointly trained query-score gate: "
                "initial_alpha=%s bounds=[%s,%s] lr=%s log_alpha_l2=%s features=%s",
                loss_config.retriever_residual_alpha,
                loss_config.retriever_residual_alpha_minimum,
                loss_config.retriever_residual_alpha_maximum,
                loss_config.retriever_residual_gate_lr,
                loss_config.retriever_residual_gate_log_alpha_l2,
                gate.feature_names,
            )
        else:
            self._residual_gate_optimizer = None
        self._teacher_weight_initial = float(loss_config.teacher_weight)
        self._teacher_weight_final = float(
            loss_config.teacher_weight
            if loss_config.teacher_weight_final is None
            else loss_config.teacher_weight_final
        )
        self._teacher_decay_steps = loss_config.teacher_decay_steps
        self._candidate_group_size = int(loss_config.candidate_group_size)
        self._robust_cycle_lambda_start = float(
            loss_config.robust_cycle_lambda_start
        )
        self._robust_cycle_lambda_final = float(
            loss_config.robust_cycle_lambda_final
        )
        self._robust_cycle_lambda_ramp_steps = int(
            loss_config.robust_cycle_lambda_ramp_steps
        )
        self._runtime_group_audited = False
        # Validation is driven one batch at a time by Cosmos-RL's worker. Keep
        # the metric accumulator alive across those batches, then publish its
        # means with the first following training report. The worker already
        # publishes the canonical aggregate objective as ``val/avg_loss``.
        self._validation_metric_step: int | None = None
        if (
            loss_config.hidden_supcon_weight
            and not self.config.policy.enable_liger_fused_cross_entropy
        ):
            raise ValueError(
                "hidden_supcon_weight requires "
                "policy.enable_liger_fused_cross_entropy=true so the loss receives "
                "final hidden states"
            )
        if self.config.policy.enable_liger_fused_cross_entropy:
            hf_model = getattr(self.model, "model", None)
            if hf_model is None or not hasattr(hf_model, "lm_head"):
                raise ValueError("Could not locate the HF LM head for binary projection")
            original_lm_head = hf_model.lm_head
            if not getattr(original_lm_head, "_pas_hidden_state_forward", False):
                original_lm_head.forward = types.MethodType(
                    _pas_hidden_state_lm_head_forward, original_lm_head
                )
                original_lm_head._pas_hidden_state_forward = True
            self.loss_fn.projection_weight = self.model.lm_head.weight
            if hasattr(self.model.lm_head, "lora_A"):
                self.loss_fn.projection_module = self.model.lm_head
                logger.info(
                    "PAS selected-row binary projection includes trainable "
                    "LM-head LoRA"
                )
            logger.info(
                "PAS reranker uses hidden-state binary projection; full-vocabulary "
                "LM-head logits are skipped"
            )
        if loss_config.binary_readout == "bf16_ste":
            logger.info(
                "PAS binary readout uses deployment-matched per-token BF16 "
                "quantization with straight-through gradients"
            )
        atomic_and_raw = self.config.custom.get("atomic_and")
        if atomic_and_raw is not None:
            atomic_and_config = AtomicAndConfig.model_validate(atomic_and_raw)
            self.loss_fn = RankPointWithAtomicAndLoss(
                self.loss_fn, atomic_and_config
            )
            logger.info(
                "PAS end-to-end atomic AND enabled: candidates=%s fields=%s "
                "aggregation=%s temperature=%s normalization=%s "
                "field_point_weight=%s; score uses CR3 atomic margins only",
                atomic_and_config.num_candidates,
                atomic_and_config.num_fields,
                atomic_and_config.aggregation,
                atomic_and_config.temperature,
                atomic_and_config.normalization,
                atomic_and_config.field_point_weight,
            )
        safe_update_raw = self.config.custom.get("safe_update")
        self._safe_update_config = (
            None
            if safe_update_raw is None
            else SafeUpdateConfig.model_validate(safe_update_raw)
        )
        if self._safe_update_config is not None:
            self.loss_fn.configure_safe_update(
                num_strata=self._safe_update_config.num_strata,
                cushion=self._safe_update_config.cushion,
            )
            logger.info(
                "PAS A-GEM safe update enabled: hard parent decisions only; "
                "strata=%s cushion=%s projection=exact_active_set_qp "
                "legacy_projection_sweeps=%s",
                self._safe_update_config.num_strata,
                self._safe_update_config.cushion,
                self._safe_update_config.projection_sweeps,
            )
        logger.info(
            "PAS reranker loss: rank_mode=%s rank_weight=%s point_BCE_weight=%s "
            "negative_point_weight=%s teacher_weight=%s teacher_weight_final=%s "
            "teacher_decay_steps=%s teacher_mode=%s teacher_repair_group_weight=%s "
            "point_mode=%s tau_rank=%s partial_negative_ratio=%s "
            "rank_margin=%s robust_pair_q=%s all_pairs_aux_weight=%s retriever_success_weight=%s "
            "retriever_error_weight=%s "
            "near_miss_k=%s near_miss_weight=%s "
            "parent_preservation_margin=%s "
            "parent_preservation_teacher_margin_slack=%s "
            "parent_preservation_weight=%s "
            "topk_preservation_k=%s topk_preservation_margin=%s "
            "topk_preservation_weight=%s tau_point=%s "
            "tau_teacher=%s group_size=%s query_cross_weight=%s "
            "transition_pooled_weight=%s transition_pooled_topk=%s "
            "transition_pooled_temperature=%s transition_pooled_margin=%s "
            "robust_cycle_rho=%s robust_cycle_lambda_start=%s "
            "robust_cycle_lambda_final=%s robust_cycle_lambda_ramp_steps=%s "
            "r69_smoothap_temperature=%s r69_soft_r1_temperature=%s "
            "r69_smoothap_weight=%s r69_soft_r1_weight=%s "
            "hidden_supcon_weight=%s hidden_supcon_temperature=%s "
            "hidden_supcon_center=%s same_label_consistency_weight=%s "
            "requirement_ordinal_weight=%s requirement_ordinal_gap=%s "
            "requirement_ordinal_temperature=%s",
            loss_config.rank_mode,
            loss_config.rank_weight,
            loss_config.point_weight,
            loss_config.negative_point_weight,
            loss_config.teacher_weight,
            self._teacher_weight_final,
            self._teacher_decay_steps,
            loss_config.teacher_mode,
            loss_config.teacher_repair_group_weight,
            loss_config.point_mode,
            loss_config.rank_temperature,
            loss_config.partial_negative_ratio,
            loss_config.rank_margin,
            loss_config.robust_pair_q,
            loss_config.all_pairs_aux_weight,
            loss_config.retriever_success_weight,
            loss_config.retriever_error_weight,
            loss_config.near_miss_k,
            loss_config.near_miss_weight,
            loss_config.parent_preservation_margin,
            loss_config.parent_preservation_teacher_margin_slack,
            loss_config.parent_preservation_weight,
            loss_config.topk_preservation_k,
            loss_config.topk_preservation_margin,
            loss_config.topk_preservation_weight,
            loss_config.point_temperature,
            loss_config.teacher_temperature,
            loss_config.candidate_group_size,
            loss_config.query_cross_weight,
            loss_config.transition_pooled_weight,
            loss_config.transition_pooled_topk,
            loss_config.transition_pooled_temperature,
            loss_config.transition_pooled_margin,
            loss_config.robust_cycle_rho,
            loss_config.robust_cycle_lambda_start,
            loss_config.robust_cycle_lambda_final,
            loss_config.robust_cycle_lambda_ramp_steps,
            loss_config.r69_smoothap_temperature,
            loss_config.r69_soft_r1_temperature,
            loss_config.r69_smoothap_weight,
            loss_config.r69_soft_r1_weight,
            loss_config.hidden_supcon_weight,
            loss_config.hidden_supcon_temperature,
            loss_config.hidden_supcon_center,
            loss_config.same_label_consistency_weight,
            loss_config.requirement_ordinal_weight,
            loss_config.requirement_ordinal_gap,
            loss_config.requirement_ordinal_temperature,
        )

        replay_config_raw = self.config.custom.get("attribute_replay")
        if replay_config_raw is not None:
            if self.config.custom.get("visual_cache") is None:
                raise ValueError("Attribute replay currently requires a visual prefix cache")
            replay_config = AttributeReplayConfig.model_validate(replay_config_raw)
            assets = load_attribute_replay_assets(
                replay_config.records_path,
                replay_config.attribute_vocab_path,
                replay_config.attribute_heads_path,
            ).to(self.device)
            hf_model = getattr(self.model, "model", None)
            visual_model = getattr(hf_model, "model", None)
            if visual_model is None or not hasattr(visual_model, "visual"):
                raise ValueError("Could not locate Qwen3-VL visual model for attribute replay")
            if assets.head_weights[0].shape[1] != int(
                visual_model.visual.config.out_hidden_size
            ):
                raise ValueError(
                    "Attribute heads do not match the selected visual representation width"
                )
            self.loss_fn = RankPointWithAttributeReplayLoss(
                self.loss_fn,
                visual_model,
                assets,
                replay_config,
            )
            logger.info(
                "PAS attribute replay enabled: fields=%s labelled_images=%s "
                "sample_fraction=%s loss_weight=%s label_smoothing=%s",
                len(assets.fields),
                len(assets.labels_by_image),
                replay_config.sample_fraction,
                replay_config.loss_weight,
                replay_config.label_smoothing,
            )

        structured_config_raw = self.config.custom.get("structured_attribute")
        self._structured_attribute_audit = bool(
            self.config.custom.get("structured_attribute_audit_predictions", False)
        )
        self._structured_attribute_audit_path: Path | None = None
        if structured_config_raw is not None:
            if not self.config.policy.enable_liger_fused_cross_entropy:
                raise ValueError(
                    "Structured attributes require hidden-state fused-CE mode"
                )
            structured_config = StructuredAttributeConfig.model_validate(
                structured_config_raw
            )
            assets = load_structured_attribute_assets(
                structured_config.train_records_path,
                structured_config.attribute_vocab_path,
                structured_config.query_records_path,
            )
            choice_loss = StructuredAttributeChoiceLoss(
                self.data_packer.tokenizer,
                self.model.lm_head.weight,
                assets.fields,
                hard_example_weight=structured_config.online_hard_example_weight,
                hard_example_fields=structured_config.online_hard_example_fields,
            )
            self.loss_fn = RankPointWithStructuredAttributeLoss(
                self.loss_fn,
                choice_loss,
                structured_config,
            )
            logger.info(
                "PAS masked structured attributes enabled: fields=%s "
                "train_labelled_images=%s loss_weight=%s online_hard_weight=%s "
                "online_hard_fields=%s seed=%s; template "
                "tokens are excluded from the objective",
                len(assets.fields),
                len(assets.labels_by_image),
                structured_config.loss_weight,
                structured_config.online_hard_example_weight,
                structured_config.online_hard_example_fields,
                structured_config.seed,
            )
            if self._structured_attribute_audit:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                self._structured_attribute_audit_path = (
                    Path(self.config.train.output_dir)
                    / f"structured_attribute_audit_rank{rank:02d}.jsonl"
                )
                self._structured_attribute_audit_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                self._structured_attribute_audit_path.unlink(missing_ok=True)

        hcr_config_raw = self.config.custom.get("hcr")
        if hcr_config_raw is not None:
            if not self.config.policy.enable_liger_fused_cross_entropy:
                raise ValueError("HCR requires hidden-state fused-CE mode")
            hcr_config = HCRConfig.model_validate(hcr_config_raw)
            if hcr_config.auxiliary_gradient_mode != "none":
                projection_weight = self.loss_fn.projection_weight
                if projection_weight is None:
                    raise ValueError(
                        "HCR gradient surgery requires the explicit frozen LM-head projection"
                    )
                if projection_weight.requires_grad:
                    raise ValueError(
                        "HCR gradient surgery is LoRA-only; the LM head must remain frozen"
                    )
            self.loss_fn = RankPointWithHCRLoss(
                self.loss_fn,
                self.data_packer.tokenizer,
                hcr_config,
                requirement_balance_group=(
                    self.parallel_dims.mesh["dp"].get_group()
                    if self.parallel_dims.dp_replicate_enabled
                    or self.parallel_dims.dp_shard_enabled
                    else None
                ),
            )
            logger.info(
                "PAS HCR enabled: mode=%s deployed_score=%s field_loss_weight=%s "
                "constraint_temperature=%s bottom_k=%s native_weight=%s "
                "logical_weight=%s logical_weight_start=%s logical_warmup=%s "
                "logical_ramp=%s native_point_loss_weight=%s "
                "logical_reduction=%s auxiliary_gradient_mode=%s; auxiliary "
                "targets are train-only hard labels",
                hcr_config.mode,
                {
                    "legacy_compatibility": hcr_config.aggregation,
                    "active_logical": "native_plus_active_logical",
                    "tristate_logical": "native_plus_tristate_logical",
                    "predecision_tristate": "late_binary_after_latent_checklist",
                    "postdecision_tristate": "plain_binary_then_attribute_auxiliary",
                    "decision_state_tristate": "plain_binary_same_state_attribute_auxiliary",
                }[hcr_config.mode],
                hcr_config.field_loss_weight,
                hcr_config.constraint_temperature,
                hcr_config.bottom_k,
                hcr_config.native_score_weight,
                hcr_config.logical_score_weight,
                hcr_config.logical_score_weight_start,
                hcr_config.logical_score_warmup_steps,
                hcr_config.logical_score_ramp_steps,
                hcr_config.native_point_loss_weight,
                hcr_config.logical_reduction,
                hcr_config.auxiliary_gradient_mode,
            )

        visual_distillation_raw = self.config.custom.get("visual_distillation")
        if visual_distillation_raw is not None:
            if self.config.custom.get("visual_cache") is None:
                raise ValueError(
                    "Visual distillation currently requires a visual prefix cache"
                )
            visual_distillation_config = VisualDistillationConfig.model_validate(
                visual_distillation_raw
            )
            visual_distillation_assets = load_visual_distillation_assets(
                visual_distillation_config.embedding_metadata_path,
                visual_distillation_config.image_index_path,
            )
            hf_model = getattr(self.model, "model", None)
            visual_model = getattr(hf_model, "model", None)
            if visual_model is None or not hasattr(visual_model, "visual"):
                raise ValueError(
                    "Could not locate Qwen3-VL visual model for visual distillation"
                )
            self.loss_fn = RankPointWithVisualDistillationLoss(
                self.loss_fn,
                visual_model,
                visual_distillation_assets,
                visual_distillation_config,
                group_size=self._candidate_group_size,
                device=self.device,
            )
            logger.info(
                "PAS relational visual distillation enabled: teacher_images=%s "
                "teacher_width=%s group_size=%s loss_weight=%s embeddings=%s",
                len(visual_distillation_assets.index_by_image_key),
                visual_distillation_assets.embeddings.shape[1],
                self._candidate_group_size,
                visual_distillation_config.loss_weight,
                visual_distillation_assets.embedding_path,
            )

        # `self.config` is the authoritative worker config. Constructing Config
        # in the entry script mints another timestamp and separates metrics from
        # the actual checkpoint directory.
        _METRICS_PATH = Path(self.config.train.output_dir) / "loss_metrics.jsonl"
        if _is_policy_master() and not self.config.train.resume:
            _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _METRICS_PATH.unlink(missing_ok=True)
        self._validation_score_audit_path: Path | None = None
        self._validation_score_audit_final_path: Path | None = None
        self._validation_score_audit_active_step: int | None = None
        val_dataset_config = self.config.custom.get("val_dataset") or {}
        if bool(val_dataset_config.get("score_audit_metadata", False)):
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            self._validation_score_audit_path = (
                Path(self.config.train.output_dir)
                / f"validation_score_audit_rank{rank:02d}.jsonl"
            )
            self._validation_score_audit_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.config.train.resume:
                self._validation_score_audit_path.unlink(missing_ok=True)
                for stale_path in self._validation_score_audit_path.parent.glob(
                    f"validation_score_audit_step*_rank{rank:02d}.jsonl*"
                ):
                    stale_path.unlink(missing_ok=True)
        init_offline_wandb(self.config)

    def _assert_runtime_candidate_groups(self, batch, *, stage: str) -> None:
        """Fail if packing or distributed dispatch broke query-group order."""

        if len(batch) % self._candidate_group_size:
            raise ValueError(
                f"PAS {stage} runtime batch has {len(batch)} candidates, not a "
                f"multiple of K={self._candidate_group_size}"
            )
        for begin in range(0, len(batch), self._candidate_group_size):
            rows = batch[begin : begin + self._candidate_group_size]
            group_ids = {row.get("_pas_group_id") for row in rows}
            labels = [row.get("_pas_binary_label") for row in rows]
            if None in group_ids or len(group_ids) != 1:
                raise ValueError(
                    f"PAS {stage} runtime group {begin}:{begin + self._candidate_group_size} "
                    f"was reordered across queries: {sorted(map(str, group_ids))}"
                )
            if any(label not in {0, 1} for label in labels) or not 0 < sum(labels) < len(labels):
                raise ValueError(
                    f"PAS {stage} runtime group {next(iter(group_ids))!r} has invalid "
                    f"binary labels after dispatch: {labels}"
                )
            if (
                getattr(self.loss_fn, "rank_mode", None)
                == "paired_view_robust_normalized_lambdaap_soft_r1"
            ):
                roles = [row.get("_pas_full_list_view_role") for row in rows]
                candidate_ids = [row.get("_pas_view_candidate_id") for row in rows]
                queries = [
                    " ".join(str(row.get("_pas_query_text") or "").casefold().split())
                    for row in rows
                ]
                ranks = [row.get("_pas_retriever_rank") for row in rows]
                if (
                    len(rows) != 40
                    or roles != ["canonical"] * 20 + ["hflip"] * 20
                    or labels[:20] != labels[20:]
                    or candidate_ids[:20] != candidate_ids[20:]
                    or any(not candidate_id for candidate_id in candidate_ids)
                    or len(set(queries)) != 1
                    or not queries[0]
                    or ranks[:20] != list(range(1, 21))
                    or ranks[20:] != list(range(1, 21))
                ):
                    raise ValueError(
                        f"PAS {stage} paired-view group has broken canonical/hflip K40 layout"
                    )
            if getattr(self.loss_fn, "rank_mode", None) == "natural20_inverse_pair":
                offsets = {
                    row.get("_pas_inverse_pair_original_offset") for row in rows
                }
                if len(rows) != 21 or len(offsets) != 1 or None in offsets:
                    raise ValueError(
                        f"PAS {stage} natural20-inverse group has invalid offset metadata"
                    )
                original_offset = int(next(iter(offsets)))
                queries = [
                    " ".join(
                        str(row.get("_pas_query_text") or "").casefold().split()
                    )
                    for row in rows
                ]
                ranks = [row.get("_pas_retriever_rank") for row in rows]
                if (
                    original_offset < 0
                    or original_offset >= 20
                    or labels[original_offset] != 0
                    or labels[20] != 1
                    or len(set(queries[:20])) != 1
                    or queries[20] == queries[0]
                    or ranks != list(range(1, 22))
                ):
                    raise ValueError(
                        f"PAS {stage} natural20-inverse group has broken K21 layout"
                    )
            if getattr(self.loss_fn, "rank_mode", None) == "weak_veto_mil":
                expected_labels = [1, 0] * (len(rows) // 2)
                expected_roles = ["full_positive", "full_negative"] + [
                    role
                    for _ in range(len(rows) // 2 - 1)
                    for role in ("field_positive", "field_negative")
                ]
                roles = [row.get("_pas_weak_veto_role") for row in rows]
                fields = [row.get("_pas_weak_veto_field") for row in rows]
                queries = [
                    " ".join(
                        str(row.get("_pas_query_text") or "").casefold().split()
                    )
                    for row in rows
                ]
                if labels != expected_labels or roles != expected_roles:
                    raise ValueError(
                        f"PAS {stage} weak-veto group has invalid pair roles"
                    )
                if fields[:2] != ["full", "full"] or queries[0] != queries[1]:
                    raise ValueError(
                        f"PAS {stage} weak-veto group has invalid full-query pair"
                    )
                atomic_fields = []
                for offset in range(2, len(rows), 2):
                    if (
                        not fields[offset]
                        or fields[offset] != fields[offset + 1]
                        or queries[offset] != queries[offset + 1]
                    ):
                        raise ValueError(
                            f"PAS {stage} weak-veto group has malformed field pair"
                        )
                    atomic_fields.append(str(fields[offset]))
                if len(set(atomic_fields)) != len(atomic_fields):
                    raise ValueError(
                        f"PAS {stage} weak-veto group repeats a field hypothesis"
                    )
            if getattr(self.loss_fn, "rank_mode", None) in {
                "plackett_luce_policy",
                "rb_plackett_luce_expected_ap",
            }:
                preserve = {row.get("_pas_policy_preserve") for row in rows}
                if preserve not in ({True}, {False}):
                    raise ValueError(
                        f"PAS {stage} PL group {next(iter(group_ids))!r} "
                        f"lost its hard preserve flag: {preserve}"
                    )
            if (
                getattr(self.loss_fn, "rank_mode", None)
                in {
                    "dynamic_top20_smoothap_soft_r1",
                    "incumbent_guarded_smoothap_soft_r1",
                    "incumbent_safe_lambda_ap",
                }
            ):
                query_identities = {row.get("_pas_query_identity") for row in rows}
                normalized_queries = {
                    " ".join(str(row.get("_pas_query_text") or "").casefold().split())
                    for row in rows
                }
                retriever_ranks = [row.get("_pas_retriever_rank") for row in rows]
                if (
                    len(query_identities) != 1
                    or None in query_identities
                    or "" in query_identities
                    or len(normalized_queries) != 1
                    or "" in normalized_queries
                ):
                    raise ValueError(
                        f"PAS {stage} r69 group {next(iter(group_ids))!r} "
                        "mixes query identities or texts"
                    )
                rank_mode = getattr(self.loss_fn, "rank_mode", None)
                valid_order = (
                    sorted(retriever_ranks) == list(range(1, 21))
                    if rank_mode
                    in {
                        "incumbent_guarded_smoothap_soft_r1",
                        "incumbent_safe_lambda_ap",
                    }
                    else retriever_ranks == list(range(1, 21))
                )
                if not valid_order:
                    order_failure = (
                        "invalid candidate permutation"
                        if rank_mode
                        in {
                            "incumbent_guarded_smoothap_soft_r1",
                            "incumbent_safe_lambda_ap",
                        }
                        else "lost retriever order"
                    )
                    raise ValueError(
                        f"PAS {stage} r69 group {next(iter(group_ids))!r} "
                        f"{order_failure}: {retriever_ranks}"
                    )
                if any(row.get("_pas_teacher_probability") is not None for row in rows):
                    raise ValueError(
                        f"PAS {stage} r69 group {next(iter(group_ids))!r} "
                        "contains a teacher target"
                    )
            if getattr(self.loss_fn, "rank_mode", None) in {
                "orthogonal_cycle",
                "deployment_weighted_robust_cycle",
            }:
                roles = [row.get("_pas_interaction_cycle_role") for row in rows]
                expected_roles = [
                    "diagonal_q_p",
                    "cross_q_n",
                    "diagonal_qn_n",
                    "cross_qn_p",
                ]
                if roles != expected_roles:
                    raise ValueError(
                        f"PAS {stage} orthogonal-cycle group "
                        f"{next(iter(group_ids))!r} has invalid role order: {roles}"
                    )
            if (
                getattr(self.loss_fn, "rank_mode", None)
                == "deployment_weighted_robust_cycle"
            ):
                weights = [row.get("_pas_cycle_ap_weight") for row in rows]
                if any(
                    value is None
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    for value in weights
                ) or len({float(value) for value in weights}) != 1:
                    raise ValueError(
                        f"PAS {stage} deployment-weighted checkerboard group "
                        f"{next(iter(group_ids))!r} has invalid AP weights: {weights}"
                    )
            if (
                getattr(self.loss_fn, "rank_mode", None)
                == "preservation_constrained_cycle_block"
            ):
                roles = [row.get("_pas_interaction_cycle_role") for row in rows]
                expected_roles = [
                    "diagonal_q_p",
                    "cross_q_n",
                    "diagonal_qn_n",
                    "cross_qn_p",
                ] * 8
                if roles != expected_roles:
                    raise ValueError(
                        f"PAS {stage} r68 block has invalid checkerboard roles"
                    )
                block_roles = {row.get("_pas_r68_block_role") for row in rows}
                if len(block_roles) != 1 or next(iter(block_roles)) not in {
                    "repair",
                    "vulnerable",
                    "diversity",
                }:
                    raise ValueError(
                        f"PAS {stage} r68 block has invalid stratum: {block_roles}"
                    )
                cycle_indices = [row.get("_pas_r68_cycle_index") for row in rows]
                if cycle_indices != [index for index in range(8) for _ in range(4)]:
                    raise ValueError(
                        f"PAS {stage} r68 block has invalid cycle indices"
                    )
        if not self._runtime_group_audited:
            logger.info(
                "PAS runtime group audit passed after packing/dispatch: stage=%s "
                "candidates=%s K=%s groups=%s",
                stage,
                len(batch),
                self._candidate_group_size,
                len(batch) // self._candidate_group_size,
            )
            self._runtime_group_audited = True

    def _queue_full_gallery_total_relevant(self, batch, *, stage: str) -> None:
        """Queue one exact gallery GT denominator for each runtime group."""

        if self.loss_fn.rank_mode != "full_gallery_lambda_ap":
            return
        totals: list[int] = []
        for begin in range(0, len(batch), self._candidate_group_size):
            rows = batch[begin : begin + self._candidate_group_size]
            values = [row.get("_pas_total_relevant") for row in rows]
            if any(value is None for value in values):
                raise ValueError(
                    f"PAS {stage} full-gallery AP group at offset {begin} is "
                    "missing its exact total-relevant denominator"
                )
            integer_values = [int(value) for value in values]
            if len(set(integer_values)) != 1 or integer_values[0] <= 0:
                raise ValueError(
                    f"PAS {stage} full-gallery AP group at offset {begin} has "
                    f"inconsistent denominators: {integer_values}"
                )
            totals.append(integer_values[0])
        self.loss_fn.set_total_relevant(totals)

    def _queue_inverse_pair_offsets(self, batch, *, stage: str) -> None:
        """Queue one referenced natural-negative slot per K21 group."""

        if self.loss_fn.rank_mode != "natural20_inverse_pair":
            return
        offsets: list[int] = []
        for begin in range(0, len(batch), self._candidate_group_size):
            rows = batch[begin : begin + self._candidate_group_size]
            values = {
                row.get("_pas_inverse_pair_original_offset") for row in rows
            }
            if len(values) != 1 or None in values:
                raise ValueError(
                    f"PAS {stage} inverse-pair group at offset {begin} "
                    "has missing or inconsistent original offsets"
                )
            offsets.append(int(next(iter(values))))
        self.loss_fn.set_inverse_pair_offsets(offsets)

    def _queue_candidate_loss_weights(self, batch, *, stage: str) -> None:
        """Queue train-only confidence weights without altering validation."""

        if stage != "train":
            return
        raw = [sample.get("_pas_rank_loss_weight") for sample in batch]
        if all(value is None for value in raw):
            return
        if any(value is None for value in raw):
            raise ValueError(
                "PAS candidate loss weights must be present on every row or none"
            )
        self.loss_fn.set_candidate_loss_weights([float(value) for value in raw])

    def _queue_same_label_consistency(self, batch, *, stage: str) -> None:
        """Queue raw/edit equality pairs; canonical validation never consumes them."""

        if stage != "train" or not self.loss_fn.same_label_consistency_weight:
            return
        offsets: list[int] = []
        for begin in range(0, len(batch), self._candidate_group_size):
            rows = batch[begin : begin + self._candidate_group_size]
            marked = [
                index
                for index, row in enumerate(rows)
                if row.get("_pas_same_label_consistency_role") is not None
            ]
            if not marked:
                offsets.append(-1)
                continue
            if marked != [0, 1]:
                raise ValueError(
                    "PAS same-label consistency pair must occupy K20 slots 0 and 1"
                )
            roles = [rows[index]["_pas_same_label_consistency_role"] for index in marked]
            if roles not in (["raw", "augmented"], ["reference", "paraphrase"]):
                raise ValueError(
                    "PAS consistency roles must be raw/augmented or "
                    f"reference/paraphrase, got {roles}"
                )
            labels = [rows[index].get("_pas_binary_label") for index in marked]
            if labels[0] != labels[1]:
                raise ValueError(
                    f"PAS consistency pair must have the same hard label, got {labels}"
                )
            offsets.append(0)
        self.loss_fn.set_same_label_consistency_offsets(offsets)

    def _queue_requirement_ordinal(self, batch, *, stage: str) -> None:
        """Queue hard atomic counts only for training; validation stays canonical."""

        if stage != "train" or not self.loss_fn.requirement_ordinal_weight:
            return
        values = [row.get("_pas_requirement_coverage") for row in batch]
        if any(value is None for value in values):
            raise ValueError(
                "PAS requirement ordinal training requires coverage metadata on every row"
            )
        self.loss_fn.set_requirement_ordinal_records(
            [(int(value[0]), int(value[1]), float(value[2])) for value in values]
        )

    def _queue_cycle_ap_weights(self, batch, *, stage: str) -> None:
        """Queue one hard-label AP weight per four-row checkerboard."""

        if self.loss_fn.rank_mode != "deployment_weighted_robust_cycle":
            return
        weights: list[float] = []
        for begin in range(0, len(batch), self._candidate_group_size):
            rows = batch[begin : begin + self._candidate_group_size]
            values = [row.get("_pas_cycle_ap_weight") for row in rows]
            if any(value is None for value in values):
                raise ValueError(
                    f"PAS {stage} robust-cycle group at offset {begin} is "
                    "missing its hard-label AP weight"
                )
            numeric = [float(value) for value in values]
            if (
                len(set(numeric)) != 1
                or not math.isfinite(numeric[0])
                or numeric[0] <= 0
            ):
                raise ValueError(
                    f"PAS {stage} robust-cycle group at offset {begin} has "
                    f"inconsistent AP weights: {numeric}"
                )
            weights.append(numeric[0])
        self.loss_fn.set_cycle_weights(weights)

    def _queue_r68_cycle_block_metadata(self, batch, *, stage: str) -> None:
        """Queue one role/mask plus eight hard-label AP weights per block."""

        if self.loss_fn.rank_mode != "preservation_constrained_cycle_block":
            return
        role_codes = {"repair": 0.0, "vulnerable": 1.0, "diversity": 2.0}
        metadata: list[float] = []
        for begin in range(0, len(batch), self._candidate_group_size):
            rows = batch[begin : begin + self._candidate_group_size]
            role_values = {row.get("_pas_r68_block_role") for row in rows}
            if len(role_values) != 1 or next(iter(role_values)) not in role_codes:
                raise ValueError(f"PAS {stage} r68 block has invalid role metadata")
            role = str(next(iter(role_values)))
            support_bits = 0
            weights: list[float] = []
            for cycle_index in range(8):
                cycle = rows[4 * cycle_index : 4 * cycle_index + 4]
                supports = {row.get("_pas_r68_support_cycle") for row in cycle}
                cycle_weights = {row.get("_pas_r68_cycle_weight") for row in cycle}
                if len(supports) != 1 or len(cycle_weights) != 1:
                    raise ValueError(
                        f"PAS {stage} r68 cycle metadata are inconsistent"
                    )
                if bool(next(iter(supports))):
                    support_bits |= 1 << cycle_index
                weight = float(next(iter(cycle_weights)))
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"PAS {stage} r68 AP weight is invalid")
                weights.append(weight)
            if role != "repair" and not support_bits:
                raise ValueError(f"PAS {stage} correct r68 block has no support edge")
            if abs(sum(weights) - 1.0) > 1e-5:
                raise ValueError(f"PAS {stage} r68 AP weights do not sum to one")
            metadata.extend((role_codes[role], float(support_bits), *weights))
        self.loss_fn.set_cycle_block_metadata(metadata)

    def _queue_safe_update_metadata(self, batch, *, stage: str) -> None:
        """Queue immutable parent decision boundaries for r268 A-GEM."""

        if self._safe_update_config is None or stage != "train":
            return
        if self.loss_fn.rank_mode in {
            "asymmetric_safe_residual_lambda_ap",
            "misordered_lambda_ap",
        }:
            cells = tuple(
                f"{dataset}|{difficulty}"
                for dataset in ("CUHK_PEDES", "PA-100K", "RSTPReid")
                for difficulty in ("easy", "hard", "medium")
            )
            cell_index = {cell: index for index, cell in enumerate(cells)}
            records = []
            for sample in batch:
                cell = (
                    f'{sample.get("_pas_dataset")}|'
                    f'{sample.get("_pas_query_type")}'
                )
                if cell not in cell_index:
                    raise ValueError(f"Unknown cell-safe residual stratum {cell!r}")
                # Only the final integer is consumed by the cell-safe loss;
                # the legacy tuple shape keeps the proven distributed
                # component-backward plumbing unchanged.
                records.append((0.0, False, False, cell_index[cell]))
            self.loss_fn.set_safe_update_records(records)
            return
        records: list[tuple[float, bool, bool, int]] = []
        for sample in batch:
            values = (
                sample.get("_pas_parent_margin"),
                sample.get("_pas_parent_anchor"),
                sample.get("_pas_parent_top1_correct"),
                sample.get("_pas_parent_margin_stratum"),
            )
            if any(value is None for value in values):
                raise ValueError("Safe-update training row lacks parent boundary metadata")
            records.append(
                (float(values[0]), bool(values[1]), bool(values[2]), int(values[3]))
            )
        self.loss_fn.set_safe_update_records(records)

    def _queue_retriever_residual_scores(self, batch, *, stage: str) -> None:
        if getattr(self.loss_fn, "retriever_residual_alpha", None) is None:
            return
        scores = [sample.get("_pas_retriever_score") for sample in batch]
        if any(value is None for value in scores):
            raise ValueError(
                f"PAS {stage} residual-fusion batch is missing retriever_score"
            )
        self.loss_fn.set_retriever_residual_scores(
            [float(value) for value in scores]
        )

    def step_training(self, *args, **kwargs):
        if bool(self.config.custom.get("validation_only", False)):
            raise RuntimeError(
                "validation_only PAS rank/point configuration refused a training step"
            )
        global_batch = args[0] if args else kwargs.get("global_batch")
        if global_batch is None:
            raise ValueError("PAS training batch is unavailable")
        if self._residual_gate_optimizer is not None:
            self._residual_gate_optimizer.zero_grad(set_to_none=True)
        train_step = int(kwargs.get("train_step", 0))
        total_steps = int(kwargs.get("total_steps", 1))
        decay_steps = self._teacher_decay_steps or total_steps
        self.loss_fn.teacher_weight = linear_teacher_weight(
            self._teacher_weight_initial,
            self._teacher_weight_final,
            train_step=train_step,
            decay_steps=decay_steps,
        )
        if self.loss_fn.rank_mode in {
            "deployment_weighted_robust_cycle",
            "preservation_constrained_cycle_block",
        }:
            self.loss_fn.robust_cycle_lambda = linear_teacher_weight(
                self._robust_cycle_lambda_start,
                self._robust_cycle_lambda_final,
                train_step=train_step,
                decay_steps=self._robust_cycle_lambda_ramp_steps,
            )
        if isinstance(self.loss_fn, RankPointWithHCRLoss):
            self.loss_fn.set_train_step(train_step)
        self._assert_runtime_candidate_groups(global_batch, stage="train")
        validation_metric_totals = (
            self.loss_fn.metric_totals()
            if self._validation_metric_step is not None
            else {}
        )
        validation_metric_step = self._validation_metric_step
        self._validation_metric_step = None
        self.loss_fn.reset_metrics()
        self._queue_full_gallery_total_relevant(global_batch, stage="train")
        self._queue_inverse_pair_offsets(global_batch, stage="train")
        self._queue_candidate_loss_weights(global_batch, stage="train")
        self._queue_same_label_consistency(global_batch, stage="train")
        self._queue_requirement_ordinal(global_batch, stage="train")
        self._queue_cycle_ap_weights(global_batch, stage="train")
        self._queue_r68_cycle_block_metadata(global_batch, stage="train")
        self._queue_safe_update_metadata(global_batch, stage="train")
        self._queue_retriever_residual_scores(global_batch, stage="train")
        if self.loss_fn.rank_mode in {
            "plackett_luce_policy",
            "rb_plackett_luce_expected_ap",
        }:
            preserve_flags = [
                sample.get("_pas_policy_preserve")
                for sample in global_batch[:: self._candidate_group_size]
            ]
            if any(value is None for value in preserve_flags):
                raise ValueError("PAS PL training group is missing its preserve flag")
            self.loss_fn.set_policy_preserve_flags(preserve_flags)
        if isinstance(self.loss_fn, RankPointWithAttributeReplayLoss):
            self.loss_fn.set_attribute_labels(
                [sample.get("_pas_attribute_labels") for sample in global_batch]
            )
        if isinstance(self.loss_fn, RankPointWithStructuredAttributeLoss):
            self.loss_fn.set_structured_records(
                [sample.get("_pas_structured_attributes") for sample in global_batch]
            )
        if isinstance(self.loss_fn, RankPointWithHCRLoss):
            self.loss_fn.set_hcr_records(
                [sample.get("_pas_hcr") for sample in global_batch]
            )
        if isinstance(self.loss_fn, RankPointWithAtomicAndLoss):
            self.loss_fn.set_atomic_and_records(
                [sample.get("_pas_atomic_and") for sample in global_batch]
            )
        if isinstance(self.loss_fn, RankPointWithVisualDistillationLoss):
            indices = [sample.get("_pas_visual_teacher_index") for sample in global_batch]
            if any(value is None for value in indices):
                raise ValueError(
                    "PAS visual-distillation sample is missing its teacher row index"
                )
            self.loss_fn.set_visual_teacher_indices([int(value) for value in indices])
        if (
            self.loss_fn.teacher_weight
            or rank_mode_requires_parent_scores(self.loss_fn.rank_mode)
        ):
            probabilities = [
                sample.get("_pas_teacher_probability") for sample in global_batch
            ]
            if any(value is None for value in probabilities):
                raise ValueError(
                    "PAS teacher-distillation sample is missing teacher probability"
                )
            self.loss_fn.set_teacher_probabilities(probabilities)
        report_data = super().step_training(*args, **kwargs)
        if self._residual_gate_optimizer is not None:
            gate = self.loss_fn.retriever_residual_gate
            assert gate is not None
            gate_parameters = list(gate.parameters())
            if any(parameter.grad is None for parameter in gate_parameters):
                raise RuntimeError("Query residual gate did not receive every gradient")
            if self.parallel_dims.dp_replicate_enabled:
                group = self.parallel_dims.mesh["dp_replicate"].get_group()
                replicas = self.parallel_dims.mesh["dp_replicate"].size()
                for parameter in gate_parameters:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
                    parameter.grad.div_(replicas)
            gate_grad_norm = torch.nn.utils.clip_grad_norm_(gate_parameters, 1.0)
            self._residual_gate_optimizer.step()
            report_data["optimizer/residual_gate_grad_norm"] = float(gate_grad_norm)
            report_data["optimizer/residual_gate_lr"] = float(
                self._residual_gate_optimizer.param_groups[0]["lr"]
            )
        report_data["train/teacher_weight"] = self.loss_fn.teacher_weight
        if self.loss_fn.rank_mode in {
            "deployment_weighted_robust_cycle",
            "preservation_constrained_cycle_block",
        }:
            report_data["train/robust_cycle_lambda"] = (
                self.loss_fn.robust_cycle_lambda
            )
        self.loss_fn.assert_teacher_probabilities_consumed()
        self.loss_fn.assert_candidate_loss_weights_consumed()
        self.loss_fn.assert_total_relevant_consumed()
        self.loss_fn.assert_cycle_weights_consumed()
        self.loss_fn.assert_cycle_block_metadata_consumed()
        self.loss_fn.assert_safe_update_records_consumed()
        self.loss_fn.assert_policy_preserve_flags_consumed()
        self.loss_fn.assert_inverse_pair_offsets_consumed()
        self.loss_fn.assert_same_label_consistency_offsets_consumed()
        self.loss_fn.assert_requirement_ordinal_records_consumed()
        self.loss_fn.assert_retriever_residual_scores_consumed()
        if isinstance(self.loss_fn, RankPointWithAttributeReplayLoss):
            self.loss_fn.assert_attribute_labels_consumed()
        if isinstance(self.loss_fn, RankPointWithStructuredAttributeLoss):
            self.loss_fn.assert_structured_records_consumed()
        if isinstance(self.loss_fn, RankPointWithHCRLoss):
            self.loss_fn.assert_hcr_records_consumed()
        if isinstance(self.loss_fn, RankPointWithAtomicAndLoss):
            self.loss_fn.assert_atomic_and_records_consumed()
        if isinstance(self.loss_fn, RankPointWithVisualDistillationLoss):
            self.loss_fn.assert_visual_teacher_embeddings_consumed()
        for name, (metric_sum, metric_count) in self.loss_fn.metric_totals().items():
            value = self._distributed_metric_mean(metric_sum, metric_count)
            if value is None:
                continue
            suffix = "_loss" if name in {"rank", "point", "teacher"} else ""
            report_data[f"train/{name}{suffix}"] = value
        for name, (metric_sum, metric_count) in validation_metric_totals.items():
            value = self._distributed_metric_mean(metric_sum, metric_count)
            if value is None:
                continue
            suffix = "_loss" if name in {"rank", "point", "teacher"} else ""
            report_data[f"val/{name}{suffix}"] = value
        if validation_metric_step is not None:
            report_data["val/metric_source_step"] = validation_metric_step
        return report_data

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def _save_residual_gate_endpoint(self, train_step: int) -> None:
        gate = self.loss_fn.retriever_residual_gate
        if gate is None or self._residual_gate_optimizer is None:
            return
        endpoint = (
            Path(self.config.train.output_dir) / "safetensors" / f"step_{train_step}"
        )
        adapter = endpoint / "adapter_model.safetensors"
        if not adapter.is_file():
            raise FileNotFoundError(
                f"Cannot publish residual gate without LoRA adapter: {adapter}"
            )
        endpoint.mkdir(parents=True, exist_ok=True)
        gate_path = endpoint / "alpha_gate.safetensors"
        gate_tmp = endpoint / "alpha_gate.safetensors.partial"
        save_file(
            {
                name: value.detach().float().cpu().contiguous()
                for name, value in gate.state_dict().items()
            },
            str(gate_tmp),
        )
        os.replace(gate_tmp, gate_path)
        gate_config = {
            "format_version": 1,
            "feature_version": gate.feature_version,
            "feature_names": list(gate.feature_names),
            "initial_alpha": gate.initial_alpha,
            "minimum": gate.minimum,
            "maximum": gate.maximum,
            "group_size": self._candidate_group_size,
            "epsilon": self.loss_fn.retriever_residual_epsilon,
            "log_alpha_l2": self.loss_fn.retriever_residual_gate_log_alpha_l2,
        }
        config_path = endpoint / "alpha_gate_config.json"
        config_tmp = endpoint / "alpha_gate_config.json.partial"
        config_tmp.write_text(
            json.dumps(gate_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(config_tmp, config_path)
        optimizer_path = endpoint / "alpha_gate_optimizer.pt"
        optimizer_tmp = endpoint / "alpha_gate_optimizer.pt.partial"
        torch.save(self._residual_gate_optimizer.state_dict(), optimizer_tmp)
        os.replace(optimizer_tmp, optimizer_path)
        manifest = {
            "format_version": 1,
            "train_step": int(train_step),
            "adapter_model_sha256": self._sha256_file(adapter),
            "alpha_gate_sha256": self._sha256_file(gate_path),
            "alpha_gate_config_sha256": self._sha256_file(config_path),
            "alpha_gate_optimizer_sha256": self._sha256_file(optimizer_path),
        }
        manifest_path = endpoint / "composite_manifest.json"
        manifest_tmp = endpoint / "composite_manifest.json.partial"
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest_path)

    def checkpointing(self, *args, **kwargs):
        train_step = int(kwargs.get("train_step", args[1] if len(args) > 1 else 0))
        save_freq = int(kwargs.get("save_freq", args[2] if len(args) > 2 else 0))
        is_last_step = bool(kwargs.get("is_last_step", False))
        do_save = bool(kwargs.get("do_save", False))
        should_save = (
            is_last_step
            or do_save
            or (save_freq > 0 and train_step > 0 and train_step % save_freq == 0)
        ) and self.config.train.ckpt.enable_checkpoint
        result = super().checkpointing(*args, **kwargs)
        if should_save and self._residual_gate_optimizer is not None:
            if self.parallel_dims.dp_replicate_coord[0] == 0:
                self._save_residual_gate_endpoint(train_step)
            dist.barrier()
        return result

    def _distributed_metric_mean(
        self, metric_sum: torch.Tensor, metric_count: int
    ) -> float | None:
        """Compute a count-weighted metric mean with identical collectives/rank."""

        totals = torch.stack(
            [metric_sum.float(), metric_sum.new_tensor(float(metric_count)).float()]
        )
        if self.parallel_dims.dp_replicate_enabled or self.parallel_dims.dp_shard_enabled:
            dist.all_reduce(
                totals,
                op=dist.ReduceOp.SUM,
                group=self.parallel_dims.mesh["dp"].get_group(),
            )
        if totals[1].item() == 0:
            return None
        return (totals[0] / totals[1]).item()

    def step_validation(self, val_global_batch, train_step: int, total_steps: int):
        if self.loss_fn.rank_mode in {
            "deployment_weighted_robust_cycle",
            "preservation_constrained_cycle_block",
        }:
            self.loss_fn.robust_cycle_lambda = linear_teacher_weight(
                self._robust_cycle_lambda_start,
                self._robust_cycle_lambda_final,
                train_step=train_step,
                decay_steps=self._robust_cycle_lambda_ramp_steps,
            )
        self._assert_runtime_candidate_groups(val_global_batch, stage="validation")
        if isinstance(self.loss_fn, RankPointWithHCRLoss):
            self.loss_fn.set_train_step(train_step)
        if self._validation_metric_step != train_step:
            self.loss_fn.reset_metrics()
            self._validation_metric_step = train_step
        self._queue_full_gallery_total_relevant(
            val_global_batch, stage="validation"
        )
        self._queue_inverse_pair_offsets(
            val_global_batch, stage="validation"
        )
        self._queue_cycle_ap_weights(val_global_batch, stage="validation")
        self._queue_r68_cycle_block_metadata(
            val_global_batch, stage="validation"
        )
        self._queue_retriever_residual_scores(
            val_global_batch, stage="validation"
        )
        if (
            self.loss_fn.teacher_weight
            or rank_mode_requires_parent_scores(self.loss_fn.rank_mode)
        ):
            probabilities = [
                sample.get("_pas_teacher_probability") for sample in val_global_batch
            ]
            if any(value is None for value in probabilities):
                raise ValueError(
                    "PAS validation sample is missing teacher probability"
                )
            self.loss_fn.set_teacher_probabilities(probabilities)
        if isinstance(self.loss_fn, RankPointWithVisualDistillationLoss):
            indices = [
                sample.get("_pas_visual_teacher_index")
                for sample in val_global_batch
            ]
            if any(value is None for value in indices):
                raise ValueError(
                    "PAS visual-distillation validation sample is missing its "
                    "teacher row index"
                )
            self.loss_fn.set_visual_teacher_indices(
                [int(value) for value in indices]
            )
        if isinstance(self.loss_fn, RankPointWithStructuredAttributeLoss):
            self.loss_fn.set_structured_records(
                [
                    sample.get("_pas_structured_attributes")
                    for sample in val_global_batch
                ]
            )
            self.loss_fn.set_attribute_audit_enabled(
                self._structured_attribute_audit
            )
        if isinstance(self.loss_fn, RankPointWithHCRLoss):
            self.loss_fn.set_hcr_records(
                [sample.get("_pas_hcr") for sample in val_global_batch]
            )
        if isinstance(self.loss_fn, RankPointWithAtomicAndLoss):
            self.loss_fn.set_atomic_and_records(
                [sample.get("_pas_atomic_and") for sample in val_global_batch]
            )
        self.loss_fn.canonical_validation = True
        try:
            score = super().step_validation(
                val_global_batch,
                train_step,
                total_steps,
            )
            if self._validation_score_audit_path is not None:
                scores = self.loss_fn.last_scores
                if scores is None or scores.numel() != len(val_global_batch):
                    raise ValueError(
                        "Validation score audit could not align scores with batch rows"
                    )
                score_values = scores.detach().float().cpu().tolist()
                with self._validation_score_audit_path.open(
                    "a", encoding="utf-8"
                ) as handle:
                    for sample, value in zip(
                        val_global_batch, score_values, strict=True
                    ):
                        metadata = sample.get("_pas_score_audit")
                        if metadata is None:
                            raise ValueError(
                                "Validation score audit metadata is missing"
                            )
                        record = dict(metadata)
                        record["score"] = float(value)
                        record["validation_step"] = int(train_step)
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            self.loss_fn.assert_teacher_probabilities_consumed()
            self.loss_fn.assert_total_relevant_consumed()
            self.loss_fn.assert_cycle_weights_consumed()
            self.loss_fn.assert_cycle_block_metadata_consumed()
            self.loss_fn.assert_inverse_pair_offsets_consumed()
            self.loss_fn.assert_retriever_residual_scores_consumed()
            if isinstance(self.loss_fn, RankPointWithStructuredAttributeLoss):
                self.loss_fn.assert_structured_records_consumed()
                if self._structured_attribute_audit:
                    if self._structured_attribute_audit_path is None:
                        raise ValueError("Structured attribute audit path is unavailable")
                    records = self.loss_fn.pop_attribute_audit_records()
                    with self._structured_attribute_audit_path.open(
                        "a", encoding="utf-8"
                    ) as handle:
                        for record in records:
                            record["validation_step"] = int(train_step)
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
            if isinstance(self.loss_fn, RankPointWithHCRLoss):
                self.loss_fn.assert_hcr_records_consumed()
            if isinstance(self.loss_fn, RankPointWithAtomicAndLoss):
                self.loss_fn.assert_atomic_and_records_consumed()
            if isinstance(self.loss_fn, RankPointWithVisualDistillationLoss):
                self.loss_fn.assert_visual_teacher_embeddings_consumed()
            return score
        finally:
            self.loss_fn.canonical_validation = False


def main() -> None:
    configure_cuda_runtime()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, encoding="utf-8") as handle:
        config_dict = toml.load(handle)
    custom = CustomConfig.model_validate(config_dict.get("custom", {}))
    attribute_replay_assets = (
        load_attribute_replay_assets(
            custom.attribute_replay.records_path,
            custom.attribute_replay.attribute_vocab_path,
            custom.attribute_replay.attribute_heads_path,
        )
        if custom.attribute_replay is not None
        else None
    )
    structured_train_assets = (
        load_structured_attribute_assets(
            custom.structured_attribute.train_records_path,
            custom.structured_attribute.attribute_vocab_path,
            custom.structured_attribute.query_records_path,
        )
        if custom.structured_attribute is not None
        else None
    )
    structured_validation_assets = (
        load_structured_attribute_assets(
            custom.structured_attribute.validation_records_path,
            custom.structured_attribute.attribute_vocab_path,
        )
        if custom.structured_attribute is not None
        else None
    )
    hcr_assets = (
        load_dense_hcr_assets(custom.hcr.asset_manifest_path)
        if custom.hcr is not None
        else None
    )
    visual_distillation_assets = (
        load_visual_distillation_assets(
            custom.visual_distillation.embedding_metadata_path,
            custom.visual_distillation.image_index_path,
        )
        if custom.visual_distillation is not None
        else None
    )

    def build_dataset(
        dataset_config: DatasetConfig,
        *,
        enable_attribute_replay: bool,
        structured_attribute_assets: StructuredAttributeAssets | None,
        enable_hcr_targets: bool,
        enable_visual_distillation: bool,
    ):
        def factory(_config):
            return PasConversationDataset(
                custom,
                dataset_config,
                attribute_replay_assets=(
                    attribute_replay_assets if enable_attribute_replay else None
                ),
                structured_attribute_assets=structured_attribute_assets,
                hcr_assets=hcr_assets if enable_hcr_targets else None,
                visual_distillation_assets=(
                    visual_distillation_assets
                    if enable_visual_distillation
                    else None
                ),
            )

        return factory

    data_packer = (
        PasCachedQwen3VLDataPacker(custom.visual_cache, custom.vision)
        if custom.visual_cache is not None
        else PasFullImageQwen3VLDataPacker()
    )
    val_data_packer = (
        (
            PasCachedQwen3VLDataPacker(custom.visual_cache, custom.vision)
            if custom.visual_cache is not None
            else PasFullImageQwen3VLDataPacker()
        )
        if custom.val_dataset is not None
        else None
    )

    grouped = custom.train_dataset.response_mode != "mcq"
    if custom.val_dataset is not None:
        val_grouped = custom.val_dataset.response_mode != "mcq"
    else:
        val_grouped = False
    cosmos_rl.launcher.worker_entry.main(
        dataset=build_dataset(
            custom.train_dataset,
            enable_attribute_replay=True,
            structured_attribute_assets=structured_train_assets,
            enable_hcr_targets=True,
            enable_visual_distillation=True,
        ),
        val_dataset=(
            build_dataset(
                custom.val_dataset,
                enable_attribute_replay=False,
                structured_attribute_assets=structured_validation_assets,
                # The deployed HCR score consumes only query text and pixels.
                # Optional targets are train-split telemetry only and cannot
                # enter canonical validation loss or the deployed readout.
                enable_hcr_targets=custom.hcr_validation_targets_for_audit,
                # This validation split is a disjoint train-only diagnostic,
                # not Val999.  Teacher rows are used only for fixed VSE
                # telemetry; canonical checkpoint selection remains rank-only.
                enable_visual_distillation=True,
            )
            if custom.val_dataset
            else None
        ),
        data_packer=data_packer,
        val_data_packer=val_data_packer,
        batch_sampler=pas_train_group_batch_sampler if grouped else None,
        val_batch_sampler=pas_val_group_batch_sampler if val_grouped else None,
        custom_logger_fns=[write_loss_metrics],
        hook_fns={
            "pre_validation_hook": begin_atomic_validation_score_audit,
            "post_validation_hook": finalize_pas_validation,
        },
    )


if __name__ == "__main__":
    main()
