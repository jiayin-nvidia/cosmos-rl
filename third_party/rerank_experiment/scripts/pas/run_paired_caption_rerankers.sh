#!/bin/bash
# Run PAS V3.1 retriever + optional reranker evaluation.
#
# Default mode is scalar_plus_accessories. Override MODES/QUERY_TYPES when needed.
#
# Run from the package root with identity or cosmos_reason.
# SHARDS and GPUS control independent query shards.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"

configure_python_cuda_libs() {
  local lib_dir paths=""
  for lib_dir in "$PWD"/.venv/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$lib_dir" ] || continue
    paths="${paths:+$paths:}$lib_dir"
  done
  if [ -n "$paths" ]; then
    export LD_LIBRARY_PATH="$paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

configure_python_cuda_libs

# Each Super shard already occupies four GPUs.  Prevent CPU BLAS/tokenizer
# worker pools from oversubscribing host cores when multiple shards run.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SUPPORTED=(identity cosmos_reason)
RUN_FAILURES=()

contains_model() {
  local needle="$1" item
  for item in "${SUPPORTED[@]}"; do
    [ "$item" = "$needle" ] && return 0
  done
  return 1
}

report_failures() {
  if [ "${#RUN_FAILURES[@]}" -gt 0 ]; then
    echo "!!!! ${#RUN_FAILURES[@]} RUN(S) FAILED: ${RUN_FAILURES[*]}"
    return 1
  fi
  echo "=== all PAS reranker runs OK ==="
  return 0
}

models=("$@")
[ ${#models[@]} -gt 0 ] || models=(identity)

PAIRS_FILE="${PAIRS_FILE:-artifacts/pas_v31_test_tao/test_pairs.json}"
IMAGE_ROOT="${IMAGE_ROOT:-artifacts/pas_v31_test_tao/images}"
IMAGE_EMBEDDINGS="${IMAGE_EMBEDDINGS:-artifacts/pas_v31_test_tao/test_pairs_source_image_embeddings.pkl}"
TEXT_EMBEDDINGS="${TEXT_EMBEDDINGS:-artifacts/pas_v31_test_tao/test_pairs_text_embeddings_lower.pkl}"
RUN_DATETIME="${RUN_DATETIME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-results/pas_v31_step01953_${RUN_DATETIME}}"
RUN_PREFIX="${RUN_PREFIX:-step01953}"

GPU="${GPU:-0}"
GPUS="${GPUS:-$GPU}"
# Space-separated CUDA_VISIBLE_DEVICES assignments, one per shard.  This is
# needed when each shard is itself a tensor-parallel model replica.
GPU_GROUPS="${GPU_GROUPS:-}"
SHARDS="${SHARDS:-1}"
DEPTH="${DEPTH:-50}"
K="${K:-5}"
QUERY_BATCH="${QUERY_BATCH:-64}"
RUNNING_LOG_EVERY="${RUNNING_LOG_EVERY:-200}"
RERANKER_BATCH="${RERANKER_BATCH:-16}"
SCORE_CHUNK="${SCORE_CHUNK:-256}"
MODES="${MODES:-scalar_plus_accessories}"
QUERY_TYPES="${QUERY_TYPES:-easy medium hard}"
MODE_LABEL="${MODE_LABEL:-${MODES// /_}}"
REASONING="${REASONING:-none}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-logit_delta}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-256}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
QUERY_SUBSET_FILE="${QUERY_SUBSET_FILE:-}"
SAVE_QUERY_RANKINGS="${SAVE_QUERY_RANKINGS:-1}"
HCR_ACTIVE_LOGICAL="${HCR_ACTIVE_LOGICAL:-0}"
HCR_NATIVE_SCORE_WEIGHT="${HCR_NATIVE_SCORE_WEIGHT:-}"
HCR_LOGICAL_SCORE_WEIGHT="${HCR_LOGICAL_SCORE_WEIGHT:-}"
HCR_LOGICAL_REDUCTION="${HCR_LOGICAL_REDUCTION:-}"
HCR_PROMPT_LOGPROBS="${HCR_PROMPT_LOGPROBS:-20}"
HCR_EXPECTED_TEMPLATE_SHA256="${HCR_EXPECTED_TEMPLATE_SHA256:-}"
HCR_CHECKPOINT_SHA256="${HCR_CHECKPOINT_SHA256:-}"
HCR_TRAINING_CONFIG_SHA256="${HCR_TRAINING_CONFIG_SHA256:-}"
ATOMIC_CONSTRAINT_AND="${ATOMIC_CONSTRAINT_AND:-0}"
ATOMIC_CONSTRAINT_REDUCTION="${ATOMIC_CONSTRAINT_REDUCTION:-sum_logprob}"
ATOMIC_CONSTRAINT_TEMPERATURE="${ATOMIC_CONSTRAINT_TEMPERATURE:-1.0}"
# This machine does not expose nvcc or /usr/local/cuda.  vLLM's FlashInfer
# sampler tries to JIT-compile kernels at startup and fails without them.
# Keep it disabled by default; callers can override with VLLM_USE_FLASHINFER_SAMPLER=1.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

COMMON_EVAL_ARGS=()

validate_inputs() {
  local model="$1" path label rc=0
  for label in PAIRS_FILE IMAGE_EMBEDDINGS TEXT_EMBEDDINGS; do
    path="${!label}"
    if [ ! -f "$path" ]; then
      echo "!! $label not found: $path" >&2
      rc=1
    fi
  done
  if [ -n "$QUERY_SUBSET_FILE" ] && [ ! -f "$QUERY_SUBSET_FILE" ]; then
    echo "!! QUERY_SUBSET_FILE not found: $QUERY_SUBSET_FILE" >&2
    rc=1
  fi
  if [ "$model" != "identity" ] && [ ! -d "$IMAGE_ROOT" ]; then
    echo "!! IMAGE_ROOT not found: $IMAGE_ROOT" >&2
    echo "!! Visual rerankers require the materialized PAS images; set IMAGE_ROOT to their directory." >&2
    rc=1
  fi
  return "$rc"
}

build_common_eval_args() {
  local model="$1" outdir="$2" run_name="$3"
  local mode_args=() query_type_args=() extra_args=()
  read -r -a mode_args <<< "$MODES"
  read -r -a query_type_args <<< "$QUERY_TYPES"

  COMMON_EVAL_ARGS=(
    --reranker "$model"
    --pairs-file "$PAIRS_FILE"
    --image-root "$IMAGE_ROOT"
    --image-embeddings "$IMAGE_EMBEDDINGS"
    --text-embeddings "$TEXT_EMBEDDINGS"
    --output-dir "$outdir"
    --run-name "$run_name"
    --modes "${mode_args[@]}"
    --query-types "${query_type_args[@]}"
    --rerank-depth "$DEPTH"
    --reranker-batch-size "$RERANKER_BATCH"
    --reranker-score-chunk-size "$SCORE_CHUNK"
    --query-batch-size "$QUERY_BATCH"
    --running-log-every "$RUNNING_LOG_EVERY"
    --reasoning "$REASONING"
    --output-format "$OUTPUT_FORMAT"
    --max-think-tokens "$MAX_THINK_TOKENS"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --jpeg-quality "$JPEG_QUALITY"
    --k "$K"
  )
  if [ "$ENFORCE_EAGER" = "1" ]; then
    COMMON_EVAL_ARGS+=(--enforce-eager)
  fi
  if [ "$HCR_ACTIVE_LOGICAL" = "1" ]; then
    if [ -z "$HCR_NATIVE_SCORE_WEIGHT" ] || [ -z "$HCR_LOGICAL_SCORE_WEIGHT" ] || \
       [ -z "$HCR_LOGICAL_REDUCTION" ] || [ -z "$HCR_EXPECTED_TEMPLATE_SHA256" ] || \
       [ -z "$HCR_CHECKPOINT_SHA256" ] || [ -z "$HCR_TRAINING_CONFIG_SHA256" ]; then
      echo "HCR_ACTIVE_LOGICAL=1 requires weights, reduction, and all SHA-256 values" >&2
      return 2
    fi
    COMMON_EVAL_ARGS+=(
      --hcr-active-logical
      --hcr-native-score-weight "$HCR_NATIVE_SCORE_WEIGHT"
      --hcr-logical-score-weight "$HCR_LOGICAL_SCORE_WEIGHT"
      --hcr-logical-reduction "$HCR_LOGICAL_REDUCTION"
      --hcr-prompt-logprobs "$HCR_PROMPT_LOGPROBS"
      --hcr-expected-template-sha256 "$HCR_EXPECTED_TEMPLATE_SHA256"
      --hcr-checkpoint-sha256 "$HCR_CHECKPOINT_SHA256"
      --hcr-training-config-sha256 "$HCR_TRAINING_CONFIG_SHA256"
    )
  fi
  if [ "$ATOMIC_CONSTRAINT_AND" = "1" ]; then
    COMMON_EVAL_ARGS+=(
      --atomic-constraint-and
      --atomic-constraint-reduction "$ATOMIC_CONSTRAINT_REDUCTION"
      --atomic-constraint-temperature "$ATOMIC_CONSTRAINT_TEMPERATURE"
    )
  fi
  [ -n "${LIMIT_PAIRS:-}" ] && COMMON_EVAL_ARGS+=(--limit-pairs "$LIMIT_PAIRS")
  [ -n "$QUERY_SUBSET_FILE" ] && \
    COMMON_EVAL_ARGS+=(--query-subset-file "$QUERY_SUBSET_FILE")
  if [ "$SAVE_QUERY_RANKINGS" != "0" ] && [ "$model" != "identity" ]; then
    COMMON_EVAL_ARGS+=(--save-query-rankings)
  fi
  if [ -n "${EXTRA:-}" ]; then
    # Backward-compatible for simple extra flags. Prefer dedicated variables
    # such as QUERY_SUBSET_FILE when a value must remain one exact argument.
    read -r -a extra_args <<< "$EXTRA"
    COMMON_EVAL_ARGS+=("${extra_args[@]}")
  fi
}

run_single() {
  local model="$1" outdir="$2" run_name="$3" log="$4"
  echo "=== PAS $MODES | model=$model depth=$DEPTH k=$K gpu=$GPU | $(date +%H:%M:%S) ===" | tee -a "$log"
  (
    set -o pipefail
    build_common_eval_args "$model" "$outdir" "$run_name"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_pas_three_modes_with_reranker.py \
      "${COMMON_EVAL_ARGS[@]}" 2>&1 | tee -a "$log"
  )
  return ${PIPESTATUS[0]}
}

run_sharded() {
  local model="$1" outdir="$2" run_name="$3" log="$4"
  local shard_count="$SHARDS"
  local shard_gpus=() shard_ids=() shard seen=" " local_index
  if [ -n "${SHARD_INDICES:-}" ]; then
    read -r -a shard_ids <<< "$SHARD_INDICES"
  else
    for ((shard=0; shard<shard_count; shard++)); do shard_ids+=("$shard"); done
  fi
  if [ "${#shard_ids[@]}" -eq 0 ]; then
    echo "!! no shard indices selected" | tee -a "$log" >&2
    return 2
  fi
  for shard in "${shard_ids[@]}"; do
    if ! [[ "$shard" =~ ^[0-9]+$ ]] || [ "$shard" -ge "$shard_count" ] || [[ "$seen" == *" $shard "* ]]; then
      echo "!! invalid or repeated shard index: $shard" | tee -a "$log" >&2
      return 2
    fi
    seen+="$shard "
  done
  if [ -n "$GPU_GROUPS" ]; then
    read -r -a shard_gpus <<< "$GPU_GROUPS"
  else
    IFS=',' read -r -a shard_gpus <<< "$GPUS"
  fi
  if [ "${#shard_gpus[@]}" -lt "${#shard_ids[@]}" ]; then
    echo "!! selected ${#shard_ids[@]} shards require that many GPU assignments; set GPUS or space-separated GPU_GROUPS." | tee -a "$log" >&2
    return 2
  fi

  echo "=== PAS $MODES | model=$model depth=$DEPTH k=$K shards=$shard_count gpus=${GPU_GROUPS:-$GPUS} | $(date +%H:%M:%S) ===" | tee -a "$log"

  local pids=() shard_dirs=() shard_logs=()
  local gpu shard_dir shard_log
  for ((local_index=0; local_index<${#shard_ids[@]}; local_index++)); do
    shard="${shard_ids[$local_index]}"
    gpu="${shard_gpus[$local_index]}"
    shard_dir="$outdir/shard_${shard}_of_${shard_count}"
    shard_log="$shard_dir/run.log"
    mkdir -p "$shard_dir"
    : > "$shard_log"
    shard_dirs+=("$shard_dir")
    shard_logs+=("$shard_log")
    echo "--- launching shard $shard/$shard_count on GPU $gpu -> $shard_dir" | tee -a "$log"
    (
      build_common_eval_args \
        "$model" "$shard_dir" "${run_name}_shard_${shard}_of_${shard_count}"
      if [ "${STREAM_SHARD_LOGS:-1}" = "0" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/evaluate_pas_three_modes_with_reranker.py \
          "${COMMON_EVAL_ARGS[@]}" \
          --shard-count "$shard_count" \
          --shard-index "$shard" > "$shard_log" 2>&1
      else
        # Write the full shard log to disk, but only stream useful progress/error lines
        # to the parent terminal.  PIPESTATUS[0] preserves the Python process rc.
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/evaluate_pas_three_modes_with_reranker.py \
          "${COMMON_EVAL_ARGS[@]}" \
          --shard-count "$shard_count" \
          --shard-index "$shard" 2>&1 \
          | tee "$shard_log" \
          | grep --line-buffered -E " RUNNING | ERROR | CRITICAL |Traceback|RuntimeError|ValueError|FileNotFoundError|!!!!|FAILED" \
          | sed -u -E 's/^.*([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:,]+ (INFO|ERROR|CRITICAL) )/\1/'
        exit "${PIPESTATUS[0]}"
      fi
    ) &
    pids+=("$!")
  done

  if [ "${STREAM_SHARD_LOGS:-1}" != "0" ]; then
    echo "--- streaming cumulative RUNNING metrics and errors from shard logs live (set STREAM_SHARD_LOGS=0 to disable)" | tee -a "$log"
  fi

  local rc=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done

  if [ "$rc" -ne 0 ] || [ "${SHOW_SHARD_TAILS:-0}" != "0" ]; then
    for ((local_index=0; local_index<${#shard_ids[@]}; local_index++)); do
      shard="${shard_ids[$local_index]}"
      echo "--- shard $shard/$shard_count log tail: ${shard_logs[$local_index]}" | tee -a "$log"
      grep -viE "pci|onnxruntime|it/s\]|s/it\]" "${shard_logs[$local_index]}" | tail -60 | tee -a "$log" || true
    done
  fi

  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi

  if [ "${SKIP_SHARD_MERGE:-0}" = 1 ]; then
    echo "=== selected shards complete; merge after all workers finish ===" | tee -a "$log"
    return 0
  fi
  if [ "${#shard_ids[@]}" -ne "$shard_count" ]; then
    echo "!! selected shard subset requires SKIP_SHARD_MERGE=1" | tee -a "$log" >&2
    return 2
  fi

  echo "=== merging $shard_count PAS shards -> $outdir ===" | tee -a "$log"
  "$PY" scripts/pas/merge_pas_three_modes_shards.py \
    --output-dir "$outdir" \
    --run-name "$run_name" \
    --k "$K" \
    --modes $MODES \
    --query-types $QUERY_TYPES \
    -- \
    "${shard_dirs[@]}" 2>&1 | tee -a "$log"
  return ${PIPESTATUS[0]}
}

run_one() {
  local model="$1"
  if ! contains_model "$model"; then
    echo "!! unknown PAS reranker '$model' (known: ${SUPPORTED[*]})" >&2
    return 2
  fi

  if ! validate_inputs "$model"; then
    RUN_FAILURES+=("$model@inputs")
    return 1
  fi

  local outdir="$OUTPUT_BASE/${MODE_LABEL}_${model}_depth${DEPTH}_k${K}"
  local run_name="${RUN_PREFIX}_${MODE_LABEL}_${model}_depth${DEPTH}"
  local log="$outdir/run.log"

  mkdir -p "$outdir"
  if [ "${SKIP_SHARD_MERGE:-0}" = 1 ] && [ -n "${SHARD_INDICES:-}" ]; then
    log="$outdir/run_shards_${SHARD_INDICES// /_}.log"
  fi
  : > "$log"

  local rc=0
  if [ "$SHARDS" -gt 1 ]; then
    run_sharded "$model" "$outdir" "$run_name" "$log"
    rc=$?
  else
    run_single "$model" "$outdir" "$run_name" "$log"
    rc=$?
  fi

  if [ "$rc" -ne 0 ]; then
    RUN_FAILURES+=("$model")
    echo "!!!! FAILED: $model rc=$rc log=$log" | tee -a "$log"
    return "$rc"
  fi

  echo "=== DONE: $model ===" | tee -a "$log"
  if [ "${SKIP_SHARD_MERGE:-0}" != 1 ]; then
    echo "summary: $outdir/${run_name}_three_modes_weighted.csv" | tee -a "$log"
  fi
}

base_score_chunk="$SCORE_CHUNK"
for model in "${models[@]}"; do
  SCORE_CHUNK="$base_score_chunk"
  run_one "$model" || true
done
SCORE_CHUNK="$base_score_chunk"

report_failures
