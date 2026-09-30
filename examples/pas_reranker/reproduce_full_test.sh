#!/usr/bin/env bash
# Full deduplicated PAS test evaluation for the table's retrieval and reranking rows.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 public_identity|public_cr3|finetuned_identity|finetuned_zeroshot|finetuned_step8000" >&2
  exit 2
fi

mode="$1"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
package="$repo/third_party/rerank_experiment"
data_root="${PAS_DATA_ROOT:-/workspace/jiayin/rerank_experiment/artifacts/pas_v31_test_tao}"
public_cache="${PUBLIC_SIGLIP_CACHE:-/workspace/jiayin/rerank_experiment/artifacts/pas_v31_public_siglip2}"
finetuned_cache="${FINETUNED_SIGLIP_CACHE:-/workspace/jiayin/rerank_experiment/artifacts/pas_v31_step03906_cache}"
output_root="${PAS_OUTPUT_ROOT:-$repo/reproduction/full_test}"
merged_base="${PAS_MERGED_BASE:-/tmp/pas-cr3-repro-step8000-merged-base}"
base_model="${CR3_BASE_MODEL:-/workspace/jiayin/models/Cosmos3-Nano-VLM}"
adapter="${PAS_ADAPTER_DIR:-$repo/checkpoints/cr3_nano_step8000}"
py="${RERANK_PYTHON:-/workspace/jiayin/rerank_experiment/.venv/bin/python}"

case "$mode" in
  public_identity|public_cr3) cache="$public_cache" ;;
  finetuned_identity|finetuned_zeroshot|finetuned_step8000) cache="$finetuned_cache" ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

for path in "$data_root/test_pairs.json" "$cache/test_pairs_source_image_embeddings.pkl" "$cache/test_pairs_text_embeddings_lower.pkl"; do
  test -s "$path" || { echo "Missing input: $path" >&2; exit 2; }
done
test -d "$data_root/images"

reranker=identity
extra="--deduplicate-scalar-plus-accessories"
if [ "$mode" = public_cr3 ] || [ "$mode" = finetuned_zeroshot ]; then
  reranker=cosmos_reason
  extra="$extra --model-id ${PUBLIC_CR3_MODEL:-nvidia/Cosmos3-Nano}"
fi
if [ "$mode" = finetuned_step8000 ]; then
  reranker=cosmos_reason
  if [ -f "$adapter/materialize.py" ]; then
    python3 "$adapter/materialize.py"
  fi
  test -s "$adapter/language/adapter_model.safetensors"
  test -s "$adapter/visual/adapter_model.safetensors"
  if [ ! -s "$merged_base/model.safetensors.index.json" ]; then
    if [ -e "$merged_base" ]; then
      echo "Incomplete merged base exists: $merged_base" >&2
      exit 2
    fi
    "${TRAIN_PYTHON_BIN:-/opt/venv/cosmos_rl/bin/python}" "$repo/examples/pas_reranker/merge_adapter.py" \
      --base-model "$base_model" --adapter "$adapter/visual" \
      --output "$merged_base" --merge-dtype float32 --save-dtype float16
  fi
  extra="$extra --model-id $merged_base --lora-path $adapter/language --max-lora-rank 32"
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
if [[ -d /usr/local/cuda-13.2/compat/lib.real ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-13.2/compat/lib.real${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cuda_dir="${CUDA_HOME:-/usr/local/cuda}"
if [[ -d "$cuda_dir/targets/x86_64-linux/include" ]]; then
  export CPATH="$cuda_dir/targets/x86_64-linux/include${CPATH:+:$CPATH}"
fi
if [[ -x "$cuda_dir/bin/ptxas" ]]; then
  export PATH="$cuda_dir/bin:$PATH"
  export TRITON_PTXAS_PATH="$cuda_dir/bin/ptxas"
fi
export TRANSFORMERS_AUTO_SWITCH=0
export PY="$py"
export PAIRS_FILE="$data_root/test_pairs.json"
export IMAGE_ROOT="$data_root/images"
export IMAGE_EMBEDDINGS="$cache/test_pairs_source_image_embeddings.pkl"
export TEXT_EMBEDDINGS="$cache/test_pairs_text_embeddings_lower.pkl"
export QUERY_SUBSET_FILE="${QUERY_SUBSET_FILE:-}"
export OUTPUT_BASE="$output_root/$mode"
export MODE_LABEL=scalar_plus_accessories_full_dedup
export RUN_PREFIX="$mode"
export SHARDS="${SHARDS:-8}"
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
if [ "$mode" = finetuned_step8000 ]; then
  # The recorded step-8000 test shared GPUs with training and used 2,048.
  default_score_chunk=64
  default_max_model_len=2048
else
  # The recorded public and zero-shot CR3 runs used the package defaults.
  default_score_chunk=256
  default_max_model_len=32768
fi
export DEPTH=20 K=5 SCORE_CHUNK="${SCORE_CHUNK:-$default_score_chunk}" RERANKER_BATCH=16 QUERY_BATCH=64
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$default_max_model_len}"
if [ "$mode" = finetuned_step8000 ]; then
  # The recorded test run shared GPUs with training and used this vLLM budget.
  default_gpu_memory_utilization=0.42
else
  default_gpu_memory_utilization=0.85
fi
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$default_gpu_memory_utilization}"
export SAVE_QUERY_RANKINGS=1
export EXTRA="$extra"

cd "$package"
exec scripts/pas/run_paired_caption_rerankers.sh "$reranker"
