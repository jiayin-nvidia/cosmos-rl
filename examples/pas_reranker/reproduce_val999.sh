#!/usr/bin/env bash
# Generate the step-8000 validation trace used to select the fusion weight.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
package="$repo/third_party/rerank_experiment"
data_root="${PAS_VAL_DATA_ROOT:-/workspace/data/PAS_Datasets/NVIDIA_PAS_Filtered_06152026_V3.1_tao_ft_metadata_base_plus_val_test_medium_attr_matched_aug_val20_accessory_v2.1}"
cache="${VAL_SIGLIP_CACHE:-/workspace/jiayin/rerank_experiment/artifacts/pas_v31_val_step03906_tao}"
subset="${VAL_QUERY_SUBSET:-/workspace/jiayin/rerank_experiment/artifacts/pas_v31_val_query_subset_scalar_plus_accessories_balanced_999/query_subset.json}"
merged_base="${PAS_MERGED_BASE:-/tmp/pas-cr3-repro-step8000-merged-base}"
adapter="${PAS_ADAPTER_DIR:-$repo/checkpoints/cr3_nano_step8000}"

if [ -f "$adapter/materialize.py" ]; then
  python3 "$adapter/materialize.py"
fi
test -s "$adapter/language/adapter_model.safetensors"
test -s "$adapter/visual/adapter_model.safetensors"
for path in "$data_root/val_pairs.json" "$cache/val_pairs_source_image_embeddings.pkl" "$cache/val_pairs_text_embeddings_lower.pkl" "$subset" "$merged_base/model.safetensors.index.json"; do
  test -s "$path" || { echo "Missing input: $path" >&2; exit 2; }
done
test -d "$data_root/images"

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
export PY="${RERANK_PYTHON:-/workspace/jiayin/rerank_experiment/.venv/bin/python}"
export PAIRS_FILE="$data_root/val_pairs.json"
export IMAGE_ROOT="$data_root/images"
export IMAGE_EMBEDDINGS="$cache/val_pairs_source_image_embeddings.pkl"
export TEXT_EMBEDDINGS="$cache/val_pairs_text_embeddings_lower.pkl"
export QUERY_SUBSET_FILE="$subset"
export OUTPUT_BASE="${PAS_OUTPUT_ROOT:-$repo/reproduction/val999}/finetuned_step8000"
export MODE_LABEL=scalar_plus_accessories_val999
export RUN_PREFIX=finetuned_step8000
export SHARDS="${SHARDS:-8}" GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export DEPTH=20 K=5 SCORE_CHUNK=64 RERANKER_BATCH=16 QUERY_BATCH=64
export MAX_MODEL_LEN=2048
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export SAVE_QUERY_RANKINGS=1
export EXTRA="--model-id $merged_base --lora-path $adapter/language --max-lora-rank 32"

cd "$package"
exec scripts/pas/run_paired_caption_rerankers.sh cosmos_reason
