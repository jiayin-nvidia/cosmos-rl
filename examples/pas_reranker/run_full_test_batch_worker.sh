#!/usr/bin/env bash
# Run half of a 16-shard PAS test on one eight-GPU batch worker, or merge both.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 public_cr3|finetuned_zeroshot|finetuned_step8000 [0|1|merge]" >&2
  exit 2
fi
mode="$1"
worker="${2:-${LEPTON_JOB_WORKER_INDEX:-}}"
case "$mode" in public_cr3|finetuned_zeroshot|finetuned_step8000) ;; *) echo "Unsupported mode: $mode" >&2; exit 2 ;; esac
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$worker" == merge ]]; then
  package="$repo/third_party/rerank_experiment"
  output_root="${PAS_OUTPUT_ROOT:-$repo/reproduction/full_test}"
  outdir="$output_root/$mode/scalar_plus_accessories_full_dedup_cosmos_reason_depth20_k5"
  run_name="${mode}_scalar_plus_accessories_full_dedup_cosmos_reason_depth20"
  shards=()
  for ((shard=0; shard<16; shard++)); do
    shard_dir="$outdir/shard_${shard}_of_16"
    test -s "$shard_dir/scalar_plus_accessories/query_rankings.jsonl" || {
      echo "Missing completed shard: $shard_dir" >&2
      exit 2
    }
    shards+=("$shard_dir")
  done
  cd "$package"
  exec "${RERANK_PYTHON:-/workspace/jiayin/rerank_experiment/.venv/bin/python}" \
    scripts/pas/merge_pas_three_modes_shards.py \
    --output-dir "$outdir" --run-name "$run_name" --k 5 \
    --modes scalar_plus_accessories --query-types easy medium hard \
    -- "${shards[@]}"
fi

case "$worker" in
  0) shard_indices="0 1 2 3 4 5 6 7" ;;
  1) shard_indices="8 9 10 11 12 13 14 15" ;;
  *) echo "Expected worker index 0 or 1; got '$worker'" >&2; exit 2 ;;
esac

if [[ "$mode" == finetuned_step8000 ]]; then
  adapter="${PAS_ADAPTER_DIR:-$repo/checkpoints/cr3_nano_step8000}"
  merged_base="${PAS_MERGED_BASE:-$repo/reproduction/merged_step8000_base}"
  marker="$merged_base/.pas_merge_complete"
  if [[ "$worker" == 0 ]]; then
    if [[ -f "$adapter/materialize.py" ]]; then
      python3 "$adapter/materialize.py"
    fi
    test -s "$adapter/visual/adapter_model.safetensors"
    adapter_sha="$(sha256sum "$adapter/visual/adapter_model.safetensors")"
    adapter_sha="${adapter_sha%% *}"
    if [[ ! -f "$marker" ]]; then
      if [[ ! -s "$merged_base/model.safetensors.index.json" ]]; then
        [[ ! -e "$merged_base" ]] || {
          echo "Incomplete merged base exists: $merged_base" >&2
          exit 2
        }
        "${TRAIN_PYTHON_BIN:-/opt/venv/cosmos_rl/bin/python}" \
          "$repo/examples/pas_reranker/merge_adapter.py" \
          --base-model "${CR3_BASE_MODEL:-/workspace/jiayin/models/Cosmos3-Nano-VLM}" \
          --adapter "$adapter/visual" --output "$merged_base" \
          --merge-dtype float32 --save-dtype float16
      fi
      printf '%s\n' "$adapter_sha" > "$marker.tmp.$$"
      mv "$marker.tmp.$$" "$marker"
    fi
  else
    for ((attempt=0; attempt<360; attempt++)); do
      [[ -f "$marker" ]] && break
      sleep 10
    done
    [[ -f "$marker" ]] || { echo "Timed out waiting for $marker" >&2; exit 2; }
    adapter_sha="$(sha256sum "$adapter/visual/adapter_model.safetensors")"
    adapter_sha="${adapter_sha%% *}"
  fi
  [[ -s "$merged_base/model.safetensors.index.json" ]] || {
    echo "Merged base is incomplete: $merged_base" >&2; exit 2;
  }
  [[ "$(cat "$marker")" == "$adapter_sha" ]] || {
    echo "Merged base adapter hash does not match $adapter" >&2; exit 2;
  }
  export PAS_MERGED_BASE="$merged_base"
fi
export SHARDS=16 SHARD_INDICES="$shard_indices" SKIP_SHARD_MERGE=1
export GPUS=0,1,2,3,4,5,6,7
exec bash "$repo/examples/pas_reranker/reproduce_full_test.sh" "$mode"
