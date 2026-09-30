#!/usr/bin/env bash
# Submit one 16-GPU, two-worker Lepton test run on the mounted PAS workspace.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 public_cr3|finetuned_zeroshot|finetuned_step8000" >&2
  exit 2
fi
mode="$1"
case "$mode" in
  public_cr3) short_mode=pubcr3 ;;
  finetuned_zeroshot) short_mode=ftzero ;;
  finetuned_step8000) short_mode=ftcr3 ;;
  *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
esac
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
lep="${LEP_BIN:-/opt/venv/cosmos_rl/bin/lep}"
name="pas-${short_mode}-16g-$(date -u +%m%d%H%M%S)"
env_args=()
for variable in \
  PAS_ADAPTER_DIR PAS_MERGED_BASE PAS_OUTPUT_ROOT CR3_BASE_MODEL \
  PAS_DATA_ROOT PUBLIC_SIGLIP_CACHE FINETUNED_SIGLIP_CACHE PUBLIC_CR3_MODEL \
  RERANK_PYTHON TRAIN_PYTHON_BIN SCORE_CHUNK MAX_MODEL_LEN \
  GPU_MEMORY_UTILIZATION CUDA_HOME; do
  if [[ -n "${!variable:-}" ]]; then
    env_args+=(--env "$variable=${!variable}")
  fi
done
if [[ "$mode" == finetuned_step8000 && -z "${PAS_MERGED_BASE:-}" ]]; then
  env_args+=(--env "PAS_MERGED_BASE=$repo/reproduction/merged_step8000_base")
fi
"$lep" job create --name "$name" \
  --container-image "${LEPTON_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.1.0-cosmos-rl}" \
  --command "/bin/bash -lc 'cd $repo && bash examples/pas_reranker/run_full_test_batch_worker.sh $mode'" \
  --resource-shape gpu.8xa100-80gb --num-workers 2 \
  "${env_args[@]}" \
  --mount "${LEPTON_WORKSPACE_MOUNT:-${LEPTON_MOUNT:-/edgeai_metropolis_metropolis-2-0:/workspace:node-nfs:amlfs}}" \
  --image-pull-secrets "${LEPTON_IMAGE_PULL_SECRET:-staging-new}" --shared-memory-size 65536 \
  --log-collection true --ttl-seconds-after-finished 259200 \
  --node-group "${LEPTON_NODE_GROUP:-az-sat-lepton-001}"
