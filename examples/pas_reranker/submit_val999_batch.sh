#!/usr/bin/env bash
# Submit one eight-A100 Val999 job after the step-8000 visual adapter is merged.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
adapter="${PAS_ADAPTER_DIR:-$repo/checkpoints/cr3_nano_step8000}"
merged_base="${PAS_MERGED_BASE:-$repo/reproduction/merged_step8000_base}"
output_root="${PAS_OUTPUT_ROOT:-$repo/reproduction/val999}"
if [[ -f "$adapter/materialize.py" ]]; then
  python3 "$adapter/materialize.py"
fi
[[ -s "$adapter/language/adapter_model.safetensors" ]] || {
  echo "Missing language adapter under $adapter" >&2; exit 2;
}
[[ -s "$merged_base/model.safetensors.index.json" ]] || {
  echo "Merge the visual adapter into $merged_base first" >&2; exit 2;
}

lep="${LEP_BIN:-/opt/venv/cosmos_rl/bin/lep}"
name="pas-cr3-val999-$(date -u +%m%d%H%M%S)"
env_args=()
for variable in \
  RERANK_PYTHON PAS_VAL_DATA_ROOT VAL_SIGLIP_CACHE VAL_QUERY_SUBSET \
  GPU_MEMORY_UTILIZATION CUDA_HOME; do
  if [[ -n "${!variable:-}" ]]; then
    env_args+=(--env "$variable=${!variable}")
  fi
done
"$lep" job create --name "$name" \
  --container-image "${LEPTON_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.1.0-cosmos-rl}" \
  --command "/bin/bash -lc 'cd $repo && bash examples/pas_reranker/reproduce_val999.sh'" \
  --resource-shape gpu.8xa100-80gb --num-workers 1 \
  --env "PAS_ADAPTER_DIR=$adapter" \
  --env "PAS_MERGED_BASE=$merged_base" \
  --env "PAS_OUTPUT_ROOT=$output_root" \
  "${env_args[@]}" \
  --mount "${LEPTON_WORKSPACE_MOUNT:-${LEPTON_MOUNT:-/edgeai_metropolis_metropolis-2-0:/workspace:node-nfs:amlfs}}" \
  --image-pull-secrets "${LEPTON_IMAGE_PULL_SECRET:-staging-new}" --shared-memory-size 65536 \
  --log-collection true --ttl-seconds-after-finished 259200 \
  --node-group "${LEPTON_NODE_GROUP:-az-sat-lepton-001}"
