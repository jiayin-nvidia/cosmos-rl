#!/usr/bin/env bash
# Submit a 16- or 32-A100 replay stage after rendering its TOMLs.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ranks="${PAS_DATA_PARALLEL_RANKS:-16}"
[[ "$ranks" == 16 || "$ranks" == 32 ]] || {
  echo "PAS_DATA_PARALLEL_RANKS must be 16 or 32" >&2; exit 2;
}
run_root="$repo/reproduction/finetune/fast$ranks"
workers=$((ranks / 8))
stage="${1:-}"
case "$stage" in
  stage0) config=train_stage0_to_step2000_replay.toml ;;
  stage1)
    config=train_stage2000_to_step8000_replay.toml
    marker="$run_root/stage0/checkpoints/step_2000/policy/.rank_0_complete"
    [[ -f "$marker" ]] || { echo "Missing complete step-2000 checkpoint: $marker" >&2; exit 2; }
    ;;
  *) echo "Usage: $0 {stage0|stage1}" >&2; exit 2 ;;
esac

config="$repo/reproduction/finetune/fast${ranks}_configs/$config"
[[ -f "$config" ]] || { echo "Render the fast$ranks TOMLs first: $config" >&2; exit 2; }
job_name="${JOB_NAME:-pas-cr3-fast${ranks}-${stage}-$(date -u +%m%d%H%M%S)}"
image="${LEPTON_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.1.0-cosmos-rl}"
node_group="${LEPTON_NODE_GROUP:-az-sat-lepton-001}"
mount="${LEPTON_MOUNT:-/edgeai_metropolis_metropolis-2-0:/workspace:node-nfs:amlfs}"
pull_secret="${LEPTON_IMAGE_PULL_SECRET:-staging-new}"

cd "$repo"
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" \
  "${COSMOS_RL_BIN:-/opt/venv/cosmos_rl/bin/cosmos-rl}" \
  --config "$config" --num-workers "$workers" --lepton-mode \
  --lepton-job-name "$job_name" \
  --lepton-container-image "$image" \
  --lepton-resource-shape gpu.8xa100-80gb \
  --lepton-node-group "$node_group" \
  --lepton-image-pull-secrets "$pull_secret" \
  --lepton-mount "$mount" \
  --lepton-shared-memory-size 65536 \
  --lepton-log-collection true \
  --lepton-ttl-seconds-after-finished 259200 \
  --lepton-queue-priority 4 \
  --lepton-env "PYTHONPATH=$repo" \
  --lepton-env WANDB_MODE=offline \
  --lepton-env PAS_DISABLE_CUDNN_CONV=1 \
  --lepton-env COSMOS_LOCAL_STATIC_RDZV=1 \
  --lepton-env COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS=1 \
  --lepton-env CUDA_HOME=/usr/local/cuda-13.2 \
  --lepton-env LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat/lib.real \
  "$repo/examples/pas_reranker/tao_pas_reranker.py"
