#!/usr/bin/env bash
# Run one replay stage without changing the original 200k-step LR horizon.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stage="${1:-}"
case "$stage" in
  stage0) target=2000 ;;
  stage1) target=8000 ;;
  *) echo "Usage: $0 {stage0|stage1}" >&2; exit 2 ;;
esac

config="$repo/reproduction/finetune/generated_configs/train_${stage}_to_step${target}_replay.toml"
if [[ "$stage" == stage1 ]]; then
  config="$repo/reproduction/finetune/generated_configs/train_stage2000_to_step8000_replay.toml"
  prior="$repo/reproduction/finetune/stage0/checkpoints/step_2000/policy/.rank_0_complete"
  [[ -f "$prior" ]] || { echo "Missing complete step-2000 checkpoint: $prior" >&2; exit 2; }
fi
[[ -f "$config" ]] || { echo "Missing rendered config: $config" >&2; exit 2; }

output="$repo/reproduction/finetune/$stage"
marker="$output/checkpoints/step_${target}/policy/.rank_0_complete"
train_python="${TRAIN_PYTHON_BIN:-/opt/venv/cosmos_rl/bin/python}"

split_final_adapter() {
  local adapter="$output/safetensors/step_8000"
  local deploy="$repo/reproduction/finetune/deploy"
  if [[ -s "$deploy/language/adapter_model.safetensors" &&
        -s "$deploy/language/adapter_config.json" &&
        -s "$deploy/visual/adapter_model.safetensors" &&
        -s "$deploy/visual/adapter_config.json" ]]; then
    echo "Step-8000 language and visual adapters already exported: $deploy"
    return
  fi
  [[ -s "$adapter/adapter_model.safetensors" ]] || {
    echo "Missing complete step-8000 LoRA export: $adapter" >&2; return 1;
  }
  if [[ -e "$deploy/language" || -e "$deploy/visual" ]]; then
    echo "Incomplete adapter split in $deploy; inspect or move those directories before retrying" >&2
    return 1
  fi
  "$train_python" "$repo/examples/pas_reranker/split_joint_lora_adapter.py" \
    --adapter "$adapter" \
    --language-output "$deploy/language" \
    --visual-output "$deploy/visual"
}

if [[ -f "$marker" ]]; then
  echo "Stage $stage already has a complete step-$target checkpoint: $marker"
  if [[ "$stage" == stage1 ]]; then split_final_adapter; fi
  exit 0
fi

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d /usr/local/cuda-13.2/compat/lib.real ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-13.2/compat/lib.real${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PAS_DISABLE_CUDNN_CONV=1
export COSMOS_LOCAL_STATIC_RDZV=1
export COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS=1
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$repo"
mkdir -p "$output"
"${COSMOS_RL_BIN:-/opt/venv/cosmos_rl/bin/cosmos-rl}" \
  --config "$config" examples/pas_reranker/tao_pas_reranker.py &
launcher_pid=$!
echo "Started $stage launcher PID $launcher_pid; waiting for $marker"

# The checkpoint completes every 500 steps. SIGUSR1 is Cosmos-RL's graceful
# stop signal; the launcher forwards it to all policy ranks after the chosen
# checkpoint exists. Stage 1 resumes the exact step-2000 optimizer state.
while kill -0 "$launcher_pid" 2>/dev/null; do
  if [[ -f "$marker" ]]; then
    if [[ "$stage" == stage1 ]]; then
      if ! "$train_python" "$repo/examples/pas_reranker/wait_adapter_export.py" \
        --adapter-dir "$output/safetensors/step_8000"; then
        kill -USR1 "$launcher_pid"
        wait "$launcher_pid" || true
        exit 1
      fi
    fi
    echo "Complete step-$target checkpoint found; requesting graceful stop"
    kill -USR1 "$launcher_pid"
    break
  fi
  sleep 10
done

status=0
wait "$launcher_pid" || status=$?
[[ -f "$marker" ]] || { echo "$stage ended without $marker (exit $status)" >&2; exit 1; }
echo "$stage finished with complete step-$target checkpoint (launcher exit $status)"

if [[ "$stage" == stage1 ]]; then
  split_final_adapter
fi
