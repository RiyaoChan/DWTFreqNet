#!/usr/bin/env bash
set -uo pipefail

# Dynamically schedule the six two-dataset DM-AWGM ablations.  A GPU is
# considered idle only when both its allocated memory and utilization are low;
# this avoids taking a card that is busy with an unrelated experiment.
PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_DIR=${DATASET_DIR:-$PROJECT/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/runs/ablation_two_datasets}
RUNNER=${RUNNER:-$PROJECT/scripts/run_dm_awgm_ablation_dataset.sh}
IDLE_MEMORY_MB=${IDLE_MEMORY_MB:-2000}
IDLE_UTILIZATION=${IDLE_UTILIZATION:-5}
POLL_SECONDS=${POLL_SECONDS:-30}

TASK_DATASETS=(
  NUAA-SIRST NUAA-SIRST NUAA-SIRST
  IRSTD-1K IRSTD-1K IRSTD-1K
)
TASK_VARIANTS=(
  dm_awgm_no_mamba dm_awgm_no_dcn dm_awgm_conv_only
  dm_awgm_no_mamba dm_awgm_no_dcn dm_awgm_conv_only
)

if [ "${#TASK_DATASETS[@]}" -ne "${#TASK_VARIANTS[@]}" ]; then
  echo "task list length mismatch" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
declare -A ACTIVE_PID ACTIVE_TASK ACTIVE_GPU
next_task=0

idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v max_mem="$IDLE_MEMORY_MB" -v max_util="$IDLE_UTILIZATION" \
      '{g=$1+0; mem=$2+0; util=$3+0; if (mem <= max_mem && util <= max_util) print g}'
}

task_output() {
  printf '%s/%s/%s' "$OUTPUT_ROOT" "$1" "$2"
}

task_running() {
  local output=$1
  ps -eo args= | awk -v out="$output" \
    'index($0, "train_one.py") && index($0, "--output-dir " out) {found=1} END {exit !found}'
}

start_task() {
  local index=$1 gpu=$2 dataset=${TASK_DATASETS[$1]} variant=${TASK_VARIANTS[$1]}
  local output
  output=$(task_output "$dataset" "$variant")
  mkdir -p "$output"
  if [ -f "$output/COMPLETED" ]; then
    echo "[$(date --iso-8601=seconds)] skip completed $dataset $variant"
    return 0
  fi
  if task_running "$output"; then
    echo "[$(date --iso-8601=seconds)] skip already-running $dataset $variant"
    return 0
  fi
  echo "[$(date --iso-8601=seconds)] launch task=$index gpu=$gpu dataset=$dataset variant=$variant"
  PROJECT="$PROJECT" PYTHON_BIN="$PYTHON_BIN" DATASET_DIR="$DATASET_DIR" \
    OUTPUT_ROOT="$OUTPUT_ROOT" "$RUNNER" "$dataset" "$variant" "$gpu" 1000 \
    >> "$output/scheduler.log" 2>&1 &
  ACTIVE_PID[$gpu]=$!
  ACTIVE_TASK[$gpu]=$index
  ACTIVE_GPU[$index]=$gpu
}

reap_finished() {
  local gpu pid task status state
  for gpu in "${!ACTIVE_PID[@]}"; do
    pid=${ACTIVE_PID[$gpu]}
    state=$(ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]')
    if [ -n "$state" ] && [[ "$state" != Z* ]]; then
      continue
    fi
    task=${ACTIVE_TASK[$gpu]}
    wait "$pid" 2>/dev/null
    status=$?
    echo "[$(date --iso-8601=seconds)] finished task=$task gpu=$gpu status=$status"
    unset 'ACTIVE_PID[$gpu]' 'ACTIVE_TASK[$gpu]'
    unset 'ACTIVE_GPU[$task]'
  done
}

while [ "$next_task" -lt "${#TASK_DATASETS[@]}" ] || [ "${#ACTIVE_PID[@]}" -gt 0 ]; do
  reap_finished
  while read -r gpu; do
    [ -n "$gpu" ] || continue
    [ -n "${ACTIVE_PID[$gpu]+set}" ] && continue
    [ "$next_task" -lt "${#TASK_DATASETS[@]}" ] || break
    start_task "$next_task" "$gpu"
    next_task=$((next_task + 1))
  done < <(idle_gpus)
  echo "[$(date --iso-8601=seconds)] queue next=$next_task/${#TASK_DATASETS[@]} active=${#ACTIVE_PID[@]}"
  [ "$next_task" -lt "${#TASK_DATASETS[@]}" ] || [ "${#ACTIVE_PID[@]}" -gt 0 ] || break
  sleep "$POLL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] all queued DM-AWGM ablations finished"
