#!/usr/bin/env bash
set -uo pipefail

# Queue the W8M variants that are not part of the first six full runs.  The
# scheduler never assumes a fixed GPU map; it only claims cards that are idle
# according to nvidia-smi and leaves existing processes untouched.
PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_DIR=${DATASET_DIR:-$PROJECT/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/runs/w8m_haar_aligned_full}
RUNNER=${RUNNER:-$PROJECT/scripts/run_w8m_stage1_dataset.sh}
IDLE_MEMORY_MB=${IDLE_MEMORY_MB:-2000}
IDLE_UTILIZATION=${IDLE_UTILIZATION:-5}
POLL_SECONDS=${POLL_SECONDS:-30}

# Priority order from DIAGONAL_WAVELET_MAMBA_EXPERIMENT_SPEC_FOR_CODEX.md:
# first dir-embedding, then diag2/pair, then independent/all.  Each variant
# is repeated for NUDT, NUAA, and IRSTD so every queued run covers all datasets.
TASK_DATASETS=(
  NUDT-SIRST NUAA-SIRST IRSTD-1K
  NUDT-SIRST NUAA-SIRST IRSTD-1K
  NUDT-SIRST NUAA-SIRST IRSTD-1K
  NUDT-SIRST NUAA-SIRST IRSTD-1K
  NUDT-SIRST NUAA-SIRST IRSTD-1K
)
TASK_VARIANTS=(
  w8m_diag4_axial_diag_shared_dir_embed w8m_diag4_axial_diag_shared_dir_embed w8m_diag4_axial_diag_shared_dir_embed
  w8m_diag2_subband_shared w8m_diag2_subband_shared w8m_diag2_subband_shared
  w8m_diag4_pair_shared w8m_diag4_pair_shared w8m_diag4_pair_shared
  w8m_diag4_independent w8m_diag4_independent w8m_diag4_independent
  w8m_diag4_all_shared w8m_diag4_all_shared w8m_diag4_all_shared
)

if [ "${#TASK_DATASETS[@]}" -ne "${#TASK_VARIANTS[@]}" ]; then
  echo "task list length mismatch" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
declare -A ACTIVE_PID ACTIVE_TASK
next_task=0

active_count() {
  set +u
  local count=${#ACTIVE_PID[@]}
  set -u
  printf '%s' "$count"
}

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
  echo "[$(date --iso-8601=seconds)] queue next=$next_task/${#TASK_DATASETS[@]} active=$(active_count)"
  [ "$next_task" -lt "${#TASK_DATASETS[@]}" ] || [ "$(active_count)" -gt 0 ] || break
  sleep "$POLL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] all missing W8M variants finished"
