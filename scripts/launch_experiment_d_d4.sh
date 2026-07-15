#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-1024}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-5}"
SEED="${SEED:-42}"
RUNTIME_LOG="$PROJECT_ROOT/runs/experiment_d_ablation/D4_no_matching/queue_runtime.tsv"

datasets=("NUAA-SIRST" "IRSTD-1K" "NUDT-SIRST")
IFS=',' read -r -a allowed_gpus <<< "$GPU_LIST"
declare -A pid_by_gpu=()
SELECTED_GPU=""

mkdir -p "$(dirname "$RUNTIME_LOG")"
if [[ ! -f "$RUNTIME_LOG" ]]; then
  printf 'timestamp\tid\tdataset\tsd_variant\tseed\tgpu\tpid\toutput_dir\tstatus\n' \
    > "$RUNTIME_LOG"
fi

refresh_jobs() {
  local gpu pid
  for gpu in "${!pid_by_gpu[@]}"; do
    pid="${pid_by_gpu[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
      unset 'pid_by_gpu[$gpu]'
    fi
  done
}

find_idle_gpu() {
  local gpu used util
  SELECTED_GPU=""
  refresh_jobs
  for gpu in "${allowed_gpus[@]}"; do
    [[ -n "${pid_by_gpu[$gpu]:-}" ]] && continue
    IFS=',' read -r used util < <(
      nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    if (( used <= MAX_USED_MIB && util <= MAX_UTIL_PERCENT )); then
      SELECTED_GPU="$gpu"
      return 0
    fi
  done
  return 1
}

output_is_active() {
  local output_dir="$1"
  ps -eo args= | grep -F 'train_experiment_d_hfe_ablation.py' | \
    grep -F -- "--output-dir $output_dir" | grep -v grep >/dev/null
}

launch_one() {
  local dataset="$1"
  local output_dir="runs/experiment_d_ablation/D4_no_matching/${dataset}/seed${SEED}"
  local gpu pid
  if output_is_active "$output_dir"; then
    printf '%s\tD4\t%s\tsd_awgm_hfe_nomatch\t%s\t-\t-\t%s\tskipped_active\n' \
      "$(date --iso-8601=seconds)" "$dataset" "$SEED" "$output_dir" | \
      tee -a "$RUNTIME_LOG"
    return
  fi
  if [[ -f "$output_dir/COMPLETED" ]]; then
    printf '%s\tD4\t%s\tsd_awgm_hfe_nomatch\t%s\t-\t-\t%s\tskipped_completed\n' \
      "$(date --iso-8601=seconds)" "$dataset" "$SEED" "$output_dir" | \
      tee -a "$RUNTIME_LOG"
    return
  fi
  until find_idle_gpu; do
    sleep "$POLL_SECONDS"
  done
  gpu="$SELECTED_GPU"
  mkdir -p "$output_dir"
  PATH="$(dirname "$PYTHON_BIN"):$PATH" PYTHON_BIN="$PYTHON_BIN" \
    bash scripts/run_experiment_d_hfe_ablation.sh \
      d4_no_matching "$dataset" "$gpu" "$SEED" "$DATASET_DIR" \
      > "$output_dir/launcher.log" 2>&1 &
  pid=$!
  pid_by_gpu[$gpu]="$pid"
  printf '%s\tD4\t%s\tsd_awgm_hfe_nomatch\t%s\t%s\t%s\t%s\trunning\n' \
    "$(date --iso-8601=seconds)" "$dataset" "$SEED" "$gpu" "$pid" \
    "$output_dir" | tee -a "$RUNTIME_LOG"
}

cd "$PROJECT_ROOT"
for dataset in "${datasets[@]}"; do
  launch_one "$dataset"
done

for gpu in "${!pid_by_gpu[@]}"; do
  wait "${pid_by_gpu[$gpu]}" || true
done
