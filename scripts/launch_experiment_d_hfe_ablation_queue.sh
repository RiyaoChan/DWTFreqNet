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
RUNTIME_LOG="$PROJECT_ROOT/runs/experiment_d_ablation/queue_runtime.tsv"

datasets=("NUAA-SIRST" "IRSTD-1K" "NUDT-SIRST")
IFS=',' read -r -a allowed_gpus <<< "$GPU_LIST"
declare -A scheduler_pid_by_gpu=()
SELECTED_GPU=""

mkdir -p "$(dirname "$RUNTIME_LOG")"
if [[ ! -f "$RUNTIME_LOG" ]]; then
  printf 'timestamp\tid\tdataset\tsd_variant\tseed\tgpu\tpid\toutput_dir\tstatus\n' \
    > "$RUNTIME_LOG"
fi

variant_dir() {
  case "$1" in
    d2_softcos_all) printf 'D2_softcos_all' ;;
    d3_scaleaware) printf 'D3_scaleaware' ;;
    *) return 2 ;;
  esac
}

sd_variant() {
  case "$1" in
    d2_softcos_all) printf 'sd_awgm_hfe_softcos' ;;
    d3_scaleaware) printf 'sd_awgm_hfe_scaleaware' ;;
    *) return 2 ;;
  esac
}

ablation_id() {
  case "$1" in
    d2_softcos_all) printf 'D2' ;;
    d3_scaleaware) printf 'D3' ;;
    *) return 2 ;;
  esac
}

output_dir_for() {
  local ablation="$1" dataset="$2"
  printf 'runs/experiment_d_ablation/%s/%s/seed%s' \
    "$(variant_dir "$ablation")" "$dataset" "$SEED"
}

refresh_running_jobs() {
  local gpu pid
  for gpu in "${!scheduler_pid_by_gpu[@]}"; do
    pid="${scheduler_pid_by_gpu[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
      unset 'scheduler_pid_by_gpu[$gpu]'
    fi
  done
}

find_idle_gpu() {
  local gpu used util
  SELECTED_GPU=""
  refresh_running_jobs
  for gpu in "${allowed_gpus[@]}"; do
    [[ -n "${scheduler_pid_by_gpu[$gpu]:-}" ]] && continue
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

launch_task() {
  local ablation="$1" dataset="$2"
  local output_dir variant id gpu pid
  output_dir="$(output_dir_for "$ablation" "$dataset")"
  variant="$(sd_variant "$ablation")"
  id="$(ablation_id "$ablation")"

  if output_is_active "$output_dir"; then
    printf '%s\t%s\t%s\t%s\t%s\t-\t-\t%s\tskipped_active\n' \
      "$(date --iso-8601=seconds)" "$id" "$dataset" "$variant" \
      "$SEED" "$output_dir" | tee -a "$RUNTIME_LOG"
    return
  fi
  if [[ -f "$output_dir/COMPLETED" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t-\t-\t%s\tskipped_completed\n' \
      "$(date --iso-8601=seconds)" "$id" "$dataset" "$variant" \
      "$SEED" "$output_dir" | tee -a "$RUNTIME_LOG"
    return
  fi

  until find_idle_gpu; do
    sleep "$POLL_SECONDS"
  done
  gpu="$SELECTED_GPU"
  mkdir -p "$output_dir"
  PATH="$(dirname "$PYTHON_BIN"):$PATH" PYTHON_BIN="$PYTHON_BIN" \
    bash scripts/run_experiment_d_hfe_ablation.sh \
      "$ablation" "$dataset" "$gpu" "$SEED" "$DATASET_DIR" \
      > "$output_dir/launcher.log" 2>&1 &
  pid=$!
  scheduler_pid_by_gpu[$gpu]="$pid"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\trunning\n' \
    "$(date --iso-8601=seconds)" "$id" "$dataset" "$variant" \
    "$SEED" "$gpu" "$pid" "$output_dir" | tee -a "$RUNTIME_LOG"
}

d2_all_usable() {
  local dataset output_dir
  for dataset in "${datasets[@]}"; do
    output_dir="$(output_dir_for d2_softcos_all "$dataset")"
    if [[ -f "$output_dir/status.tsv" ]] && \
      tail -n 1 "$output_dir/status.tsv" | grep -qE \
        'failed|oom_batch4|non_finite'; then
      return 2
    fi
    [[ -s "$output_dir/best_metrics.json" ]] || return 1
    if grep -qiE 'non-finite|out of memory|CUDA.*memory' \
      "$output_dir/train.log" 2>/dev/null; then
      return 2
    fi
  done
  return 0
}

cd "$PROJECT_ROOT"

# Phase 1: D2 is always scheduled first in NUAA, IRSTD, NUDT order.
for dataset in "${datasets[@]}"; do
  launch_task d2_softcos_all "$dataset"
done

# Global usability barrier: all three D2 runs must produce their first best
# checkpoint (evaluation starts at epoch 100) without NaN, OOM or process failure.
while true; do
  set +e
  d2_all_usable
  barrier_status=$?
  set -e
  if (( barrier_status == 0 )); then
    printf '%s\tD2-BARRIER\tALL\tsd_awgm_hfe_softcos\t%s\t-\t-\t-\tpassed\n' \
      "$(date --iso-8601=seconds)" "$SEED" | tee -a "$RUNTIME_LOG"
    break
  fi
  if (( barrier_status == 2 )); then
    printf '%s\tD2-BARRIER\tALL\tsd_awgm_hfe_softcos\t%s\t-\t-\t-\tfailed\n' \
      "$(date --iso-8601=seconds)" "$SEED" | tee -a "$RUNTIME_LOG"
    exit 1
  fi
  refresh_running_jobs
  sleep "$POLL_SECONDS"
done

# Phase 2: D3 becomes eligible only after the D2 usability barrier passes.
for dataset in "${datasets[@]}"; do
  launch_task d3_scaleaware "$dataset"
done

for gpu in "${!scheduler_pid_by_gpu[@]}"; do
  wait "${scheduler_pid_by_gpu[$gpu]}" || true
done
