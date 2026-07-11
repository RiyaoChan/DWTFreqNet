#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/SIRST-5K-main/dataset}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-1024}"
RUNTIME_LOG="$PROJECT_ROOT/runs/experiment_a_v2/queue_runtime.tsv"

tasks=(
  "A0-NUAA|NUAA-SIRST|dwtfreqnet_original"
  "A1-NUAA|NUAA-SIRST|dwtfreqnet_wulle_a"
  "A0-NUDT|NUDT-SIRST|dwtfreqnet_original"
  "A0-IRSTD|IRSTD-1k|dwtfreqnet_original"
  "A1-NUDT|NUDT-SIRST|dwtfreqnet_wulle_a"
  "A1-IRSTD|IRSTD-1k|dwtfreqnet_wulle_a"
)

IFS=',' read -r -a allowed_gpus <<< "$GPU_LIST"
declare -A scheduler_pid_by_gpu=()

mkdir -p "$(dirname "$RUNTIME_LOG")"
if [[ ! -f "$RUNTIME_LOG" ]]; then
  printf 'timestamp\tid\tdataset\tmodel_variant\tgpu\tpid\toutput_dir\n' > "$RUNTIME_LOG"
fi

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
  refresh_running_jobs
  for gpu in "${allowed_gpus[@]}"; do
    [[ -n "${scheduler_pid_by_gpu[$gpu]:-}" ]] && continue
    IFS=',' read -r used util < <(
      nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    if (( used <= MAX_USED_MIB && util <= 5 )); then
      printf '%s\n' "$gpu"
      return 0
    fi
  done
  return 1
}

cd "$PROJECT_ROOT"
for task in "${tasks[@]}"; do
  IFS='|' read -r experiment_id dataset model_variant <<< "$task"
  gpu=""
  until gpu="$(find_idle_gpu)"; do
    sleep "$POLL_SECONDS"
  done

  output_dir="runs/experiment_a_v2/$dataset/$model_variant/awgm_original"
  mkdir -p "$output_dir"
  PATH="$(dirname "$PYTHON_BIN"):$PATH" \
    bash scripts/run_experiment_a_v2.sh \
      "$dataset" "$model_variant" awgm_original "$gpu" "$DATASET_DIR" \
      > "$output_dir/launcher.log" 2>&1 &
  pid=$!
  scheduler_pid_by_gpu[$gpu]="$pid"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$experiment_id" "$dataset" \
    "$model_variant" "$gpu" "$pid" "$output_dir" | tee -a "$RUNTIME_LOG"
done

for gpu in "${!scheduler_pid_by_gpu[@]}"; do
  wait "${scheduler_pid_by_gpu[$gpu]}" || true
done
