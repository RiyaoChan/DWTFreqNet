#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT/analysis/phase1_task_prior_validation}"
E1_ROOT="${E1_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS/runs/experiment_e_lfss_awgm/E1_lfss_resblock}"
H_ROOT="${H_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_H_DECODER_LFP/runs/experiment_h_decoder_lfp}"
PYTHON="${PYTHON:-python}"
POLL_SECONDS="${POLL_SECONDS:-30}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-1000}"
IDLE_UTILIZATION="${IDLE_UTILIZATION:-10}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
NUM_RANDOM_REPEATS="${NUM_RANDOM_REPEATS:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-$ANALYSIS_ROOT/metadata/formal_run_manifest.json}"

read_manifest_pid() {
  local field="$1"
  [[ -s "$FORMAL_MANIFEST" ]] || return 0
  "$PYTHON" - "$FORMAL_MANIFEST" "$field" <<'PY' 2>/dev/null || true
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(sys.argv[2], "")
print(value if isinstance(value, int) and value > 0 else "")
PY
}

P2_PRIMARY_PID="${P2_PRIMARY_PID:-$(read_manifest_pid p2_h_pid)}"
P3_PRIMARY_PID="${P3_PRIMARY_PID:-$(read_manifest_pid p3_pid)}"
P2_PRIMARY_GPU="${P2_PRIMARY_GPU:-0}"
P3_PRIMARY_GPU="${P3_PRIMARY_GPU:-5}"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCHEDULER_ROOT="$ANALYSIS_ROOT/idle_gpu_scheduler"
CLAIM_ROOT="$SCHEDULER_ROOT/claims"
SLOT_ROOT="$SCHEDULER_ROOT/slots"
TEMP_ROOT="$SCHEDULER_ROOT/task_outputs"
BACKUP_ROOT="$SCHEDULER_ROOT/replaced_outputs"
LOG_ROOT="$SCHEDULER_ROOT/logs"
STATE_ROOT="$SCHEDULER_ROOT/state"
MONITOR_LOG="$LOG_ROOT/monitor.log"

mkdir -p "$CLAIM_ROOT" "$SLOT_ROOT" "$TEMP_ROOT" "$BACKUP_ROOT" "$LOG_ROOT" "$STATE_ROOT"
export ROOT DATASET_DIR ANALYSIS_ROOT E1_ROOT H_ROOT PYTHON POLL_SECONDS
export IDLE_MEMORY_MIB IDLE_UTILIZATION BOOTSTRAP NUM_RANDOM_REPEATS MAX_SAMPLES
export FORMAL_MANIFEST P2_PRIMARY_PID P3_PRIMARY_PID P2_PRIMARY_GPU P3_PRIMARY_GPU

log_message() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MONITOR_LOG"
}

task_fields() {
  local key="$1" remainder
  TASK_FAMILY="${key%%__*}"
  remainder="${key#*__}"
  TASK_DATASET="${remainder%%__*}"
  TASK_SPLIT="${remainder#*__}"
  export TASK_FAMILY TASK_DATASET TASK_SPLIT
}

task_output() {
  local key="$1"
  task_fields "$key"
  case "$TASK_FAMILY" in
    p2) printf '%s\n' "$ANALYSIS_ROOT/P2_wavelet_consistency/$TASK_DATASET/$TASK_SPLIT" ;;
    p3) printf '%s\n' "$ANALYSIS_ROOT/P3_sampling_geometry/$TASK_DATASET/$TASK_SPLIT" ;;
    hcross) printf '%s\n' "$ANALYSIS_ROOT/H_cross_analysis/$TASK_DATASET" ;;
    *) return 1 ;;
  esac
}

h_cross_output_is_valid() {
  local dataset="$1"
  local output="$ANALYSIS_ROOT/H_cross_analysis/$dataset"
  local instances="$output/h_prior_alignment_instances.csv"
  [[ -s "$output/summary.json" && -s "$instances" ]] || return 1
  head -n 1 "$instances" | grep -q 'P1_R2' || return 1
  head -n 1 "$instances" | grep -q 'C_joint' || return 1
}

task_is_complete() {
  local key="$1" output
  task_fields "$key"
  output=$(task_output "$key")
  if [[ "$TASK_FAMILY" == "hcross" ]]; then
    h_cross_output_is_valid "$TASK_DATASET"
  else
    [[ -s "$output/summary.json" ]]
  fi
}

final_runner_is_active() {
  local key="$1" output
  output=$(task_output "$key")
  ps -u "$(id -un)" -o args= |
    grep -F -- "--output-dir $output" |
    grep -E 'tools/phase1/(validate_wavelet_directional_consistency|compare_sampling_geometries|analyze_h_lfp_prior_alignment)\.py' |
    grep -v grep >/dev/null
}

h_dataset_is_complete() {
  local dataset="$1" variant log
  for variant in \
    H1_rawll_attention H1_decoder_attention \
    H2_rawll_fixed_gaussian H2_decoder_fixed_gaussian \
    H3_rawll_adaptive_gaussian H3_decoder_adaptive_gaussian; do
    log="$H_ROOT/$variant/$dataset/seed42/train.log"
    [[ -s "$log" ]] || return 1
    "$PYTHON" - "$log" <<'PY' >/dev/null 2>&1 || return 1
import json
import pathlib
import sys

rows = []
for line in pathlib.Path(sys.argv[1]).read_text(errors="ignore").splitlines()[-5:]:
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
if not rows or int(rows[-1].get("epoch", 0)) != 1000:
    raise SystemExit(1)
PY
  done
}

task_dependencies_are_ready() {
  local key="$1"
  task_fields "$key"
  case "$TASK_FAMILY" in
    p2|p3)
      if [[ "$TASK_FAMILY" == "p3" && "$TASK_SPLIT" == "test" ]]; then
        [[ -s "$ANALYSIS_ROOT/P3_sampling_geometry/$TASK_DATASET/train/selected_radii.json" ]]
      else
        return 0
      fi
      ;;
    hcross)
      [[ -s "$ANALYSIS_ROOT/P1_gaussian_geometry/$TASK_DATASET/test/gaussian_instance_metrics.csv" ]] || return 1
      [[ -s "$ANALYSIS_ROOT/P2_wavelet_consistency/$TASK_DATASET/test/instance_consistency_metrics.csv" ]] || return 1
      h_dataset_is_complete "$TASK_DATASET"
      ;;
    *) return 1 ;;
  esac
}

gpu_is_idle() {
  local index="$1" line memory utilization compute_count
  if [[ "$index" == "$P2_PRIMARY_GPU" && -n "$P2_PRIMARY_PID" ]] &&
      kill -0 "$P2_PRIMARY_PID" 2>/dev/null; then
    return 1
  fi
  if [[ "$index" == "$P3_PRIMARY_GPU" && -n "$P3_PRIMARY_PID" ]] &&
      kill -0 "$P3_PRIMARY_PID" 2>/dev/null; then
    return 1
  fi
  line=$(nvidia-smi --id="$index" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  memory=$(echo "$line" | cut -d, -f1 | tr -d ' ')
  utilization=$(echo "$line" | cut -d, -f2 | tr -d ' ')
  compute_count=$(nvidia-smi pmon -c 1 | awk -v gpu="$index" '$1==gpu && $3=="C" {count++} END {print count+0}')
  (( memory < IDLE_MEMORY_MIB && utilization < IDLE_UTILIZATION && compute_count == 0 ))
}

slot_is_available() {
  local gpu="$1" slot="$SLOT_ROOT/gpu_${gpu}.pid" pid
  [[ -s "$slot" ]] || return 0
  pid=$(cat "$slot")
  if kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  rm -f "$slot"
  return 0
}

all_task_keys() {
  local dataset split
  for dataset in NUAA-SIRST IRSTD-1K NUDT-SIRST; do
    for split in train test; do
      printf 'p2__%s__%s\n' "$dataset" "$split"
    done
  done
  for dataset in NUAA-SIRST IRSTD-1K NUDT-SIRST; do
    printf 'p3__%s__train\n' "$dataset"
  done
  for dataset in NUAA-SIRST IRSTD-1K NUDT-SIRST; do
    printf 'p3__%s__test\n' "$dataset"
  done
  for dataset in NUAA-SIRST IRSTD-1K NUDT-SIRST; do
    printf 'hcross__%s__test\n' "$dataset"
  done
}

next_ready_task() {
  local key claim pid
  while read -r key; do
    task_is_complete "$key" && continue
    claim="$CLAIM_ROOT/$key"
    if [[ -d "$claim" ]]; then
      pid=""
      [[ -s "$claim/pid" ]] && pid=$(cat "$claim/pid")
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      rm -f "$claim/pid"
      rmdir "$claim" 2>/dev/null || continue
    fi
    final_runner_is_active "$key" && continue
    task_dependencies_are_ready "$key" || continue
    printf '%s\n' "$key"
    return 0
  done < <(all_task_keys)
  return 1
}

all_tasks_are_complete() {
  local key
  while read -r key; do
    task_is_complete "$key" || return 1
  done < <(all_task_keys)
}

make_h_checkpoint_map() {
  local dataset="$1" output="$2"
  "$PYTHON" - "$H_ROOT" "$dataset" "$output" <<'PY'
import json
import pathlib
import sys

root, dataset, output = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
mapping = {
    "h1_rawll_attention": root / "H1_rawll_attention" / dataset / "seed42" / "best.pth.tar",
    "h1_decoder_attention": root / "H1_decoder_attention" / dataset / "seed42" / "best.pth.tar",
    "h2_rawll_fixed_gaussian": root / "H2_rawll_fixed_gaussian" / dataset / "seed42" / "best.pth.tar",
    "h2_decoder_fixed_gaussian": root / "H2_decoder_fixed_gaussian" / dataset / "seed42" / "best.pth.tar",
    "h3_rawll_adaptive_gaussian": root / "H3_rawll_adaptive_gaussian" / dataset / "seed42" / "best.pth.tar",
    "h3_decoder_adaptive_gaussian": root / "H3_decoder_adaptive_gaussian" / dataset / "seed42" / "best.pth.tar",
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({key: str(value) for key, value in mapping.items()}, indent=2), encoding="utf-8")
PY
}

publish_output() {
  local key="$1" temporary="$2" final backup_stamp
  final=$(task_output "$key")
  mkdir -p "$(dirname "$final")"

  exec 8>"$SCHEDULER_ROOT/publish.lock"
  flock 8
  if task_is_complete "$key"; then
    log_message "$key completed by another runner; retaining scheduler output at $temporary"
    return 0
  fi
  if final_runner_is_active "$key"; then
    log_message "$key final-directory runner is active; retaining scheduler output at $temporary"
    return 0
  fi
  if [[ -e "$final" || -L "$final" ]]; then
    backup_stamp="$(date '+%Y%m%d_%H%M%S')"
    mv -- "$final" "$BACKUP_ROOT/${key}_${backup_stamp}"
  fi
  ln -s "$temporary" "$final"
  log_message "$key published from $temporary to $final"
}

run_task() {
  local key="$1" gpu="$2" timestamp temporary log_file map rc=0
  task_fields "$key"
  timestamp=$(date '+%Y%m%d_%H%M%S')
  temporary="$TEMP_ROOT/${key}_${timestamp}_gpu${gpu}_pid$$"
  log_file="$LOG_ROOT/${key}_${timestamp}_gpu${gpu}.log"
  mkdir -p "$temporary"
  cd "$ROOT"

  log_message "starting $key on GPU $gpu; temporary output $temporary"
  case "$TASK_FAMILY" in
    p2)
      "$PYTHON" tools/phase1/validate_wavelet_directional_consistency.py \
        --dataset-dir "$DATASET_DIR" --dataset-name "$TASK_DATASET" --split "$TASK_SPLIT" \
        --checkpoint "$E1_ROOT/$TASK_DATASET/seed42/best.pth.tar" \
        --device "cuda:$gpu" --output-dir "$temporary" \
        --bootstrap "$BOOTSTRAP" --max-samples "$MAX_SAMPLES" >"$log_file" 2>&1 || rc=$?
      ;;
    p3)
      args=()
      if [[ "$TASK_SPLIT" == "test" ]]; then
        args+=(--selected-radii "$ANALYSIS_ROOT/P3_sampling_geometry/$TASK_DATASET/train/selected_radii.json")
      fi
      "$PYTHON" tools/phase1/compare_sampling_geometries.py \
        --dataset-dir "$DATASET_DIR" --dataset-name "$TASK_DATASET" --split "$TASK_SPLIT" \
        --checkpoint "$E1_ROOT/$TASK_DATASET/seed42/best.pth.tar" \
        --device "cuda:$gpu" --output-dir "$temporary" \
        --num-random-repeats "$NUM_RANDOM_REPEATS" --max-samples "$MAX_SAMPLES" \
        "${args[@]}" >"$log_file" 2>&1 || rc=$?
      ;;
    hcross)
      map="$SCHEDULER_ROOT/H_checkpoint_map_${TASK_DATASET}.json"
      make_h_checkpoint_map "$TASK_DATASET" "$map"
      "$PYTHON" tools/phase1/analyze_h_lfp_prior_alignment.py \
        --dataset-dir "$DATASET_DIR" --dataset-name "$TASK_DATASET" --split test \
        --e1-checkpoint "$E1_ROOT/$TASK_DATASET/seed42/best.pth.tar" \
        --h-checkpoint-map "$map" --device "cuda:$gpu" --output-dir "$temporary" \
        --p1-metrics "$ANALYSIS_ROOT/P1_gaussian_geometry/$TASK_DATASET/test/gaussian_instance_metrics.csv" \
        --p2-metrics "$ANALYSIS_ROOT/P2_wavelet_consistency/$TASK_DATASET/test/instance_consistency_metrics.csv" \
        --max-samples "$MAX_SAMPLES" >"$log_file" 2>&1 || rc=$?
      ;;
    *) rc=64 ;;
  esac

  if (( rc == 0 )) && [[ -s "$temporary/summary.json" ]]; then
    publish_output "$key" "$temporary"
  else
    log_message "$key failed with code $rc; see $log_file"
  fi
  rm -f "$CLAIM_ROOT/$key/pid"
  rmdir "$CLAIM_ROOT/$key" 2>/dev/null || true
  local slot="$SLOT_ROOT/gpu_${gpu}.pid"
  if [[ -s "$slot" && "$(cat "$slot")" == "$$" ]]; then
    rm -f "$slot"
  fi
  return "$rc"
}

if [[ "${1:-}" == "--run-task" ]]; then
  run_task "$2" "$3"
  exit $?
fi

if [[ "${1:-}" == "--status" ]]; then
  while read -r status_key; do
    if task_is_complete "$status_key"; then
      printf 'complete %s\n' "$status_key"
    elif final_runner_is_active "$status_key"; then
      printf 'running  %s\n' "$status_key"
    elif task_dependencies_are_ready "$status_key"; then
      printf 'ready    %s\n' "$status_key"
    else
      printf 'blocked  %s\n' "$status_key"
    fi
  done < <(all_task_keys)
  exit 0
fi

exec 9>"$SCHEDULER_ROOT/monitor.lock"
if ! flock -n 9; then
  echo "A Phase 1 idle-GPU monitor is already running." >&2
  exit 2
fi

log_message "idle-GPU monitor started: memory<$IDLE_MEMORY_MIB MiB, utilization<$IDLE_UTILIZATION%, poll=${POLL_SECONDS}s"
while true; do
  if all_tasks_are_complete; then
    log_message "all GPU-capable Phase 1 tasks are complete; monitor exiting"
    exit 0
  fi

  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  for ((gpu=0; gpu<gpu_count; gpu++)); do
    slot_is_available "$gpu" || continue
    gpu_is_idle "$gpu" || continue
    key=$(next_ready_task || true)
    [[ -n "$key" ]] || continue
    mkdir "$CLAIM_ROOT/$key" 2>/dev/null || continue
    nohup bash "$SELF" --run-task "$key" "$gpu" \
      >"$LOG_ROOT/${key}_worker_stdout.log" 2>&1 < /dev/null &
    worker_pid=$!
    printf '%s\n' "$worker_pid" >"$CLAIM_ROOT/$key/pid"
    printf '%s\n' "$worker_pid" >"$SLOT_ROOT/gpu_${gpu}.pid"
    log_message "dispatched $key to GPU $gpu as PID $worker_pid"
  done
  sleep "$POLL_SECONDS"
done
