#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_K_E1_CENTERED}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/experiment_k_e1_centered}"
DECISION_JSON="${DECISION_JSON:-${PROJECT_ROOT}/analysis/experiment_k/K_A_DECISION.json}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-0,1,2,3,4,5,6}"
MAX_CONCURRENT="${MAX_CONCURRENT:-7}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-1000}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-10}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/queue.instance.v1.lock"
if ! flock -n 9; then echo "Another Experiment K queue is active" >&2; exit 2; fi
QUEUE_LOG="${OUTPUT_ROOT}/queue.log"
RUNTIME_TSV="${OUTPUT_ROOT}/queue_runtime.tsv"
if [[ ! -s "${RUNTIME_TSV}" ]]; then
  printf 'timestamp\tvariant\tdataset\tseed\tgpu\twrapper_pid\toutput_dir\tstatus\n' > "${RUNTIME_TSV}"
fi
IFS=',' read -r -a GPUS <<< "${GPU_ALLOWLIST}"

output_name() {
  case "$1" in
    k2_dose_calibrated) printf K2_dose_calibrated ;;
    k3_gr_raw_all) printf K3_gr_raw_all ;;
    k4_gr_lfss_s123) printf K4_gr_lfss_s123 ;;
    k5_gr_guided_s123) printf K5_gr_guided_s123 ;;
    k6_gr_selected_hybrid) printf K6_gr_selected_hybrid ;;
  esac
}
task_dir() { printf '%s/%s/%s/seed%s' "${OUTPUT_ROOT}" "$(output_name "$1")" "$2" "${SEED}"; }
task_state() {
  local dir="$1" pid
  [[ -f "${dir}/TRAINING_COMPLETE" ]] && { printf complete; return; }
  [[ -f "${dir}/FAILED" ]] && { printf failed; return; }
  if [[ -f "${dir}/RUNNING.lock" ]]; then
    pid="$(tr -dc '0-9' < "${dir}/RUNNING.lock" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then printf running; else touch "${dir}/FAILED"; printf failed; fi
    return
  fi
  if [[ -s "${dir}/latest.pth.tar" || -s "${dir}/best.pth.tar" ]]; then touch "${dir}/FAILED"; printf failed; return; fi
  printf queued
}
load_tasks() {
  mapfile -t TASK_LINES < <("${PYTHON_BIN}" - "${DECISION_JSON}" <<'PY'
import json, os, sys
path = sys.argv[1]
decision = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
tasks = [("k2_dose_calibrated", "NUAA-SIRST"), ("k2_dose_calibrated", "IRSTD-1K")]
if decision.get("discovery_complete") and decision.get("confirmation_complete"):
    for variant in decision.get("variants_to_train", []):
        if variant == "k2_dose_calibrated":
            continue
        tasks.extend([(variant, "NUAA-SIRST"), (variant, "IRSTD-1K")])
for variant in decision.get("global_candidates_for_nudt", []):
    tasks.append((variant, "NUDT-SIRST"))
seen = set()
for task in tasks:
    if task not in seen:
        print("\t".join(task)); seen.add(task)
PY
  )
}
active_count() {
  local count=0 line variant dataset dir
  for line in "${TASK_LINES[@]}"; do
    IFS=$'\t' read -r variant dataset <<< "${line}"
    dir="$(task_dir "${variant}" "${dataset}")"
    [[ "$(task_state "${dir}")" == running ]] && count=$((count + 1))
  done
  printf '%s' "${count}"
}
gpu_reserved() {
  local gpu="$1" line variant dataset dir reserved
  for line in "${TASK_LINES[@]}"; do
    IFS=$'\t' read -r variant dataset <<< "${line}"
    dir="$(task_dir "${variant}" "${dataset}")"
    [[ "$(task_state "${dir}")" == running && -f "${dir}/gpu.id" ]] || continue
    reserved="$(tr -dc '0-9' < "${dir}/gpu.id" || true)"
    [[ "${reserved}" == "${gpu}" ]] && return 0
  done
  return 1
}
gpu_is_free() {
  local gpu="$1" stats used util pids
  gpu_reserved "${gpu}" && return 1
  stats="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" || return 1
  IFS=',' read -r used util <<< "${stats}"
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  [[ "${used}" -lt "${MAX_USED_MIB}" && "${util}" -lt "${MAX_UTIL_PERCENT}" && -z "${pids}" ]]
}

printf '%s\tqueue_started\tallowlist=%s\tmax_concurrent=%s\n' "$(date --iso-8601=seconds)" "${GPU_ALLOWLIST}" "${MAX_CONCURRENT}" >> "${QUEUE_LOG}"
while true; do
  load_tasks
  active="$(active_count)"
  if [[ "${active}" -lt "${MAX_CONCURRENT}" ]]; then
    for line in "${TASK_LINES[@]}"; do
      IFS=$'\t' read -r variant dataset <<< "${line}"
      dir="$(task_dir "${variant}" "${dataset}")"
      [[ "$(task_state "${dir}")" == queued ]] || continue
      selected_gpu=""
      for gpu in "${GPUS[@]}"; do
        if gpu_is_free "${gpu}"; then selected_gpu="${gpu}"; break; fi
      done
      [[ -n "${selected_gpu}" ]] || break
      PROJECT_ROOT="${PROJECT_ROOT}" DATASET_DIR="${DATASET_DIR}" PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${OUTPUT_ROOT}" DECISION_JSON="${DECISION_JSON}" \
        nohup bash "${PROJECT_ROOT}/scripts/run_experiment_k.sh" "${variant}" "${dataset}" "${selected_gpu}" "${SEED}" >/dev/null 2>&1 9>&- &
      wrapper_pid=$!
      timestamp="$(date --iso-8601=seconds)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\trunning\n' "${timestamp}" "${variant}" "${dataset}" "${SEED}" "${selected_gpu}" "${wrapper_pid}" "${dir}" >> "${RUNTIME_TSV}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\trunning\n' "${timestamp}" "${variant}" "${dataset}" "${SEED}" "${selected_gpu}" "${wrapper_pid}" "${dir}" >> "${QUEUE_LOG}"
      active=$((active + 1))
      [[ "${active}" -ge "${MAX_CONCURRENT}" ]] && break
      sleep 2
    done
  fi
  sleep "${POLL_SECONDS}"
done
