#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_K_E1_CENTERED}"
PYTHON="${PYTHON:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-0,1,2,3,4,5,6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_CONCURRENT="${MAX_CONCURRENT:-7}"
SPLIT="${SPLIT:-train}"

QUEUE_ROOT="${ROOT}/analysis/experiment_k/diagnosis_queue_${SPLIT}"
mkdir -p "${QUEUE_ROOT}"
exec 9>"${QUEUE_ROOT}/queue.lock"
if ! flock -n 9; then
    echo "Experiment K diagnosis queue is already running for split=${SPLIT}." >&2
    exit 0
fi

echo "$$" > "${QUEUE_ROOT}/queue.pid"
trap 'rm -f "${QUEUE_ROOT}/queue.pid"' EXIT

IFS=',' read -r -a GPUS <<< "${GPU_ALLOWLIST}"
DATASETS=("NUAA-SIRST" "IRSTD-1K")
MODES=("fidelity" "prior" "treatment" "counterfactual" "mad")

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${QUEUE_ROOT}/queue.log"
}

output_dir() {
    local mode="$1" dataset="$2"
    case "${mode}" in
        fidelity)       echo "${ROOT}/analysis/experiment_k/compactness_fidelity/${dataset}/${SPLIT}" ;;
        prior)          echo "${ROOT}/analysis/experiment_k/prior_drift/${dataset}/${SPLIT}" ;;
        treatment)      echo "${ROOT}/analysis/experiment_k/treatment_effect/${dataset}/${SPLIT}" ;;
        counterfactual) echo "${ROOT}/analysis/experiment_k/j1_counterfactual/${dataset}/${SPLIT}" ;;
        mad)            echo "${ROOT}/analysis/experiment_k/mad_gaussian_audit/${dataset}/${SPLIT}" ;;
        *) return 1 ;;
    esac
}

e1_checkpoint() {
    local dataset="$1"
    echo "/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS/runs/experiment_e_lfss_awgm/E1_lfss_resblock/${dataset}/seed42/best.pth.tar"
}

j_checkpoint_map() {
    local dataset="$1"
    echo "${ROOT}/analysis/experiment_k/checkpoint_maps/${dataset}.json"
}

task_state() {
    local mode="$1" dataset="$2" out pid
    out="$(output_dir "${mode}" "${dataset}")"
    if [[ -f "${out}/DIAGNOSIS_COMPLETE" ]]; then echo complete; return; fi
    if [[ -f "${out}/FAILED" ]]; then echo failed; return; fi
    if [[ -f "${out}/launcher.pid" ]]; then
        pid="$(cat "${out}/launcher.pid" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then echo running; return; fi
    fi
    echo pending
}

gpu_is_idle() {
    local gpu="$1" mem util compute
    [[ -z "${RESERVED_GPUS[${gpu}]:-}" ]] || return 1
    IFS=',' read -r mem util < <(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    compute="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
    [[ "${mem}" -lt 1000 && "${util}" -lt 10 && -z "${compute}" ]]
}

running_count() {
    local count=0 mode dataset
    for dataset in "${DATASETS[@]}"; do
        for mode in "${MODES[@]}"; do
            [[ "$(task_state "${mode}" "${dataset}")" == running ]] && count=$((count + 1))
        done
    done
    echo "${count}"
}

dependency_ready() {
    local mode="$1" dataset="$2"
    if [[ "${mode}" == prior ]]; then
        [[ "$(task_state fidelity "${dataset}")" == complete ]]
        return
    fi
    if [[ "${mode}" == mad ]]; then
        [[ "$(task_state treatment "${dataset}")" == complete ]]
        return
    fi
    return 0
}

launch_task() {
    local mode="$1" dataset="$2" gpu="$3" out e1 map
    out="$(output_dir "${mode}" "${dataset}")"
    e1="$(e1_checkpoint "${dataset}")"
    map="$(j_checkpoint_map "${dataset}")"
    mkdir -p "${out}"
    rm -f "${out}/launcher.pid"
    (
        cd "${ROOT}"
        exec env PROJECT_ROOT="${ROOT}" PYTHON_BIN="${PYTHON}" DATASET_DIR="${DATASET_DIR}" \
            bash scripts/run_experiment_k_diagnosis.sh \
            "${mode}" "${dataset}" "${SPLIT}" "${gpu}" "${e1}" "${map}"
    ) >> "${out}/launcher.log" 2>&1 9>&- &
    local pid=$!
    echo "${pid}" > "${out}/launcher.pid"
    log "started mode=${mode} dataset=${dataset} split=${SPLIT} gpu=${gpu} pid=${pid}"
}

if [[ "${SPLIT}" == test ]]; then
    decision="${ROOT}/analysis/experiment_k/K_A_DECISION.json"
    if ! "${PYTHON}" - "${decision}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if not payload.get("discovery_complete", False):
    raise SystemExit(1)
PY
    then
        log "test confirmation is locked until discovery_complete=true in K_A_DECISION.json"
        exit 2
    fi
fi

log "queue started split=${SPLIT} gpus=${GPU_ALLOWLIST} max_concurrent=${MAX_CONCURRENT}"
while true; do
    declare -A RESERVED_GPUS=()
    total_terminal=0
    total_failed=0
    for mode in "${MODES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            state="$(task_state "${mode}" "${dataset}")"
            [[ "${state}" == complete || "${state}" == failed ]] && total_terminal=$((total_terminal + 1))
            [[ "${state}" == failed ]] && total_failed=$((total_failed + 1))
        done
    done
    expected_terminal=$((${#MODES[@]} * ${#DATASETS[@]}))
    if [[ "${total_terminal}" -eq "${expected_terminal}" ]]; then
        log "queue finished split=${SPLIT} failed=${total_failed}"
        exit "${total_failed}"
    fi

    active="$(running_count)"
    if [[ "${active}" -lt "${MAX_CONCURRENT}" ]]; then
        for mode in "${MODES[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                [[ "$(task_state "${mode}" "${dataset}")" == pending ]] || continue
                dependency_ready "${mode}" "${dataset}" || continue
                for gpu in "${GPUS[@]}"; do
                    if gpu_is_idle "${gpu}"; then
                        launch_task "${mode}" "${dataset}" "${gpu}"
                        RESERVED_GPUS["${gpu}"]=1
                        active=$((active + 1))
                        break
                    fi
                done
                [[ "${active}" -lt "${MAX_CONCURRENT}" ]] || break 2
            done
        done
    fi
    sleep "${POLL_SECONDS}"
done
