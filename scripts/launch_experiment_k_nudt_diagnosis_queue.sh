#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_K_E1_CENTERED}"
PYTHON="${PYTHON:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
J_ROOT="${J_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-0,1,2,3,4,5,6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_CONCURRENT="${MAX_CONCURRENT:-7}"
DATASET="NUDT-SIRST"
SPLIT="train"

E1_CHECKPOINT="/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS/runs/experiment_e_lfss_awgm/E1_lfss_resblock/${DATASET}/seed42/best.pth.tar"
IMMEDIATE_MAP="${ROOT}/analysis/experiment_k/checkpoint_maps/${DATASET}.immediate.json"
J2J3_MAP="${ROOT}/analysis/experiment_k/checkpoint_maps/${DATASET}.j2j3.json"
QUEUE_ROOT="${ROOT}/analysis/experiment_k/nudt_staged/queue"

# A5 is intentionally first because it is much shorter than the two dense
# sweeps. A6 and A7 are dependency-gated so they cannot consume incomplete
# fidelity/treatment outputs.
TASK_STAGES=(
    "immediate" "immediate" "immediate" "immediate" "immediate"
    "j2j3_completed" "j2j3_completed"
)
TASK_MODES=(
    "treatment" "fidelity" "counterfactual" "prior" "mad"
    "fidelity" "prior"
)
TASK_LABELS=(
    "A5_J1" "A1_E1_J1" "A3_A4_J1" "A6_E1_J1" "A7_J1"
    "A1_A2_J2_J3" "A6_J2_J3"
)

mkdir -p "${QUEUE_ROOT}"
# v2 adds A6/A7. The original queue's already-running A5 child may retain the
# v1 lock descriptor after the parent scheduler is replaced, so use a
# versioned scheduler lock without interrupting that child.
exec 9>"${QUEUE_ROOT}/queue.v2.lock"
if ! flock -n 9; then
    echo "Experiment K NUDT staged diagnosis queue is already running." >&2
    exit 0
fi
echo "$$" > "${QUEUE_ROOT}/queue.pid"
trap 'rm -f "${QUEUE_ROOT}/queue.pid"' EXIT

IFS=',' read -r -a GPUS <<< "${GPU_ALLOWLIST}"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${QUEUE_ROOT}/queue.log"
}

mode_dir() {
    case "$1" in
        fidelity) echo compactness_fidelity ;;
        prior) echo prior_drift ;;
        treatment) echo treatment_effect ;;
        counterfactual) echo j1_counterfactual ;;
        mad) echo mad_gaussian_audit ;;
        *) return 1 ;;
    esac
}

stage_root() {
    echo "${ROOT}/analysis/experiment_k/nudt_staged/$1"
}

output_dir() {
    local stage="$1" mode="$2"
    echo "$(stage_root "${stage}")/$(mode_dir "${mode}")/${DATASET}/${SPLIT}"
}

checkpoint_map() {
    if [[ "$1" == immediate ]]; then
        echo "${IMMEDIATE_MAP}"
    else
        echo "${J2J3_MAP}"
    fi
}

j_output_name() {
    case "$1" in
        j1_bandwise_noise_calibrated) echo J1_bandwise_noise_calibrated ;;
        j2_rawll_compactness) echo J2_rawll_compactness ;;
        j2_decoder_compactness) echo J2_decoder_compactness ;;
        j3_dual_evidence_fixed) echo J3_dual_evidence_fixed ;;
        j3_dual_evidence_reliability) echo J3_dual_evidence_reliability ;;
        *) return 1 ;;
    esac
}

j_task_complete() {
    local variant="$1" name out
    name="$(j_output_name "${variant}")"
    out="${J_ROOT}/${name}/${DATASET}/seed42"
    [[ -f "${out}/best.pth.tar" ]] || return 1
    [[ -f "${out}/status.tsv" ]] || return 1
    grep -qi $'\tcomplete' "${out}/status.tsv"
}

stage_ready() {
    local stage="$1" variant
    [[ -f "${E1_CHECKPOINT}" ]] || return 1
    if [[ "${stage}" == immediate ]]; then
        j_task_complete j1_bandwise_noise_calibrated
        return
    fi
    for variant in \
        j2_rawll_compactness \
        j2_decoder_compactness \
        j3_dual_evidence_fixed \
        j3_dual_evidence_reliability
    do
        j_task_complete "${variant}" || return 1
    done
}

dependency_ready() {
    local index="$1" stage mode prerequisite
    stage="${TASK_STAGES[index]}"
    mode="${TASK_MODES[index]}"
    stage_ready "${stage}" || return 1
    case "${mode}" in
        prior)
            prerequisite="$(output_dir "${stage}" fidelity)"
            [[ -f "${prerequisite}/DIAGNOSIS_COMPLETE" ]]
            ;;
        mad)
            prerequisite="$(output_dir "${stage}" treatment)"
            [[ -f "${prerequisite}/DIAGNOSIS_COMPLETE" ]]
            ;;
        *)
            return 0
            ;;
    esac
}

task_state() {
    local index="$1" out pid
    out="$(output_dir "${TASK_STAGES[index]}" "${TASK_MODES[index]}")"
    if [[ -f "${out}/DIAGNOSIS_COMPLETE" ]]; then echo complete; return; fi
    if [[ -f "${out}/FAILED" || -f "${out}/ORPHANED" ]]; then echo failed; return; fi
    if [[ -f "${out}/launcher.pid" ]]; then
        pid="$(cat "${out}/launcher.pid" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo running
            return
        fi
        if find "${out}" -maxdepth 1 -type f \
            ! -name launcher.pid ! -name launcher.log ! -name diagnosis.instance.v1.lock \
            -print -quit 2>/dev/null | grep -q .; then
            touch "${out}/ORPHANED"
            log "orphaned label=${TASK_LABELS[index]} output=${out}; preserving partial output"
            echo failed
            return
        fi
    fi
    echo pending
}

gpu_is_idle() {
    local gpu="$1" mem util compute
    [[ -z "${RESERVED_GPUS[${gpu}]:-}" ]] || return 1
    IFS=',' read -r mem util < <(
        nvidia-smi -i "${gpu}" \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits | tr -d ' '
    )
    compute="$(
        nvidia-smi -i "${gpu}" \
            --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true
    )"
    [[ "${mem}" -lt 1000 && "${util}" -lt 10 && -z "${compute}" ]]
}

running_count() {
    local count=0 index
    for index in "${!TASK_LABELS[@]}"; do
        [[ "$(task_state "${index}")" == running ]] && count=$((count + 1))
    done
    echo "${count}"
}

launch_task() {
    local index="$1" gpu="$2" stage mode label out map
    stage="${TASK_STAGES[index]}"
    mode="${TASK_MODES[index]}"
    label="${TASK_LABELS[index]}"
    out="$(output_dir "${stage}" "${mode}")"
    map="$(checkpoint_map "${stage}")"
    mkdir -p "${out}"
    (
        cd "${ROOT}"
        exec env \
            PROJECT_ROOT="${ROOT}" \
            PYTHON_BIN="${PYTHON}" \
            DATASET_DIR="${DATASET_DIR}" \
            OUTPUT_ROOT="$(stage_root "${stage}")" \
            bash scripts/run_experiment_k_diagnosis.sh \
                "${mode}" "${DATASET}" "${SPLIT}" "${gpu}" \
                "${E1_CHECKPOINT}" "${map}"
    ) >> "${out}/launcher.log" 2>&1 9>&- &
    local pid=$!
    echo "${pid}" > "${out}/launcher.pid"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${stage}" "${mode}" \
        "${gpu}" "${pid}" "${out}" >> "${QUEUE_ROOT}/queue_runtime.tsv"
    log "started label=${label} stage=${stage} mode=${mode} gpu=${gpu} pid=${pid}"
}

log "queue started dataset=${DATASET} gpus=${GPU_ALLOWLIST} max_concurrent=${MAX_CONCURRENT}"
while true; do
    declare -A RESERVED_GPUS=()
    terminal=0
    failed=0
    for index in "${!TASK_LABELS[@]}"; do
        state="$(task_state "${index}")"
        [[ "${state}" == complete || "${state}" == failed ]] && terminal=$((terminal + 1))
        [[ "${state}" == failed ]] && failed=$((failed + 1))
    done
    if [[ "${terminal}" -eq "${#TASK_LABELS[@]}" ]]; then
        log "queue finished failed=${failed}"
        exit "${failed}"
    fi

    active="$(running_count)"
    if [[ "${active}" -lt "${MAX_CONCURRENT}" ]]; then
        for index in "${!TASK_LABELS[@]}"; do
            [[ "$(task_state "${index}")" == pending ]] || continue
            dependency_ready "${index}" || continue
            for gpu in "${GPUS[@]}"; do
                if gpu_is_idle "${gpu}"; then
                    launch_task "${index}" "${gpu}"
                    RESERVED_GPUS["${gpu}"]=1
                    active=$((active + 1))
                    break
                fi
            done
            [[ "${active}" -lt "${MAX_CONCURRENT}" ]] || break
        done
    fi
    sleep "${POLL_SECONDS}"
done
