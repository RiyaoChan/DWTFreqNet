#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?DENP variant is required}"
DATASET="${2:?dataset name is required}"
GPU="${3:?GPU id is required}"
SEED="${4:-42}"

PROJECT_ROOT="${PROJECT_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/experiment_j_denp}"

case "${VARIANT}" in
  j1_bandwise_noise_calibrated) OUTPUT_NAME="J1_bandwise_noise_calibrated" ;;
  j2_rawll_compactness) OUTPUT_NAME="J2_rawll_compactness" ;;
  j2_decoder_compactness) OUTPUT_NAME="J2_decoder_compactness" ;;
  j3_dual_evidence_fixed) OUTPUT_NAME="J3_dual_evidence_fixed" ;;
  j3_dual_evidence_reliability) OUTPUT_NAME="J3_dual_evidence_reliability" ;;
  *) echo "J0 is regression-only or unknown Experiment J variant: ${VARIANT}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}/${DATASET}/seed${SEED}"
RUNNING_LOCK="${OUTPUT_DIR}/RUNNING.lock"
FAILED_MARKER="${OUTPUT_DIR}/FAILED"
COMPLETE_MARKER="${OUTPUT_DIR}/TRAINING_COMPLETE"
mkdir -p "${OUTPUT_DIR}"
exec 8>"${OUTPUT_DIR}/task.instance.v1.lock"
if ! flock -n 8; then echo "Task already claimed: ${OUTPUT_DIR}" >&2; exit 2; fi
if [[ -f "${COMPLETE_MARKER}" ]]; then echo "Already complete: ${OUTPUT_DIR}"; exit 0; fi
if [[ -f "${FAILED_MARKER}" ]]; then echo "FAILED marker exists: ${OUTPUT_DIR}" >&2; exit 3; fi
if [[ -f "${RUNNING_LOCK}" ]]; then
  EXISTING_PID="$(tr -dc '0-9' < "${RUNNING_LOCK}" || true)"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "Already running: ${OUTPUT_DIR}"; exit 0
  fi
  touch "${FAILED_MARKER}"; echo "Stale RUNNING.lock: ${OUTPUT_DIR}" >&2; exit 4
fi
if [[ -s "${OUTPUT_DIR}/latest.pth.tar" || -s "${OUTPUT_DIR}/best.pth.tar" ]]; then
  touch "${FAILED_MARKER}"; echo "Orphan checkpoint: ${OUTPUT_DIR}" >&2; exit 5
fi

echo "$$" > "${RUNNING_LOCK}"
echo "$$" > "${OUTPUT_DIR}/launcher.pid"
echo "${GPU}" > "${OUTPUT_DIR}/gpu.id"
rm -f "${FAILED_MARKER}" "${COMPLETE_MARKER}"
cleanup() {
  local status=$?
  rm -f "${RUNNING_LOCK}"
  if [[ ${status} -eq 0 && -s "${OUTPUT_DIR}/latest.pth.tar" ]]; then
    touch "${COMPLETE_MARKER}"
    printf '%s\tcomplete\n' "$(date --iso-8601=seconds)" >> "${OUTPUT_DIR}/status.tsv"
  else
    touch "${FAILED_MARKER}"
    printf '%s\tfailed\t%s\n' "$(date --iso-8601=seconds)" "${status}" >> "${OUTPUT_DIR}/status.tsv"
  fi
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"
printf '%s\trunning\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${VARIANT}" "${DATASET}" "${GPU}" >> "${OUTPUT_DIR}/status.tsv"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_experiment_j_denp.py \
  --denp-variant "${VARIANT}" --dataset-name "${DATASET}" \
  --dataset-dir "${DATASET_DIR}" --output-dir "${OUTPUT_DIR}" \
  --epochs 1000 --batch-size 4 --patch-size 256 --workers 0 --lr 1e-3 \
  --eval-start 100 --eval-every 1 --save-every 20 --threshold 0.5 --seed "${SEED}" \
  > >(tee -a "${OUTPUT_DIR}/train.log") 2>&1 &
PYTHON_PID=$!
echo "${PYTHON_PID}" > "${OUTPUT_DIR}/python.pid"
wait "${PYTHON_PID}"
