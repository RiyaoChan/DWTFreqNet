#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?encoder variant is required}"
DATASET="${2:?dataset name is required}"
GPU="${3:?GPU id is required}"
SEED="${4:-42}"

PROJECT_ROOT="${PROJECT_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
PYTHON_BIN="${PYTHON_BIN:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/experiment_e_lfss_awgm}"

case "${VARIANT}" in
  e1_lfss_resblock) OUTPUT_NAME="E1_lfss_resblock" ;;
  e2_lfss_transition) OUTPUT_NAME="E2_lfss_transition" ;;
  *) echo "Unknown Experiment E variant: ${VARIANT}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}/${DATASET}/seed${SEED}"
RUNNING_LOCK="${OUTPUT_DIR}/RUNNING.lock"
FAILED_MARKER="${OUTPUT_DIR}/FAILED"
COMPLETE_MARKER="${OUTPUT_DIR}/TRAINING_COMPLETE"
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${COMPLETE_MARKER}" ]]; then
  echo "Already complete: ${OUTPUT_DIR}"
  exit 0
fi
if [[ -f "${FAILED_MARKER}" ]]; then
  echo "FAILED marker exists; manual inspection required: ${OUTPUT_DIR}" >&2
  exit 3
fi
if [[ -f "${RUNNING_LOCK}" ]]; then
  EXISTING_PID="$(tr -dc '0-9' < "${RUNNING_LOCK}" || true)"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "Task is already running with wrapper PID ${EXISTING_PID}: ${OUTPUT_DIR}"
    exit 0
  fi
  echo "Stale RUNNING.lock detected; marking FAILED for manual inspection" >&2
  touch "${FAILED_MARKER}"
  exit 4
fi
if [[ -s "${OUTPUT_DIR}/latest.pth.tar" || -s "${OUTPUT_DIR}/best.pth.tar" ]]; then
  echo "Checkpoint exists without a live lock; refusing to overwrite: ${OUTPUT_DIR}" >&2
  touch "${FAILED_MARKER}"
  exit 5
fi

echo "$$" > "${RUNNING_LOCK}"
echo "$$" > "${OUTPUT_DIR}/launcher.pid"
echo "${GPU}" > "${OUTPUT_DIR}/gpu.id"
rm -f "${FAILED_MARKER}" "${COMPLETE_MARKER}"

cleanup() {
  local status=$?
  rm -f "${RUNNING_LOCK}"
  # A killed or otherwise incomplete trainer must not look like a valid run.
  # Formal training always writes latest.pth.tar before returning successfully.
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
printf '%s\trunning\t%s\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "${VARIANT}" "${DATASET}" "${GPU}" \
  >> "${OUTPUT_DIR}/status.tsv"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_experiment_e_lfss_awgm.py \
  --encoder-variant "${VARIANT}" \
  --dataset-name "${DATASET}" \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 1000 \
  --batch-size 4 \
  --patch-size 256 \
  --workers 0 \
  --lr 1e-3 \
  --eval-start 100 \
  --eval-every 1 \
  --save-every 20 \
  --threshold 0.5 \
  --seed "${SEED}" \
  > >(tee -a "${OUTPUT_DIR}/train.log") 2>&1 &

PYTHON_PID=$!
echo "${PYTHON_PID}" > "${OUTPUT_DIR}/python.pid"
wait "${PYTHON_PID}"
