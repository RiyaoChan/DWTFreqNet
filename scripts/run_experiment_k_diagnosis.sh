#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?mode: fidelity|prior|treatment|mad|counterfactual}"
DATASET="${2:?dataset name is required}"
SPLIT="${3:?train or test is required}"
GPU="${4:?GPU id is required}"
E1_CHECKPOINT="${5:?E1 checkpoint is required}"
J_CHECKPOINT_MAP="${6:?J checkpoint-map JSON is required}"

PROJECT_ROOT="${PROJECT_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_K_E1_CENTERED}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
PHASE1_ROOT="${PHASE1_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_PHASE1_TASK_PRIOR_VALIDATION}"
PYTHON_BIN="${PYTHON_BIN:-/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/analysis/experiment_k}"
case "${MODE}" in
  fidelity) MODE_DIR="compactness_fidelity" ;;
  prior) MODE_DIR="prior_drift" ;;
  treatment) MODE_DIR="treatment_effect" ;;
  mad) MODE_DIR="mad_gaussian_audit" ;;
  counterfactual) MODE_DIR="j1_counterfactual" ;;
  *) echo "Unknown diagnosis mode: ${MODE}" >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_ROOT}/${MODE_DIR}/${DATASET}/${SPLIT}"
mkdir -p "${OUTPUT_DIR}"
exec 8>"${OUTPUT_DIR}/diagnosis.instance.v1.lock"
if ! flock -n 8; then echo "Diagnosis already running: ${OUTPUT_DIR}" >&2; exit 2; fi
if [[ -f "${OUTPUT_DIR}/DIAGNOSIS_COMPLETE" ]]; then echo "Already complete: ${OUTPUT_DIR}"; exit 0; fi
if [[ -f "${OUTPUT_DIR}/FAILED" ]]; then echo "FAILED marker exists: ${OUTPUT_DIR}" >&2; exit 3; fi

J1_CHECKPOINT="$("${PYTHON_BIN}" - "${J_CHECKPOINT_MAP}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["j1_bandwise_noise_calibrated"])
PY
)"
cd "${PROJECT_ROOT}"
COMMON_ARGS=(--dataset-dir "${DATASET_DIR}" --dataset-name "${DATASET}" --split "${SPLIT}" --phase1-root "${PHASE1_ROOT}" --device cuda:0 --output-dir "${OUTPUT_DIR}")
set +e
case "${MODE}" in
  fidelity)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u tools/experiment_k/audit_compactness_fidelity.py \
      "${COMMON_ARGS[@]}" --e1-checkpoint "${E1_CHECKPOINT}" --checkpoint-map "${J_CHECKPOINT_MAP}" \
      > "${OUTPUT_DIR}/diagnosis.log" 2>&1
    ;;
  prior)
    STATISTICS="${OUTPUT_ROOT}/compactness_fidelity/${DATASET}/${SPLIT}/operator_statistics.csv"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u tools/experiment_k/audit_feature_prior_drift.py \
      "${COMMON_ARGS[@]}" --e1-checkpoint "${E1_CHECKPOINT}" --checkpoint-map "${J_CHECKPOINT_MAP}" \
      --statistics "${STATISTICS}" > "${OUTPUT_DIR}/diagnosis.log" 2>&1
    ;;
  treatment)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u tools/experiment_k/audit_compactness_treatment_effect.py \
      "${COMMON_ARGS[@]}" --j1-checkpoint "${J1_CHECKPOINT}" \
      > "${OUTPUT_DIR}/diagnosis.log" 2>&1
    ;;
  mad)
    TREATMENT="${OUTPUT_ROOT}/treatment_effect/${DATASET}/${SPLIT}/treatment_instances.csv"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u tools/experiment_k/audit_mad_gaussian_effect.py \
      "${COMMON_ARGS[@]}" --j1-checkpoint "${J1_CHECKPOINT}" --treatment-instances "${TREATMENT}" \
      > "${OUTPUT_DIR}/diagnosis.log" 2>&1
    ;;
  counterfactual)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u tools/experiment_k/run_j1_counterfactual_protection.py \
      "${COMMON_ARGS[@]}" --j1-checkpoint "${J1_CHECKPOINT}" \
      > "${OUTPUT_DIR}/diagnosis.log" 2>&1
    ;;
esac
status=$?
set -e
if [[ ${status} -eq 0 ]]; then
  touch "${OUTPUT_DIR}/DIAGNOSIS_COMPLETE"
else
  touch "${OUTPUT_DIR}/FAILED"
fi
exit "${status}"
