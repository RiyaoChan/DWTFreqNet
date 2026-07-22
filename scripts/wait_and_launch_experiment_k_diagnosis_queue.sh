#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_K_E1_CENTERED}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-0,1,2,3,4,5,6}"
MAX_CONCURRENT="${MAX_CONCURRENT:-7}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SPLIT="${SPLIT:-train}"
QUEUE_ROOT="${ROOT}/analysis/experiment_k/diagnosis_queue_${SPLIT}"
mkdir -p "${QUEUE_ROOT}"

exec 8>"${QUEUE_ROOT}/handoff.lock"
if ! flock -n 8; then
    echo "A diagnosis-queue handoff watcher is already active." >&2
    exit 0
fi
echo "$$" > "${QUEUE_ROOT}/handoff.pid"
trap 'rm -f "${QUEUE_ROOT}/handoff.pid"' EXIT

printf '[%s] waiting for previous queue lock to be released\n' "$(date '+%F %T')" \
    >> "${QUEUE_ROOT}/handoff.log"
while ! flock -n "${QUEUE_ROOT}/queue.lock" true; do
    sleep 30
done

cd "${ROOT}"
nohup env ROOT="${ROOT}" GPU_ALLOWLIST="${GPU_ALLOWLIST}" \
    MAX_CONCURRENT="${MAX_CONCURRENT}" POLL_SECONDS="${POLL_SECONDS}" SPLIT="${SPLIT}" \
    bash scripts/launch_experiment_k_diagnosis_queue.sh \
    > "${QUEUE_ROOT}/nohup.log" 2>&1 8>&- &
new_pid=$!
printf '[%s] launched updated diagnosis queue pid=%s\n' "$(date '+%F %T')" "${new_pid}" \
    >> "${QUEUE_ROOT}/handoff.log"
