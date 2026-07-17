#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DATASET GPU SEED [DATASET_DIR]" >&2
  exit 2
fi

DATASET="$1"
GPU="$2"
SEED="$3"
DATASET_DIR="${4:-${DATASET_DIR:-./datasets}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="runs/experiment_d/${DATASET}/sd_awgm_hfe/seed${SEED}"

mkdir -p "$OUTPUT_DIR"
set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u train_experiment_d.py \
  --dataset-name "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 1000 \
  --batch-size 4 \
  --patch-size 256 \
  --workers 0 \
  --lr 1e-3 \
  --eval-start 100 \
  --eval-every 1 \
  --save-every 20 \
  --threshold 0.5 \
  --seed "$SEED" \
  --resume auto \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

if (( status == 0 )); then
  touch "$OUTPUT_DIR/COMPLETED"
  printf '%s\tcompleted\n' "$(date --iso-8601=seconds)" >> "$OUTPUT_DIR/status.tsv"
else
  failure="failed"
  if tail -n 200 "$OUTPUT_DIR/train.log" | grep -qiE 'out of memory|CUDA.*memory'; then
    failure="oom_batch4"
  fi
  printf '%s\t%s\texit=%s\n' \
    "$(date --iso-8601=seconds)" "$failure" "$status" >> "$OUTPUT_DIR/status.tsv"
fi
exit "$status"
