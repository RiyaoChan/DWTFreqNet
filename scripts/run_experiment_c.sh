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
OUTPUT_DIR="runs/experiment_c/${DATASET}/sd_awgm_ldrc/seed${SEED}"

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" python -u train_experiment_c.py \
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
touch "$OUTPUT_DIR/COMPLETED"
