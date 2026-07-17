#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 DATASET SD_VARIANT GPU SEED [DATASET_DIR]" >&2
  exit 2
fi

DATASET="$1"
SD_VARIANT="$2"
GPU="$3"
SEED="$4"
DATASET_DIR="${5:-${DATASET_DIR:-./datasets}}"
OUTPUT_DIR="runs/experiment_b/${DATASET}/${SD_VARIANT}/seed${SEED}"

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" python -u train_experiment_b.py \
  --dataset-name "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --sd-variant "$SD_VARIANT" \
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
