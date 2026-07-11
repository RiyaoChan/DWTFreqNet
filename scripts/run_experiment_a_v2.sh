#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 DATASET MODEL_VARIANT AWGM_VARIANT GPU [DATASET_DIR]" >&2
  exit 2
fi

DATASET="$1"
MODEL_VARIANT="$2"
AWGM_VARIANT="$3"
GPU="$4"
DATASET_DIR="${5:-${DATASET_DIR:-./datasets}}"
OUTPUT_DIR="runs/experiment_a_v2/${DATASET}/${MODEL_VARIANT}/${AWGM_VARIANT}"

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" python -u train_one.py \
  --model-variant "$MODEL_VARIANT" \
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
  --seed 42 \
  --awgm-variant "$AWGM_VARIANT" \
  --resume auto \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"
