#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <awgm_variant> <dataset_name> <physical_gpu_index> [seed]" >&2
  exit 2
fi

VARIANT=$1
DATASET=$2
GPU=$3
SEED=${4:-42}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(dirname "$SCRIPT_DIR")}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_DIR=${DATASET_DIR:-$PROJECT/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/runs/w8m_full_three_datasets}
OUTPUT=$OUTPUT_ROOT/$DATASET/$VARIANT/seed_$SEED

mkdir -p "$OUTPUT"
if [ -f "$OUTPUT/COMPLETED" ]; then
  echo "$VARIANT $DATASET seed $SEED is already complete"
  exit 0
fi

cd "$PROJECT"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train_one.py \
  --dataset-name "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT" \
  --epochs 1000 \
  --batch-size 4 \
  --patch-size 256 \
  --workers 0 \
  --eval-start 100 \
  --eval-every 1 \
  --save-every 20 \
  --threshold 0.5 \
  --seed "$SEED" \
  --resume auto \
  --awgm-variant "$VARIANT"
status=$?
if [ "$status" -eq 0 ]; then
  touch "$OUTPUT/COMPLETED"
fi
exit "$status"
