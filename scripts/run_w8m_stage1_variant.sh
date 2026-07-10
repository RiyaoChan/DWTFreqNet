#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <awgm_variant> <physical_gpu_index> [stop_epoch]" >&2
  exit 2
fi

VARIANT=$1
GPU=$2
STOP_EPOCH=${3:-400}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(dirname "$SCRIPT_DIR")}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_DIR=${DATASET_DIR:-$PROJECT/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/runs/w8m_stage1_nudt}
OUTPUT=$OUTPUT_ROOT/$VARIANT

mkdir -p "$OUTPUT"
if [ -f "$OUTPUT/COMPLETED" ]; then
  echo "$VARIANT is already complete"
  exit 0
fi

cd "$PROJECT"
echo "[$(date --iso-8601=seconds)] starting $VARIANT on physical GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train_one.py \
  --dataset-name NUDT-SIRST \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT" \
  --epochs 1000 \
  --stop-after-epoch "$STOP_EPOCH" \
  --batch-size 4 \
  --patch-size 256 \
  --workers 0 \
  --eval-start 100 \
  --eval-every 1 \
  --save-every 20 \
  --threshold 0.5 \
  --seed 42 \
  --resume auto \
  --awgm-variant "$VARIANT"
status=$?

if [ "$status" -eq 0 ]; then
  if [ "$STOP_EPOCH" -ge 1000 ]; then
    touch "$OUTPUT/COMPLETED"
  else
    touch "$OUTPUT/STAGE1_COMPLETED"
  fi
  echo "[$(date --iso-8601=seconds)] finished $VARIANT at stage boundary $STOP_EPOCH"
else
  echo "[$(date --iso-8601=seconds)] $VARIANT exited with status $status" >&2
fi
exit "$status"
