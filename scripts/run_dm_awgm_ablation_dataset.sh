#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <dataset_name> <awgm_variant> <physical_gpu_index> [stop_epoch]" >&2
  exit 2
fi

DATASET=$1
VARIANT=$2
GPU=$3
STOP_EPOCH=${4:-1000}
case "$VARIANT" in
  dm_awgm_no_mamba|dm_awgm_no_dcn|dm_awgm_conv_only) ;;
  *) echo "Unsupported ablation variant: $VARIANT" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(dirname "$SCRIPT_DIR")}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_DIR=${DATASET_DIR:-$PROJECT/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/runs/ablation_two_datasets}
OUTPUT=$OUTPUT_ROOT/$DATASET/$VARIANT

mkdir -p "$OUTPUT"
if [ -f "$OUTPUT/COMPLETED" ]; then
  echo "$DATASET $VARIANT is already complete"
  exit 0
fi

cd "$PROJECT"
echo "[$(date --iso-8601=seconds)] starting $DATASET $VARIANT on physical GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train_one.py \
  --dataset-name "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT" \
  --epochs 1000 \
  --batch-size 4 \
  --patch-size 256 \
  --workers 0 \
  --eval-start 100 \
  --eval-every 5 \
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
  echo "[$(date --iso-8601=seconds)] finished $DATASET $VARIANT at epoch $STOP_EPOCH"
else
  echo "[$(date --iso-8601=seconds)] $DATASET $VARIANT exited with status $status" >&2
fi
exit "$status"
