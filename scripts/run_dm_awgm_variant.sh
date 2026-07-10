#!/usr/bin/env bash
set -u

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <awgm_variant> <physical_gpu_index>" >&2
  exit 2
fi

VARIANT=$1
GPU=$2
PROJECT=/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM
PYTHON=/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python
OUTPUT=$PROJECT/runs/stage1_nudt/$VARIANT

mkdir -p "$OUTPUT"
if [ -f "$OUTPUT/COMPLETED" ]; then
  echo "$VARIANT is already complete"
  exit 0
fi

cd "$PROJECT"
while true; do
  echo "[$(date --iso-8601=seconds)] starting $VARIANT on physical GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON" train_one.py \
    --dataset-name NUDT-SIRST \
    --dataset-dir ./datasets \
    --output-dir "$OUTPUT" \
    --epochs 1000 \
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
    touch "$OUTPUT/COMPLETED"
    echo "[$(date --iso-8601=seconds)] completed $VARIANT"
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] $VARIANT exited with $status; retrying in 60s"
  sleep 60
done
