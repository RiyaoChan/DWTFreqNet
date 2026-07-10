#!/usr/bin/env bash
set -u

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <dataset_name> <physical_gpu_index>" >&2
  exit 2
fi

DATASET=$1
GPU=$2
PROJECT=/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM
PYTHON=/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python
OUTPUT=$PROJECT/runs/full_three_datasets/$DATASET/dm_awgm_full

mkdir -p "$OUTPUT"
if [ -f "$OUTPUT/COMPLETED" ]; then
  echo "$DATASET dm_awgm_full is already complete"
  exit 0
fi

cd "$PROJECT"
while true; do
  echo "[$(date --iso-8601=seconds)] starting $DATASET dm_awgm_full on physical GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON" train_one.py \
    --dataset-name "$DATASET" \
    --dataset-dir ./datasets \
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
    --awgm-variant dm_awgm_full
  status=$?
  if [ "$status" -eq 0 ]; then
    touch "$OUTPUT/COMPLETED"
    echo "[$(date --iso-8601=seconds)] completed $DATASET dm_awgm_full"
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] $DATASET exited with $status; retrying in 60s"
  sleep 60
done
