#!/usr/bin/env bash
set -u

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <dataset_name> <awgm_variant> <physical_gpu_index> <output_root>" >&2
  exit 2
fi

DATASET=$1
VARIANT=$2
GPU=$3
OUTPUT_ROOT=$4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(dirname "$SCRIPT_DIR")}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_DIR=${DATASET_DIR:-$PROJECT/datasets}
OUTPUT=$OUTPUT_ROOT/$VARIANT

while [ ! -f "$OUTPUT/STAGE1_COMPLETED" ]; do
  if [ -f "$OUTPUT/COMPLETED" ]; then
    exit 0
  fi
  sleep 60
done

PROJECT="$PROJECT" PYTHON_BIN="$PYTHON_BIN" DATASET_DIR="$DATASET_DIR" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  "$SCRIPT_DIR/run_w8m_stage1_dataset.sh" "$DATASET" "$VARIANT" "$GPU" 1000
