#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <physical_gpu_index> [physical_gpu_index ...]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(dirname "$SCRIPT_DIR")}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/runs/w8m_stage0}
GPUS=("$@")
VARIANTS=(
  w8m_diag4_subband_shared
  w8m_diag4_axial_diag_shared
  w8m_diag4_axial_diag_shared_dir_embed
  w8m_diag2_subband_shared
  w8m_diag4_pair_shared
  w8m_diag4_independent
  w8m_diag4_all_shared
)

mkdir -p "$OUTPUT_ROOT"
pids=()
for gpu_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$gpu_index]}
  (
    for ((variant_index=gpu_index; variant_index<${#VARIANTS[@]}; variant_index+=${#GPUS[@]})); do
      variant=${VARIANTS[$variant_index]}
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
        "$PROJECT/tools/smoke_test_dm_awgm.py" \
        --awgm_variant "$variant" \
        --batch-size 1 \
        --timing-iters 3 \
        --skip-flops \
        >"$OUTPUT_ROOT/$variant.log" 2>&1
      touch "$OUTPUT_ROOT/$variant.COMPLETED"
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
