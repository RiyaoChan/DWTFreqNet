#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <physical_gpu_index> [physical_gpu_index ...]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(dirname "$SCRIPT_DIR")}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT/runs/w8m_stage1_nudt}
STOP_EPOCH=${STOP_EPOCH:-400}
GPUS=("$@")
VARIANTS=(
  w8m_diag4_subband_shared
  w8m_diag4_axial_diag_shared
  w8m_diag4_axial_diag_shared_dir_embed
  w8m_diag2_subband_shared
  w8m_diag4_pair_shared
  w8m_diag4_independent
  w8m_diag4_all_shared
  awgm_original
  dm_awgm_full
  dm_awgm_no_dcn
)

mkdir -p "$OUTPUT_ROOT/queues"
for gpu_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$gpu_index]}
  queue_log="$OUTPUT_ROOT/queues/gpu_${gpu}.log"
  (
    for ((variant_index=gpu_index; variant_index<${#VARIANTS[@]}; variant_index+=${#GPUS[@]})); do
      variant=${VARIANTS[$variant_index]}
      "$SCRIPT_DIR/run_w8m_stage1_variant.sh" "$variant" "$gpu" "$STOP_EPOCH"
    done
  ) >"$queue_log" 2>&1 &
  pid=$!
  echo "$pid" > "$OUTPUT_ROOT/queues/gpu_${gpu}.pid"
  echo "GPU $gpu queue started as PID $pid; log: $queue_log"
done
