#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 HFE_ABLATION DATASET GPU [SEED] [DATASET_DIR]" >&2
  exit 2
fi

HFE_ABLATION="$1"
DATASET="$2"
GPU="$3"
SEED="${4:-42}"
DATASET_DIR="${5:-${DATASET_DIR:-./datasets}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$HFE_ABLATION" in
  d2_softcos_all)
    ABLATION_DIR="D2_softcos_all"
    SD_VARIANT="sd_awgm_hfe_softcos"
    ;;
  d3_scaleaware)
    ABLATION_DIR="D3_scaleaware"
    SD_VARIANT="sd_awgm_hfe_scaleaware"
    ;;
  d4_no_matching)
    ABLATION_DIR="D4_no_matching"
    SD_VARIANT="sd_awgm_hfe_nomatch"
    ;;
  *)
    echo "Unsupported HFE ablation: $HFE_ABLATION" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="runs/experiment_d_ablation/${ABLATION_DIR}/${DATASET}/seed${SEED}"
mkdir -p "$OUTPUT_DIR"
set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
  train_experiment_d_hfe_ablation.py \
  --hfe-ablation "$HFE_ABLATION" \
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
  --seed "$SEED" \
  --resume auto \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

if (( status == 0 )); then
  touch "$OUTPUT_DIR/COMPLETED"
  printf '%s\tcompleted\t%s\n' \
    "$(date --iso-8601=seconds)" "$SD_VARIANT" >> "$OUTPUT_DIR/status.tsv"
else
  failure="failed"
  if tail -n 200 "$OUTPUT_DIR/train.log" | grep -qiE \
    'out of memory|CUDA.*memory'; then
    failure="oom_batch4"
  elif tail -n 200 "$OUTPUT_DIR/train.log" | grep -qiE \
    'non-finite|nan|infinity'; then
    failure="non_finite"
  fi
  printf '%s\t%s\t%s\texit=%s\n' \
    "$(date --iso-8601=seconds)" "$failure" "$SD_VARIANT" "$status" \
    >> "$OUTPUT_DIR/status.tsv"
fi
exit "$status"
