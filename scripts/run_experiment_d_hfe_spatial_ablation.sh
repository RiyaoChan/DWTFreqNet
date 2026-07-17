#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 SPATIAL_HFE_ABLATION DATASET GPU [SEED] [DATASET_DIR]" >&2
  exit 2
fi

ABLATION="$1"
DATASET="$2"
GPU="$3"
SEED="${4:-42}"
DATASET_DIR="${5:-${DATASET_DIR:-./datasets}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$ABLATION" in
  d5_same_position)
    ABLATION_DIR="D5_same_position"
    SD_VARIANT="sd_awgm_hfe_samepos"
    ;;
  d6_neighborhood)
    ABLATION_DIR="D6_neighborhood"
    SD_VARIANT="sd_awgm_hfe_neighborhood"
    ;;
  d7_target_neighborhood)
    ABLATION_DIR="D7_target_neighborhood"
    SD_VARIANT="sd_awgm_hfe_targetlocal"
    ;;
  *)
    echo "Unsupported spatial HFE ablation: $ABLATION" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="runs/experiment_d_spatial_ablation/${ABLATION_DIR}/${DATASET}/seed${SEED}"
LOCK_DIR="$OUTPUT_DIR/.run.lock"
mkdir -p "$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/COMPLETED" ]]; then
  echo "Already completed: $OUTPUT_DIR" >&2
  exit 0
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Output directory is locked: $OUTPUT_DIR" >&2
  exit 3
fi
cleanup() {
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT INT TERM
rm -f "$OUTPUT_DIR/FAILED"
printf '%s\t%s\t%s\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "$ABLATION" "$DATASET" "$GPU" "$$" \
  > "$LOCK_DIR/owner.tsv"

set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u \
  train_experiment_d_hfe_spatial_ablation.py \
  --spatial-hfe-ablation "$ABLATION" \
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
  printf '%s\n' "$failure" > "$OUTPUT_DIR/FAILED"
  printf '%s\t%s\t%s\texit=%s\n' \
    "$(date --iso-8601=seconds)" "$failure" "$SD_VARIANT" "$status" \
    >> "$OUTPUT_DIR/status.tsv"
fi
exit "$status"

