#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT/analysis/phase1_task_prior_validation}"
E1_ROOT="${E1_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS/runs/experiment_e_lfss_awgm/E1_lfss_resblock}"
H_ROOT="${H_ROOT:-/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_H_DECODER_LFP/runs/experiment_h_decoder_lfp}"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-NUAA-SIRST IRSTD-1K NUDT-SIRST}"
SPLITS="${SPLITS:-train test}"
DEVICE="${DEVICE:-auto}"
CPU_WORKERS="${CPU_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_P3="${RUN_P3:-1}"
RUN_H_CROSS="${RUN_H_CROSS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
NUM_RANDOM_REPEATS="${NUM_RANDOM_REPEATS:-20}"
BOOTSTRAP="${BOOTSTRAP:-1000}"

mkdir -p "$ANALYSIS_ROOT/metadata" "$ANALYSIS_ROOT/final" "$ANALYSIS_ROOT/logs"
cd "$ROOT"

should_skip() {
  local marker="$1"
  [[ "$SKIP_EXISTING" == "1" && -s "$marker" ]]
}

pick_idle_gpu() {
  if [[ "$DEVICE" != "auto" ]]; then
    echo "$DEVICE"
    return
  fi
  while true; do
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    local index
    for ((index=0; index<gpu_count; index++)); do
      local line memory utilization compute_count
      line=$(nvidia-smi --id="$index" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
      memory=$(echo "$line" | cut -d, -f1 | tr -d ' ')
      utilization=$(echo "$line" | cut -d, -f2 | tr -d ' ')
      compute_count=$(nvidia-smi pmon -c 1 | awk -v gpu="$index" '$1==gpu && $3=="C" {count++} END {print count+0}')
      if (( memory < 1000 && utilization < 10 && compute_count == 0 )); then
        echo "cuda:$index"
        return
      fi
    done
    echo "[$(date '+%F %T')] no idle GPU; waiting ${GPU_POLL_SECONDS}s" >&2
    sleep "$GPU_POLL_SECONDS"
  done
}

h_dataset_complete() {
  local dataset="$1" variant log epoch
  for variant in H1_rawll_attention H1_decoder_attention H2_rawll_fixed_gaussian H2_decoder_fixed_gaussian H3_rawll_adaptive_gaussian H3_decoder_adaptive_gaussian; do
    log="$H_ROOT/$variant/$dataset/seed42/train.log"
    [[ -s "$log" ]] || return 1
    epoch=$(tail -n 1 "$log" | "$PYTHON" -c 'import json,sys; print(json.loads(sys.stdin.read()).get("epoch",0))' 2>/dev/null || echo 0)
    [[ "$epoch" == "1000" ]] || return 1
  done
  return 0
}

make_h_map() {
  local dataset="$1" output="$2"
  "$PYTHON" - "$H_ROOT" "$dataset" "$output" <<'PY'
import json, pathlib, sys
root, dataset, output = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
mapping = {
    "h1_rawll_attention": root / "H1_rawll_attention" / dataset / "seed42" / "best.pth.tar",
    "h1_decoder_attention": root / "H1_decoder_attention" / dataset / "seed42" / "best.pth.tar",
    "h2_rawll_fixed_gaussian": root / "H2_rawll_fixed_gaussian" / dataset / "seed42" / "best.pth.tar",
    "h2_decoder_fixed_gaussian": root / "H2_decoder_fixed_gaussian" / dataset / "seed42" / "best.pth.tar",
    "h3_rawll_adaptive_gaussian": root / "H3_rawll_adaptive_gaussian" / dataset / "seed42" / "best.pth.tar",
    "h3_decoder_adaptive_gaussian": root / "H3_decoder_adaptive_gaussian" / dataset / "seed42" / "best.pth.tar",
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({key: str(value) for key, value in mapping.items()}, indent=2), encoding="utf-8")
PY
}

if [[ "$RUN_P1" == "1" ]]; then
  for dataset in $DATASETS; do
    for split in $SPLITS; do
      output="$ANALYSIS_ROOT/P1_gaussian_geometry/$dataset/$split"
      if should_skip "$output/summary.json"; then
        continue
      fi
      mkdir -p "$output"
      extra=()
      if [[ "$split" == "test" ]]; then
        extra+=(--thresholds "$ANALYSIS_ROOT/P1_gaussian_geometry/$dataset/train/thresholds.json")
      fi
      "$PYTHON" tools/phase1/validate_gaussian_geometry.py \
        --dataset-dir "$DATASET_DIR" --dataset-name "$dataset" --split "$split" \
        --output-dir "$output" --workers "$CPU_WORKERS" --bootstrap "$BOOTSTRAP" \
        --max-samples "$MAX_SAMPLES" "${extra[@]}" \
        >"$ANALYSIS_ROOT/logs/P1_${dataset}_${split}.log" 2>&1
    done
  done
fi

if [[ "$RUN_P2" == "1" ]]; then
  for dataset in $DATASETS; do
    checkpoint="$E1_ROOT/$dataset/seed42/best.pth.tar"
    for split in $SPLITS; do
      output="$ANALYSIS_ROOT/P2_wavelet_consistency/$dataset/$split"
      if should_skip "$output/summary.json"; then
        continue
      fi
      device=$(pick_idle_gpu)
      mkdir -p "$output"
      "$PYTHON" tools/phase1/validate_wavelet_directional_consistency.py \
        --dataset-dir "$DATASET_DIR" --dataset-name "$dataset" --split "$split" \
        --checkpoint "$checkpoint" --device "$device" --output-dir "$output" \
        --bootstrap "$BOOTSTRAP" --max-samples "$MAX_SAMPLES" \
        >"$ANALYSIS_ROOT/logs/P2_${dataset}_${split}.log" 2>&1
    done
  done
fi

if [[ "$RUN_P3" == "1" ]]; then
  for dataset in $DATASETS; do
    checkpoint="$E1_ROOT/$dataset/seed42/best.pth.tar"
    train_output="$ANALYSIS_ROOT/P3_sampling_geometry/$dataset/train"
    if ! should_skip "$train_output/summary.json"; then
      device=$(pick_idle_gpu)
      mkdir -p "$train_output"
      "$PYTHON" tools/phase1/compare_sampling_geometries.py \
        --dataset-dir "$DATASET_DIR" --dataset-name "$dataset" --split train \
        --checkpoint "$checkpoint" --device "$device" --output-dir "$train_output" \
        --num-random-repeats "$NUM_RANDOM_REPEATS" --max-samples "$MAX_SAMPLES" \
        >"$ANALYSIS_ROOT/logs/P3_${dataset}_train.log" 2>&1
    fi
    test_output="$ANALYSIS_ROOT/P3_sampling_geometry/$dataset/test"
    if ! should_skip "$test_output/summary.json"; then
      device=$(pick_idle_gpu)
      mkdir -p "$test_output"
      "$PYTHON" tools/phase1/compare_sampling_geometries.py \
        --dataset-dir "$DATASET_DIR" --dataset-name "$dataset" --split test \
        --checkpoint "$checkpoint" --device "$device" --output-dir "$test_output" \
        --selected-radii "$train_output/selected_radii.json" \
        --num-random-repeats "$NUM_RANDOM_REPEATS" --max-samples "$MAX_SAMPLES" \
        >"$ANALYSIS_ROOT/logs/P3_${dataset}_test.log" 2>&1
    fi
  done
fi

if [[ "$RUN_H_CROSS" == "1" ]]; then
  for dataset in $DATASETS; do
    output="$ANALYSIS_ROOT/H_cross_analysis/$dataset"
    if should_skip "$output/summary.json"; then
      continue
    fi
    if [[ "$dataset" == "NUDT-SIRST" ]] && ! h_dataset_complete "$dataset"; then
      echo "[$(date '+%F %T')] skip NUDT H cross-analysis until all six H runs reach epoch 1000" \
        >>"$ANALYSIS_ROOT/logs/H_cross_NUDT-SIRST.log"
      continue
    fi
    device=$(pick_idle_gpu)
    mkdir -p "$output"
    map="$ANALYSIS_ROOT/metadata/H_checkpoint_map_${dataset}.json"
    make_h_map "$dataset" "$map"
    "$PYTHON" tools/phase1/analyze_h_lfp_prior_alignment.py \
      --dataset-dir "$DATASET_DIR" --dataset-name "$dataset" --split test \
      --e1-checkpoint "$E1_ROOT/$dataset/seed42/best.pth.tar" \
      --h-checkpoint-map "$map" --device "$device" --output-dir "$output" \
      --p1-metrics "$ANALYSIS_ROOT/P1_gaussian_geometry/$dataset/test/gaussian_instance_metrics.csv" \
      --p2-metrics "$ANALYSIS_ROOT/P2_wavelet_consistency/$dataset/test/instance_consistency_metrics.csv" \
      --max-samples "$MAX_SAMPLES" \
      >"$ANALYSIS_ROOT/logs/H_cross_${dataset}.log" 2>&1
  done
fi

"$PYTHON" tools/phase1/aggregate_phase1_report.py \
  --analysis-root "$ANALYSIS_ROOT" --output-dir "$ANALYSIS_ROOT/final" \
  >"$ANALYSIS_ROOT/logs/aggregate.log" 2>&1

echo "Phase 1 execution pass finished at $(date '+%F %T')"
