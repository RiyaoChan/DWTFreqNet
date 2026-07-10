#!/usr/bin/env bash
set -euo pipefail

PROJECT=/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM
RUNNER=$PROJECT/scripts/run_dm_awgm_variant.sh
ROOT=$PROJECT/runs/stage1_nudt
mkdir -p "$ROOT"

launch_now() {
  variant=$1
  gpu=$2
  mkdir -p "$ROOT/$variant"
  nohup "$RUNNER" "$variant" "$gpu" \
    > "$ROOT/$variant/supervisor.log" 2>&1 < /dev/null &
  echo "$!" > "$ROOT/$variant/supervisor.pid"
  echo "$variant GPU=$gpu supervisor=$!"
}

launch_now dm_awgm_full 2
launch_now dm_awgm_no_mamba 3
launch_now dm_awgm_no_dcn 4

mkdir -p "$ROOT/dm_awgm_conv_only"
nohup bash -c "
  while [ ! -f '$ROOT/dm_awgm_no_mamba/COMPLETED' ]; do sleep 60; done
  exec '$RUNNER' dm_awgm_conv_only 3
" > "$ROOT/dm_awgm_conv_only/supervisor.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/dm_awgm_conv_only/supervisor.pid"
echo "dm_awgm_conv_only queued on GPU=3 supervisor=$!"
