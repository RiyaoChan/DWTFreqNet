# Experiment A v2 — Wavelet U-Net Local-frequency Branch

## Reproducibility

- Base commit: `71dfeb348878517775af3df0767b54747f692c5d`
- Implementation branch: `codex/experiment-a-wulle-v2`
- Original model file: unchanged (`model/DWTFreqNet.py`)
- New model: `model/DWTFreqNet_WULLE.py`
- Haar direction mapping: H/LH responds to vertical structure; V/HL responds to horizontal structure.

## Code change

WULLE keeps the four first-column local encoders E1–E4 and replaces the six nested
local nodes with three IDWT-based decoder nodes D3–D1. D1 and D2 are transformed by
a second DWT and fed back to the unchanged dense global branch. AWGM/W8M, LDRC and
the final decoder remain in their original positions.

Removed instance modules:

`local_encoder1_2`, `local_encoder2_2`, `local_encoder3_2`, `local_encoder1_3`,
`local_encoder2_3`, `local_encoder1_4`, `global_channel1_3`, `global_channel2_3`,
`global_channel1_4`.

Added modules: `wulle_decoder3` (16C→8C), `wulle_decoder2` (12C→4C), and
`wulle_decoder1` (6C→2C).

## Complexity gate

| Model | Parameters | Local parameters | FLOPs | Latency (1×256) | FPS | Inference peak | Training peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| dwtfreqnet_original | 37,434,599 | 6,850,368 | 66.87G | 20.77 ms | 48.14 | 617.35 MiB | 2501.57 MiB |
| dwtfreqnet_wulle_a | 35,399,143 | 5,111,360 | 54.69G | 19.66 ms | 50.87 | 554.82 MiB | 2493.21 MiB |

The parameter gate passed: total parameters decrease by 2,035,456 (5.44%) and
FLOPs decrease by 12.18G (18.21%). Runtime numbers above are a three-repeat local
CUDA check; the reproducible profiler can write a longer server-side measurement.

## Formal first-stage matrix

All six jobs use seed 42, batch size 4, patch size 256, 1000 epochs, Adam at 1e-3,
CosineAnnealingLR with eta_min 1e-5, evaluation every epoch from epoch 100,
checkpoint every 20 epochs, and threshold 0.5. No fallback backend or pretrained
original-model checkpoint is used.

The 226-server queue uses only GPUs 0, 1, 2 and 6 because GPUs 3, 4 and 5 were
already occupied when the queue was launched. It starts both NUAA models plus the
NUDT and IRSTD baselines first, then automatically starts the two remaining WULLE
jobs as soon as one of those four GPUs becomes idle. Each candidate GPU must have
at most 1024 MiB allocated and at most 5% utilization at dispatch time.

| ID | Dataset | Model | AWGM | GPU | PID | Output | Status / epoch | Best metrics |
|---|---|---|---|---:|---:|---|---|---|
| A0-NUAA | NUAA-SIRST | dwtfreqnet_original | awgm_original | | | `runs/experiment_a_v2/NUAA-SIRST/dwtfreqnet_original/awgm_original` | queued | |
| A1-NUAA | NUAA-SIRST | dwtfreqnet_wulle_a | awgm_original | | | `runs/experiment_a_v2/NUAA-SIRST/dwtfreqnet_wulle_a/awgm_original` | queued | |
| A0-NUDT | NUDT-SIRST | dwtfreqnet_original | awgm_original | | | `runs/experiment_a_v2/NUDT-SIRST/dwtfreqnet_original/awgm_original` | queued | |
| A1-NUDT | NUDT-SIRST | dwtfreqnet_wulle_a | awgm_original | | | `runs/experiment_a_v2/NUDT-SIRST/dwtfreqnet_wulle_a/awgm_original` | queued | |
| A0-IRSTD | IRSTD-1k | dwtfreqnet_original | awgm_original | | | `runs/experiment_a_v2/IRSTD-1k/dwtfreqnet_original/awgm_original` | queued | |
| A1-IRSTD | IRSTD-1k | dwtfreqnet_wulle_a | awgm_original | | | `runs/experiment_a_v2/IRSTD-1k/dwtfreqnet_wulle_a/awgm_original` | queued | |

Second-stage A2/A3 W8M experiments are intentionally not started until A0/A1 finish
and one W8M variant is selected under the protocol in the specification.

## Validation record

- Structural, 2×256 output-shape, six-output training-mode and gradient tests: passed.
- Four-variant smoke tests: passed (local Mamba-unavailable variants used the explicit test-only fallback; formal A0/A1 use no fallback).
- Haar synthetic line/step mapping and W8M routing checks: passed.
- Server profile: pending.
- 226-server deployment and launch: pending.
