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

The loader root is `/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets`. Its image
directories resolve to the requested `/DATA20T/bip/cry/code/SIRST-5K-main/dataset`
data, while the wrapper supplies the repository-specific 50/50 `img_idx` split
files required by `dataset.py` (IRSTD uses its canonical loader name `IRSTD-1K`).

| ID | Dataset | Model | AWGM | GPU | PID | Output | Status / epoch | Best metrics |
|---|---|---|---|---:|---:|---|---|---|
| A0-NUAA | NUAA-SIRST | dwtfreqnet_original | awgm_original | 0 | 2491245 | `runs/experiment_a_v2/NUAA-SIRST/dwtfreqnet_original/awgm_original` | stopped by request after epoch 46 | historical baseline used |
| A1-NUAA | NUAA-SIRST | dwtfreqnet_wulle_a | awgm_original | 1 | 2491260 | `runs/experiment_a_v2/NUAA-SIRST/dwtfreqnet_wulle_a/awgm_original` | running; epoch 62 | evaluation starts at 100 |
| A0-NUDT | NUDT-SIRST | dwtfreqnet_original | awgm_original | 2 | 2491399 | `runs/experiment_a_v2/NUDT-SIRST/dwtfreqnet_original/awgm_original` | stopped by request after epoch 15 | historical baseline used |
| A1-NUDT | NUDT-SIRST | dwtfreqnet_wulle_a | awgm_original | 0 | 2500662 | `runs/experiment_a_v2/NUDT-SIRST/dwtfreqnet_wulle_a/awgm_original` | running; epoch 3 | evaluation starts at 100 |
| A0-IRSTD | IRSTD-1K | dwtfreqnet_original | awgm_original | 6 | 2491471 | `runs/experiment_a_v2/IRSTD-1K/dwtfreqnet_original/awgm_original` | stopped by request after epoch 12 | historical baseline used |
| A1-IRSTD | IRSTD-1K | dwtfreqnet_wulle_a | awgm_original | 2 | 2500672 | `runs/experiment_a_v2/IRSTD-1K/dwtfreqnet_wulle_a/awgm_original` | running; epoch 2 | evaluation starts at 100 |

The original plan to delay A2/A3 until A0/A1 completion was superseded after the
historical A0 baselines were accepted and the user requested concurrent W8M runs.

## WULLE + eight-direction Mamba selection

After the historical Original baselines were accepted, the three repeated A0 jobs
were stopped and excluded from formal comparison. The three `awgm_original` WULLE
jobs have priority. The released capacity is also used for a three-dataset WULLE +
W8M run selected from the corrected-direction experiments on the new server.

| Variant | NUAA best mIoU | NUDT best mIoU | IRSTD current best mIoU | Three-dataset mean |
|---|---:|---:|---:|---:|
| w8m_diag4_subband_shared | 0.777742 | 0.952406 | 0.657319 | 0.795822 |
| w8m_diag4_axial_diag_shared | 0.785339 | 0.948034 | 0.653019 | 0.795464 |
| w8m_diag4_axial_diag_shared_dir_embed | 0.774292 | 0.950249 | 0.625487 | 0.783342 |

`w8m_diag4_subband_shared` is selected because it currently wins on NUDT and IRSTD
and has the highest cross-dataset mean. NUAA and NUDT have completion markers for
this variant; the new-server IRSTD run is still in progress, so this selection is
provisional until its final epoch.

| ID | Dataset | Model | AWGM | GPU | PID | Output | Status |
|---|---|---|---|---:|---:|---|---|
| A2-NUAA | NUAA-SIRST | dwtfreqnet_wulle_a | w8m_diag4_subband_shared | 6 | 2502361 | `runs/experiment_a_v2/NUAA-SIRST/dwtfreqnet_wulle_a/w8m_diag4_subband_shared` | running; epoch 1 |
| A2-NUDT | NUDT-SIRST | dwtfreqnet_wulle_a | w8m_diag4_subband_shared | 3 | 2725159 | `runs/experiment_a_v2/NUDT-SIRST/dwtfreqnet_wulle_a/w8m_diag4_subband_shared` | running; just started |
| A2-IRSTD | IRSTD-1K | dwtfreqnet_wulle_a | w8m_diag4_subband_shared | 5 | 2725160 | `runs/experiment_a_v2/IRSTD-1K/dwtfreqnet_wulle_a/w8m_diag4_subband_shared` | running; just started |

After GPU 3 and GPU 5 became available, the waiting W8M scheduler was stopped and
these two jobs were dispatched directly, so all six 226 GPUs permitted for this
experiment set are now occupied by the planned WULLE jobs.

## Validation record

- Structural, 2×256 output-shape, six-output training-mode and gradient tests: passed.
- Four-variant smoke tests: passed (local Mamba-unavailable variants used the explicit test-only fallback; formal A0/A1 use no fallback).
- Haar synthetic line/step mapping and W8M routing checks: passed.
- Server full-shape/backward and no-fallback `awgm_original` checks: passed.
- 226-server deployment: `/DATA20T/bip/cry/code/DWTFreqNet_WULLE_A_V2`.
- Queue scheduler PID: `2491226`; launched at `2026-07-11T23:28:31+08:00`.
- W8M queue scheduler PID: `2502325`; launched at `2026-07-11T23:49:09+08:00`.
- The pre-training dataset-root validation failure is archived separately at
  `runs/experiment_a_v2_failed_dataset_root_20260711_2326`; it ran zero epochs and
  is excluded from all formal results.

## Latest status snapshot (2026-07-12)

### 226 server

The NUAA `awgm_original` WULLE-A and WULLE+Mamba jobs reached epoch 1000. The
other four jobs are still training; the values below are the latest evaluated
records unless explicitly marked as a best checkpoint.

| Dataset | Model/AWGM | Epoch | Latest mIoU | Best mIoU observed |
|---|---|---:|---:|---:|
| NUAA-SIRST | WULLE-A / awgm_original | 1000 | 0.773623 | 0.798215 |
| NUDT-SIRST | WULLE-A / awgm_original | 830 | 0.936626 | 0.941684 |
| IRSTD-1K | WULLE-A / awgm_original | 574 | 0.650734 | ≥0.650734 |
| NUAA-SIRST | WULLE-A / w8m_diag4_subband_shared | 1000 | 0.772083 | 0.781281 |
| NUDT-SIRST | WULLE-A / w8m_diag4_subband_shared | 408 | 0.945403 | pending final |
| IRSTD-1K | WULLE-A / w8m_diag4_subband_shared | 326 | 0.641232 | pending final |

At this snapshot GPUs 1, 4 and 6 were idle; GPUs 0, 2, 3 and 5 were occupied.

### New server W8M matrix

The following are intermediate best checkpoints from
`runs/w8m_haar_aligned_full`; none is treated as a final 1000-epoch result yet.

| Variant | NUAA mIoU | NUDT mIoU | IRSTD mIoU |
|---|---:|---:|---:|
| w8m_diag4_subband_shared | 0.777742 | 0.952406 | 0.657319 |
| w8m_diag4_axial_diag_shared | 0.785339 | 0.948034 | 0.653019 |
| w8m_diag4_axial_diag_shared_dir_embed | 0.774292 | 0.950249 | 0.657663 |
| w8m_diag2_subband_shared | 0.783085 | 0.945438 | 0.646300 |
| w8m_diag4_pair_shared | 0.780801 | 0.947805 | 0.611054 |

The current per-dataset leaders are axial-diag-shared on NUAA, subband-shared on
NUDT, and axial-diag-shared-dir-embed on IRSTD. Subband-shared remains the highest
three-dataset mean among the currently observed checkpoints. The new server had no
GPU satisfying the idle dispatch threshold at this snapshot. Several W8M output
directories also showed more than one active process; therefore these values must
be de-duplicated and revalidated before being used as final paper results.
