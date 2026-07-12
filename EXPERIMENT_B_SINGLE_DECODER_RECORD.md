# Experiment B — Single Decoder Directional Pyramid

## Reproducibility

- Base commit: `b98bb4e25b425d9fdf5f2ccbadca6f76af38b539`
- Branch: `codex/experiment-b-single-decoder-directional-pyramid`
- Model: `model/DWTFreqNet_SingleDecoder.py`
- Original and WULLE model files: unchanged
- Dataset split root: `/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets`
- Seed: 42
- Input patch: 256×256
- Epochs: 1000
- Evaluation: every epoch from epoch 100

## Variants

| Variant | Stage-wise AWGM | Directional pyramid | Coefficients |
|---|---:|---:|---|
| sd_raw | No | No | aligned raw H/V/D |
| sd_awgm | Yes | No | aligned raw H/V/D |
| sd_pyramid | No | Yes | raw + directional residual |
| sd_full | Yes | Yes | raw + directional residual |

All variants use exactly four encoder DWT calls, four decoder IDWT calls and one
wavelet decoder. They contain no nested local nodes, dense global nodes, WULLE
decoder, post-AWGM, transformer fusion or LDRC modules.

## Validation

- Syntax/import: passed
- 2×256 train/test output shapes: passed (six train outputs, one test output)
- Intermediate X0/E1–E4/L3–L0 and P1–P4 shapes: passed
- DWT/IDWT count: passed (4/4 for every variant)
- Gradient and bypass checks: passed for all four variants
- Haar H/LH→vertical and V/HL→horizontal: passed; routing aligned
- Complexity profile: passed; both required complexity gates hold

## Complexity

| Model | Parameters | FLOPs | Latency | FPS | Inference peak | Training peak |
|---|---:|---:|---:|---:|---:|---:|
| Original | 37,434,599 | 66.87G | 20.65 ms | 48.43 | 617.35 MiB | 2501.31 MiB |
| WULLE-A | 35,399,143 | 54.69G | 21.80 ms | 45.87 | 550.26 MiB | 2490.46 MiB |
| sd_raw | 5,471,275 | 13.99G | 3.23 ms | 309.34 | 305.54 MiB | 480.01 MiB |
| sd_awgm | 5,925,687 | 14.38G | 5.15 ms | 194.22 | 326.31 MiB | 566.32 MiB |
| sd_pyramid | 11,304,715 | 26.36G | 6.62 ms | 151.05 | 361.50 MiB | 707.34 MiB |
| sd_full | 11,486,007 | 26.51G | 8.27 ms | 120.93 | 369.95 MiB | 745.79 MiB |

The runtime figures are a three-repeat local CUDA check with input `[1,1,256,256]`.
`sd_raw` has 84.55% fewer parameters and 74.41% fewer FLOPs than WULLE-A.
`sd_full` remains 69.31% smaller in parameters and 60.36% lower in FLOPs than
Original, so formal training is allowed to proceed.

## Formal experiment queue

| ID | Dataset | Variant | AWGM | Pyramid | GPU | PID | Output | Status | Best metrics |
|---|---|---|---:|---:|---:|---:|---|---|---|
| B0-NUAA | NUAA-SIRST | sd_raw | No | No | | | `runs/experiment_b/NUAA-SIRST/sd_raw/seed42` | queued | |
| B1-NUAA | NUAA-SIRST | sd_awgm | Yes | No | | | `runs/experiment_b/NUAA-SIRST/sd_awgm/seed42` | queued | |
| B2-NUAA | NUAA-SIRST | sd_pyramid | No | Yes | | | `runs/experiment_b/NUAA-SIRST/sd_pyramid/seed42` | queued | |
| B3-NUAA | NUAA-SIRST | sd_full | Yes | Yes | | | `runs/experiment_b/NUAA-SIRST/sd_full/seed42` | queued | |
| B0-NUDT | NUDT-SIRST | sd_raw | No | No | | | `runs/experiment_b/NUDT-SIRST/sd_raw/seed42` | queued | |
| B3-NUDT | NUDT-SIRST | sd_full | Yes | Yes | | | `runs/experiment_b/NUDT-SIRST/sd_full/seed42` | queued | |
| B1-NUDT | NUDT-SIRST | sd_awgm | Yes | No | | | `runs/experiment_b/NUDT-SIRST/sd_awgm/seed42` | queued | |
| B2-NUDT | NUDT-SIRST | sd_pyramid | No | Yes | | | `runs/experiment_b/NUDT-SIRST/sd_pyramid/seed42` | queued | |
| B0-IRSTD | IRSTD-1K | sd_raw | No | No | | | `runs/experiment_b/IRSTD-1K/sd_raw/seed42` | queued | |
| B3-IRSTD | IRSTD-1K | sd_full | Yes | Yes | | | `runs/experiment_b/IRSTD-1K/sd_full/seed42` | queued | |

## Results

| Dataset | Variant | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | sd_raw | | | | | | |
| NUAA-SIRST | sd_awgm | | | | | | |
| NUAA-SIRST | sd_pyramid | | | | | | |
| NUAA-SIRST | sd_full | | | | | | |
| NUDT-SIRST | sd_raw | | | | | | |
| NUDT-SIRST | sd_awgm | | | | | | |
| NUDT-SIRST | sd_pyramid | | | | | | |
| NUDT-SIRST | sd_full | | | | | | |
| IRSTD-1K | sd_raw | | | | | | |
| IRSTD-1K | sd_full | | | | | | |
