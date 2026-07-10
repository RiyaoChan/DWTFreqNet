# DWTFreqNet 实验记录

本文件记录服务器 `cry@202.38.209.226` 上完成或正在运行的 DWTFreqNet 实验。训练数据位于 `/DATA20T/bip/cry/code/SIRST-5K-main/dataset/`，项目目录位于 `/DATA20T/bip/cry/code/`。指标均来自测试集评估，Fa 使用绝对比例记录为 `e-6`。

## 1. 统一实验设置

- 代码项目：`DWTFreqNet`（原始 AWGM）和 `DWTFreqNet_DM_AWGM`（DM-AWGM 变体）。
- 训练环境：服务器 `202.38.209.226`，Python 环境 `/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python`。
- 输入裁剪分辨率：`256 x 256`（`--patch-size 256`）。
- batch size：`4`；workers：`0`；seed：`42`。
- 最大训练轮数：`1000`；从第 `100` 轮开始评估，每 `5` 轮评估一次；每 `20` 轮保存一次检查点。
- 指标：mIoU、nIoU、F1、Pd、Fa。
- 数据集划分规模：NUAA-SIRST `213/214`、NUDT-SIRST `663/664`、IRSTD-1K `800/201`（train/test）。
- 权重、日志、TensorBoard 事件和运行产物不纳入 Git，见 `.gitignore`。

## 2. 原始 AWGM baseline

| 数据集 | 状态/最佳轮次 | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 已完成，best epoch 540 | 77.5118% | 78.6100% | 87.3314% | 95.0382% | 21.5407e-6 |
| NUDT-SIRST | 运行中，best epoch 585 | 94.9807% | 95.2359% | 97.4257% | 99.1534% | 4.2513e-6 |
| IRSTD-1K | 运行中，best epoch 490 | 65.3450% | 64.1243% | 79.0408% | 91.9192% | 15.5435e-6 |

NUAA baseline 已完成 1000 轮；其最终一轮指标为 mIoU `76.0502%`、nIoU `77.7194%`、F1 `86.3961%`、Pd `95.8015%`、Fa `26.4113e-6`。

## 3. DM-AWGM full 实验

| 数据集 | 状态/最新轮次 | 最佳轮次 | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 已完成（1000） | 460 | **79.3581%** | 79.0560% | 88.4913% | 96.9466% | 17.1502e-6 |
| NUDT-SIRST | 运行中（527） | 460 | **94.7862%** | 95.0772% | 97.3233% | 99.3651% | 3.7458e-6 |
| IRSTD-1K | 运行中（385） | 330 | **64.1932%** | 63.4869% | 78.1923% | 90.2357% | 15.8281e-6 |

NUAA DM-AWGM full 相比原始 baseline 的最佳 mIoU 提升 `1.8461` 个百分点，Fa 从 `21.5407e-6` 降至 `17.1502e-6`。NUDT 和 IRSTD 的 full 实验仍在继续运行，表中为当前已出现的最佳结果。

## 4. NUDT-SIRST 消融实验

| 变体 | 状态/最新轮次 | 最佳轮次 | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dm_awgm_no_mamba` | 运行中（338） | 290 | 94.7354% | 94.9422% | 97.2965% | 99.3651% | 4.5041e-6 |
| `dm_awgm_no_dcn` | 运行中（319） | 265 | **95.0498%** | **95.2229%** | **97.4621%** | 99.2593% | 4.9867e-6 |
| `dm_awgm_conv_only` | 运行中（79） | — | — | — | — | — | — |

`dm_awgm_conv_only` 尚未到首次评估轮次，因此目前没有可报告的测试指标。

## 5. 预训练权重推理

文件 `DWTFreqNet_NUDT.pth.tar`（checkpoint epoch 315）已在 NUDT-SIRST 测试集上完成推理：

| mIoU | nIoU | F1 | Pd | Fa |
|---:|---:|---:|---:|---:|
| 94.8818% | 94.9596% | 97.3737% | 99.2593% | 4.1594e-6 |

## 6. 模块检查与复杂度

DM-AWGM 的四个变体均已完成 forward/backward、尺寸、有限值检查。正式服务器环境使用 `mamba_ssm.Mamba` 和 `torchvision.ops.DeformConv2d` 后端。

| 变体 | 参数量 | FLOPs | 单张推理耗时 |
|---|---:|---:|---:|
| `awgm_original` | 37.43M | 66.87G | 35.25 ms |
| `dm_awgm_full` | 41.56M | 68.15G | 45.45 ms |
| `dm_awgm_no_mamba` | 39.18M | 67.53G | 46.58 ms |
| `dm_awgm_no_dcn` | 40.70M | 68.06G | 46.38 ms |
| `dm_awgm_conv_only` | 38.32M | 67.44G | 35.80 ms |

## 7. 运行目录

- baseline：`DWTFreqNet_NUAA-SIRST/runs/train_1000`、`DWTFreqNet_NUDT-SIRST/runs/train_1000`、`DWTFreqNet_IRSTD-1k/runs/train_1000`
- full：`DWTFreqNet_DM_AWGM/runs/full_three_datasets/{NUAA-SIRST,NUDT-SIRST,IRSTD-1K}/dm_awgm_full`
- 消融：`DWTFreqNet_DM_AWGM/runs/stage1_nudt/dm_awgm_{no_mamba,no_dcn,conv_only}`

记录快照日期：2026-07-10。正在运行的任务会继续写入服务器上的 `metrics.jsonl`；本文件是本次提交时的可追溯快照。
