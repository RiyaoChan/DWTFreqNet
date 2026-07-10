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

## 8. W8M-AWGM 新实验计划

W8M-AWGM 将 D 分支的卷积/DCN 替换为斜对角 Mamba，并保持 H/V/D 经 `DirectionFusionGate` 引导低频分量的方式不变。默认对角序列使用 `snake` 顺序；多条对角线拼成一条序列，因此第一阶段允许状态跨对角线边界传播。

### Stage 0：功能验证

- 验证 3x3 固定索引、任意矩形尺寸的完整性和逆排列。
- 验证共享对象数量、共享梯度、route 输出差异和 checkpoint 严格加载。
- 对每个 variant 记录参数量、FLOPs、延迟、FPS 和峰值显存。

### Stage 1：三数据集统一 1000-epoch 实验

所有任务使用 seed 42、patch size 256、batch size 4 和 1000-epoch 余弦调度。epoch 400 只作为中途检查点和阶段记录，不作为淘汰条件，所有任务都会自动续训到 epoch 1000。原先的变体优先级如下，当前六个实际运行方案均保留：

1. `w8m_diag4_subband_shared`
2. `w8m_diag4_axial_diag_shared`
3. `w8m_diag4_axial_diag_shared_dir_embed`
4. `w8m_diag2_subband_shared`
5. `w8m_diag4_pair_shared`
6. `w8m_diag4_independent`
7. `w8m_diag4_all_shared`
8. `awgm_original`
9. `dm_awgm_full`
10. `dm_awgm_no_dcn`

所有当前方案都会从 epoch 400 的 checkpoint 原地恢复到 epoch 1000，最终以 1000 epoch 内的最佳指标进行比较。

### Stage 2 与 Stage 3

- Stage 2：汇总六个 1000-epoch 结果，并按数据集选择最优 W8M 方案，与原始 AWGM/DM-AWGM baseline 对比。
- Stage 3：最终最佳 W8M 使用 seed 42、3407、2026 复验并报告五项指标的 mean ± standard deviation。

每个输出目录的 `run_config.json` 记录 variant、数据集、seed、参数量、后端、共享方式、对角方向数、对角顺序、direction embedding 和 checkpoint/log 路径；`metrics.jsonl` 额外记录 `mean_G_H/V/D`、attention mean/std 以及 axial/diagonal feature norm。

## 9. W8M 新服务器部署与 Stage 0 结果

部署时间：2026-07-10；服务器：`connect.nmb2.seetacloud.com:31570`；项目目录：`/root/autodl-tmp/DWTFreqNet_W8M`；环境：Python 3.10、PyTorch 2.8.0+cu128、torchvision 0.23.0、mamba_ssm 2.3.2.post1；GPU：6 x RTX 3090 24GB。数据集从原实验服务器完整复制，共 5516 个文件且没有残留软链接。

Stage 0 的 6 项单元测试全部通过，包括 3x3 固定索引、矩形尺寸完整性、逆排列、2-Mamba/1-Mamba 共享关系、共享梯度和不同 route 输出。七个 W8M 变体均完成真实 CUDA forward/backward、finite gradient、checkpoint round-trip、延迟和峰值显存测试。

| Variant | Mamba 数/每个 AWGM | 总参数 | AWGM 参数 | 延迟 | FPS | batch-1 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| `w8m_diag2_subband_shared` | 3 | 40.10M | 2.68M | 50.39 ms | 19.85 | 2.82GB |
| `w8m_diag4_independent` | 8 | 43.08M | 5.66M | 55.09 ms | 18.15 | 2.91GB |
| `w8m_diag4_pair_shared` | 4 | 40.70M | 3.27M | 52.12 ms | 19.19 | 2.90GB |
| `w8m_diag4_subband_shared` | 3 | 40.10M | 2.68M | 54.72 ms | 18.27 | 2.90GB |
| `w8m_diag4_axial_diag_shared` | 2 | 39.50M | 2.08M | 53.86 ms | 18.57 | 2.90GB |
| `w8m_diag4_axial_diag_shared_dir_embed` | 2 | 39.51M | 2.09M | 52.83 ms | 18.93 | 2.90GB |
| `w8m_diag4_all_shared` | 1 | 38.90M | 1.48M | 54.05 ms | 18.50 | 2.90GB |

`w8m_diag4_subband_shared` 的 THOP 近似 FLOPs 为 68.70G；THOP 不精确统计 selective-scan 自定义 CUDA kernel，因此该值仅用于同一工具口径下对比。batch size 4 的整网 forward/backward 峰值显存为 11.04GB，正式训练无需降低 batch size。

## 10. W8M Stage 1 已启动安排

Stage 1 于 2026-07-10 11:49 CST 启动。原先六个任务全部使用 NUDT-SIRST；后按要求修正为三个数据集各两个任务。所有任务使用 seed 42、batch size 4、patch size 256、1000-epoch 余弦调度，epoch 400 仅记录中途结果，随后自动续训到 epoch 1000。

| GPU | 当前任务 | 后续队列 |
|---:|---|---|
| 0 | `w8m_diag4_subband_shared` | `w8m_diag4_all_shared` |
| 1 | `w8m_diag4_axial_diag_shared` | `awgm_original` |
| 2 | `w8m_diag4_axial_diag_shared_dir_embed` | `dm_awgm_full` |
| 3 | `w8m_diag2_subband_shared` | `dm_awgm_no_dcn` |
| 4 | `w8m_diag4_pair_shared` | — |
| 5 | `w8m_diag4_independent` | — |

运行输出：`/root/autodl-tmp/DWTFreqNet_W8M/runs/w8m_stage1_nudt/<variant>`；队列日志：`/root/autodl-tmp/DWTFreqNet_W8M/runs/w8m_stage1_nudt/queues/gpu_<id>.log`。当前六个首批任务均已进入训练，单卡占用约 12.3–12.7GB。

## 11. 多数据集调度修正

根据实验执行要求，Stage 1 已从“六个 NUDT 变体”修正为“三个数据集各两个优先变体”。保留 NUDT 的两个任务，停止另外四个尚未到首次评估的 NUDT 任务，并重新分配如下：

| GPU | 数据集 | 变体 |
|---:|---|---|
| 0 | NUDT-SIRST | `w8m_diag4_subband_shared` |
| 1 | NUDT-SIRST | `w8m_diag4_axial_diag_shared` |
| 2 | NUAA-SIRST | `w8m_diag4_subband_shared` |
| 3 | NUAA-SIRST | `w8m_diag4_axial_diag_shared` |
| 4 | IRSTD-1K | `w8m_diag4_subband_shared` |
| 5 | IRSTD-1K | `w8m_diag4_axial_diag_shared` |

六个任务均使用 seed 42、batch size 4、patch size 256 和 1000-epoch scheduler。每个任务在 epoch 400 写入 `STAGE1_COMPLETED` 后，由 watcher 自动用同一输出目录的 checkpoint 继续到 epoch 1000；新任务输出目录为 `/root/autodl-tmp/DWTFreqNet_W8M/runs/w8m_stage1_multi/<dataset>/<variant>`，并会继续记录五项性能指标以及方向统计。

重新分配时停止的四个 NUDT 任务只完成了约 16–18 个 epoch，均未到首次评估轮次，不作为最终实验结论；对应目录写入 `STOPPED_DATASET_REALLOCATION` 标记。当前有效运行目录只有上表的六个数据集-变体组合。
