# DWTFreqNet 实验记录

> Experiment H：固定 Experiment E1，在四级 Decoder IDWT 前加入 LFP 高频净化，
> 比较 raw LL 与 E4/L3/L2/L1 两种低频来源以及 H1/H2/H3 三种净化策略。
> 六个变体 × 三个数据集共18项已在 226 动态队列执行，详见
> `EXPERIMENT_H_DECODER_LFP_RECORD.md`。

> Experiment E：LFSS 预处理的 AWGM Encoder 已建立独立分支并在 226 服务器
> 按 E1/E2 × NUAA-SIRST、IRSTD-1K、NUDT-SIRST 共 6 项正式实验排队执行。
> 完整中文方案、验证、复杂度和实时结果见 `EXPERIMENT_E_LFSS_AWGM_RECORD.md`。

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

## 12. Haar H/V 方向对应检查

使用奇数坐标处的合成单像素水平线、垂直线以及水平/垂直阶跃边缘检查当前 `HaarWaveletTransform`。结果一致为：

- 代码返回的 `H=LH`：垂直线响应 16，水平线响应 0，即实际对应垂直结构；
- 代码返回的 `V=HL`：水平线响应 16，垂直线响应 0，即实际对应水平结构；
- 轴对齐合成信号的 `D=HH` 响应为 0。

原 W8M 实现中的 `H -> horizontal scan`、`V -> vertical scan` 与该 Haar 实现的真实方向相反。现已修正为 `H=LH -> TB/BT vertical scan`、`V=HL -> LR/RL horizontal scan`，而 H/V 在融合器中的槽位与 DWT/IDWT 参数顺序均保持不变。`tools/check_haar_direction_mapping.py --require-aligned-routing` 与路由输入捕获单元测试共同防止后续滤波器或路由变化再次静默交换方向。

## 13. Haar 对齐修正版重启

2026-07-10 按要求停止新服务器上此前采用反向 H/V 扫描对应关系的六个 W8M 任务。旧目录、日志和 checkpoint 全部保留，并写入 `INVALID_HV_SCAN_ROUTING` 标记；这些运行只可作为错误路由对照，不纳入正式结果。修正版使用全新输出根目录 `/root/autodl-tmp/DWTFreqNet_W8M/runs/w8m_haar_aligned_full`，所有任务从 epoch 0 开始直接训练到 epoch 1000，不从旧 checkpoint 恢复：

旧任务停止快照如下（Fa 以 `×10^-6` 表示，全部为无效路由结果）：

| 数据集 | 变体 | 停止 epoch | 最佳 epoch | mIoU (%) | nIoU (%) | F1 (%) | Pd (%) | Fa |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUDT-SIRST | `w8m_diag4_subband_shared` | 186 | 175 | 90.6536 | 91.3422 | 95.0977 | 99.0476 | 11.9037 |
| NUDT-SIRST | `w8m_diag4_axial_diag_shared` | 183 | 175 | 92.8789 | 92.8863 | 96.3080 | 99.2593 | 4.7569 |
| NUAA-SIRST | `w8m_diag4_subband_shared` | 505 | 495 | 78.3510 | 78.8973 | 87.8616 | 95.8015 | 14.4748 |
| NUAA-SIRST | `w8m_diag4_axial_diag_shared` | 499 | 435 | 77.4831 | 78.6239 | 87.3132 | 96.5649 | 19.8256 |
| IRSTD-1K | `w8m_diag4_subband_shared` | 137 | 105 | 61.4813 | 61.3829 | 76.1466 | 90.2357 | 21.9393 |
| IRSTD-1K | `w8m_diag4_axial_diag_shared` | 137 | 135 | 64.7704 | 61.5094 | 78.6190 | 89.5623 | 42.1894 |

| GPU | 数据集 | 变体 |
|---:|---|---|
| 0 | NUDT-SIRST | `w8m_diag4_subband_shared` |
| 1 | NUDT-SIRST | `w8m_diag4_axial_diag_shared` |
| 2 | NUAA-SIRST | `w8m_diag4_subband_shared` |
| 3 | NUAA-SIRST | `w8m_diag4_axial_diag_shared` |
| 4 | IRSTD-1K | `w8m_diag4_subband_shared` |
| 5 | IRSTD-1K | `w8m_diag4_axial_diag_shared` |

重启前必须同时通过方向严格检查、8 项 W8M 单元测试和真实 CUDA forward/backward smoke test。六项任务统一使用 seed 42、batch size 4、patch size 256、1000-epoch cosine scheduler，并持续记录 mIoU、nIoU、F1、Pd、Fa 与方向门控统计。

上述检查均已通过；真实 CUDA smoke test 使用 `mamba_ssm.Mamba`，验证了有限梯度、非零 Mamba 梯度和 checkpoint round-trip。六项修正版任务于 2026-07-10 15:45:37 CST 启动，启动日志中的 `haar_routing_aligned` 均为 `true`，每卡初始显存约 12.6GB。

## 14. 修正版评估频率调整

新服务器上的 Haar 对齐修正版实验统一从 epoch 100 开始评估，并从原来的每 5 epoch 评估一次改为每个 epoch 都评估一次，即 `--eval-start 100 --eval-every 1`。调整时六项任务均未到首次评估轮次；为保证 `metrics.jsonl`、`run_config.json` 和启动参数不混合两种频率，初始短运行完整归档到 `/root/autodl-tmp/DWTFreqNet_W8M/runs/w8m_haar_aligned_eval5_initial_20260710_154537`，正式目录从 epoch 0 重新启动。归档时 NUAA-SIRST 两项到 epoch 23、NUDT-SIRST 两项到 epoch 7、IRSTD-1K 两项到 epoch 5，均不纳入正式结果。六个正式任务的进程参数与 `run_config.json` 已逐项确认 `eval_start=100`、`eval_every=1`。

## 15. 旧服务器跨数据集 DM-AWGM 消融动态队列

旧服务器上已有 NUDT-SIRST 的 `dm_awgm_no_mamba`、`dm_awgm_no_dcn`、`dm_awgm_conv_only` 消融。新增 NUAA-SIRST 和 IRSTD-1K 的同三项消融，共 6 个新实验。使用 `scripts/schedule_dm_awgm_ablation_idle_gpus.sh` 动态检查 GPU 显存和利用率：仅当显存不超过 2GB 且 GPU 利用率不超过 5% 时占用该卡；当前任务完成释放 GPU 后，队列自动启动下一项，不固定绑定 GPU 0/2，也不抢占已有实验。每项使用 batch size 4、patch size 256、seed 42、1000 epoch，并从 epoch 100 开始每 5 epoch 评估一次。

队列于 2026-07-10 23:31:17 CST 启动，首批为 GPU 0 的 NUAA `dm_awgm_no_mamba` 和 GPU 2 的 NUAA `dm_awgm_no_dcn`。此前 23:29 的一次启动尝试因旧服务器 `train_one.py` 不支持 `--stop-after-epoch` 而在参数解析阶段退出，没有产生训练指标，目录已归档至 `/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/runs/ablation_two_datasets_failed_stop_after_20260710_232919`。

## 16. W8M 优先级队列补齐

规范中的 W8M 方案共有 7 个，覆盖 3 个数据集时应形成 21 个 full 实验。此前新服务器只启动了 `w8m_diag4_subband_shared` 和 `w8m_diag4_axial_diag_shared` 两个方案（6 个实验），遗漏了第一优先级的 `w8m_diag4_axial_diag_shared_dir_embed` 以及第二、第三优先级的 4 个方案。现已保留正在运行的 6 个任务，并在同一修正版输出根目录启动 `scripts/schedule_w8m_missing_variants_idle_gpus.sh`，将遗漏的 15 个实验按规范顺序动态排队：

1. `w8m_diag4_axial_diag_shared_dir_embed`（3 个数据集）
2. `w8m_diag2_subband_shared`、`w8m_diag4_pair_shared`（各 3 个数据集）
3. `w8m_diag4_independent`、`w8m_diag4_all_shared`（各 3 个数据集）

调度器于 2026-07-10 23:44:14 CST 启动；当时 6 张 GPU 全部被正式任务占用，因此队列保持等待状态，不抢占当前任务。首次获得空闲 GPU 后发现 runner 的 `OUTPUT_ROOT` 需要按数据集展开，导致两个短暂的 `dir_embed` 进程错误共用变体目录；这些权重和日志已归档至 `/root/autodl-tmp/DWTFreqNet_W8M/runs/w8m_invalid_output_collision_20260711_090000` 并标记为无效。修正后的调度器于 2026-07-11 09:01:17 CST 重新启动，当前 NUDT-SIRST 与 NUAA-SIRST 的 `w8m_diag4_axial_diag_shared_dir_embed` 已分别写入独立目录；后续任务继续按空闲 GPU 自动排队。

## 17. 结果快照（2026-07-11 20:10 CST）

以下快照直接对应服务器上的 `metrics.jsonl` 和 `best_metrics.json`；mIoU、nIoU、F1、Pd 为百分比，Fa 为 `×10^-6`。尚未开始测试集评估的任务以 `—` 表示。

### 新服务器 W8M

| 数据集 | 方案 | 当前 epoch | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `w8m_diag4_subband_shared` | 1000 | 294 | 77.7742 | 78.0461 | 87.4977 | 95.8015 | 32.6540 |
| NUAA-SIRST | `w8m_diag4_axial_diag_shared` | 1000 | 350 | 78.5339 | 78.4651 | 87.9765 | 95.8015 | 19.2082 |
| NUAA-SIRST | `w8m_diag4_axial_diag_shared_dir_embed` | 1000 | 546 | 77.4292 | 78.0457 | 87.2790 | 95.4198 | 18.6594 |
| NUDT-SIRST | `w8m_diag4_subband_shared` | 950 | 337 | 95.2406 | 95.3566 | 97.5623 | 99.4709 | 3.0334 |
| NUDT-SIRST | `w8m_diag4_axial_diag_shared` | 951 | 491 | 94.8034 | 95.2712 | 97.3324 | 99.3651 | 6.0897 |
| NUDT-SIRST | `w8m_diag4_axial_diag_shared_dir_embed` | 396 | 392 | 95.0249 | 95.2632 | 97.4490 | 99.4709 | 3.0104 |
| IRSTD-1K | `w8m_diag4_subband_shared` | 726 | 599 | 65.7319 | 65.1970 | 79.3232 | 91.5825 | 18.8647 |
| IRSTD-1K | `w8m_diag4_axial_diag_shared` | 728 | 700 | 65.3019 | 65.3385 | 79.0092 | 92.2559 | 13.3230 |
| IRSTD-1K | `w8m_diag4_axial_diag_shared_dir_embed` | 46 | — | — | — | — | — | — |

其余 12 个 W8M 方案-数据集组合仍在动态队列中，未开始训练。

### 旧服务器 baseline 与 DM-AWGM full

| 数据集 | 方案 | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | baseline | 540 | 77.5118 | 78.6100 | 87.3314 | 95.0382 | 21.5407 |
| NUAA-SIRST | `dm_awgm_full` | 460 | 79.3581 | 79.0560 | 88.4913 | 96.9466 | 17.1502 |
| NUDT-SIRST | baseline | 585 | 94.9807 | 95.2359 | 97.4257 | 99.1534 | 4.2513 |
| NUDT-SIRST | `dm_awgm_full` | 530 | 94.8446 | 95.2489 | 97.3541 | 99.5767 | 6.2506 |
| IRSTD-1K | baseline | 490 | 65.3450 | 64.1243 | 79.0408 | 91.9192 | 15.5435 |
| IRSTD-1K | `dm_awgm_full` | 885 | 65.9503 | 65.7152 | 79.4820 | 91.5825 | 13.8923 |

### 旧服务器 DM 消融

| 数据集 | 方案 | 当前 epoch | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUDT-SIRST | `dm_awgm_no_mamba` | 1000 | 635 | 95.3122 | 95.4920 | 97.5999 | 99.3651 | 3.6309 |
| NUDT-SIRST | `dm_awgm_no_dcn` | 1000 | 535 | 95.2509 | 95.3150 | 97.5677 | 99.0476 | 3.4930 |
| NUDT-SIRST | `dm_awgm_conv_only` | 1000 | 510 | 95.2752 | 95.2868 | 97.5804 | 99.2593 | 3.6079 |
| NUAA-SIRST | `dm_awgm_no_mamba` | 1000 | 225 | 77.3049 | 77.3722 | 87.2000 | 95.8015 | 23.4615 |
| NUAA-SIRST | `dm_awgm_no_dcn` | 1000 | 855 | 78.2857 | 78.1710 | 87.8205 | 96.5649 | 14.2690 |
| NUAA-SIRST | `dm_awgm_conv_only` | 1000 | 395 | 78.3583 | 78.8942 | 87.8662 | 95.8015 | 16.9444 |
| IRSTD-1K | `dm_awgm_no_mamba` | 671 | 450 | 65.2340 | 64.9133 | 78.9595 | 92.5926 | 24.8240 |
| IRSTD-1K | `dm_awgm_no_dcn` | 596 | 580 | 65.5795 | 65.8119 | 79.2121 | 93.9394 | 14.7464 |
| IRSTD-1K | `dm_awgm_conv_only` | 637 | 625 | 65.9995 | 66.1821 | 79.5177 | 92.5926 | 13.9493 |

## 18. 新服务器 W8M/Mamba 最新结果回填（停止前快照，2026-07-13）

新服务器上仍有此前部署的 W8M-AWGM（Mamba）实验。以下结果从
`/root/autodl-tmp/DWTFreqNet_W8M/runs/w8m_haar_aligned_full` 下各实验的
`best_metrics.json` 和 `metrics.jsonl` 回填；指标为百分比，Fa 为 `×10^-6`。
旧的 H/V 方向相反实验已标记为无效，不纳入本表。

| 数据集 | 方案 | 状态/当前 epoch | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `w8m_diag2_subband_shared` | 完成/1000 | 553 | 78.3085 | 78.4080 | 87.8348 | 96.1832 | 15.5038 |
| NUAA-SIRST | `w8m_diag4_subband_shared` | 完成/1000 | 294 | 77.7742 | 78.0461 | 87.4977 | 95.8015 | 32.6540 |
| NUAA-SIRST | `w8m_diag4_axial_diag_shared` | 完成/1000 | 350 | 78.5339 | 78.4651 | 87.9765 | 95.8015 | 19.2082 |
| NUAA-SIRST | `w8m_diag4_axial_diag_shared_dir_embed` | 完成/1000 | 546 | 77.4292 | 78.0457 | 87.2790 | 95.4198 | 18.6594 |
| NUAA-SIRST | `w8m_diag4_pair_shared` | 完成/1000 | 449 | 78.0801 | 78.3486 | 87.6910 | 95.4198 | 16.8758 |
| NUAA-SIRST | `w8m_diag4_independent` | 完成/1000 | 355 | 78.5683 | 78.8043 | 87.9981 | 96.1832 | 33.4772 |
| NUAA-SIRST | `w8m_diag4_all_shared` | 进行中/663 | 334 | 78.1965 | 79.5442 | 87.7643 | 95.4198 | 30.9390 |
| NUDT-SIRST | `w8m_diag2_subband_shared` | 完成/1000 | 337 | 94.5438 | 95.1337 | 97.1954 | 99.5767 | 7.6294 |
| NUDT-SIRST | `w8m_diag4_subband_shared` | 完成/1000 | 337 | 95.2406 | 95.3566 | 97.5623 | 99.4709 | 3.0334 |
| NUDT-SIRST | `w8m_diag4_axial_diag_shared` | 完成/1000 | 491 | 94.8034 | 95.2712 | 97.3324 | 99.3651 | 6.0897 |
| NUDT-SIRST | `w8m_diag4_axial_diag_shared_dir_embed` | 完成/1000 | 392 | 95.0249 | 95.2632 | 97.4490 | 99.4709 | 3.0104 |
| NUDT-SIRST | `w8m_diag4_pair_shared` | 完成/1000 | 510 | 94.9605 | 95.2263 | 97.4151 | 99.1534 | 3.8836 |
| NUDT-SIRST | `w8m_diag4_independent` | 进行中/961 | 271 | 94.9061 | 94.9621 | 97.3865 | 99.1534 | 3.2862 |
| NUDT-SIRST | `w8m_diag4_all_shared` | 进行中/327 | 295 | 95.0420 | 95.2982 | 97.4580 | 99.3651 | 3.6079 |
| IRSTD-1K | `w8m_diag2_subband_shared` | 完成/1000 | 458 | 66.1710 | 66.4951 | 79.6420 | 92.9293 | 16.1508 |
| IRSTD-1K | `w8m_diag4_subband_shared` | 完成/1000 | 599 | 65.7319 | 65.1970 | 79.3232 | 91.5825 | 18.8647 |
| IRSTD-1K | `w8m_diag4_axial_diag_shared` | 完成/1000 | 700 | 65.3019 | 65.3385 | 79.0092 | 92.2559 | 13.3230 |
| IRSTD-1K | `w8m_diag4_axial_diag_shared_dir_embed` | 完成/1000 | 699 | 66.0666 | 65.9325 | 79.5664 | 92.9293 | 16.2267 |
| IRSTD-1K | `w8m_diag4_pair_shared` | 进行中/806 | 694 | 65.9442 | 66.0148 | 79.4776 | 92.5926 | 14.1391 |
| IRSTD-1K | `w8m_diag4_independent` | 进行中/277 | 251 | 64.5102 | 64.1935 | 78.4270 | 89.2256 | 30.0052 |
| IRSTD-1K | `w8m_diag4_all_shared` | 进行中/28 | — | — | — | — | — | — |

因此，新服务器上的 Mamba/W8M 实验结果已经有历史快照记录；本节补充了
2026-07-13 停止前的最新完整结果和当时进行中的任务。停止前已完成 15/21
个 W8M 组合，其余 6 个的训练均已停止，训练日志和 checkpoint 均保存在对应输出目录。

## 19. 新服务器任务停止记录（2026-07-13）

按最新调度要求，新服务器上的全部训练任务和 W8M/Experiment B 调度器已停止。
停止前的 W8M/Mamba 结果已在第 18 节记录；所有 checkpoint、`metrics.jsonl`、
`best_metrics.json` 和日志文件均保留，后续如需恢复可从 `latest.pth.tar` 继续。

## 20. Phase 1红外小目标任务先验验证

从 Experiment H 最新提交 `8cfd7a97bd460b07efbad28ca7b709d7277cdd1b`
建立独立分支 `codex/phase1-irst-task-prior-validation`。本阶段不训练新模型，
也不修改任何 `model/*` 或训练代码；只通过 P1 Gaussian/椭圆紧凑性、P2
LL/H/V/D方向一致性、P3公平结构化采样几何和 Experiment H 交叉分析验证
任务先验。完整实现、资源规则、PID、错误与结果见
`PHASE1_TASK_PRIOR_VALIDATION_RECORD.md`，最终输出位于
`analysis/phase1_task_prior_validation/final/`。
