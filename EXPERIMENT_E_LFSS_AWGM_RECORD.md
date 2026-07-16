# Experiment E：LFSS 预处理的 AWGM Encoder 实验记录

更新时间：2026-07-16 18:11（CST）

## 1. 实验目的与方案

本实验只改变单解码器 `sd_awgm` 基线的 Encoder 低频主路：先使用
Wave-Mamba 的 LFSSBlock 建模每一级 DWT 的 LL，再把 LFSS 输出送入原有
StageWiseAWGM。H/V/D 始终是原始 Haar 系数，Decoder、skip fusion、deep
supervision、损失、数据划分和评价指标均不改变。

| ID | 每一级 Encoder 的低频路径 | 含义 |
|---|---|---|
| E0 | `LL -> AWGM -> 原 Res_block` | 已完成的 Experiment B `sd_awgm` 参考基线 |
| E1 | `LL -> LFSS -> AWGM -> 原 Res_block` | 只验证 AWGM 前增加 LFSS 是否有效 |
| E2 | `LL -> LFSS -> AWGM -> Conv1x1(no bias)+BN+GELU` | 验证 LFSS 能否同时替代原 Res_block 的主要提取职责 |

E1/E2 的四级通道为 32/64/128/256，每级一个 LFSSBlock，统一采用
`d_state=16`、`expand=2`、`drop_path=0`、`attn_drop_rate=0`。LFSS 外部没有
额外 residual、LayerScale、gamma、beta 或残差混合，实际执行的是
`refined_ll = lfss(raw_ll)`。

明确未加入：Decoder HFE、D1-D7 relation、LDRC、Directional Pyramid、
第二次 DWT、额外 targetness prior 和额外 loss。模型仍为 4 次 DWT、4 次
IDWT 和单 Wavelet Decoder。

## 2. 分支、来源与部署

- 仓库：`RiyaoChan/DWTFreqNet`
- 基础分支：`codex/experiment-b-single-decoder-directional-pyramid`
- 实际基础 HEAD：`435ab1827ecee4c6b83b669789bb9833a5fd5320`
- Experiment E 分支：`codex/experiment-e-lfss-before-awgm`
- 实现与正式启动提交：`d36b07205146fef128e6edddefeb2cf7df64ee21`
- 独立 Draft PR：`https://github.com/RiyaoChan/DWTFreqNet/pull/4`
- 226 独立部署目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS`
- 数据集目录：`/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets`

LFSS 取自 Wave-Mamba 官方仓库 `AlexZou14/Wave-Mamba` 的
`basicsr/archs/wavemamba_arch.py`，实际 main commit 为
`7e8c63f37af7640e228345c410c2e2165e216117`。本项目只适配了 NCHW 输入，
未改变 `SimpleGate`、`ffn`、`SS2D` 和 `LFSSBlock` 的核心计算。官方 LICENSE
已逐字保存为 `model/third_party/WAVE_MAMBA_LICENSE`，来源和改动说明见
`model/third_party/WAVE_MAMBA_NOTICE.md`；许可证为 CC BY-NC-SA 4.0。

服务器实际环境：PyTorch 2.8.0+cu128、CUDA 12.8、mamba-ssm
2.3.2.post1、einops 0.8.2、timm 1.0.19；selective scan 使用
`mamba_ssm.ops.selective_scan_interface.selective_scan_fn`。服务器原环境缺少
`timm`，本次已在现有 `mirfd_mamba` 环境中补装 1.0.19。

## 3. E0 基线一致性核验

226 上 Experiment B 项目的当前提交虽然包含后续记录更新，但以下四个决定
E0 模型、数据和训练过程的文件，其 Git blob 与本分支基础版本完全一致：

| 文件 | Git blob |
|---|---|
| `model/DWTFreqNet_SingleDecoder.py` | `0ce770c9829f374141b6ca0297eff52508dd145f` |
| `train_experiment_b.py` | `948d578ae0934d977e0e02cacdd93f82b6f44b4e` |
| `dataset.py` | `be2c7faf349592a1a9df2e4671438ec864b34470` |
| `train_one.py` | `8812d3bb8b90db74bc032cb8f203b7941469dfb0` |

历史 E0 均为随机初始化、seed 42、1000 epoch、batch size 4、256×256、Adam
1e-3、10 epoch warmup + cosine（eta_min 1e-5）、epoch 100 起每 epoch 评估、
每 20 epoch 保存。因此不重复占卡训练 E0，直接使用已完成参考结果：

| 数据集 | best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 |
| NUDT-SIRST | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 |
| IRSTD-1K | 894 | 0.6561 | 0.6477 | 0.7924 | 0.9091 | 1.537e-5 |

## 4. 验证结果

服务器 CUDA 全量测试已通过：

- E1/E2 初始化后 LFSS 参数保护前后最大绝对差均为 0；每个模型有 24 个
  LFSS 特殊参数被显式保护。
- 四级 hook 顺序均严格为 `LFSS -> AWGM -> post encoder`；AWGM 收到的张量
  与 LFSS 输出完全相同，并且与原始 LL 不同。
- E2 不经过原 Res_block；E1/E2 的 Decoder 模块名称和结构与 E0 完全一致。
- E1/E2 均通过 FP32 和 AMP 的 `2x1x256x256` 前后向；SSM 的 `A_logs`、
  `Ds`、dt 投影、in/out projection、FFN、skip scale，以及 Encoder、AWGM、
  Decoder、side heads 均获得梯度。
- 两个模型均保持输出尺寸、四次 DWT/四次 IDWT，并通过 Haar H/V 方向检查。
- NUAA 真数据 batch=4、256×256 optimizer step 通过：直接模型测试 loss 分别为
  E1 6.6627、E2 5.6246；正式训练器单 batch 冒烟 loss 分别为 E1 7.1346、
  E2 6.3736。

## 5. 复杂度

测试条件：RTX 3090、FP32、输入 `1x1x256x256`、warmup 5 次、重复 20 次。
THOP 不完整统计 selective-scan CUDA 算子，所以 FLOPs 只适合在同一脚本下
比较，不能视为 LFSS 的精确理论 FLOPs。

| 方案 | 参数量 | 相对 E0 | THOP FLOPs | 相对 E0 | 延迟 | FPS | 推理峰值 | 训练峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 5,925,687 | — | 14.377G | — | 8.316 ms | 120.25 | 178.05 MiB | 403.52 MiB |
| E1 | 7,013,527 | +18.36% | 15.032G | +4.56% | 13.160 ms | 75.99 | 190.33 MiB | 627.99 MiB |
| E2 | 4,734,039 | -20.11% | 12.014G | -16.44% | 11.977 ms | 83.49 | 215.53 MiB | 611.89 MiB |

E1/E2 各含 1,087,840 个 LFSS 参数。E2 删除原四级 Res_block 后，总参数和
THOP FLOPs 均低于 E0；但 selective-scan 激活仍使实测训练显存和延迟高于
E0。

## 6. 正式实验安排与实时状态

所有正式任务均使用随机初始化、seed 42、1000 epoch、batch size 4、256×256，
epoch 100 起每 epoch 评估 mIoU、nIoU、F1、Pd、Fa。调度器仅在显存小于
1000 MiB、利用率小于 10% 且没有 compute PID 时使用 GPU，最多并发 2 个。

固定队列顺序为：

1. E1 NUAA-SIRST、E2 NUAA-SIRST；
2. E1 IRSTD-1K、E2 IRSTD-1K；
3. E1 NUDT-SIRST、E2 NUDT-SIRST。

| 任务 | GPU | 状态（2026-07-16 18:11 CST） | 输出目录 |
|---|---:|---|---|
| E1 NUAA-SIRST | 5 | 运行中，约 epoch 37；尚未到首次评估 | `runs/experiment_e_lfss_awgm/E1_lfss_resblock/NUAA-SIRST/seed42` |
| E2 NUAA-SIRST | 6 | 运行中，约 epoch 17；尚未到首次评估 | `runs/experiment_e_lfss_awgm/E2_lfss_transition/NUAA-SIRST/seed42` |
| E1 IRSTD-1K | 自动 | 排队 | `runs/experiment_e_lfss_awgm/E1_lfss_resblock/IRSTD-1K/seed42` |
| E2 IRSTD-1K | 自动 | 排队 | `runs/experiment_e_lfss_awgm/E2_lfss_transition/IRSTD-1K/seed42` |
| E1 NUDT-SIRST | 自动 | 排队 | `runs/experiment_e_lfss_awgm/E1_lfss_resblock/NUDT-SIRST/seed42` |
| E2 NUDT-SIRST | 自动 | 排队 | `runs/experiment_e_lfss_awgm/E2_lfss_transition/NUDT-SIRST/seed42` |

首次启动时发现调度器存在约两秒的 CUDA 登记竞态，E1/E2 wrapper 曾短暂同时
选择 GPU 5。E2 的这次无效启动已停止并完整归档到
`runs/experiment_e_lfss_awgm/_invalid_startup_gpu_collision_20260716_1803/`，不纳入
任何结果。调度器已增加任务级 GPU 预留判断并关闭子进程继承的队列文件锁；
正式 E2 已重新在 GPU 6 从 epoch 1 随机初始化启动，E1 未中断。

训练完成后运行 `tools/analyze_experiment_e_low_frequency.py`，从 best checkpoint
统计各级 raw LL、LFSS 输出和 AWGM 引导后特征的目标/背景响应比、LFSS 改变量、
AWGM gate 和 H/V/D 权重；该诊断不参与训练或模型选择。
