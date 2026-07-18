# Experiment G：解码器 DSHF 语义目标性恢复实验记录

## 1. 实验目的

Experiment G 固定使用 Experiment E 的 E1 编码器（四阶段 `LFSS → AWGM → 原始 Res_block`），编码器结构和训练设置均不改变，只研究在每一级 IDWT 前恢复高频系数是否有效。

| 编号 | 方案 | 含义 |
|---|---|---|
| G0 | `g0_e1_passthrough` | E1 严格直通回归，仅用于等价性和复杂度基准，不重复训练。 |
| G1 | `g1_decoder_dshf` | 对一次对齐后的原始 H/V/D 系数使用 F2 多尺度稀疏 DSHF，并以 `beta=1e-3` 的有界残差恢复。 |
| G2 | `g2_decoder_dshf_semantic` | 在 G1 上增加当前解码层低频语义引导的 H/V/D 方向门控。 |
| G3 | `g3_decoder_dshf_targetness` | 在 G2 上使用已有原生侧输出的 `sigmoid(logit).detach()` 作为目标性提示；不新增监督头和损失。 |

## 2. 固定设置

- 基线提交：`68ede894be748c8842427e140898f007dbe67953`
- F2 Core 来源提交：`3034408051d3742d80473650fe9d198fc37e48ab`
- 数据集：NUAA-SIRST、IRSTD-1K、NUDT-SIRST
- 随机种子 42，batch size 4，patch 256，训练 1000 epoch
- Adam，初始学习率 `1e-3`，10 epoch warmup + CosineAnnealingLR，`eta_min=1e-5`
- epoch 100 开始每个 epoch 评估；每 20 epoch 保存；阈值 0.5
- 所有方案随机初始化，不加载预训练权重
- DWT/IDWT 均固定为 4 次；无第二次 DWT、方向金字塔、LDRC 或额外通道匹配

## 3. 运行顺序

共 9 个正式训练任务，动态检测空闲显卡后按以下顺序启动：

1. G1 / G2 / G3 × NUAA-SIRST
2. G1 / G2 / G3 × IRSTD-1K
3. G1 / G2 / G3 × NUDT-SIRST

G0 使用 E1 已有参考结果，不重复训练：

| 数据集 | best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 595 | 0.7842 | 0.7960 | 0.8790 | 0.9656 | 2.717e-5 |
| IRSTD-1K | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 |
| NUDT-SIRST | 456 | 0.9516 | 0.9482 | 0.9752 | 0.9947 | 1.724e-6 |

## 4. 验证、复杂度和训练状态

### 4.1 服务器验证

验证环境：服务器 `202.38.209.226`，NVIDIA GeForce RTX 3090，CUDA FP32/AMP，输入分辨率 256×256。

- G0 与 E1 的 451 个 state-dict 键完全一致，严格加载成功；最终输出和 A、A_lfss、A_guided、E 中间量逐元素完全一致，最大差异为 0。
- `beta=0` 时 G1 与 E1 严格等价；初始化时 G2 与 G1、G3 与 G2 等价。
- G3 的 targetness 只来自 `gt_conv5/4/3/2` 原生侧输出，使用 `sigmoid(logit).detach()`；没有新建预测头、标签或损失。
- 稀疏门控初始阈值比例为 0.5；语义方向门控初始缩放为 1；`beta_h/v/d` 初值均为 `1e-3`。
- H/V/D 方向、一次对齐顺序、系数形状、有符号高频保护、禁止 `cdist/topk`、4 次 DWT/4 次 IDWT 均通过检查。
- G1/G2/G3 的 FP32、AMP、完整梯度，以及 NUAA-SIRST 真实 batch=4 连续两步更新全部通过。
- 独立训练入口额外完成 G3/NUAA 单批 smoke test：epoch 1 loss 为 7.2771，warmup 后学习率为 `1e-4`，checkpoint 正常写出。

### 4.2 复杂度

统一条件：RTX 3090、FP32、输入 `1×1×256×256`、预热 5 次、计时 20 次。THOP 对 selective scan 的统计能力有限，因此 FLOPs 作为本仓库统一口径的相对比较值。

| 方案 | 参数量 | Decoder DSHF 参数 | THOP FLOPs | 相对 G0 参数 | 相对 G0 FLOPs | 延迟(ms) | FPS | 推理峰值显存 | 训练峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 7,013,527 | 0 | 15.032G | — | — | 12.730 | 78.56 | 190.4 MiB | 1,254.8 MiB |
| G1 | 8,069,151 | 1,055,624 | 16.428G | +15.05% | +9.29% | 17.864 | 55.98 | 337.7 MiB | 1,775.0 MiB |
| G2 | 8,115,315 | 1,101,788 | 16.503G | +15.71% | +9.79% | 19.887 | 50.28 | 369.0 MiB | 1,790.1 MiB |
| G3 | 8,116,107 | 1,102,580 | 16.505G | +15.72% | +9.80% | 20.024 | 49.94 | 314.1 MiB | 1,766.1 MiB |

每个方案的实测 DWT/IDWT 计数均为 4/4。

### 4.3 正式训练启动快照

正式启动时间：2026-07-18 23:58（Asia/Shanghai）。23:50 的首次启动在约 epoch 3–4 时因进一步对齐规范中的“残差能量门控”和“G0 完整子类 forward”而主动停止，原目录已保留为 `experiment_g_decoder_dshf_pre_spec_fix_*`，不会混入正式结果。动态队列不会停止或抢占其他实验/用户进程，只在显存小于 1000 MiB、利用率小于 10% 且无计算进程时占用显卡。

| GPU | 当前任务 | 状态 |
|---:|---|---|
| 0 | G1 / NUAA-SIRST | 运行中 |
| 1 | G2 / NUAA-SIRST | 运行中 |
| 2 | G3 / NUAA-SIRST | 运行中 |
| 3 | G1 / IRSTD-1K | 运行中 |
| 4 | G2 / IRSTD-1K | 运行中 |
| 5 | Experiment F 的 F3 / NUDT-SIRST | 未占用，等待其释放 |
| 6 | G3 / IRSTD-1K | 运行中 |

G1/G2/G3 的三个 NUDT-SIRST 任务按既定优先级处于队列中，任一显卡释放后自动启动。

## 5. 正式结果

训练进行中；epoch 100 开始产生 mIoU、nIoU、F1、Pd、Fa，最终结果将从各任务的 `best_metrics.json` 汇总。
