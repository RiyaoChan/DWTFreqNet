# Experiment D：Decoder HFE 空间一致性消融记录（D5–D7）

## 1. 实验定位

D5、D6、D7 是 Experiment D：`SD-AWGM + Decoder-side HFE` 的内部空间关系
消融，不是新的主实验，也不建立 Experiment E。三者均基于 D4 的 Direct Low
Fusion，删除全图高低频通道匹配，研究同位置或局部邻域的高低频空间一致性。

- 开始时实际 HEAD：`e5747b7f35fed3ecd3702a5f45332a8c35be8bd3`
- 工作分支：`codex/experiment-d-hfe-matching-ablation-d2-d3`
- GitHub：Draft PR #3
- 服务器：`202.38.209.226`
- 项目：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_D_ABLATION`
- 数据集：`/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets`

规范文件：

```text
experiment_guide/EXPERIMENT_D_D5_D6_D7_SPATIAL_CONSISTENCY_FOR_CODEX.md
```

## 2. 三个方案的含义

| ID | 训练标识 | 模型标识 | 空间关系 | 偏移搜索 | 目标先验 |
|---|---|---|---|---:|---:|
| D5 | `d5_same_position` | `sd_awgm_hfe_samepos` | 同位置余弦一致性 + 低频局部对比 | 否 | 否 |
| D6 | `d6_neighborhood` | `sd_awgm_hfe_neighborhood` | 3×3 邻域局部注意力 | 是 | 否 |
| D7 | `d7_target_neighborhood` | `sd_awgm_hfe_targetlocal` | targetness 引导的 3×3 邻域注意力 | 是 | 是 |

共同保留 Stage-wise AWGM Encoder、原始 H/V/D 系数对齐、SKFF、HFE Channel
Attention/FFN、三个方向残差头、`beta=1e-3`、四级 IDWT、Single Decoder 和现有
Deep Supervision。均不使用 `torch.cdist`、`torch.topk`、Top-1/Top-k、argmin、
候选通道索引或 `[B,C,C]` 高频—低频通道相似度矩阵。

## 3. D5：Same-position Consistency

D5 将高频与当前低频分别投影到固定嵌入通道，在每个空间位置计算余弦一致性；
同时从低频 1×1 响应与其 3×3 局部均值之差得到局部对比。两者经空间门控生成：

```text
spatial_scale: [B,1,H,W]
conditioned_low = spatial_scale × low_feature
```

空间门控为 `Conv3×3(2→8) + GELU + Conv3×3(8→1)`，最后一层权重和偏置均
初始化为零，因此 `spatial_scale = 2×sigmoid(0) = 1`。训练入口执行统一
`init_weights` 后会再次恢复该零初始化，确保正式实验初始时 D5 精确退化为 D4。

## 4. D6：3×3 Neighborhood Consistency

D6 的固定 offset 顺序为：

```text
(-1,-1), (-1,0), (-1,1),
( 0,-1), ( 0,0), ( 0,1),
( 1,-1), ( 1,0), ( 1,1)
```

中心索引为 `4`。实现采用 replicate padding 和逐 offset 切片，定义为
`output[p] = input[p + (dy,dx)]`。只保留 `[B,9,H,W]` 的局部注意力，不构造
`[B,C,9,H,W]` 的长期 value tensor；低频 value 按 offset 逐项加权累加。

## 5. D7：Target-aware Neighborhood Consistency

D7 复用现有 Deep Supervision side head，不新增 head 或 loss：

| Stage | 当前低频 | Side head |
|---:|---|---|
| 4 | E4 | `gt_conv5` |
| 3 | L3 | `gt_conv4` |
| 2 | L2 | `gt_conv3` |
| 1 | L1 | `gt_conv2` |

每级使用 `targetness = sigmoid(side_logit).detach()`，HFE 不通过 prior 反向操纵
side head；side head 仍由原有 Deep Supervision Loss 获得梯度。目标先验以
`softplus(raw_targetness_scale) × log(targetness)` 加入 D6 的 3×3 局部 logits，
`targetness_scale` 初值精确为 1。

## 6. 实现文件

新增：

```text
model/DWTFreqNet_SingleDecoder_HFE_SpatialAblation.py
train_experiment_d_hfe_spatial_ablation.py
tools/test_experiment_d_hfe_spatial_ablation.py
tools/profile_experiment_d_hfe_spatial_ablation.py
tools/analyze_experiment_d_spatial_consistency.py
scripts/run_experiment_d_hfe_spatial_ablation.sh
scripts/launch_experiment_d_hfe_spatial_ablation_queue.sh
EXPERIMENT_D_HFE_SPATIAL_CONSISTENCY_RECORD.md
```

以下原始模型文件未修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
model/DWTFreqNet_SingleDecoder.py
model/DWTFreqNet_SingleDecoder_LDRC.py
model/DWTFreqNet_SingleDecoder_HFE.py
```

## 7. 单元测试与真数据验证

| 检查项 | 结果 |
|---|---|
| Python 编译 | 通过 |
| Linux Bash `-n` | 两个启动脚本均通过 |
| CPU 小尺寸前向/反向 | D5/D6/D7 全部通过 |
| CUDA `2×1×256×256` FP32 | 三方案前向、反向全部通过 |
| CUDA AMP | 三方案均无 NaN/Inf/OOM |
| D5 spatial scale | 四级形状正确，初始化最小值/最大值均为 1 |
| D5 → D4 初始退化 | 最大绝对误差 `0` |
| D6 offset 方向 | 9 个 offset 合成坐标图检查全部通过 |
| D6 attention | `[B,9,H,W]`，通道和误差不超过 `2.384e-7` |
| D6 中心权重 → D4 | 最大绝对误差 `0` |
| D7 均匀 prior → D6 | 完整 CUDA 测试最大绝对误差 `8.941e-8` |
| D7 prior detach | HFE loss 不回传 side logit；四级 prior 均 `requires_grad=False` |
| D7 side head 梯度 | `gt_conv5/4/3/2` 均通过原深监督获得非零梯度 |
| `torch.cdist/topk` monkeypatch | D5/D6/D7 完整前向均通过 |
| Matching 模块扫描 | D5/D6/D7 均不存在全局 Matching 模块 |
| beta 全置零 → D0 | 三方案最大绝对误差均为 `0` |
| DWT/IDWT | 三方案均为 `4/4` |
| NUAA 真数据 smoke | 三方案 batch=4、256、单步 optimizer.step 全部通过 |
| run_config/checkpoint | 变体、结构、prior、DWT/IDWT 元数据逐项通过 |

NUAA 单步 smoke 的初始损失分别为 D5 `7.1742`、D6 `6.9934`、D7 `6.6994`；
这里只用于确认数据流和优化器有效，不能作为模型优劣结论。

## 8. 参数、FLOPs、速度和显存

统一在 RTX 3090、`1×1×256×256`、warmup=5、repeat=20 下测量。THOP 可能不
统计 shift、局部相似度和注意力加权，因此同时报告实测延迟和峰值显存。

| ID | 模型 | Params | Relation params | THOP FLOPs | Latency | FPS | Infer peak | Train peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | `sd_awgm` | 5.926M | 0 | 14.377G | 8.40ms | 119.07 | 178.05MiB | 403.52MiB |
| D1 | `sd_awgm_hfe` | 10.181M | 1,846,784 | 20.473G | 19.22ms | 52.04 | 260.61MiB | 923.01MiB |
| D2 | `sd_awgm_hfe_softcos` | 10.181M | 1,846,792 | 20.473G | 19.88ms | 50.31 | 269.22MiB | 1007.78MiB |
| D3 | `sd_awgm_hfe_scaleaware` | 10.225M | 1,890,820 | 20.769G | 19.25ms | 51.94 | 308.43MiB | 1095.19MiB |
| D4 | `sd_awgm_hfe_nomatch` | 10.181M | 1,846,784 | 20.473G | 17.59ms | 56.84 | 347.72MiB | 1041.34MiB |
| D5 | `sd_awgm_hfe_samepos` | 10.335M | 2,001,544 | 20.704G | 21.12ms | 47.35 | 391.64MiB | 1116.41MiB |
| D6 | `sd_awgm_hfe_neighborhood` | 10.332M | 1,998,344 | 20.691G | 26.25ms | 38.09 | 429.07MiB | 1296.45MiB |
| D7 | `sd_awgm_hfe_targetlocal` | 10.332M | 1,998,352 | 20.691G | 29.15ms | 34.30 | 430.93MiB | 1343.60MiB |

D6/D7 的局部邻域计算没有显著增加 THOP FLOPs，但实测延迟和显存高于 D5，说明
只看 THOP 会低估局部 shift、相似度、softmax 和逐 offset 聚合的实际成本。

## 9. 正式训练设置

九项实验统一为：seed42、patch256、batch4、1000 epoch、Adam、初始学习率
`1e-3`、现有 10 epoch warmup + CosineAnnealingLR、`eta_min=1e-5`、epoch100 起
每个 epoch 评估、每20 epoch 保存、阈值0.5。全部随机初始化，不加载 D0–D4
权重，不修改数据划分、loss、增强、scheduler 或指标实现。

## 10. GPU 动态调度

启动时间：`2026-07-15 23:08:19 CST`。队列 PID：`1270595`。D4 的 GPU3/4/5
任务保持运行，没有停止或抢占。首次空闲检查得到 GPU0/1/2/6，因此首批安排为：

| 顺序 | ID | 数据集 | GPU | Wrapper PID | Python PID | 输出目录 | 状态 |
|---:|---|---|---:|---:|---:|---|---|
| 1 | D5 | NUAA-SIRST | 0 | 1270612 | 1270629 | `runs/experiment_d_spatial_ablation/D5_same_position/NUAA-SIRST/seed42` | 训练中（确认epoch13） |
| 2 | D5 | IRSTD-1K | 1 | 1270700 | 1270717 | `runs/experiment_d_spatial_ablation/D5_same_position/IRSTD-1K/seed42` | 训练中（确认epoch2） |
| 3 | D5 | NUDT-SIRST | 2 | 1270788 | 1270805 | `runs/experiment_d_spatial_ablation/D5_same_position/NUDT-SIRST/seed42` | 训练中（确认epoch4） |
| 4 | D6 | NUAA-SIRST | 6 | 1270885 | 1270902 | `runs/experiment_d_spatial_ablation/D6_neighborhood/NUAA-SIRST/seed42` | 训练中（确认epoch11） |
| 5 | D6 | IRSTD-1K | 动态 | — | — | `runs/experiment_d_spatial_ablation/D6_neighborhood/IRSTD-1K/seed42` | 排队 |
| 6 | D6 | NUDT-SIRST | 动态 | — | — | `runs/experiment_d_spatial_ablation/D6_neighborhood/NUDT-SIRST/seed42` | 排队 |
| 7 | D7 | NUAA-SIRST | 动态 | — | — | `runs/experiment_d_spatial_ablation/D7_target_neighborhood/NUAA-SIRST/seed42` | 排队 |
| 8 | D7 | IRSTD-1K | 动态 | — | — | `runs/experiment_d_spatial_ablation/D7_target_neighborhood/IRSTD-1K/seed42` | 排队 |
| 9 | D7 | NUDT-SIRST | 动态 | — | — | `runs/experiment_d_spatial_ablation/D7_target_neighborhood/NUDT-SIRST/seed42` | 排队 |

队列每60秒检查 GPU 已分配显存、利用率、compute PID、输出目录锁、完成标记和
失败标记。任意 GPU 满足空闲条件后按 D5→D6→D7、每个方案 NUAA→IRSTD→NUDT
顺序启动下一项，不中止已有正式任务。

## 11. 结果表

以下结果均来自服务器 `best_metrics.json`。除 D7/NUDT-SIRST 按用户要求在
epoch888 提前停止外，其余 D1–D7 任务均已训练到1000 epoch。

| Dataset | ID | Relation | Best epoch | mIoU | nIoU | F1 | Pd | Fa | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| NUAA | D0 | None | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 | 完成 |
| NUAA | D1 | Hard L2 Top-1 | 286 | 0.7747 | 0.7809 | 0.8731 | 0.9695 | 2.394e-5 | 完成 |
| NUAA | D2 | Soft Cosine Top-k | 350 | 0.773369 | 0.782203 | 0.872203 | 0.973282 | 3.739e-5 | 完成 |
| NUAA | D3 | Gate + Soft Cosine | 549 | 0.776066 | 0.785939 | 0.873916 | 0.973282 | 3.211e-5 | 完成 |
| NUAA | D4 | Direct Low Fusion | 393 | 0.785242 | 0.783468 | 0.879704 | 0.958015 | 3.128e-5 | 已完成1000 epoch |
| NUAA | D5 | Same-position | 441 | 0.774032 | 0.775192 | 0.872625 | 0.969466 | 2.573e-5 | 已完成1000 epoch |
| NUAA | D6 | 3×3 Neighborhood | 512 | 0.780663 | 0.782047 | 0.876823 | 0.969466 | 3.416e-5 | 已完成1000 epoch |
| NUAA | D7 | Target-aware 3×3 | 563 | 0.786192 | 0.788801 | 0.880300 | 0.961832 | 2.140e-5 | 已完成1000 epoch |
| NUDT | D0 | None | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 | 完成 |
| NUDT | D1 | Hard L2 Top-1 | 694 | 0.943166 | 0.949027 | 0.970752 | 0.991534 | 4.343e-6 | 完成 |
| NUDT | D2 | Soft Cosine Top-k | 419 | 0.946951 | 0.947689 | 0.972753 | 0.990476 | 1.953e-6 | 完成 |
| NUDT | D3 | Gate + Soft Cosine | 513 | 0.947825 | 0.949940 | 0.973214 | 0.995767 | 1.540e-6 | 完成 |
| NUDT | D4 | Direct Low Fusion | 441 | 0.944108 | 0.946479 | 0.971251 | 0.988360 | 3.033e-6 | 已完成1000 epoch |
| NUDT | D5 | Same-position | 685 | 0.942888 | 0.948031 | 0.970604 | 0.989418 | 4.044e-6 | 已完成1000 epoch |
| NUDT | D6 | 3×3 Neighborhood | 584 | 0.945232 | 0.947691 | 0.971845 | 0.990476 | 1.746e-6 | 已完成1000 epoch |
| NUDT | D7 | Target-aware 3×3 | 840 | 0.943812 | 0.947057 | 0.971094 | 0.992593 | 1.976e-6 | 用户取消于epoch888，未完成1000 epoch |
| IRSTD | D0 | None | 894 | 0.6561 | 0.6477 | 0.7924 | 0.9091 | 1.537e-5 | 完成 |
| IRSTD | D1 | Hard L2 Top-1 | 556 | 0.657358 | 0.658863 | 0.793260 | 0.922559 | 1.395e-5 | 完成 |
| IRSTD | D2 | Soft Cosine Top-k | 464 | 0.658223 | 0.659029 | 0.793890 | 0.915825 | 1.704e-5 | 完成 |
| IRSTD | D3 | Gate + Soft Cosine | 735 | 0.657163 | 0.660729 | 0.793119 | 0.936027 | 1.782e-5 | 完成 |
| IRSTD | D4 | Direct Low Fusion | 601 | 0.662392 | 0.657617 | 0.796914 | 0.922559 | 1.640e-5 | 已完成1000 epoch |
| IRSTD | D5 | Same-position | 118 | 0.676958 | 0.620624 | 0.807364 | 0.932660 | 2.289e-5 | 已完成1000 epoch |
| IRSTD | D6 | 3×3 Neighborhood | 298 | 0.665016 | 0.631655 | 0.798810 | 0.946128 | 6.069e-5 | 已完成1000 epoch |
| IRSTD | D7 | Target-aware 3×3 | 191 | 0.673834 | 0.649221 | 0.805138 | 0.925926 | 3.272e-5 | 已完成1000 epoch |

## 12. 离线 target/background 诊断

`tools/analyze_experiment_d_spatial_consistency.py` 已实现基于验证集和最佳 checkpoint
的离线诊断。GT 使用 `adaptive_max_pool2d` 对齐各级，避免深层小目标消失；记录
D5 的 target/background spatial scale 与 similarity、D6/D7 的 entropy、中心权重、
邻域选择率、offset 距离，以及 H/V/D 高频残差目标/背景比。D7 额外记录 targetness
分离度和 scale。诊断只在正式 checkpoint 产生后执行，不参与训练或 loss。

## 13. 当前结论边界

当前只能确认结构、数值等价性、梯度、复杂度、真数据数据流和正式任务均有效。
epoch100 前没有 mIoU、nIoU、F1、Pd、Fa；训练到1000 epoch 后才能比较 D4–D7。
三数据集平均 mIoU 差异小于约0.002时只记录为轻微数值波动，不表述为确定优势。

## 14. 最新训练结果快照（2026-07-16 15:32 CST）

以下指标均来自服务器最新 `best_metrics.json`；运行中的任务不是最终结果。
Fa 使用绝对比例记录。

| 实验 | 数据集 | 最新 epoch | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| D4 | NUAA-SIRST | 1000 | 393 | 0.785242 | 0.783468 | 0.879704 | 0.958015 | 3.128e-5 | 完成 |
| D4 | IRSTD-1K | 1000 | 601 | 0.662392 | 0.657617 | 0.796914 | 0.922559 | 1.640e-5 | 完成 |
| D4 | NUDT-SIRST | 929 | 441 | 0.944108 | 0.946479 | 0.971251 | 0.988360 | 3.033e-6 | 运行中 |
| D5 | NUAA-SIRST | 1000 | 441 | 0.774032 | 0.775192 | 0.872625 | 0.969466 | 2.573e-5 | 完成 |
| D5 | IRSTD-1K | 926 | 118 | 0.676958 | 0.620624 | 0.807364 | 0.932660 | 2.289e-5 | 运行中 |
| D5 | NUDT-SIRST | 764 | 685 | 0.942888 | 0.948031 | 0.970604 | 0.989418 | 4.044e-6 | 运行中 |
| D6 | NUAA-SIRST | 1000 | 512 | 0.780663 | 0.782047 | 0.876823 | 0.969466 | 3.416e-5 | 完成 |
| D6 | IRSTD-1K | 535 | 298 | 0.665016 | 0.631655 | 0.798810 | 0.946128 | 6.069e-5 | 运行中 |
| D6 | NUDT-SIRST | 382 | 343 | 0.941007 | 0.946215 | 0.969607 | 0.994709 | 3.746e-6 | 运行中 |
| D7 | NUAA-SIRST | 855 | 563 | 0.786192 | 0.788801 | 0.880300 | 0.961832 | 2.140e-5 | 运行中 |
| D7 | IRSTD-1K | 67 | — | — | — | — | — | — | 评估尚未开始 |
| D7 | NUDT-SIRST | — | — | — | — | — | — | — | 排队中 |

226 服务器 GPU0–6 当前计算进程均属于用户 `cry`，未发现其他用户的 GPU 计算进程。
登录会话中虽有 `cgzkproj`、`zja`，但未检测到其占用 GPU。

## 15. 最新同步快照（2026-07-17 16:26 CST）

本节覆盖前一节快照，数据直接读取 226 服务器各输出目录的
`metrics.jsonl` 和 `best_metrics.json`。表中“最新 epoch”是日志最后一条评估记录，
其余指标是当前最佳 checkpoint（按 mIoU 保存）；Fa 保持原始比例。

| 实验 | 数据集 | 最新 epoch | 最佳 epoch | mIoU | nIoU | F1 | Pd | Fa | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| D5 | NUAA-SIRST | 1000 | 441 | 0.774032 | 0.775192 | 0.872625 | 0.969466 | 2.573e-5 | 已完成 |
| D5 | IRSTD-1K | 1000 | 118 | 0.676958 | 0.620624 | 0.807364 | 0.932660 | 2.289e-5 | 已完成 |
| D5 | NUDT-SIRST | 1000 | 685 | 0.942888 | 0.948031 | 0.970604 | 0.989418 | 4.044e-6 | 已完成 |
| D6 | NUAA-SIRST | 1000 | 512 | 0.780663 | 0.782047 | 0.876823 | 0.969466 | 3.416e-5 | 已完成 |
| D6 | IRSTD-1K | 1000 | 298 | 0.665016 | 0.631655 | 0.798810 | 0.946128 | 6.069e-5 | 已完成 |
| D6 | NUDT-SIRST | 1000 | 584 | 0.945232 | 0.947691 | 0.971845 | 0.990476 | 1.746e-6 | 已完成 |
| D7 | NUAA-SIRST | 1000 | 563 | 0.786192 | 0.788801 | 0.880300 | 0.961832 | 2.140e-5 | 已完成 |
| D7 | IRSTD-1K | 1000 | 191 | 0.673834 | 0.649221 | 0.805138 | 0.925926 | 3.272e-5 | 已完成 |
| D7 | NUDT-SIRST | 888 | 840 | 0.943812 | 0.947057 | 0.971094 | 0.992593 | 1.976e-6 | 用户取消，未完成1000 epoch |

按用户要求，D7/NUDT-SIRST 已停止并将 GPU1 让给 Experiment F；服务器已确认
对应 Python PID、wrapper PID 和 Experiment D 队列 PID 均不存在。该任务没有
训练到1000 epoch，因此表中数值只能作为“截至 epoch888 的最佳结果”，不能标成
完整最终结果。`latest.pth.tar`、`best.pth.tar`、`best_metrics.json`、训练日志和
`CANCELLED_BY_USER` 标记均保留在原输出目录。
