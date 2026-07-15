# Experiment D：Decoder HFE Matching 消融记录（D2 → D3）

## 1. 实验定位

本轮是 Experiment D 的内部消融，不是新的主实验，也不建立 Experiment E。
唯一消融轴是 Decoder HFE 中高频与当前 decoder 低频的 relation/matching 方式：

| ID | 模型 | Stage 1/2 | Stage 3/4 |
|---|---|---|---|
| D0 | `sd_awgm` | 无 HFE | 无 HFE |
| D1 | `sd_awgm_hfe` | Hard L2 Top-1 | Hard L2 Top-1 |
| D2 | `sd_awgm_hfe_softcos` | Soft Cosine Top-k | Soft Cosine Top-k |
| D3 | `sd_awgm_hfe_scaleaware` | Local Correlation Gate | Soft Cosine Top-k |

严格执行 `D2 → D3`：D2 先验证 L2/硬 Top-1 是否限制了泛化，D3 再只替换
浅层 Stage 1/2 relation，验证浅层全局通道匹配是否造成背景误增强。

## 2. 代码基线与隔离

- Repository：`RiyaoChan/DWTFreqNet`
- 实际 base branch：`codex/experiment-d-sd-awgm-decoder-hfe`
- 实际 base commit：`6fb19768dd7013aff536447b39652a44c1538912`
- 消融分支：`codex/experiment-d-hfe-matching-ablation-d2-d3`
- 服务器独立目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_D_ABLATION`

新增文件：

```text
model/DWTFreqNet_SingleDecoder_HFE_Ablation.py
train_experiment_d_hfe_ablation.py
tools/test_experiment_d_hfe_matching_ablation.py
tools/profile_experiment_d_hfe_matching_ablation.py
scripts/run_experiment_d_hfe_ablation.sh
scripts/launch_experiment_d_hfe_ablation_queue.sh
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
```

D0/D1 的模型文件、训练入口、checkpoint 和输出目录均未修改。D2/D3 继承 D1
模型主路径，只替换 `decoder_hfe1–4`，输出写入独立的
`runs/experiment_d_ablation/`。

## 3. 固定结构配置

D2 四级均为 `SoftCosineTopKMatching(topk=8, temperature=0.1)`：

| Stage | Channels | Heads | Relation | Top-k | 初始温度 |
|---:|---:|---:|---|---:|---:|
| 1 | 64 | 1 | Soft Cosine Top-k | 8 | 0.1 |
| 2 | 128 | 2 | Soft Cosine Top-k | 8 | 0.1 |
| 3 | 256 | 4 | Soft Cosine Top-k | 8 | 0.1 |
| 4 | 256 | 4 | Soft Cosine Top-k | 8 | 0.1 |

D3 只改变浅层：

| Stage | Channels | Heads | Relation | Top-k | 初始温度 |
|---:|---:|---:|---|---:|---:|
| 1 | 64 | 1 | Local Correlation Gate | — | — |
| 2 | 128 | 2 | Local Correlation Gate | — | — |
| 3 | 256 | 4 | Soft Cosine Top-k | 8 | 0.1 |
| 4 | 256 | 4 | Soft Cosine Top-k | 8 | 0.1 |

D2/D3 的 Stage 3/4 模块类型、超参数和参数形状已逐项比较并严格一致；Attention
和 FFN 各自创建独立 relation 实例，不共享参数。

## 4. 实现要点

- Soft Cosine 使用 float32 余弦相似度、可学习温度和 Top-k softmax 权重；没有
  `torch.cdist`、L2、`argmin` 或 Hard Top-1。
- 正式聚合采用 `scatter_add + bmm`，避免浅层 `expand + gather` 的大显存开销；
  单测用 gather 参考实现验证两者数值一致。
- Local Correlation Gate 使用 `[H,L,H×L,|H-L|]` 局部关系特征，在高、低频
  value 分支之间逐像素逐通道门控。
- 完整 relation 熵、候选使用率、温度和 gate 统计只在评估阶段计算；训练阶段
  不执行逐 batch GPU→CPU 同步。
- SubbandSelectiveFusion、Attention/FFN 主体、三个方向残差头、`beta=1e-3`、
  四级 IDWT、单解码器和深监督均与 D1 保持一致。

## 5. 测试结果

本地 CPU 小尺寸测试和 226 服务器 RTX 3090 CUDA 完整测试均已通过。CUDA 测试
使用 `2×1×256×256` 输入。

| 检查项 | 结果 |
|---|---|
| D2 Stage 1–4 类型 | 全部 `SoftMatchingTransformation` |
| D3 Stage 1/2 类型 | 全部 `LocalCorrelationGate` |
| D3 Stage 3/4 类型 | 与 D2 完全相同的 `SoftMatchingTransformation` |
| 训练输出 | `6 × [2,1,256,256]` |
| 测试输出 | `[2,1,256,256]` |
| Soft Cosine shape | Stage1 `[2,64,64]`，Stage2 `[2,128,128]`，Stage3/4 `[2,256,256]` |
| Top-k shape/权重 | `[B,C,8]`，权重和误差小于 `1e-6` |
| Gate shape | Stage1 `[2,64,128,128]`，Stage2 `[2,128,64,64]` |
| Gate 范围 | `[0,1]` |
| 禁止 `torch.cdist` | Monkeypatch 后 D2/D3 完整 forward 通过 |
| beta 置零退化 | D2/D3 与 `sd_awgm` 最大绝对误差均为 `0` |
| 梯度 | relation、temperature、Attention、FFN、方向头、beta、主路径均非零 |
| AMP | CUDA autocast 前后向无 NaN/Inf |
| DWT/IDWT | D0/D1/D2/D3 均为 `4/4` |
| 真实数据 smoke | NUAA batch=4，D2/D3 单步训练均通过，无 OOM |

## 6. 复杂度与实测速度

设备为 RTX 3090，输入为 `1×1×256×256`，warmup=5、重复=20。FLOPs 使用
THOP 统一口径；THOP 不统计代码中直接完成的余弦、matching 和 attention 矩阵乘法，
因此同时报告实测延迟和显存。

| ID | 模型 | Params | FLOPs | 延迟 | FPS | 推理峰值显存 | 训练峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|
| D0 | `sd_awgm` | 5.926M | 14.38G | 8.76ms | 114.22 | 178.05MiB | 403.52MiB |
| D1 | `sd_awgm_hfe` | 10.181M | 20.47G | 19.80ms | 50.50 | 260.61MiB | 932.08MiB |
| D2 | `sd_awgm_hfe_softcos` | 10.181M | 20.47G | 19.67ms | 50.85 | 269.22MiB | 1009.85MiB |
| D3 | `sd_awgm_hfe_scaleaware` | 10.225M | 20.77G | 19.02ms | 52.59 | 308.43MiB | 1097.26MiB |

D1→D2 仅增加 8 个可学习温度标量，实测延迟略低于 D1；D3 比 D2 增加约
44K 参数和 0.30G THOP FLOPs，但本次实测延迟略低。延迟差异应结合多次复测理解。

## 7. 统一训练设置

```text
seed=42
patch size=256
batch size=4
epochs=1000
optimizer=Adam
initial lr=1e-3
scheduler=CosineAnnealingLR, eta_min=1e-5
eval start=100
eval every=1
save every=20
threshold=0.5
```

数据划分、loss、增强、优化器、学习率和阈值均与 D1 相同。

## 8. 正式实验安排与启动状态

启动时间：2026-07-14 23:23:32 CST。动态队列 PID 为 `564224`。

### 8.1 Phase 1：D2 已启动

| 优先级 | 数据集 | GPU | Wrapper PID | Python PID | 状态 | 输出目录 |
|---:|---|---:|---:|---:|---|---|
| 1 | NUAA-SIRST | 1 | 564242 | 564252 | 已完成1000 epoch | `runs/experiment_d_ablation/D2_softcos_all/NUAA-SIRST/seed42` |
| 2 | IRSTD-1K | 2 | 564328 | 564338 | 训练中 | `runs/experiment_d_ablation/D2_softcos_all/IRSTD-1K/seed42` |
| 3 | NUDT-SIRST | 5 | 564426 | 564435 | 训练中 | `runs/experiment_d_ablation/D2_softcos_all/NUDT-SIRST/seed42` |

三个任务均已完成真实数据首个训练 step，无 NaN/Inf、无 OOM。D2 NUAA 已完成
1000 epoch；D2 IRSTD 和 D2 NUDT 也已完成 1000 epoch。GPU0/3
上的 D1、GPU4 上的 Experiment C 均未停止或覆盖。

### 8.2 Phase 2：D3 三数据集已完成

原队列会等待三个 D2 均满足：

```text
完成 epoch 100 后首次正式评估
best_metrics.json 正常生成
日志中无 NaN/Inf
无 OOM 或失败状态
```

2026-07-14 23:41 CST，按实验调度要求，将空闲 GPU6 直接用于 D3
NUAA-SIRST，不再等待全局门槛。随后三个 D2 在 2026-07-15 00:59 CST
通过 epoch 100 首次评估、best_metrics、NaN/Inf 和 OOM 门槛，队列继续启动
D3 的 IRSTD-1K（GPU1）和 NUDT-SIRST（GPU6）。D3 不抢占或停止其他实验，
并通过活动输出目录检查跳过已在运行或已完成的 NUAA 任务。

| 数据集 | GPU | Wrapper PID | Python PID | 启动确认 | 状态 |
|---|---:|---:|---:|---|---|
| NUAA-SIRST | 6 | 573177 | 573179 | 已完成1000 epoch，best epoch 549 | 已完成 |
| IRSTD-1K | 1 | 752866 | 752874 | 已完成1000 epoch，无 NaN/Inf、无 OOM | 已完成 |
| NUDT-SIRST | 6 | 755891 | 755897 | 已完成1000 epoch，无 NaN/Inf、无 OOM | 已完成 |

输出目录分别为：

```text
runs/experiment_d_ablation/D3_scaleaware/NUAA-SIRST/seed42
runs/experiment_d_ablation/D3_scaleaware/IRSTD-1K/seed42
runs/experiment_d_ablation/D3_scaleaware/NUDT-SIRST/seed42
```

## 9. 结果表

以下 D0 为已有基线，D1 为启动 D2/D3 时可用的最佳 checkpoint；D2/D3 的指标
为三项任务完成 1000 epoch 后的最终 `best_metrics.json` 快照。

| Dataset | ID | Model | Stage1/2 | Stage3/4 | Best epoch | mIoU | nIoU | F1 | Pd | Fa | 状态 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| NUAA | D0 | `sd_awgm` | None | None | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 | 已完成 |
| NUAA | D1 | `sd_awgm_hfe` | Hard L2 | Hard L2 | 286 | 0.7747 | 0.7809 | 0.8731 | 0.9695 | 2.394e-5 | 已完成1000 epoch |
| NUAA | D2 | `sd_awgm_hfe_softcos` | Soft Cosine | Soft Cosine | 350 | 0.773369 | 0.782203 | 0.872203 | 0.973282 | 3.739e-5 | 已完成1000 epoch |
| NUAA | D3 | `sd_awgm_hfe_scaleaware` | Correlation Gate | Soft Cosine | 549 | 0.776066 | 0.785939 | 0.873916 | 0.973282 | 3.211e-5 | 已完成1000 epoch |
| NUDT | D0 | `sd_awgm` | None | None | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 | 已完成 |
| NUDT | D1 | `sd_awgm_hfe` | Hard L2 | Hard L2 | 694 | 0.943166 | 0.949027 | 0.970752 | 0.991534 | 4.343e-6 | 已完成1000 epoch |
| NUDT | D2 | `sd_awgm_hfe_softcos` | Soft Cosine | Soft Cosine | 419 | 0.946951 | 0.947689 | 0.972753 | 0.990476 | 1.953e-6 | 已完成1000 epoch |
| NUDT | D3 | `sd_awgm_hfe_scaleaware` | Correlation Gate | Soft Cosine | 513 | 0.947825 | 0.949940 | 0.973214 | 0.995767 | 1.540e-6 | 已完成1000 epoch |
| IRSTD | D0 | `sd_awgm` | None | None | 894 | 0.6561 | 0.6477 | 0.7924 | 0.9091 | 1.537e-5 | 已完成 |
| IRSTD | D1 | `sd_awgm_hfe` | Hard L2 | Hard L2 | 556 | 0.657358 | 0.658863 | 0.793260 | 0.922559 | 1.395e-5 | 已完成1000 epoch |
| IRSTD | D2 | `sd_awgm_hfe_softcos` | Soft Cosine | Soft Cosine | 464 | 0.658223 | 0.659029 | 0.793890 | 0.915825 | 1.704e-5 | 已完成1000 epoch |
| IRSTD | D3 | `sd_awgm_hfe_scaleaware` | Correlation Gate | Soft Cosine | 735 | 0.657163 | 0.660729 | 0.793119 | 0.936027 | 1.782e-5 | 已完成1000 epoch |

D2/D3 最终均训练至 1000 epoch；从 epoch 100 起每个 epoch 评估并按 mIoU 保存
最佳 checkpoint。完成后再依据 D0→D1、D1→D2、D2→D3 顺序给出消融结论。
