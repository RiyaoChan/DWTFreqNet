# Experiment F：DSHF 高频方向稀疏特征提取实验记录

更新时间：2026-07-17（CST）

## 1. 实验定位

Experiment F 固定以 Experiment E 的 E1 为 F0 基线，只替换 Encoder 中送入
StageWiseAWGM 的高频编码器：原 `DirectionalBandEncoder` 替换为
`DSHFBlock`。LFSS、AWGM、AWGM 后的原 `Res_block`、原始 Haar 系数保存、
Decoder、IDWT、单解码器、深监督、损失、数据划分和训练设置保持不变。

严格执行顺序为：

```text
DWT → LFSS(raw LL) → DSHF(raw H/V/D, optional LFSS LL)
    → AWGM(LFSS LL, DSHF H/V/D) → 原 Res_block
```

DSHF 输出只参与 AWGM 和离线诊断；Decoder 仍使用原始 Haar H/V/D 经 1×1 Conv
对齐后的系数，不使用 DSHF 输出。

## 2. 分支与隔离

- 仓库：`RiyaoChan/DWTFreqNet`
- 基础分支：`codex/experiment-e-lfss-before-awgm`
- 实际基础 HEAD：`68ede894be748c8842427e140898f007dbe67953`
- Experiment F 分支：`codex/experiment-f-dshf-high-frequency-encoder`
- 已验证并部署提交：`51b2d070a1dad7ab6b400b25b81a3022338b00bd`
- 独立 Draft PR：`https://github.com/RiyaoChan/DWTFreqNet/pull/5`
- 226 部署目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_F_DSHF`
- 输出根目录：`runs/experiment_f_dshf`
- 数据集目录：`/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets`

Experiment D/E 的项目目录、代码、进程、checkpoint 和输出目录均不修改或覆盖。

## 3. 四个变体

| ID | 训练变体 | 多尺度方向提取 | Sparse Gate | Cross-direction Gate | LFSS低频引导 |
|---|---|---|---|---|---|
| F1 | `f1_multiscale` | 是 | 否 | 否 | 否 |
| F2 | `f2_sparse` | 是 | 是 | 否 | 否 |
| F3 | `f3_cross_direction` | 是 | 是 | 是 | 否 |
| F4 | `f4_low_guided_full` | 是 | 是 | 是 | 是 |

正式启动优先级为 `F1 → F4 → F2 → F3`，但最终消融解释固定为
`F1-F0`、`F2-F1`、`F3-F2`、`F4-F3`。

## 4. 模块实现

四级通道固定为 32/64/128/256，每一级使用独立 DSHFBlock，三个方向也不共享
权重：H 使用 3×1 与 5×1 深度卷积分支；V 使用 1×3 与 1×5；D 使用 3×3
`dilation=1` 与 3×3 `dilation=2`。双分支拼接后统一经过 1×1 Conv、BN、GELU。

F2–F4 的 Sparse Gate 使用每通道平均幅值和 MLP 预测软阈值，最后一层权重及
偏置均零初始化，使初始 `threshold_ratio=0.5`。F3/F4 的 Cross-direction Gate
根据 H/V/D 有界能量和联合能量输出三个空间尺度，最后一层零初始化，使初始
尺度严格为1。F4 额外从同一级 `LFSS(raw LL)` 计算 1×1 响应与 3×3 局部对比，
不使用 raw LL、AWGM 输出、GT 或 side head。

## 5. 新增和最小修改文件

```text
model/DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM.py
train_experiment_f_dshf.py
tools/test_experiment_f_dshf.py
tools/profile_experiment_f_dshf.py
tools/analyze_experiment_f_high_frequency.py
scripts/run_experiment_f_dshf.sh
scripts/launch_experiment_f_dshf_queue.sh
experiment_guide/EXPERIMENT_F_DSHF_HIGH_FREQUENCY_ENCODER_FOR_CODEX.md
EXPERIMENT_F_DSHF_RECORD.md
EXPERIMENT_RECORD.md
README.md
```

规范禁止修改的 E1、Haar/DWT、数据、训练公共逻辑和 Wave-Mamba 文件保持不变。

## 6. 固定训练设置

三个数据集均使用 seed42、patch 256、batch size 4、1000 epoch、Adam、初始
学习率 `1e-3`、10 epoch warmup、CosineAnnealingLR、`eta_min=1e-5`，从
epoch100 起每个 epoch 评估、每20 epoch 保存、阈值0.5。全部随机初始化，不
加载 E1/E2/D0–D7 或 Wave-Mamba 预训练权重。

## 7. 验证记录

2026-07-17 在 226 的 RTX 3090 与 `mirfd_mamba` 环境完成：

| 检查项 | 结果 |
|---|---|
| E1基线文件与结构 | 禁止修改文件相对基础 HEAD 无差异；原 E1 仍为 DirectionalBandEncoder + Res_block |
| 模块替换 | F1–F4 四级均为 DSHFBlock，模型内无残留 DirectionalBandEncoder |
| 初始化 | LFSS特殊参数最大差0；Sparse/Cross最后一层权重和偏置均为0 |
| 执行顺序 | 四级严格为 LFSS→DSHF→AWGM→原Res_block |
| AWGM输入 | low=LFSS输出，H/V/D=DSHF输出 |
| Decoder系数 | 四级 H/V/D 均由 raw Haar 系数经 align Conv 得到，未使用DSHF输出 |
| F1退化 | 三方向融合输出置零后与 raw H/V/D 逐元素完全相同 |
| Sparse Gate | support范围(0,1)，threshold为[B,C,1,1]，初始ratio=0.5，符号保持 |
| F3→F2 | 复制36项共享状态后，零初始化Cross Gate使输出一致 |
| F4→F3 | 复制39项共享状态后，零初始化Cross Gate使输出一致 |
| F4低频来源 | 四级输入均等于同级LFSS输出且不同于raw LL |
| DWT/IDWT | 四变体均为4/4 |
| 输出 | 训练6×[2,1,256,256]；测试[2,1,256,256] |
| 梯度 | LFSS、DSHF对应分支、AWGM、Res_block、Decoder和side head均非零 |
| 禁止操作/模块 | cdist/topk monkeypatch下完整forward通过；无Matching/HFE/Pyramid |
| CUDA | 四变体FP32和AMP前后向全部通过，无NaN/Inf/OOM |
| NUAA真实batch4 smoke | F1/F2/F3/F4初始损失为4.2408/7.3392/6.2861/7.1495，均完成optimizer.step |
| Haar方向 | H实际响应垂直结构、V响应水平结构，当前方向卷积路由对齐 |

### 7.1 RTX 3090复杂度

输入 `1×1×256×256`，FP32 eval，warmup=5、repeat=20；THOP 对 selective scan
可能存在漏计，但所有模型使用同一统计口径。

| 模型 | Params | DSHF Params | Encoder Params | THOP FLOPs | Latency | FPS | Infer peak | Train peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0/E1 | 7.014M | 0 | 3.942M | 15.032G | 12.810ms | 78.06 | 190.33MiB | 627.99MiB |
| F1 | 7.282M | 0.541M | 4.210M | 15.248G | 13.126ms | 76.18 | 225.33MiB | 647.20MiB |
| F4 | 7.354M | 0.613M | 4.282M | 15.268G | 17.782ms | 56.24 | 225.62MiB | 696.93MiB |
| F2 | 7.350M | 0.609M | 4.278M | 15.248G | 15.629ms | 63.98 | 225.60MiB | 681.25MiB |
| F3 | 7.353M | 0.612M | 4.281M | 15.264G | 17.506ms | 57.12 | 225.61MiB | 696.42MiB |

## 8. F0/E1 复用结果

| 数据集 | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 595 | 0.7842 | 0.7960 | 0.8790 | 0.9656 | 2.717e-5 |
| IRSTD-1K | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 |
| NUDT-SIRST | 456 | 0.9516 | 0.9482 | 0.9752 | 0.9947 | 1.724e-6 |

F0 不重新训练；正式部署前核验模型、LFSS、AWGM、Res_block、Decoder、数据、
loss、评价、训练参数和数据划分与 E1 一致。

## 9. 12项正式任务

| 优先级 | 变体 | 数据集 | 输出目录 | GPU/PID | 状态 |
|---:|---|---|---|---|---|
| 1 | F1 | NUAA-SIRST | `F1_multiscale/NUAA-SIRST/seed42` | GPU0 / 2158100 / 2158108 | 已完成1000 epoch |
| 2 | F4 | NUAA-SIRST | `F4_low_guided_full/NUAA-SIRST/seed42` | GPU2 / 2158294 / 2158302 | 已完成1000 epoch |
| 3 | F2 | NUAA-SIRST | `F2_sparse/NUAA-SIRST/seed42` | GPU3 / 2158597 / 2158605 | 已完成1000 epoch |
| 4 | F3 | NUAA-SIRST | `F3_cross_direction/NUAA-SIRST/seed42` | GPU4 / 2158925 / 2158933 | 已完成1000 epoch |
| 5 | F1 | IRSTD-1K | `F1_multiscale/IRSTD-1K/seed42` | GPU5 / 2159332 / 2159340 | 已完成1000 epoch |
| 6 | F1 | NUDT-SIRST | `F1_multiscale/NUDT-SIRST/seed42` | GPU6 / 2165890 / 2165898 | 已完成1000 epoch |
| 7 | F4 | IRSTD-1K | `F4_low_guided_full/IRSTD-1K/seed42` | GPU1 / 2169933 / 2169940 | 已完成1000 epoch |
| 8 | F4 | NUDT-SIRST | `F4_low_guided_full/NUDT-SIRST/seed42` | GPU0 / 2509026 / 2509034 | 已完成1000 epoch |
| 9 | F2 | IRSTD-1K | `F2_sparse/IRSTD-1K/seed42` | GPU3 / 2533658 / 2533666 | 已完成1000 epoch |
| 10 | F2 | NUDT-SIRST | `F2_sparse/NUDT-SIRST/seed42` | GPU4 / 2561065 / 2561073 | 已完成1000 epoch |
| 11 | F3 | IRSTD-1K | `F3_cross_direction/IRSTD-1K/seed42` | GPU2 / 2564255 / 2564263 | 已完成1000 epoch |
| 12 | F3 | NUDT-SIRST | `F3_cross_direction/NUDT-SIRST/seed42` | GPU5 / 3323500 / 3323508 | 运行中（epoch664） |

动态队列当前每10秒检查允许列表中的空闲 GPU，要求显存占用小于1000 MiB、利用率
小于10%且无 compute PID；不会停止或抢占 Experiment D/E 及其他用户任务。

正式队列最初于 `2026-07-17 16:09:29 CST` 启动。按用户在
`2026-07-17 16:15 CST` 的新要求，已停止原队列和 GPU6 上的 F4/IRSTD-1K；
该任务完成的前2个 epoch 未作为正式结果，完整中断现场保存在
`runs/cancelled_20260717_1615/F4_low_guided_full/IRSTD-1K/seed42`，正式输出目录已
释放，后续会从头训练。

队列已改为同一变体的 IRSTD-1K 与 NUDT-SIRST 相邻排序：
`F1 IRSTD+NUDT -> F4 IRSTD+NUDT -> F2 IRSTD+NUDT -> F3 IRSTD+NUDT`。
这里的“同步”表示顺序补齐，不要求同时空出两张显卡：某变体的 IRSTD-1K 已
启动后，下一张空闲卡优先启动同变体 NUDT-SIRST，再进入下一变体。

GPU6 于 `2026-07-17 16:17:24 CST` 启动 F1/NUDT-SIRST，补齐 F1。按用户要求，
Experiment D 的 D7 target_neighborhood/NUDT-SIRST 于 epoch888 停止并释放
GPU1；它在停止前的最佳结果为 epoch840：mIoU 0.943812、nIoU 0.947057、
F1 0.971094、Pd 0.992593、Fa 1.976289e-6，checkpoint 和日志均保留在原目录。
新队列于 `2026-07-17 16:23:07 CST` 启动，队列 PID `2169345`，随后在 GPU1
从头启动 F4/IRSTD-1K；下一张空闲 GPU 将优先启动 F4/NUDT-SIRST。其余6个
Experiment F 进程均未停止或重启。以上运行进度快照采集于
`2026-07-17 16:24 CST`。

最新同步快照采集于 `2026-07-18 22:41 CST`：11项已完成、1项运行中、0项排队、
0项失败。当前只有 F3/NUDT-SIRST 尚未完成，运行至 epoch664；其余任务均已
训练到1000 epoch。

## 10. 结果表

| Dataset | ID | 高频模块 | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | Latency | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NUAA | F0/E1 | 原DirectionalBandEncoder | 595 | 0.7842 | 0.7960 | 0.8790 | 0.9656 | 2.717e-5 | 7.014M | 12.810ms | 基线 |
| NUAA | F1 | Multi-scale | 739 | 0.789485 | 0.797091 | 0.882360 | 0.965649 | 1.653e-5 | 7.282M | 13.126ms | 已完成1000 epoch |
| NUAA | F2 | Multi-scale + Sparse | 762 | 0.790276 | 0.796704 | 0.882854 | 0.958015 | 1.475e-5 | 7.350M | 15.629ms | 已完成1000 epoch |
| NUAA | F3 | F2 + Cross-direction | 585 | 0.792278 | 0.800332 | 0.884102 | 0.965649 | 3.087e-5 | 7.353M | 17.506ms | 已完成1000 epoch |
| NUAA | F4 | F3 + LFSS Low Guidance | 477 | 0.794258 | 0.798127 | 0.885333 | 0.969466 | 1.640e-5 | 7.354M | 17.782ms | 已完成1000 epoch |
| IRSTD | F0/E1 | 原DirectionalBandEncoder | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 | 7.014M | 12.810ms | 基线 |
| IRSTD | F1 | Multi-scale | 640 | 0.665652 | 0.671340 | 0.799269 | 0.942761 | 1.382e-5 | 7.282M | 13.126ms | 已完成1000 epoch |
| IRSTD | F2 | Multi-scale + Sparse | 775 | 0.667652 | 0.666804 | 0.800709 | 0.922559 | 1.110e-5 | 7.350M | 15.629ms | 已完成1000 epoch |
| IRSTD | F3 | F2 + Cross-direction | 675 | 0.670769 | 0.668084 | 0.802947 | 0.919192 | 1.040e-5 | 7.353M | 17.506ms | 已完成1000 epoch |
| IRSTD | F4 | F3 + LFSS Low Guidance | 604 | 0.661037 | 0.653661 | 0.795933 | 0.922559 | 1.264e-5 | 7.354M | 17.782ms | 已完成1000 epoch |
| NUDT | F0/E1 | 原DirectionalBandEncoder | 456 | 0.9516 | 0.9482 | 0.9752 | 0.9947 | 1.724e-6 | 7.014M | 12.810ms | 基线 |
| NUDT | F1 | Multi-scale | 358 | 0.951807 | 0.950017 | 0.975308 | 0.992593 | 1.149e-6 | 7.282M | 13.126ms | 已完成1000 epoch |
| NUDT | F2 | Multi-scale + Sparse | 310 | 0.955089 | 0.955679 | 0.977029 | 0.991534 | 2.505e-6 | 7.350M | 15.629ms | 已完成1000 epoch |
| NUDT | F3 | F2 + Cross-direction | 476 | 0.951798 | 0.952349 | 0.975304 | 0.994709 | 1.540e-6 | 7.353M | 17.506ms | 运行中（epoch664） |
| NUDT | F4 | F3 + LFSS Low Guidance | 580 | 0.956631 | 0.955744 | 0.977835 | 0.991534 | 1.218e-6 | 7.354M | 17.782ms | 已完成1000 epoch |

F3/NUDT 的指标来自当前 `best_metrics.json`，不是1000 epoch最终结果。当前
NUAA 上 F4 的 mIoU/F1 最好；IRSTD 上 F3 的 mIoU/F1/Fa 最好，F1 的 nIoU/Pd
最好；NUDT 上 F4 的 mIoU/nIoU/F1 最好，F1 的 Fa 最低。

## 11. 高频离线诊断

`tools/analyze_experiment_f_high_frequency.py` 只对最佳 checkpoint 离线运行，记录
各级 raw/multiscale/sparse/cross/final H/V/D 的目标背景响应比、Sparse support、
跨方向尺度和联合能量、方向熵、F4 低频局部对比、DSHF残差强度及AWGM联动。
诊断不参与训练、loss 或 checkpoint 选择。
