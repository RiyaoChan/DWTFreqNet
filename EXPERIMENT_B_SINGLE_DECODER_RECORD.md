# Experiment B：单一小波解码器与方向金字塔实验记录

## 1. 复现信息

- 基础 commit：`b98bb4e25b425d9fdf5f2ccbadca6f76af38b539`
- 实验分支：`codex/experiment-b-single-decoder-directional-pyramid`
- 实现 commit：`2ee0878c2e5ef8bd4639f58282cf41ce4e5dc44c`
- 服务器工程：`/DATA20T/bip/cry/code/DWTFreqNet_SINGLE_DECODER_B`
- 新模型：`model/DWTFreqNet_SingleDecoder.py`
- 独立训练入口：`train_experiment_b.py`
- 数据划分：`/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets`
- 原始 `model/DWTFreqNet.py`：未修改
- Experiment A 的 `model/DWTFreqNet_WULLE.py`：未修改

统一训练设置：seed 42、输入 256×256、batch size 4、1000 epoch、Adam、
初始学习率 1e-3、CosineAnnealingLR、eta_min 1e-5、从 epoch 100 开始每个
epoch 评估一次、每 20 epoch 保存一次、阈值 0.5，不使用预训练权重。

## 2. 这些实验是什么意思

Experiment B 要验证的是：原模型和 WULLE 中是否存在重复解码，以及只用一个
Wavelet Decoder 能否在大幅降低复杂度的同时保持检测精度。

新模型只有一条编码—解码路径：

```text
输入 → X0 → 4级DWT编码器 → E4
                         ↓
               唯一Wavelet Decoder
                         ↓
                L3 → L2 → L1 → L0 → 预测
```

编码阶段每次 DWT 都产生同源的低频 A 和方向高频 H/V/D：

- AWGM 用同一次 DWT 的 H/V/D 调制低频 A；
- Directional Pyramid 跨尺度聚合 H/V/D；
- Pyramid 只校准原始 Haar 高频系数，不直接伪造小波系数；
- 全模型固定调用 4 次 DWT 和 4 次 IDWT；
- 不再包含第二次 DWT、WULLE 内部 decoder、Dense Global、后置 AWGM、
  TransTo/LDRC 或 Mamba。

## 3. 四个变体分别验证什么

| 编号 | Variant | AWGM | 方向金字塔 | 实验含义 |
|---|---|---:|---:|---|
| B0 | `sd_raw` | 否 | 否 | 单 decoder 的结构基线，只使用通道对齐后的原始 H/V/D 系数 |
| B1 | `sd_awgm` | 是 | 否 | 在 B0 上加入同源高频调制低频，用于单独判断 AWGM 贡献 |
| B2 | `sd_pyramid` | 否 | 是 | 在 B0 上加入方向金字塔，用于单独判断跨尺度系数校准贡献 |
| B3 | `sd_full` | 是 | 是 | 同时使用 AWGM 和 Pyramid，验证二者能否互补 |

关键对比关系：

- `sd_raw` 对比 WULLE-A：判断 WULLE 内部 decoder 是否可以删除；
- `sd_awgm` 对比 `sd_raw`：判断同源 H/V/D 调制同源 A 是否有效；
- `sd_pyramid` 对比 `sd_raw`：判断方向高频跨尺度融合是否有效；
- `sd_full` 对比 `sd_awgm`：判断加入 Pyramid 后是否进一步提升；
- `sd_full` 对比 `sd_pyramid`：判断加入 AWGM 后是否进一步提升；
- 如果 B1、B2 分别提升但 B3 下降，说明两种机制可能重复增强高频。

## 4. 方向模块定义

当前 Haar 方向已经由合成水平线和垂直线验证：

```text
H = LH：响应垂直结构，使用 5×1 depthwise conv
V = HL：响应水平结构，使用 1×5 depthwise conv
D = HH：响应对角结构，使用 3×3 depthwise conv
```

H/V/D 三条路径从方向编码、Top-down Pyramid 到系数 delta head 始终相互独立。

Stage-wise AWGM 使用 softmax 生成 H/V/D 权重，通过 `tanh` gate 和可学习
alpha 对低频 A 进行增强或抑制。Pyramid 使用可学习 beta 将方向残差加入通道
对齐后的原始 Haar 系数。

## 5. 测试结果

- Python 编译与导入：通过；
- 本地 2×256 训练/测试前向：通过；
- 226 服务器 2×256 前向与反向：通过；
- 训练模式 6 个 `[2,1,256,256]` 输出：通过；
- 测试模式 1 个 `[2,1,256,256]` 输出：通过；
- X0、E1–E4、L3–L0 中间形状：通过；
- Pyramid 的 P1–P4 三方向形状：通过；
- 四个变体 DWT/IDWT 调用数均为 4/4：通过；
- 编码器、decoder、系数 head、输出 head 梯度：通过；
- AWGM 的 direction encoder、gate、alpha 梯度：通过；
- Pyramid、delta head、beta 梯度：通过；
- `sd_raw/sd_awgm/sd_pyramid` 旁路关系：通过；
- Haar `H/LH→vertical`、`V/HL→horizontal`：通过；
- 四个变体真实数据单 batch smoke test：通过。

## 6. 复杂度结果

统一输入 `[1,1,256,256]`，延迟为本地 CUDA 三次重复的阶段性测量。

| 模型 | 参数量 | FLOPs | 推理延迟 | FPS | 推理峰值显存 | 训练峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| Original | 37,434,599 | 66.87G | 20.65 ms | 48.43 | 617.35 MiB | 2501.31 MiB |
| WULLE-A | 35,399,143 | 54.69G | 21.80 ms | 45.87 | 550.26 MiB | 2490.46 MiB |
| sd_raw | 5,471,275 | 13.99G | 3.23 ms | 309.34 | 305.54 MiB | 480.01 MiB |
| sd_awgm | 5,925,687 | 14.38G | 5.15 ms | 194.22 | 326.31 MiB | 566.32 MiB |
| sd_pyramid | 11,304,715 | 26.36G | 6.62 ms | 151.05 | 361.50 MiB | 707.34 MiB |
| sd_full | 11,486,007 | 26.51G | 8.27 ms | 120.93 | 369.95 MiB | 745.79 MiB |

结论：

- `sd_raw` 相比 WULLE-A 参数量减少 84.55%，FLOPs 减少 74.41%；
- `sd_full` 相比 Original 参数量减少 69.31%，FLOPs 减少 60.36%；
- `sd_raw < WULLE-A` 和 `sd_full < Original` 两项正式训练门槛均已通过。

## 7. 正式实验安排

### Phase I：NUAA、NUDT 的 2×2 消融

NUAA 和 NUDT 都运行 `sd_raw/sd_awgm/sd_pyramid/sd_full`，共 8 项。这样可以
分别计算 AWGM 与 Pyramid 的独立贡献以及组合后的交互作用。

### Phase II：IRSTD 泛化验证

IRSTD 至少运行 `sd_raw` 和 `sd_full`，用于判断最简单结构和完整结构在更困难、
训练更慢的数据集上是否仍然成立。如果 Phase I 中 B1 或 B2 优于 B3，再追加
对应的 IRSTD 实验。

### Phase III：多 seed

只有当至少两个数据集不低于 WULLE-A，或者平均 mIoU 下降不超过 0.3 个百分点
且复杂度明显下降时，才追加 seed 3407 和 2026。

## 8. 当前队列状态

原调度器 PID `3092160` 已在不终止已有训练进程的前提下停止；新的调度器
PID 为 `3097664`，启动时间：`2026-07-12 21:10:55 +08:00`。
调度器只使用显存占用不超过 1024 MiB、利用率不超过 5% 的 GPU，不终止或覆盖
Experiment A。调度顺序已改为先让 NUAA、NUDT、IRSTD 各启动一个代表性
`sd_raw`，再继续 NUAA 的其他变体和 NUDT/IRSTD 的剩余变体；重启调度器时会
识别已有输出目录对应的活动进程，不会重复启动 NUAA 任务。

| ID | 数据集 | Variant | GPU | Python PID | 当前状态 | 输出目录 |
|---|---|---|---:|---:|---|---|
| B0-NUAA | NUAA-SIRST | sd_raw | 1 | 3092182 | 运行中，epoch 39 | `runs/experiment_b/NUAA-SIRST/sd_raw/seed42` |
| B1-NUAA | NUAA-SIRST | sd_awgm | 4 | 3092264 | 运行中，epoch 34 | `runs/experiment_b/NUAA-SIRST/sd_awgm/seed42` |
| B2-NUAA | NUAA-SIRST | sd_pyramid | 6 | 3092353 | 运行中，epoch 30 | `runs/experiment_b/NUAA-SIRST/sd_pyramid/seed42` |
| B3-NUAA | NUAA-SIRST | sd_full | 自动 | 待分配 | 排队 | `runs/experiment_b/NUAA-SIRST/sd_full/seed42` |
| B0-NUDT | NUDT-SIRST | sd_raw | 自动 | 待分配 | 排队 | `runs/experiment_b/NUDT-SIRST/sd_raw/seed42` |
| B3-NUDT | NUDT-SIRST | sd_full | 自动 | 待分配 | 排队 | `runs/experiment_b/NUDT-SIRST/sd_full/seed42` |
| B1-NUDT | NUDT-SIRST | sd_awgm | 自动 | 待分配 | 排队 | `runs/experiment_b/NUDT-SIRST/sd_awgm/seed42` |
| B2-NUDT | NUDT-SIRST | sd_pyramid | 自动 | 待分配 | 排队 | `runs/experiment_b/NUDT-SIRST/sd_pyramid/seed42` |
| B0-IRSTD | IRSTD-1K | sd_raw | 自动 | 待分配 | 排队 | `runs/experiment_b/IRSTD-1K/sd_raw/seed42` |
| B3-IRSTD | IRSTD-1K | sd_full | 自动 | 待分配 | 排队 | `runs/experiment_b/IRSTD-1K/sd_full/seed42` |

当前所有已启动任务都未到 epoch 100，因此尚无 mIoU、nIoU、F1、Pd、Fa；这属于
正常状态，不应使用早期训练 loss 判断最终优劣。

调度器重排后，NUDT 和 IRSTD 的 `sd_raw` 会在下一张空闲 GPU 上优先启动，随后
才继续 NUAA 的 `sd_pyramid/sd_full` 和其它消融，因此三个数据集不会被 NUAA
独占整个队列。

## 9. 最终结果表

| 数据集 | Variant | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
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

## 10. 指标与判定方法

- mIoU、nIoU、F1、Pd 越高越好；Fa 越低越好；
- 判断单 decoder：重点比较 `sd_raw` 与 WULLE-A；
- 判断 AWGM：同时看 `sd_awgm - sd_raw` 和 `sd_full - sd_pyramid`；
- 判断 Pyramid：同时看 `sd_pyramid - sd_raw` 和 `sd_full - sd_awgm`；
- Pyramid 有效的期望是 Pd 提升且 Fa 不明显恶化；
- 如果精度基本持平而复杂度显著下降，则单 decoder 方案成立；
- 只有训练到 1000 epoch 后的 best checkpoint 才进入最终结论。

AWGM 实验还会记录 H/V/D 方向权重、gate mean/std、alpha、A 与 A_guided norm；
Pyramid 实验还会记录 P1–P4 norm、raw coefficient norm、delta norm、beta 和
final/raw coefficient norm，用于判断模块是否真正工作或发生高频重复增强。
