# Experiment K v2：以 E1 为中心的剂量校准净化实验记录

> 状态：核心实现与启动前验证已完成；第一批 `K2 × NUAA/IRSTD` 已于2026-07-23 00:19启动。K-A train discovery正在独立GPU上运行，尚未锁定，因此K3–K6保持禁用。所有正式训练均为1000 epoch，epoch 100起每epoch评估。

## 1. 版本与隔离

- Repository：`RiyaoChan/DWTFreqNet`
- 实际 Experiment J base HEAD：`048b01aa0f9c13edfb75a3081dde003e4e9aef4b`
- Phase 1 参考 HEAD：`e7980e064acc4eca06237a23914adc77cabf94fe`
- 分支：`codex/experiment-k-e1-centered-dose-purification`
- 本地工作区：`G:/DWTFreqNet-main/DWTFreqNet-experiment-k`
- 服务器目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_K_E1_CENTERED`
- Draft PR：[#10 Experiment K: E1-centered dose-calibrated purification](https://github.com/RiyaoChan/DWTFreqNet/pull/10)，base=`codex/experiment-j-dual-evidence-denp`
- Experiment J：未停止、未覆盖、未修改其代码或输出目录

## 2. 核心代码改动

本实验未修改规范禁止修改的 E1/J 文件，新建独立实现：

- `model/decoder_k_dose.py`：复用J1的稳健MAD与band Gaussian；新增12个stage/band独立alpha、Gaussian-radial Compactness及最多4个stage rho。
- `model/DWTFreqNet_SingleDecoder_LFSS_AWGM_K.py`：固定E1 encoder/decoder数据流，仅在每级IDWT前对一次对齐后的有符号H/V/D施加剂量校准Gaussian残差。
- `train_experiment_k.py`：固定seed42、patch256、batch4、1000 epoch、Adam、10 epoch warmup+cosine、epoch100起逐epoch评估。
- `tools/test_experiment_k.py`：K0/E1、K1/J1、alpha=0、alpha=1、rho=0严格退化测试，offset、stop-gradient、CUDA/AMP、两步梯度和真实NUAA smoke。
- `tools/profile_experiment_k.py`：参数、FLOPs、延迟、FPS、推理/训练显存、MAD/Gaussian/Gaussian-radial组件耗时。
- `tools/experiment_k/`：K-A1–A7诊断及跨数据集discovery/confirmation决策聚合工具。
- `scripts/run_experiment_k.sh`、`scripts/launch_experiment_k_queue.sh`：任务锁、失败/完成标记与严格空闲GPU调度。

### 2.1 K2剂量公式

```text
dose = alpha(stage, band) × N
purified = aligned + dose × (Gaussian(aligned) - aligned)
```

- `alpha∈(0,1)`，共12个，初始化0.05；
- `alpha=0`严格回退E1；
- `alpha=1,rho=0`严格回退J1。

### 2.2 Compactness有限保护

```text
dose = alpha × N × (1-rho×P_GR)
rho∈(0,0.5)，初始化0.05
```

- 32个Gaussian-radial offset，Rmax=2；
- inner 14点、outer 18点；
- `grid_sample`为bilinear、border、align_corners=False，chunk=8；
- prior low在Compactness入口强制detach；
- K4/K5的Stage4不创建rho参数。

## 3. 变体与当前放行状态

| 变体 | 含义 | 当前状态 |
|---|---|---|
| K0 | E1严格直通 | 仅回归/profile，不训练 |
| K1 | J1满剂量机制对照 | 仅回归/profile，不训练 |
| K2 | 可学习alpha，无Compactness | **必须执行；第一批NUAA/IRSTD** |
| K3 | Gaussian-radial Raw LL，S1–4 | 等待K-A5 Go |
| K4 | Gaussian-radial LFSS LL，S1–3 | 等待K-A5 Go |
| K5 | Gaussian-radial Guided LL，S1–3 | 等待K-A5 Go |
| K6 | train discovery预注册的stage混合 | 当前禁用 |

初始 `analysis/experiment_k/K_A_DECISION.json` 明确设置 discovery/confirmation 为false，只允许K2。队列不会在诊断完成前启动K3–K6。

## 4. 测试记录

### 4.1 构造与公式测试

服务器环境首轮构造测试已通过：

| 检查 | 结果 |
|---|---:|
| Gaussian-radial vs Phase 1 offset最大误差 | `0.0` |
| inner / outer点数 | `14 / 18` |
| 最大半径 | `2.0`（浮点值`1.9999999999999998`） |
| alpha初始化 | `0.0500000007` |
| rho初始化 | `0.0500000007` |
| stop-gradient | `protection.requires_grad=false` |

参数量（构造阶段）：

| 变体 | 参数量 |
|---|---:|
| K0/E1 | 7,013,527 |
| K1/J1 | 7,013,551 |
| K2 | 7,013,563 |
| K3 | 7,013,567 |
| K4 | 7,013,566 |
| K5 | 7,013,566 |

### 4.2 完整CUDA/退化/真实数据验证

以下验证均已通过：

| 检查 | 最大误差/结果 |
|---|---:|
| K0→E1 | `0.0` |
| alpha=0→E1 | `0.0` |
| K1→J1 | `0.0` |
| alpha=1、rho=0→J1 | `0.0` |
| K3 rho=0→K2 | `0.0` |
| K2/K3/K4/K5 CUDA FP32与AMP | 全部有限输出与梯度 |
| K5连续两步梯度loss | `3.3383 → 2.7884` |
| NUAA batch4 K2真实数据两步smoke | checkpoint已写出，loss有限 |
| Haar方向检查 | H纵向扫描、V横向扫描，routing aligned |

### 4.3 RTX 3090复杂度

统一条件：`1×1×256×256`、FP32、eval、warmup 5、repeat 20。THOP未必统计median、`grid_sample`和selective scan，因此FLOPs只用于同脚本相对比较。

| 变体 | Params | THOP FLOPs | Latency | FPS | Infer peak | Train peak |
|---|---:|---:|---:|---:|---:|---:|
| K0/E1 | 7,013,527 | 15.032 G | 15.868 ms | 63.02 | 194.1 MiB | 628.0 MiB |
| K1/J1 | 7,013,551 | 15.083 G | 26.199 ms | 38.17 | 316.2 MiB | 791.6 MiB |
| K2 | 7,013,563 | 15.083 G | 26.205 ms | 38.16 | 337.9 MiB | 813.3 MiB |
| K3 | 7,013,567 | 15.083 G | 41.546 ms | 24.07 | 338.4 MiB | 861.3 MiB |
| K4 | 7,013,566 | 15.083 G | 38.767 ms | 25.79 | 364.1 MiB | 860.6 MiB |
| K5 | 7,013,566 | 15.083 G | 44.455 ms | 22.49 | 311.6 MiB | 835.1 MiB |

K2相对K1只增加12个alpha；K3相对K2增加4个rho，K4/K5各增加3个rho。Gaussian-radial四stage/三stage隔离耗时分别约16.96 ms与14.10–21.54 ms。

## 5. K-A诊断状态

| 诊断 | 状态 | 输出 |
|---|---|---|
| A1/A2 算子忠实度与低频source错配 | NUAA、IRSTD train均已完成 | C_P2/C_square/C_GR、四source、四stage |
| A3/A4 stage/rho反事实与Gate强度 | NUAA、IRSTD train均运行中 | source×operator×stage集合×rho、阈值0.1–0.9 |
| A5 Gaussian处理效应 | 首次运行因未开启`debug_tensors`失败；修复后已排队自动重跑 | Gaussian-on/off局部Delta loss/IoU/概率/Fa |
| A6 先验反馈漂移 | NUAA、IRSTD train均已完成 | gap/AUC/分布漂移、逐样本D_feat与stop-gradient审计 |
| A7 MAD/Gaussian扩散 | 等同数据集A5完成后运行 | global/per-channel/local MAD与三层ring扩散 |

发现集只使用train；锁定决策后才运行test confirmation。NUDT的部分J2/J3 checkpoint尚未完成时，不会伪造或提前锁定三数据集K-A结论。

服务器端发现集队列覆盖A1–A7；A6依赖同数据集A1/A2完成，A7依赖同数据集A5完成。调度按诊断类型优先跨NUAA/IRSTD配对，再进入下一类型。它与正式训练使用相同的空闲判定：显存`<1000 MiB`、利用率`<10%`且无compute PID，不抢占现有Experiment J/K任务。

### 5.1 已完成的train discovery结果

A1/A2已覆盖NUAA的213张train图像和IRSTD的800张train图像。下表是跨E1/J1/J2/J3 checkpoint与四个stage汇总后，Dense算子相对Phase 1实例级`C_P2`的Spearman中位数：

| 数据集 | source | C_square | C_GR |
|---|---|---:|---:|
| NUAA | Raw LL | 0.5436 | **0.6391** |
| NUAA | LFSS LL | 0.5219 | **0.6347** |
| NUAA | Guided LL | 0.5270 | **0.6376** |
| NUAA | Decoder low | 0.2750 | **0.6277** |
| IRSTD | Raw LL | **0.6361** | 0.5052 |
| IRSTD | LFSS LL | **0.5273** | 0.4609 |
| IRSTD | Guided LL | **0.5286** | 0.4728 |
| IRSTD | Decoder low | 0.3246 | **0.7234** |

当前没有跨数据集一致的算子赢家：NUAA四种source均更支持`C_GR`，IRSTD的Raw/LFSS/Guided更支持`C_square`，只有Decoder low更支持`C_GR`。目标与困难背景区分上，IRSTD的`C_square`中位ROC-AUC较高（Raw/LFSS/Guided分别约`0.9244/0.9125/0.8685`）；NUAA最明显的是Decoder low的`C_square≈0.7440`。这些只是描述/忠实度结果，不能替代A5处理效应判定。

A6每个数据集完成240组gap/AUC/分布比较，并记录NUAA `17,040`条、IRSTD `64,000`条逐样本`D_feat`。以下数值均来自正式`prior_distribution_drift.csv`、`feature_drift_summary.csv`和梯度审计，而不是文字快照。

#### 5.1.1 A6 prior分布漂移

表中均为相对E1、跨source/stage/operator后的中位变化；负的gap漂移表示target与hard-negative的Compactness分离度相对E1减小。

| 数据集 | 变体 | Target-hard gap漂移 | ROC-AUC漂移 | Target分布漂移 | Hard-negative分布漂移 |
|---|---|---:|---:|---:|---:|
| NUAA | J1 | +0.005446 | +0.007274 | +0.005650 | -0.000016 |
| NUAA | J2-R | -0.002644 | -0.009983 | -0.002826 | -0.002075 |
| NUAA | J2-D | +0.002026 | +0.003032 | -0.001399 | -0.003241 |
| NUAA | J3-F | +0.000562 | +0.000893 | -0.002130 | -0.004293 |
| NUAA | J3-R | +0.000146 | -0.018090 | -0.003990 | -0.001637 |
| IRSTD | J1 | +0.005451 | +0.003691 | -0.025574 | -0.023122 |
| IRSTD | J2-R | -0.037814 | -0.008906 | -0.049189 | -0.008896 |
| IRSTD | J2-D | -0.028121 | -0.004410 | -0.040363 | -0.007229 |
| IRSTD | J3-F | -0.032268 | -0.010670 | -0.039002 | -0.007148 |
| IRSTD | J3-R | -0.025451 | -0.009114 | -0.024685 | -0.006356 |

IRSTD中J2-R/J2-D/J3-F/J3-R的gap和AUC漂移均为负，说明训练后Compactness先验的target/hard-negative区分总体弱于E1；NUAA的变化较小且方向不统一。

#### 5.1.2 A6 特征漂移 `D_feat`

| 数据集 | 变体 | `D_feat`中位数 | 最大`D_feat` | 最大漂移source | Stage |
|---|---|---:|---:|---|---:|
| NUAA | J1 | 0.829745 | 0.938987 | Raw LL | 4 |
| NUAA | J2-R | 0.844825 | 0.923243 | Raw LL | 4 |
| NUAA | J2-D | 0.855632 | 0.966028 | LFSS LL | 4 |
| NUAA | J3-F | 0.860877 | 0.963034 | Raw LL | 4 |
| NUAA | J3-R | 0.868122 | 0.992219 | Raw LL | 4 |
| IRSTD | J1 | 1.032316 | 1.270032 | Guided LL | 2 |
| IRSTD | J2-R | 0.959574 | 1.081257 | LFSS LL | 3 |
| IRSTD | J2-D | 0.990831 | **1.951730** | LFSS LL | 3 |
| IRSTD | J3-F | 0.960495 | 1.770811 | LFSS LL | 3 |
| IRSTD | J3-R | 0.983146 | 1.365369 | LFSS LL | 3 |

`D_feat`中位数普遍接近1，IRSTD的J2-D Stage3 LFSS LL最大达到`1.951730`，说明J2/J3训练后的低频表征已经明显偏离E1。

#### 5.1.3 A6 梯度审计

| 路径 | Low source梯度 | 输出是否可求导 | 解释 |
|---|---:|---|---|
| Square正常路径 | L2 norm约`0.00305` | 是 | Compactness会向low source及自身参数回传 |
| Square使用`low_source.detach()` | `None` | 是 | 输出梯度仅来自Square自身斜率/阈值参数，不回到low source |
| Gaussian-radial正式路径 | `None` | 否 | `C_GR`无可学习参数且source已detach |
| Experiment K正式实现 | `None` | 不适用 | 所有Compactness source统一stop-gradient |

A6因此支持K正式实现始终对Compactness source使用stop-gradient；原摘要中的`unexpected_grad_path`实际是把Square自身参数梯度误当作低频source梯度，已在审计代码中纠正。

A5首次尝试在读取source时发现J1-as-K没有开启`debug_tensors`，两数据集均在生成正式CSV前失败。失败输出已分别归档到`attempts/attempt_001_debug_tensors_disabled`；修复commit为`4b988e6`，两个任务已恢复为queued，等待满足空闲门槛的GPU自动重跑。当前尚无A5结果，因此`K_A_DECISION.json`仍保持No-Go锁定状态。

## 6. 正式实验安排

第一批固定：

```text
K2 × NUAA-SIRST × seed42 × 1000 epoch
K2 × IRSTD-1K × seed42 × 1000 epoch
```

第二批仅在K-A Go后：在NUAA/IRSTD运行`K_A_DECISION.json`放行的K3/K4/K5/K6。

第三批：根据NUAA和IRSTD选择同一个全局配置的1–2个候选，再运行NUDT；不得为三个数据集分别挑不同配置。

GPU调度门槛：显存低于1000 MiB、利用率低于10%、无compute PID。队列不得停止、抢占或覆盖Experiment J及其他用户任务。

## 7. 正式任务状态

启动时间：`2026-07-23 00:19`（Asia/Shanghai）。队列PID `952620`，每60秒重新检查GPU，仅使用显存低于1000 MiB、利用率低于10%且无compute PID的卡。

| 变体 | 数据集 | GPU | Wrapper PID | Python PID | 状态 |
|---|---|---:|---:|---:|---|
| K2 | NUAA-SIRST | 3 | 952684 | 952691 | 运行，epoch 807/1000 |
| K2 | IRSTD-1K | 4 | 952888 | 952895 | 运行，epoch 270/1000 |

状态更新时间：`2026-07-23 09:10`（Asia/Shanghai）。当前epoch指标来自`train.log`最后一条正式评估记录，历史最佳指标来自对应`best_metrics.json`，二者严格分开记录。

### 7.1 当前epoch性能

| 数据集 | 变体 | 状态 | 当前epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | K2 | 运行 | 807 | 0.792627 | 0.793240 | 0.884319 | 0.969466 | 2.91554e-05 |
| IRSTD-1K | K2 | 运行 | 270 | 0.626016 | 0.637622 | 0.770000 | 0.925926 | 2.93029e-05 |

### 7.2 历史最佳性能

| 数据集 | 变体 | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | K2 | 607 | 0.801101 | 0.795325 | 0.889568 | 0.965649 | 2.54509e-05 |
| IRSTD-1K | K2 | 158 | 0.649640 | 0.603540 | 0.787614 | 0.892256 | 2.67978e-05 |

K3–K6没有启动；NUDT任务也没有提前启动。它们分别受K-A Go和NUAA/IRSTD同一全局配置筛选约束。正式指标从epoch 100开始产生，后续记录当前epoch、best epoch及mIoU/nIoU/F1/Pd/Fa。
