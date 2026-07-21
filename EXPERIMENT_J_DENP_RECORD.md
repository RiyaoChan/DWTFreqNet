# Experiment J：双证据噪声校准净化（DENP）实验记录

> 状态：代码实现与本地语法检查已完成；服务器 CUDA/真实数据验证、复杂度测试和正式训练待更新。严禁用占位值冒充实验结果。

## 1. 版本与隔离

- Repository：`RiyaoChan/DWTFreqNet`
- 固定模型基线：Experiment E1，`68ede894be748c8842427e140898f007dbe67953`
- 实际开发起点（Experiment H HEAD）：`ba96c1f119ea1cc4e8f0fdf2dc3818291ff48449`
- Phase 1 参考 HEAD：`356ac5611b4797452d2aeedd954993948e06750d`
- 分支：`codex/experiment-j-dual-evidence-denp`
- 服务器独立目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP`
- Draft PR：待创建
- 最终 commit：待更新

本实验不修改 E1、Experiment H、Phase 1 的已有模型文件；所有 DENP 实现、测试、训练、诊断和队列均位于独立新文件中。

## 2. 研究问题与方案

Experiment J 研究两个问题：高频系数是否弱到更像噪声，以及原始 LL/当前 Decoder low 是否支持当前位置为紧凑小目标。只有噪声证据要求净化、且低频证据没有要求保护时，才用同 band 的 3×3 depthwise Gaussian 平滑替换一部分有符号高频。

| ID | 训练变体 | 含义 |
|---|---|---|
| J0 | `j0_e1_passthrough` | E1 严格直通，只用于回归和复杂度基准，不重复训练 |
| J1 | `j1_bandwise_noise_calibrated` | 每样本、每 stage、每 H/V/D band 用稳健 MAD 标定噪声尺度 |
| J2-R | `j2_rawll_compactness` | J1 + 同一次 DWT 的 raw LL 紧凑性保护 |
| J2-D | `j2_decoder_compactness` | J1 + 当前 Decoder low 紧凑性保护 |
| J3-F | `j3_dual_evidence_fixed` | 同时使用 raw LL 与 Decoder low，固定乘积 gate |
| J3-R | `j3_dual_evidence_reliability` | 双证据 gate，并为每 stage/band 学习两组可靠性指数 |

固定数据流：E1 encoder 不变；每个 decoder stage 仅使用一次对齐后的 raw H/V/D；DENP 在 IDWT 前净化；仍保持 4 次 DWT、4 次 IDWT。未引入第二次 DWT、方向金字塔、LDRC、DSHF、额外 loss 或 hard threshold。

## 3. 代码文件

- `model/decoder_denp.py`：MAD 噪声尺度、band 独立 Gaussian、低频紧凑性、DENP gate。
- `model/DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP.py`：固定 E1 主体与 J0–J3 六种变体。
- `train_experiment_j_denp.py`：1000 epoch 正式训练入口。
- `tools/test_experiment_j_denp.py`：严格回归、公式、梯度、CUDA/AMP、真实数据 smoke。
- `tools/profile_experiment_j_denp.py`：参数、THOP FLOPs、延迟、显存和组件耗时。
- `tools/analyze_experiment_j_denp.py`：复用 Phase 1 困难背景候选规则的区域/逐样本/阈值诊断。
- `scripts/run_experiment_j_denp.sh`：单任务执行器。
- `scripts/launch_experiment_j_denp_queue.sh`：15 项动态空闲 GPU 队列。

## 4. 核心公式与初始化

- 稳健尺度：每通道空间 MAD 后再做通道中位数，`sigma_hat = 1.4826 × median_channel(MAD)`；FP32 计算并 `detach`。
- 阈值：`tau = lambda × sigma_hat`，`lambda ∈ (0.5, 3.0)`，初始值 1.0。
- 弱高频概率：`N = sigmoid((tau - |B|) / (0.15 × tau + 1e-6))`。
- Gaussian：每 stage/band 一个标量 `sigma ∈ (0.5, 2.0)`，初始值 1.0；通道内共享、band 间不共享。
- Compactness：通道 RMS 能量上的 3×3 中心与 7×7 纯外环对比；斜率范围 `(1, 20)`、初始 5；阈值范围 `(-0.5, 0.5)`、初始 0。
- J3-R：每 stage/band 有 `gamma_R` 和 `gamma_D`，范围 `(0.5, 2.0)`、初始 1；总计 24 个额外参数。

## 5. 测试记录

| 检查 | 状态 | 证据 |
|---|---|---|
| Python 语法编译 | 通过 | 六个 Python 新文件 `py_compile` 无错误 |
| 本地完整构造测试 | 环境不适用 | 本地没有 `mamba_ssm`；必须在 226 的既有 PyTorch/Mamba 环境运行 |
| J0/E1 strict state-dict | 待服务器验证 | 不允许记录推测结果 |
| J0 输出与中间量逐元素一致 | 待服务器验证 | 不允许记录推测结果 |
| MAD、outlier、N、Gaussian、Compactness、gate 公式 | 待服务器脚本正式运行 | 测试代码已实现 |
| CUDA FP32/AMP、两步梯度 | 待服务器验证 | 五个训练变体均覆盖 |
| NUAA batch=4、256×256、连续两步、checkpoint | 待服务器验证 | 五个训练变体均覆盖 |

服务器测试输出将写入：`artifacts/experiment_j_tests/`。

## 6. 训练协议

- seed 42，patch 256，batch 4，1000 epochs。
- Adam，初始学习率 `1e-3`，10 epoch warmup + CosineAnnealingLR，`eta_min=1e-5`。
- epoch 100 开始评估，此后每 epoch 评估一次；每 20 epoch 保存；阈值 0.5。
- 数据划分、增强、loss、deep supervision 权重、优化器、调度器和评价代码与 E1/Experiment H 保持一致。

## 7. 动态 GPU 队列与正式任务

队列只会选择同时满足以下三项的 GPU：显存占用小于 1000 MiB、利用率小于 10%、没有 compute PID。它不会停止、抢占或覆盖 Experiment H、Phase 1 或其他用户任务；通过 `flock`、任务 claim、`RUNNING.lock`、PID 和完成/失败标记防止重复运行。

任务总数：5 个训练变体 × 3 个数据集 = 15。每个数据集按 J1 → J2-R → J2-D → J3-F → J3-R 排序；数据集按 NUAA-SIRST → IRSTD-1K → NUDT-SIRST 排序。队列能在多张空闲卡上并发领取后续任务。

| 数据集 | J1 | J2-R | J2-D | J3-F | J3-R |
|---|---|---|---|---|---|
| NUAA-SIRST | 待启动 | 待启动 | 待启动 | 待启动 | 待启动 |
| IRSTD-1K | 待启动 | 待启动 | 待启动 | 待启动 | 待启动 |
| NUDT-SIRST | 待启动 | 待启动 | 待启动 | 待启动 | 待启动 |

运行 PID、GPU、当前 epoch、best 指标和失败信息将在正式队列启动后更新，运行时权重不提交 GitHub。

## 8. 复杂度

统一条件：RTX 3090、`1×1×256×256`、FP32、eval、warmup 5、repeat 20。除参数/FLOPs/延迟/FPS/推理与训练峰值显存外，还单独报告四级 MAD、Compactness、Gaussian 与完整 DENP 的隔离耗时。结果待服务器验证后填写。

| Variant | Params | THOP FLOPs | Latency (ms) | FPS | Infer peak | Train peak |
|---|---:|---:|---:|---:|---:|---:|
| E1/J0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| J1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| J2-R | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| J2-D | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| J3-F | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| J3-R | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 9. 正式性能结果

下列 E1/J0 是既有基线，不是本轮重复训练结果；J1–J3 只在正式评估文件产生后更新。

| Dataset | Variant | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA | E1/J0 | 595 | 0.7842 | 0.7960 | 0.8790 | 0.9656 | 2.717e-5 |
| NUAA | J1 | 待训练 | — | — | — | — | — |
| NUAA | J2-R | 待训练 | — | — | — | — | — |
| NUAA | J2-D | 待训练 | — | — | — | — | — |
| NUAA | J3-F | 待训练 | — | — | — | — | — |
| NUAA | J3-R | 待训练 | — | — | — | — | — |
| IRSTD | E1/J0 | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 |
| IRSTD | J1 | 待训练 | — | — | — | — | — |
| IRSTD | J2-R | 待训练 | — | — | — | — | — |
| IRSTD | J2-D | 待训练 | — | — | — | — | — |
| IRSTD | J3-F | 待训练 | — | — | — | — | — |
| IRSTD | J3-R | 待训练 | — | — | — | — | — |
| NUDT | E1/J0 | 456 | 0.9516 | 0.9482 | 0.9752 | 0.9947 | 1.724e-6 |
| NUDT | J1 | 待训练 | — | — | — | — | — |
| NUDT | J2-R | 待训练 | — | — | — | — | — |
| NUDT | J2-D | 待训练 | — | — | — | — | — |
| NUDT | J3-F | 待训练 | — | — | — | — | — |
| NUDT | J3-R | 待训练 | — | — | — | — | — |

## 10. 诊断与 Go/No-Go

正式 checkpoint 产生后，诊断工具将复用 Phase 1 的 target exclusion、局部极大值和强度匹配困难背景规则，输出 Target Interior、Boundary、Hard Negative、Near/Far Background 的 `sigma_hat/lambda/tau/N/C/P/M`、IDWT 差异、逐样本 E1 配对和 0.1–0.9 阈值扫描。

- J1 Go：至少两个数据集 mIoU/nIoU 提升，或主指标持平且 Fa 明显下降；lambda 不全部撞边界，N 不接近全 0/全 1。
- J2 Go：J2-R 或 J2-D 相对 J1 稳定改善，且 `P_target > P_hard-negative`。
- J3-F Go：相对两种单低频来源跨数据集更稳定，或接近各数据集单源最优。
- J3-R Go：相对 J3-F 缓解过度保守，gamma 未全部撞边界且存在有意义的 stage/band 差异。

当前尚无正式训练与诊断结果，因此 Go/No-Go 结论为“待实验”，不是 No-Go。
