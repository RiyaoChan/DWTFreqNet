# Experiment J：双证据噪声校准净化（DENP）实验记录

> 状态：代码、服务器 CUDA/真实数据验证和 RTX 3090 复杂度测试均已完成；15 项正式训练队列持续运行。NUAA的J1/J2-R/J2-D已完成1000 epoch；J3-R首次失败输出已独立归档，设备校验已修正，并已在GPU 0重跑。所有表格数值直接来自对应 `best_metrics.json`，严禁用占位值或当前epoch指标冒充best。

## 1. 版本与隔离

- Repository：`RiyaoChan/DWTFreqNet`
- 固定模型基线：Experiment E1，`68ede894be748c8842427e140898f007dbe67953`
- 实际开发起点（Experiment H HEAD）：`ba96c1f119ea1cc4e8f0fdf2dc3818291ff48449`
- Phase 1 参考 HEAD：`356ac5611b4797452d2aeedd954993948e06750d`
- 分支：`codex/experiment-j-dual-evidence-denp`
- 服务器独立目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP`
- Draft PR：[#9](https://github.com/RiyaoChan/DWTFreqNet/pull/9)
- 初始实现 commit：`168a037da4a34abb5cedbb5bc9bb419a6c99a0bc`
- 最终记录 commit：见本文件最近一次提交

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
| 本地完整构造测试 | 环境不适用 | 本地没有 `mamba_ssm`；服务器环境已替代验证 |
| 服务器构造测试 | 通过 | 六个变体均成功构造，初始化和参数关系断言通过 |
| J0/E1 strict state-dict | 通过 | key 顺序与 shape 完全一致，strict load 成功，J0 无 DENP 参数 |
| J0 输出与中间量逐元素一致 | 通过 | 最终输出、A/A_lfss/A_guided/E、四级 decoder 输入/输出最大绝对误差均为 0 |
| MAD、outlier、N、Gaussian、Compactness、gate 公式 | 通过 | Gaussian 尺度误差 0.102%；1% 极端离群点使 MAD 变化 1.234%，std 变化 407.367%；其余逐元素断言通过 |
| CUDA FP32/AMP、两步梯度 | 通过 | 五个训练变体均为有限输出/梯度；J3-R 两步 loss 为 3.1445、2.5220 |
| NUAA batch=4、256×256、连续两步、checkpoint | 通过 | 五个变体各完成两步并写出约 81 MiB smoke checkpoint |

服务器测试输出位于：`artifacts/experiment_j_tests/construct.json`、`cuda_full_real_smoke.json`、`profile_rtx3090.json`。smoke checkpoint 仅保存在服务器，不提交 GitHub。

## 6. 训练协议

- seed 42，patch 256，batch 4，1000 epochs。
- Adam，初始学习率 `1e-3`，10 epoch warmup + CosineAnnealingLR，`eta_min=1e-5`。
- epoch 100 开始评估，此后每 epoch 评估一次；每 20 epoch 保存；阈值 0.5。
- 数据划分、增强、loss、deep supervision 权重、优化器、调度器和评价代码与 E1/Experiment H 保持一致。

## 7. 动态 GPU 队列与正式任务

队列只会选择同时满足以下三项的 GPU：显存占用小于 1000 MiB、利用率小于 10%、没有 compute PID。它不会停止、抢占或覆盖 Experiment H、Phase 1 或其他用户任务；通过 `flock`、任务 claim、`RUNNING.lock`、PID 和完成/失败标记防止重复运行。

任务总数：5 个训练变体 × 3 个数据集 = 15。每个数据集按 J1 → J2-R → J2-D → J3-F → J3-R 排序；数据集按 NUAA-SIRST → IRSTD-1K → NUDT-SIRST 排序。队列能在多张空闲卡上并发领取后续任务。队列 PID 为 `3889915`，每 60 秒重新检查空闲 GPU。

| 数据集 | J1 | J2-R | J2-D | J3-F | J3-R |
|---|---|---|---|---|---|
| NUAA-SIRST | **已完成1000 epoch** | **已完成1000 epoch** | **已完成1000 epoch** | GPU 4运行 | GPU 0重跑运行中，best epoch 248 |
| IRSTD-1K | GPU 6运行，best epoch 354 | GPU 1运行，尚未评估 | GPU 2运行，尚未评估 | GPU 3运行，尚未评估 | 待启动 |
| NUDT-SIRST | 待启动 | 待启动 | 待启动 | 待启动 | 待启动 |

状态更新时间：`2026-07-21 23:47`（Asia/Shanghai）。队列PID `3889915` 正常存活；当前6项运行、3项完成、6项排队，另保留J3-R–NUAA的首次失败归档。J3-R重跑已于22:32在GPU 0启动；IRSTD的J2-R/J2-D/J3-F已分别在GPU 1/2/3启动。运行时权重不提交GitHub。

### 7.1 J3-R–NUAA 首次启动失败记录

- 时间：2026-07-21 19:00:32，GPU 6。
- 原因：`validate_initialization` 用 CPU `torch.ones(3)` 比较 CUDA gamma，触发 device mismatch；模型 forward、梯度和 CUDA smoke 本身此前均已通过。
- 处理：失败目录、`FAILED`、`train.log`、PID 和 `status.tsv` 原样保留；没有 checkpoint，因此没有覆盖权重；队列按规则继续领取其他任务。
- 修正：比较张量改为 `torch.ones_like(processor.gamma_raw/gamma_decoder)`；只修正校验设备，不改变模型、初始化值或训练协议。
- 重跑：用户已明确授权自动重跑。首次失败目录已原样移动到 `seed42_attempt1_failed_20260721T190032`，其中 `FAILED`、`train.log`、PID与状态文件保持不变；干净的正式 `seed42` 路径已于22:32在GPU 0启动重跑，首次失败没有checkpoint，重跑不会覆盖权重。

## 8. 复杂度

统一条件：RTX 3090、`1×1×256×256`、FP32、eval、warmup 5、repeat 20。THOP 对 3×3 depthwise Gaussian 使用显式 handler；median、比较操作和 selective scan 可能不计入 FLOPs，因此 FLOPs 仅用于同脚本内相对比较。

| Variant | Params | THOP FLOPs | Latency (ms) | FPS | Infer peak | Train peak |
|---|---:|---:|---:|---:|---:|---:|
| E1/J0 | 7,013,527 | 15.032 G | 14.404 | 69.43 | 190.3 MiB | 628.0 MiB |
| J1 | 7,013,551 | 15.083 G | 21.346 | 46.85 | 290.7 MiB | 769.8 MiB |
| J2-R | 7,013,559 | 15.083 G | 24.026 | 41.62 | 312.8 MiB | 792.7 MiB |
| J2-D | 7,013,559 | 15.083 G | 22.596 | 44.26 | 312.8 MiB | 818.2 MiB |
| J3-F | 7,013,567 | 15.083 G | 23.912 | 41.82 | 338.9 MiB | 841.1 MiB |
| J3-R | 7,013,591 | 15.083 G | 27.227 | 36.73 | 287.2 MiB | 816.1 MiB |

参数关系已验证：J2-R 与 J2-D 完全一致；J3-R 相对 J3-F 只增加 24 个 gamma 参数。组件耗时是隔离的四级 microbenchmark，不是可相加的端到端归因：

| Variant | MAD (ms) | Compactness (ms) | Gaussian (ms) | 完整 DENP (ms) |
|---|---:|---:|---:|---:|
| J1 | 1.302 | 0.000 | 3.027 | 6.904 |
| J2-R | 1.278 | 1.225 | 3.157 | 8.528 |
| J2-D | 1.440 | 1.189 | 2.915 | 8.361 |
| J3-F | 1.274 | 2.361 | 3.224 | 10.125 |
| J3-R | 1.516 | 2.388 | 3.006 | 11.544 |

## 9. 正式/阶段最佳性能结果

下列 E1/J0 是既有基线，不是本轮重复训练结果。表中所有数值均为截至 `2026-07-21 23:47` 从各正式运行目录的 `best_metrics.json` 读取的最佳epoch及其同epoch五项指标；没有产生该文件的任务明确标记为“运行未评估”或“待训练”，不会用当前epoch指标冒充best。标记“最终”的任务已经完成1000 epoch，标记“阶段”的任务仍在运行。

| Dataset | Variant | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA | E1/J0 | 595 | 0.7842 | 0.7960 | 0.8790 | 0.9656 | 2.717e-5 |
| NUAA | **J1** | **572（最终）** | **0.7962** | **0.8004** | **0.8865** | 0.9733 | **1.955e-5** |
| NUAA | J2-R | 658（最终） | 0.7856 | 0.7956 | 0.8799 | 0.9618 | **1.825e-5** |
| NUAA | J2-D | 738（最终） | 0.7882 | 0.7953 | 0.8816 | 0.9733 | 2.133e-5 |
| NUAA | J3-F | 568（阶段） | 0.7944 | 0.7941 | 0.8854 | **0.9771** | 3.608e-5 |
| NUAA | J3-R | 248（阶段） | 0.7658 | 0.7790 | 0.8673 | 0.9618 | 2.922e-5 |
| IRSTD | E1/J0 | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 |
| IRSTD | J1 | 354（阶段） | 0.6494 | 0.6381 | 0.7874 | 0.8990 | 1.141e-5 |
| IRSTD | J2-R | 运行未评估 | — | — | — | — | — |
| IRSTD | J2-D | 运行未评估 | — | — | — | — | — |
| IRSTD | J3-F | 运行未评估 | — | — | — | — | — |
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

NUAA的J1、J2-R、J2-D已经完成1000 epoch：J1最终best为 mIoU/nIoU/F1/Pd/Fa `0.7962/0.8004/0.8865/0.9733/1.955e-5`，相对E1/J0的mIoU提升 `0.0119`；J2-R取得当前最低Fa `1.825e-5`；J2-D的mIoU为 `0.7882`。仍在运行的J3-F阶段best mIoU为 `0.7944`，J3-R重跑阶段best为 `0.7658`。IRSTD-J1阶段best mIoU为 `0.6494`，低于E1/J0的 `0.6647`，但Fa由 `1.230e-5`降至 `1.141e-5`；其余三项IRSTD任务刚启动、尚未产生best。跨数据集Go/No-Go仍需等待后续任务完成。
