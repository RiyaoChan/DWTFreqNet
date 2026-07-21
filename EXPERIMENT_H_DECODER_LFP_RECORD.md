# Experiment H：E1 Decoder LFP 双低频来源高频净化实验记录

## 1. 实验目的与变体

Experiment H 固定使用 Experiment E1 编码器和 Decoder 主体，只在每级 IDWT 前净化一次对齐的原始 H/V/D。它不使用 Experiment G 的 DSHF、语义方向门控或 Targetness。

| ID | 变体 | 低频来源 | Gaussian | 阈值 |
|---|---|---|---|---|
| H0 | `h0_e1_passthrough` | 无 | 无 | 无；只做严格回归和复杂度，不训练 |
| H1-R | `h1_rawll_attention` | 同一次 DWT 的 raw LL | 无 | 无 |
| H1-D | `h1_decoder_attention` | 当前 Decoder low semantic | 无 | 无 |
| H2-R | `h2_rawll_fixed_gaussian` | raw LL | 3×3 depthwise | 固定0.5硬掩码 |
| H2-D | `h2_decoder_fixed_gaussian` | Decoder low | 3×3 depthwise | 固定0.5硬掩码 |
| H3-R | `h3_rawll_adaptive_gaussian` | raw LL | 3×3 depthwise | 自适应软掩码 |
| H3-D | `h3_decoder_adaptive_gaussian` | Decoder low | 3×3 depthwise | 自适应软掩码 |

## 2. 代码改动

- 新增 `model/decoder_lfp.py`：实现 channel mean/max → 7×7 Conv → sigmoid 的低频空间注意力；实现每级单个正 sigma、3×3、replicate padding、逐通道 Gaussian；实现 H3 的逐样本逐通道自适应阈值。
- 新增 `model/DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP.py`：严格继承 E1，不改 Encoder 和 Decoder 主体；完整重写在线 Decoder 顺序，在每级 IDWT 前只对齐一次 raw H/V/D，再执行 LFP。Raw-LL 系列使用同级 DWT 的 LL，Decoder-low 系列依次使用 E4/L3/L2/L1。
- H0 没有任何 LFP 参数，只用于和 E1 做 strict state-dict、输出及 Decoder 中间量回归。
- H1 仅做低频注意力；H2 使用固定 `tau=0.5` 硬掩码选择 Gaussian；H3 使用初始 ratio=0.5 的自适应阈值和温度软掩码。四级均启用，模型始终只有4次 DWT 和4次 IDWT。
- 新增独立训练入口、单任务脚本和动态 GPU 队列；新增结构/数值/梯度/CUDA/AMP/真实数据测试、复杂度脚本和离线区域诊断工具。
- 未修改 `model/DWTFreqNet.py`、E1 模型、Experiment F/G 模型，也未加载任何既有权重。

## 3. 来源与固定设置

- E1 基线分支：`codex/experiment-e-lfss-before-awgm`
- E1 实际基线提交：`68ede894be748c8842427e140898f007dbe67953`
- NS-FPN 官方仓库：`https://github.com/mengduann/NS-FPN`
- NS-FPN 参考提交：`b857bef068ba48f1258b62de6bf082f73dbafde4`
- 参考文件与模块：`model/NS_FPN.py` / `wav_Enhance`
- H 分支：`codex/experiment-h-decoder-lfp-purification`
- 训练：seed 42、batch 4、patch 256、1000 epoch、Adam `1e-3`、10 epoch warmup + Cosine、`eta_min=1e-5`
- epoch 100 起每个 epoch 评估；每20 epoch保存；阈值0.5；全部随机初始化

## 4. 正式任务与执行顺序

六个正式变体 × NUAA-SIRST、IRSTD-1K、NUDT-SIRST，共18项。每个数据集内顺序固定为 H1-R、H1-D、H2-R、H2-D、H3-R、H3-D；数据集顺序为 NUAA、IRSTD、NUDT。动态队列仅使用满足显存、利用率和无 compute PID 三项空闲条件的显卡。

服务器项目目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_H_DECODER_LFP`；正式输出目录：`runs/experiment_h_decoder_lfp/`。队列不会停止或抢占其他用户/其他 Experiment 任务。

## 5. 验证结果（2026-07-19）

- Python 编译和 shell 语法：通过。
- H0 strict 回归：state-dict 键完全一致；E1/H0 的输出、X0、E1–E4、U3–U0、L3–L0 最大绝对差异均为 `0.0`。
- 六个正式变体：`2×1×256×256` CUDA FP32 和 AMP forward/backward 全通过；训练输出均为6张同分辨率概率图。
- 低频来源、Attention、H2固定阈值/硬掩码、H3自适应阈值/软掩码、Gaussian kernel/sigma/depthwise、正负高频、4 DWT/4 IDWT、Encoder不变、禁用 `torch.cdist/topk` 均通过。
- H3 两步梯度：threshold predictor 末层首步梯度绝对和 `1.9351e-3`；前层第二步梯度绝对和 `4.3421e-4`，符合零初始化末层的预期传播顺序。
- 六个变体均使用真实 NUAA、batch 4、patch 256 连续完成2个 optimizer step，loss 有限，并分别写出检查点。
- 六个变体又通过正式训练入口的2-batch/1-epoch检查，`latest.pth.tar` 全部成功写出。

验证文件只保存在服务器 `validation/`，检查点与日志不提交 Git。

## 6. 复杂度（RTX 3090，FP32，1×1×256×256）

统一 warmup 5 次、计时20次；THOP 为 Gaussian depthwise 3×3 增加显式统计，Selective Scan 仍可能被 THOP 低估。

| 变体 | 参数量 | LFP参数 | THOP FLOPs | 延迟(ms) | FPS | 推理峰值(MiB) | 训练峰值(MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0/E1 | 7,013,527 | 0 | 15.0318G | 12.664 | 78.96 | 213.1 | 650.7 |
| H1-R | 7,013,919 | 392 | 15.0339G | 13.190 | 75.81 | 247.4 | 658.2 |
| H1-D | 7,013,919 | 392 | 15.0339G | 13.409 | 74.58 | 247.4 | 683.8 |
| H2-R | 7,013,923 | 396 | 15.0852G | 15.161 | 65.96 | 313.4 | 794.6 |
| H2-D | 7,013,923 | 396 | 15.0852G | 14.999 | 66.67 | 313.4 | 769.3 |
| H3-R | 7,357,291 | 343,764 | 15.0856G | 18.386 | 54.39 | 314.0 | 792.0 |
| H3-D | 7,357,291 | 343,764 | 15.0856G | 19.455 | 51.40 | 314.0 | 820.0 |

R/D 三组配对的参数量与 FLOPs 完全一致，低频来源本身没有引入额外参数或运算。

## 7. E1参考结果

| 数据集 | best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 595 | 78.42% | 79.60% | 87.90% | 96.56% | 27.17e-6 |
| IRSTD-1K | 551 | 66.47% | 66.30% | 79.85% | 91.58% | 12.30e-6 |
| NUDT-SIRST | 456 | 95.16% | 94.82% | 97.52% | 99.47% | 1.724e-6 |

## 8. 运行状态与结果

结果更新时间：2026-07-21 11:46（Asia/Shanghai）。NUAA/IRSTD 的最终结果以及 NUDT 的阶段性结果均直接读取服务器对应 `train.log` 的 JSON 评估记录；`best epoch/mIoU` 为截至当前记录的历史最佳，Fa 越低越好。当前共 13 项完成、5 项运行、0 项排队，无失败或缺失任务；NUDT 的 H1-R 已完成 1000 epoch，其余五项继续训练至 1000 epoch。

### 8.1 NUAA-SIRST 最终结果

H1-R、H1-D、H2-R、H2-D、H3-R、H3-D 均已完成 1000 epoch。

| 方案 | best epoch | mIoU | nIoU | F1 | Pd | Fa | 相对 E1 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1/H0 | 595 | 0.7842 | **0.7960** | 0.8790 | 0.9656 | 2.717e-5 | — |
| H1-R | 458 | 0.7902 | 0.7939 | 0.8828 | 0.9542 | 2.840e-5 | +0.0060 |
| H1-D | 655 | 0.7907 | 0.7910 | 0.8831 | 0.9580 | **1.736e-5** | +0.0065 |
| H2-R | 393 | 0.7914 | 0.7957 | 0.8836 | 0.9618 | 2.380e-5 | +0.0072 |
| **H2-D** | 493 | **0.7963** | 0.7948 | **0.8866** | 0.9580 | 2.319e-5 | **+0.0121** |
| H3-R | 438 | 0.7926 | 0.7931 | 0.8843 | 0.9695 | 1.749e-5 | +0.0084 |
| H3-D | 622 | 0.7828 | 0.7911 | 0.8782 | **0.9809** | 3.361e-5 | -0.0014 |

NUAA 上 H2-D 的 mIoU/F1 最好，mIoU 比 E1 提高 0.0121；H1-D 的 Fa 最低；H3-R 在 mIoU、Pd 和 Fa 之间最均衡。H3-D 虽然 Pd 最高，但 mIoU 低于 E1 且 Fa 明显升高，不推荐。

### 8.2 IRSTD-1K 最终结果

H1-R、H1-D、H2-R、H2-D、H3-R、H3-D 均已完成 1000 epoch。

| 方案 | best epoch | mIoU | nIoU | F1 | Pd | Fa | 相对 E1 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1/H0 | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 | — |
| H1-R | 625 | 0.6579 | 0.6570 | 0.7937 | 0.9158 | 1.177e-5 | -0.0068 |
| H1-D | 780 | 0.6717 | 0.6701 | 0.8036 | 0.9125 | **8.768e-6** | +0.0070 |
| **H2-R** | 820 | **0.6726** | **0.6703** | **0.8043** | 0.9259 | 9.167e-6 | **+0.0079** |
| H2-D | 743 | 0.6611 | 0.6567 | 0.7960 | 0.9057 | 1.272e-5 | -0.0036 |
| H3-R | 828 | 0.6653 | 0.6660 | 0.7990 | **0.9327** | 1.289e-5 | +0.0006 |
| H3-D | 645 | 0.6641 | 0.6651 | 0.7982 | **0.9327** | 1.258e-5 | -0.0006 |

IRSTD 上 H2-R 的 mIoU、nIoU 和 F1 最好，且相比 E1 同时提高 Pd、降低 Fa，是综合最优方案。H1-D 与 H2-R 的主指标接近，并取得最低 Fa。

### 8.3 NUDT-SIRST 阶段性结果

H1-R 已完成 1000 epoch；其余五项仍在训练，因此除 H1-R 外本节仍是阶段性结果。

| 方案 | 当前 epoch | best epoch | mIoU | nIoU | F1 | Pd | Fa | 相对 E1 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1/H0 | 1000 | 456 | 0.9516 | 0.9482 | 0.9752 | **0.9947** | 1.724e-6 | — |
| **H1-R** | **1000** | 653 | 0.9568 | **0.9570** | 0.9779 | 0.9926 | 2.781e-6 | **+0.0052** |
| H1-D | 563 | 488 | 0.9555 | 0.9546 | 0.9772 | 0.9937 | 1.884e-6 | +0.0039 |
| H2-R | 523 | 471 | 0.9557 | 0.9558 | 0.9774 | 0.9905 | 2.344e-6 | +0.0041 |
| H2-D | 483 | 376 | 0.9570 | 0.9551 | 0.9780 | 0.9937 | 1.609e-6 | +0.0053 |
| **H3-R** | 454 | 437 | **0.9576** | 0.9567 | **0.9783** | 0.9894 | **1.241e-6** | **+0.0060** |
| H3-D | 439 | 433 | 0.9517 | 0.9495 | 0.9753 | 0.9926 | 2.413e-6 | +0.0001 |

截至当前，H3-R 的历史最佳 mIoU/F1/Fa 分别为 `0.9576/0.9783/1.241e-6`，H1-R 已完成 1000 epoch 且保持 `0.9568` 的最佳 mIoU。H1-R、H1-D、H2-R、H2-D、H3-R 的主指标均已超过 E1；由于 H1-D/H2-R/H2-D/H3-R/H3-D 尚未完成 1000 epoch，暂不作最终排序。

### 8.4 当前任务分配

| GPU | 当前任务 | 状态 |
|---:|---|---|
| 0 | Phase 1 P2：NUAA-SIRST directional consistency | 运行中 |
| 1 | H2-D / NUDT-SIRST | 训练中 |
| 2 | H1-D / NUDT-SIRST | 训练中 |
| 3 | H2-R / NUDT-SIRST | 训练中 |
| 4 | H3-R / NUDT-SIRST | 训练中 |
| 5 | Phase 1 P3：NUAA-SIRST sampling geometry | 运行中 |
| 6 | H3-D / NUDT-SIRST | 训练中 |

所有未完成的 NUDT 方案仍按要求训练到 1000 epoch，后续最终结果继续更新本节。Phase 1 使用 GPU 0 和 GPU 5 的低显存诊断任务未停止或抢占 H 的 GPU 1/2/3/4/6。
