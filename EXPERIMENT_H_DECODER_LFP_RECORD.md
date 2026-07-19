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

## 8. 运行状态与结果（2026-07-19 18:58 CST）

队列 PID `2087851` 已启动。启动时 GPU 0/2/5 正在运行既有 Experiment G，队列没有触碰它们；检测到 GPU 1/3/4/6 空闲后自动启动：

| GPU | 当前正式任务 | 状态 | 评估结果 |
|---:|---|---|---|
| 1 | H1-R / NUAA-SIRST | 训练中（1000 epoch） | epoch 100 后产生 |
| 3 | H1-D / NUAA-SIRST | 训练中（1000 epoch） | epoch 100 后产生 |
| 4 | H2-R / NUAA-SIRST | 训练中（1000 epoch） | epoch 100 后产生 |
| 6 | H2-D / NUAA-SIRST | 训练中（1000 epoch） | epoch 100 后产生 |

其余14项按既定优先级排队；任一显卡满足 `memory.used < 1000 MiB`、`utilization < 10%` 且无 compute PID 时，队列就会自动补任务。H0 不重复训练。最佳 mIoU 对应的 mIoU、nIoU、F1、Pd、Fa 将在这里持续更新。
