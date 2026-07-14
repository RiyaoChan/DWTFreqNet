# Experiment D：SD-AWGM + Decoder-side HFE 实验记录

## 1. 实验定位

本实验以 Experiment B 的 `sd_awgm` 为唯一基础，在单一 Wavelet Decoder 的
四次 IDWT 前加入 Decoder-side High-Frequency Enhancement（HFE）。Encoder 仍由
H/V/D 通过 Stage-wise AWGM 引导同源低频；Decoder HFE 使用当前低频语义匹配并
残差校正同尺度 H/V/D 系数。

HFE 设计受 Wave-Mamba 的低频引导高频思想启发，但模块为独立重写，没有复制其
完整源码文件。

## 2. 代码隔离

- 基础分支：`codex/experiment-b-single-decoder-directional-pyramid`
- 实际基础提交：`435ab1827ecee4c6b83b669789bb9833a5fd5320`
- Experiment D 分支：`codex/experiment-d-sd-awgm-decoder-hfe`
- 新模型：`model/DWTFreqNet_SingleDecoder_HFE.py`
- 新训练入口：`train_experiment_d.py`
- 未修改 `model/DWTFreqNet.py`、`model/DWTFreqNet_WULLE.py`、
  `model/DWTFreqNet_SingleDecoder.py` 和
  `model/DWTFreqNet_SingleDecoder_LDRC.py`。

本轮只实现 `sd_awgm_hfe`，不加入 LDRC、Directional Pyramid、第二次 DWT、
W8M、Mamba、LFSSBlock、新损失或新数据增强。

## 3. Decoder HFE 结构

每级先将原始 DWT 高频通过 1×1 卷积对齐到 IDWT 所需通道，再执行：

```text
aligned H/V/D
  → channel-SKFF
  → shared HF
  → Channel-Matched Attention + Channel-Matched FFN
  → refined shared HF
  → 三个独立方向残差头
  → base + beta × delta
  → IDWT
```

硬匹配矩阵为 `[B,C,C]`：每个高频通道在当前低频特征中选择 L2 距离最小的
一个低频通道。`torch.cdist` 使用 float32，索引不参与梯度，选出的低频特征保留
梯度。H/V/D 三个 beta 均初始化为 `1e-3`。

| Stage | 低频引导 | H/V/D 对齐通道 | 空间尺寸 | Heads |
|---|---|---:|---:|---:|
| 4 | E4 | 256 | 16×16 | 4 |
| 3 | L3 | 256 | 32×32 | 4 |
| 2 | L2 | 128 | 64×64 | 2 |
| 1 | L1 | 64 | 128×128 | 1 |

固定结构：DWT=4、IDWT=4、第二次 DWT=0、单 decoder=True、Pyramid=False、
LDRC=False、Mamba=False。

## 4. 统一训练配置

| 项目 | 设置 |
|---|---|
| seed | 42 |
| 输入/patch | 256×256 |
| batch size | 4 |
| epochs | 1000 |
| optimizer | Adam |
| 初始学习率 | 1e-3 |
| scheduler | CosineAnnealingLR，eta_min=1e-5 |
| 评估 | epoch 100 开始，每 epoch 一次 |
| checkpoint | 每20 epoch；best/latest |
| 阈值 | 0.5 |

正式实验从随机初始化开始，不加载任何 `sd_awgm`、LDRC、WULLE 或其他权重。
数据集顺序为 NUDT-SIRST、NUAA-SIRST、IRSTD-1K；调度器仅在显存不超过
1024 MiB 且利用率不超过5%时启动，不终止 Experiment A/B/C 或 W8M 任务。

## 5. 测试与复杂度

已完成：

- `tools/test_sd_awgm_hfe_experiment_d.py --full`
- `tools/check_haar_direction_mapping.py --require-aligned-routing`
- `tools/profile_sd_awgm_hfe_experiment_d.py`

| 模型 | 参数量 | FLOPs | 延迟 | FPS | 推理峰值显存 | 训练峰值显存 | DWT/IDWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 37.435M | 66.87G | 33.06ms | 30.24 | 618.35MiB | 2500.06MiB | 12/15 |
| WULLE-A | 35.399M | 54.69G | 31.93ms | 31.31 | 549.26MiB | 2377.33MiB | 11/12 |
| `sd_awgm` | 5.926M | 14.38G | 8.28ms | 120.78 | 194.36MiB | 443.59MiB | 4/4 |
| `sd_awgm_hfe` | 10.181M | 20.47G | 28.17ms | 35.50 | 268.79MiB | 970.67MiB | 4/4 |

`sd_awgm_hfe` 中 HFE 参数量为 4,254,991。规范要求的 CUDA 完整测试使用
`2×1×256×256` 输入，训练模式得到6个同尺寸输出，测试模式得到单一输出；所有
E1–E4、HFE1–4、L3–L0 形状均符合规范。四级 Attention/FFN 的 Matching 矩阵
分别为 `[2,64,64]`、`[2,128,128]`、`[2,256,256]`、`[2,256,256]`，索引分别为
`[2,64]`、`[2,128]`、`[2,256]`、`[2,256]`。

复制 `sd_awgm` 共同参数并将12个 beta 临时置零后，两模型输出最大绝对误差为0。
Stem、四级 encoder/方向编码/AWGM、全部对齐层、四级 HFE 的 SKFF/Attention/FFN/
三个方向头/beta、decoder fuse 和输出头均获得非零梯度。Haar 检查结果为
H/LH→vertical、V/HL→horizontal、`routing_aligned=true`。真实 NUDT 数据 batch=4、
256×256 的单步前向/反向通过，loss 为6.65097，未发生 OOM。

THOP 不能准确统计 `torch.cdist` 和代码中直接完成的注意力矩阵乘法，因此 FLOPs
仅作为统一工具口径的近似值，另行报告实测延迟和显存。

## 6. 正式实验结果

| Dataset | Model | 状态 | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | FLOPs | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `sd_awgm` | 已有结果 | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 | 5.926M | 14.38G | 5.15ms |
| NUAA-SIRST | `sd_awgm_hfe` | GPU1训练中（当前 epoch 981） | 286 | 0.7747 | 0.7809 | 0.8731 | 0.9695 | 2.394e-5 | 10.181M | 20.47G | 28.17ms |
| NUDT-SIRST | `sd_awgm` | 已有结果 | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 | 5.926M | 14.38G | 5.15ms |
| NUDT-SIRST | `sd_awgm_hfe` | GPU0训练中（当前 epoch 365） | 318 | 0.9429 | 0.9455 | 0.9706 | 0.9947 | 2.045e-6 | 10.181M | 20.47G | 28.17ms |
| IRSTD-1K | `sd_awgm` | 已有结果 | 894 | 0.6561 | 0.6477 | 0.7924 | 0.9091 | 1.537e-5 | 5.926M | 14.38G | 5.15ms |
| IRSTD-1K | `sd_awgm_hfe` | GPU3训练中（当前 epoch 406） | 311 | 0.6488 | 0.6261 | 0.7870 | 0.9226 | 3.522e-5 | 10.181M | 20.47G | 28.17ms |

## 7. 输出目录

```text
runs/experiment_d/NUDT-SIRST/sd_awgm_hfe/seed42
runs/experiment_d/NUAA-SIRST/sd_awgm_hfe/seed42
runs/experiment_d/IRSTD-1K/sd_awgm_hfe/seed42
```

每次评估除五项性能指标外，还记录各级 SKFF 权重、通道匹配复用率、共享/增强
高频范数、beta、delta/base、beta×delta/base 和 final/base 比率。

## 8. 服务器启动记录

- 服务器项目目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_D`
- 启动时间：2026-07-14 18:06:10 CST
- 动态调度器 PID：374225
- NUDT-SIRST：GPU0，wrapper PID 374236，Python PID 374246；启动确认 epoch 9
- NUAA-SIRST：GPU1，wrapper PID 374315，Python PID 374325；启动确认 epoch 30
- IRSTD-1K：GPU3，wrapper PID 374397，Python PID 374403；启动确认 epoch 6
- 单任务训练显存约4.68GB；GPU2、GPU4 上的 Experiment C 未被终止或覆盖。

启动命令：

```bash
PROJECT_ROOT=/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_D \
DATASET_DIR=/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets \
PYTHON_BIN=/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python \
GPU_LIST=0,1,2,3,4,5,6 POLL_SECONDS=60 \
MAX_USED_MIB=1024 MAX_UTIL_PERCENT=5 SEED=42 \
bash scripts/launch_experiment_d_queue.sh
```

三个任务均使用 batch size 4、1000 epoch、epoch 100 开始每 epoch 评估一次。

### 8.1 阶段性结果快照（2026-07-14 22:18:22 CST）

三项任务均已进入正式评估阶段；下表记录的是当前 `best_metrics.json` 中的最佳
checkpoint，同时注明日志中的最新训练 epoch。由于三项任务尚未完成 1000 epoch，
这些是阶段性结果，不应视为最终结果。

| 数据集 | GPU | 最新 epoch | 当前最佳 epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUDT-SIRST | 0 | 365 | 318 | 0.9429 | 0.9455 | 0.9706 | 0.9947 | 2.045e-6 |
| NUAA-SIRST | 1 | 981 | 286 | 0.7747 | 0.7809 | 0.8731 | 0.9695 | 2.394e-5 |
| IRSTD-1K | 3 | 406 | 311 | 0.6488 | 0.6261 | 0.7870 | 0.9226 | 3.522e-5 |
