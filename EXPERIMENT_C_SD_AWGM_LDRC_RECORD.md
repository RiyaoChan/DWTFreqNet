# Experiment C：SD-AWGM + Encoder-side LDRC 实验记录

## 1. 实验目标

本实验以 Experiment B 的 `sd_awgm` 为唯一基础，只在四级 AWGM 编码特征
`E1/E2/E3/E4` 与单一小波解码器之间增加 Encoder-side LDRC，验证原始模型与
`sd_awgm` 的性能差距是否主要来自同尺度自注意力和跨尺度关系建模。

本轮唯一新方案为 `sd_awgm_ldrc`，不加入 Directional Pyramid、第二次 DWT、
后置 AWGM、Dense Global Encoder、Mamba-LDRC、新位置编码或新损失。

## 2. 代码隔离

- 基础分支：`codex/experiment-b-single-decoder-directional-pyramid`
- 基础提交：`435ab1827ecee4c6b83b669789bb9833a5fd5320`
- Experiment C 分支：`codex/experiment-c-sd-awgm-encoder-ldrc`
- 新模型：`model/DWTFreqNet_SingleDecoder_LDRC.py`
- 新训练入口：`train_experiment_c.py`
- 原 `model/DWTFreqNet.py`、`model/DWTFreqNet_WULLE.py`、
  `model/DWTFreqNet_SingleDecoder.py` 和 `train_experiment_b.py` 均不修改。

## 3. 结构改动

`DWTFreqNet_SingleDecoder_LDRC` 继承 Experiment B 的单解码器模型，并把父类固定为
`sd_variant="sd_awgm"`。四级 DWT、DirectionalBandEncoder、Stage-wise AWGM、
原始 H/V/D 系数对齐和 Single Wavelet Decoder 与 `sd_awgm` 保持一致。

唯一新增的 `EncoderLDRC` 包含：

1. 将 E1–E4 投影到统一的 128 维 token；
2. 按 E4→E3→E2→E1 顺序执行一层原始 `TransFuseModel`；
3. 每层保持 SAM→CAM→FFL 及其残差、LayerNorm、4头注意力和 dropout 0.2；
4. 通过 1×1 回投影和初始化为 1e-3 的 gamma 残差注入原特征；
5. 高频 IDWT 系数仍使用通道对齐后的原始 H/V/D，不引入 Pyramid 残差。

固定结构约束：DWT=4、IDWT=4、第二次 DWT=0、Pyramid=0、Mamba=0。

## 4. 统一训练设置

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

数据集优先级为 NUDT-SIRST、NUAA-SIRST、IRSTD-1K。调度器只使用显存不超过
1024 MiB 且利用率不超过5%的 GPU，不终止或覆盖 Experiment A、B、W8M 任务。

## 5. 测试与复杂度

已完成：

- `tools/test_sd_awgm_ldrc_experiment_c.py --full`
- `tools/check_haar_direction_mapping.py --require-aligned-routing`
- `tools/profile_sd_awgm_ldrc_experiment_c.py`

本地 CPU 快速测试、226服务器 CUDA 快速测试和规范要求的 `2×256×256` 完整测试
均通过。gamma1–4 临时置零后，新模型与 `sd_awgm` 的输出最大绝对误差为0；
所有指定编码器、AWGM、LDRC、系数对齐、decoder 和输出 head 均获得非零梯度；
H/LH→vertical、V/HL→horizontal 且 `routing_aligned=true`。真实 NUDT 数据的
batch size 4、256×256 单步前向/反向也通过。

| 模型 | 参数量 | FLOPs | 延迟 | FPS | 推理峰值显存 | 训练峰值显存 | DWT/IDWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 37.435M | 66.87G | 34.83 ms | 28.71 | 618.35 MiB | 2500.06 MiB | 12/15 |
| WULLE-A | 35.399M | 54.69G | 32.75 ms | 30.53 | 549.26 MiB | 2377.33 MiB | 11/12 |
| `sd_awgm` | 5.926M | 14.38G | 8.03 ms | 124.57 | 194.80 MiB | 443.09 MiB | 4/4 |
| `sd_awgm_ldrc` | 13.293M | 19.24G | 23.97 ms | 41.72 | 392.31 MiB | 2093.28 MiB | 4/4 |

`sd_awgm_ldrc` 中 LDRC 参数量为 7,367,552；相对 `sd_awgm` 增加 7,367,552
参数和约 4.86G THOP FLOPs。THOP 不统计代码中直接使用 `@` 完成的注意力矩阵乘法，
因此该 FLOPs 是同工具口径下的近似值。四级 SAM/CAM 在 batch 1、4头下合计
358,875,136 个注意力元素，对应概念上的 1369 MiB FP32 矩阵；实现按 head 顺序
执行，实测推理峰值显存为392.31 MiB。

## 6. 正式实验结果

表中 `sd_awgm` 为现有 Experiment B 对照结果；Experiment C 结果训练后回填。

| Dataset | Model | 状态 | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `sd_awgm` | 已有结果 | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 |
| NUAA-SIRST | `sd_awgm_ldrc` | 226 GPU3训练中 | — | — | — | — | — | — |
| NUDT-SIRST | `sd_awgm` | 已有结果 | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 |
| NUDT-SIRST | `sd_awgm_ldrc` | 226 GPU2训练中 | — | — | — | — | — | — |
| IRSTD-1K | `sd_awgm` | 226服务器训练中 | — | — | — | — | — | — |
| IRSTD-1K | `sd_awgm_ldrc` | 226 GPU4训练中 | — | — | — | — | — | — |

## 7. 输出目录

```text
runs/experiment_c/NUDT-SIRST/sd_awgm_ldrc/seed42
runs/experiment_c/NUAA-SIRST/sd_awgm_ldrc/seed42
runs/experiment_c/IRSTD-1K/sd_awgm_ldrc/seed42
```

除五项性能指标外，还记录 gamma1–4 mean、E1e/E1 至 E4e/E4 norm ratio、
SAM/CAM/FFL output norm、DWT/IDWT 调用次数和 LDRC 参数量。

## 8. 226服务器启动记录

- 项目目录：`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_C`
- 启动时间：2026-07-13 23:30 CST
- 调度器 PID：4030672
- NUDT-SIRST：GPU2，Python PID 4030699
- NUAA-SIRST：GPU3，Python PID 4030784
- IRSTD-1K：GPU4，Python PID 4030865
- 单任务显存占用约9.44GB；没有终止或覆盖 GPU0/1 上的 Experiment B 或 GPU5
  上的 WULLE-A/W8M 实验。
