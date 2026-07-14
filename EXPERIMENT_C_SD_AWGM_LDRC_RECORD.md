# Experiment C：SD-AWGM + Encoder-side LDRC 实验记录

## 实验定位与代码隔离

Experiment C 从 Experiment B 的 `sd_awgm` 单解码器基线开始，只在四级 AWGM
编码特征 E1–E4 与单一小波解码器之间加入 Encoder-side LDRC。未加入 Pyramid、
第二次 DWT、后置 AWGM、Mamba 或新损失。

- 基础提交：`435ab1827ecee4c6b83b669789bb9833a5fd5320`
- 实验分支：`codex/experiment-c-sd-awgm-encoder-ldrc`
- 模型：`model/DWTFreqNet_SingleDecoder_LDRC.py`
- 训练入口：`train_experiment_c.py`
- 输出目录：`runs/experiment_c/<dataset>/sd_awgm_ldrc/seed42`

Encoder LDRC 将 E1–E4 投影到统一 token 维度，按 E4→E3→E2→E1 执行同尺度/跨
尺度关系建模，并用初始化为 `1e-3` 的 gamma 残差注入原始编码特征。固定
DWT/IDWT=4/4，输入 patch 为 256×256，batch size=4，seed=42，训练1000 epoch，
从 epoch100 开始每个 epoch 评估一次。

## 测试与复杂度

CPU、226 CUDA、`2×1×256×256` 完整结构/梯度测试、Haar 方向检查和真实 NUDT
batch=4 单步训练均通过；gamma 置零时与 `sd_awgm` 输出最大绝对误差为0。

| 模型 | 参数量 | FLOPs | 延迟 | FPS | 推理峰值显存 | 训练峰值显存 | DWT/IDWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 37.435M | 66.87G | 34.83ms | 28.71 | 618.35MiB | 2500.06MiB | 12/15 |
| WULLE-A | 35.399M | 54.69G | 32.75ms | 30.53 | 549.26MiB | 2377.33MiB | 11/12 |
| `sd_awgm` | 5.926M | 14.38G | 8.03ms | 124.57 | 194.80MiB | 443.09MiB | 4/4 |
| `sd_awgm_ldrc` | 13.293M | 19.24G | 23.97ms | 41.72 | 392.31MiB | 2093.28MiB | 4/4 |

## 正式实验结果

以下为 226 服务器最新已记录的最佳 checkpoint 指标：

| 数据集 | 方案 | 状态 | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | `sd_awgm` | 已完成 | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 |
| NUAA-SIRST | `sd_awgm_ldrc` | 已完成/1000 | 565 | 0.7755 | 0.7914 | 0.8736 | 0.9618 | 2.401e-5 |
| NUDT-SIRST | `sd_awgm` | 已完成 | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 |
| NUDT-SIRST | `sd_awgm_ldrc` | 已记录阶段结果 | 382 | **0.9564** | **0.9564** | **0.9777** | **0.9958** | **2.091e-6** |
| IRSTD-1K | `sd_awgm` | 已完成 | 894 | 0.6561 | 0.6477 | 0.7924 | 0.9091 | 1.537e-5 |
| IRSTD-1K | `sd_awgm_ldrc` | 已记录阶段结果 | 382 | 0.6508 | 0.6512 | 0.7885 | 0.9461 | 1.585e-5 |

NUDT 的 LDRC 方案明显高于 `sd_awgm`；NUAA 和 IRSTD 的 LDRC 结果低于对应
基线，最终结论以当时保存的最佳 checkpoint 为准。226 启动目录为
`/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_C`。
