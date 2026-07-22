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
| K2 | NUAA-SIRST | 3 | 952684 | 952691 | 运行，epoch 712/1000 |
| K2 | IRSTD-1K | 4 | 952888 | 952895 | 运行，epoch 239/1000 |

`2026-07-23 00:40:55 CST`快照：NUAA当前epoch 117；当前mIoU/nIoU/F1/Pd/Fa为`0.695771/0.734625/0.820596/0.931298/3.82793e-05`。当前best位于epoch 103，mIoU/nIoU/F1/Pd/Fa为`0.712572/0.732147/0.832166/0.946565/3.36830e-05`。IRSTD当前epoch 33，因评估从epoch 100开始，暂时没有正式指标或best checkpoint。

`2026-07-23 07:42:37 CST`快照：

- NUAA当前epoch 712，当前mIoU/nIoU/F1/Pd/Fa为`0.781852/0.792023/0.877572/0.973282/1.84536e-05`；best位于epoch 607，为`0.801101/0.795325/0.889568/0.965649/2.54509e-05`。
- IRSTD当前epoch 239，当前mIoU/nIoU/F1/Pd/Fa为`0.610102/0.593804/0.757842/0.841751/8.17977e-06`；best位于epoch 158，为`0.649640/0.603540/0.787614/0.892256/2.67978e-05`。

K3–K6没有启动；NUDT任务也没有提前启动。它们分别受K-A Go和NUAA/IRSTD同一全局配置筛选约束。正式指标从epoch 100开始产生，后续记录当前epoch、best epoch及mIoU/nIoU/F1/Pd/Fa。
