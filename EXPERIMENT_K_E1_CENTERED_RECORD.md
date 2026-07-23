# Experiment K v2：以 E1 为中心的剂量校准净化实验记录

> 状态：核心实现与启动前验证已完成。K2–NUAA已完成1000 epoch；K2–IRSTD和K2–NUDT分别运行到epoch 613和240。NUAA/IRSTD的K-A A1/A2/A6及修正版A5/A7已完成，修正版A3/A4仍在运行。NUDT第一轮K-A的A1和A5已完成，A3/A4正在GPU 5运行，A6/A7等待空闲GPU；J2/J3第二轮等待NUDT的J3-F/J3-R完成。三数据集A5均未达到预注册处理效应门槛，K3–K6继续禁用；该结论是当前证据下的强化No-Go，但在A3/A4、A6/A7全部完成前不伪装成最终锁定结论。所有正式训练均为1000 epoch，epoch 100起每epoch评估。

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
| A3/A4 stage/rho反事实与Gate强度 | 旧NUAA失败、旧IRSTD已停止并归档；修正版等待A5/A7后自动重跑 | `decoder_low`改为当前IDWT之前已存在的`E4/L3/L2/L1` |
| A5 Gaussian处理效应 | 修正版在GPU 0/5重跑 | 严格J1回归仅在无空间override时启用；局部Gaussian on/off已通过真实checkpoint smoke |
| A6 先验反馈漂移 | NUAA、IRSTD train均已完成 | gap/AUC/分布漂移、逐样本D_feat与stop-gradient审计 |
| A7 MAD/Gaussian扩散 | 旧描述统计已归档；修正版等待A5完成后自动重算 | global/per-channel/local MAD与三层ring扩散，并关联修正后的A5处理效应 |

发现集只使用train；锁定决策后才运行test confirmation。NUDT的部分J2/J3 checkpoint尚未完成时，不会伪造或提前锁定三数据集K-A结论。

服务器端发现集队列覆盖A1–A7；A6依赖同数据集A1/A2完成，A7依赖同数据集A5完成。调度按诊断类型优先跨NUAA/IRSTD配对，再进入下一类型。它与正式训练使用相同的空闲判定：显存`<1000 MiB`、利用率`<10%`且无compute PID，不抢占现有Experiment J/K任务。

### 5.1 已完成的train discovery结果

A1/A2已覆盖NUAA的213张train图像和IRSTD的800张train图像。A1固定同一个checkpoint、source和stage，比较Dense `C_square`/`C_GR`与Phase 1实例级`C_P2`对候选点排序的Spearman相关；A2再比较同一个checkpoint和stage下四种source的变化。Spearman越接近1表示与`C_P2`的排序越一致，接近0表示不一致，负值表示排序方向局部反转；它是算子忠实度而非最终检测性能。

明细来自服务器正式输出`analysis/experiment_k/compactness_fidelity/<dataset>/train/operator_fidelity.csv`；逐候选原始值位于同目录`compactness_instances.csv`，target/hard-negative统计位于`operator_statistics.csv`。

Stage和source在本诊断中的精确定义如下：

| 名称 | 精确定义 |
|---|---|
| Stage 1/2/3/4 | 编码器第1/2/3/4次DWT后的尺度，即相对pad后输入为`H/2`、`H/4`、`H/8`、`H/16`；对256×256输入分别为128、64、32、16 |
| Raw LL | 当前stage的原始DWT低频`A[stage]` |
| LFSS LL | Raw LL经LFSS残差块后的`A_lfss[stage]` |
| Guided LL | LFSS LL再经AWGM高频引导后的`A_guided[stage]` |
| Decoder low | Stage 1/2/3分别为`L1/L2/L3`，Stage 4为瓶颈`E4` |

#### 5.1.1 A1/A2使用的具体checkpoint

所有checkpoint都是对应任务的`best.pth.tar`，不是`last`或任意中间快照。表内路径是服务器绝对路径。

| 数据集 | Checkpoint | Best epoch | 服务器checkpoint |
|---|---|---:|---|
| NUAA | E1 | 595 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS/runs/experiment_e_lfss_awgm/E1_lfss_resblock/NUAA-SIRST/seed42/best.pth.tar` |
| NUAA | J1 | 572 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J1_bandwise_noise_calibrated/NUAA-SIRST/seed42/best.pth.tar` |
| NUAA | J2-R | 658 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J2_rawll_compactness/NUAA-SIRST/seed42/best.pth.tar` |
| NUAA | J2-D | 738 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J2_decoder_compactness/NUAA-SIRST/seed42/best.pth.tar` |
| NUAA | J3-F | 568 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J3_dual_evidence_fixed/NUAA-SIRST/seed42/best.pth.tar` |
| NUAA | J3-R | 298 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J3_dual_evidence_reliability/NUAA-SIRST/seed42/best.pth.tar` |
| IRSTD | E1 | 551 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS/runs/experiment_e_lfss_awgm/E1_lfss_resblock/IRSTD-1K/seed42/best.pth.tar` |
| IRSTD | J1 | 441 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J1_bandwise_noise_calibrated/IRSTD-1K/seed42/best.pth.tar` |
| IRSTD | J2-R | 668 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J2_rawll_compactness/IRSTD-1K/seed42/best.pth.tar` |
| IRSTD | J2-D | 635 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J2_decoder_compactness/IRSTD-1K/seed42/best.pth.tar` |
| IRSTD | J3-F | 613 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J3_dual_evidence_fixed/IRSTD-1K/seed42/best.pth.tar` |
| IRSTD | J3-R | 878 | `/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_J_DENP/runs/experiment_j_denp/J3_dual_evidence_reliability/IRSTD-1K/seed42/best.pth.tar` |

#### 5.1.2 NUAA逐checkpoint、逐stage结果

每个单元格为`C_square / C_GR`相对`C_P2`的Spearman；每个组合包含810个候选点（270个target、540个hard negative）。

| Checkpoint | Stage | Raw Sq/GR | LFSS Sq/GR | Guided Sq/GR | Decoder Sq/GR |
|---|---:|---:|---:|---:|---:|
| E1 | 1 | 0.6643 / 0.3605 | 0.6829 / 0.3783 | 0.6804 / 0.3774 | 0.6390 / 0.2969 |
| E1 | 2 | 0.6010 / 0.5868 | 0.6145 / 0.5947 | 0.6284 / 0.5979 | 0.4737 / 0.5892 |
| E1 | 3 | 0.2912 / 0.6965 | 0.4417 / 0.6850 | 0.4521 / 0.6859 | 0.1734 / 0.6484 |
| E1 | 4 | 0.5475 / 0.9305 | 0.3105 / 0.8902 | 0.2784 / 0.8885 | 0.1685 / 0.8810 |
| J1 | 1 | 0.6290 / 0.3433 | 0.6841 / 0.4052 | 0.6818 / 0.4008 | 0.6150 / 0.3493 |
| J1 | 2 | 0.6070 / 0.5804 | 0.6046 / 0.5715 | 0.6219 / 0.5789 | 0.4355 / 0.5998 |
| J1 | 3 | 0.3472 / 0.6740 | 0.4717 / 0.6760 | 0.4773 / 0.6785 | 0.2159 / 0.6753 |
| J1 | 4 | 0.5312 / 0.9330 | 0.3431 / 0.9082 | 0.2617 / 0.9046 | 0.1045 / 0.8957 |
| J2-R | 1 | 0.6557 / 0.3789 | 0.6766 / 0.4288 | 0.6745 / 0.4294 | 0.6361 / 0.4324 |
| J2-R | 2 | 0.5968 / 0.6042 | 0.6051 / 0.6045 | 0.6214 / 0.6069 | 0.4550 / 0.6069 |
| J2-R | 3 | 0.3323 / 0.6794 | 0.4858 / 0.6990 | 0.4786 / 0.6984 | 0.2123 / 0.6573 |
| J2-R | 4 | 0.5255 / 0.9220 | 0.2177 / 0.8740 | 0.1563 / 0.8744 | 0.0987 / 0.8929 |
| J2-D | 1 | 0.6590 / 0.3490 | 0.6623 / 0.3583 | 0.6580 / 0.3670 | 0.6608 / 0.3612 |
| J2-D | 2 | 0.5701 / 0.5347 | 0.6160 / 0.5780 | 0.6297 / 0.5901 | 0.3340 / 0.5361 |
| J2-D | 3 | 0.3457 / 0.6856 | 0.4783 / 0.6741 | 0.4830 / 0.6826 | 0.1901 / 0.6595 |
| J2-D | 4 | 0.4672 / 0.9202 | 0.2261 / 0.8625 | 0.1789 / 0.8672 | 0.0699 / 0.8726 |
| J3-F | 1 | 0.6609 / 0.3830 | 0.6798 / 0.3863 | 0.6748 / 0.3863 | 0.6622 / 0.4602 |
| J3-F | 2 | 0.5975 / 0.5514 | 0.6077 / 0.5479 | 0.6230 / 0.5560 | 0.4496 / 0.5828 |
| J3-F | 3 | 0.3035 / 0.6741 | 0.4159 / 0.6649 | 0.4181 / 0.6683 | 0.1827 / 0.6636 |
| J3-F | 4 | 0.4299 / 0.9129 | 0.3614 / 0.8931 | 0.3007 / 0.8904 | 0.1487 / 0.8924 |
| J3-R | 1 | 0.6673 / 0.3598 | 0.6878 / 0.3981 | 0.6715 / 0.3998 | 0.5424 / 0.3441 |
| J3-R | 2 | 0.5397 / 0.5195 | 0.5580 / 0.5363 | 0.5711 / 0.5475 | 0.4484 / 0.5752 |
| J3-R | 3 | 0.2780 / 0.6793 | 0.4185 / 0.6801 | 0.4193 / 0.6839 | 0.2070 / 0.6732 |
| J3-R | 4 | 0.3710 / 0.9160 | 0.2478 / 0.8861 | 0.2257 / 0.8912 | 0.1266 / 0.8901 |

#### 5.1.3 IRSTD逐checkpoint、逐stage结果

每个单元格仍为`C_square / C_GR`；每个组合包含3,574个候选点（1,195个target、2,379个hard negative）。

| Checkpoint | Stage | Raw Sq/GR | LFSS Sq/GR | Guided Sq/GR | Decoder Sq/GR |
|---|---:|---:|---:|---:|---:|
| E1 | 1 | 0.7041 / 0.1598 | 0.6759 / 0.1625 | 0.6315 / 0.1414 | 0.1832 / 0.2763 |
| E1 | 2 | 0.6421 / 0.4110 | 0.6128 / 0.3575 | 0.6083 / 0.3478 | 0.6380 / 0.6799 |
| E1 | 3 | 0.4618 / 0.5160 | 0.4671 / 0.5358 | 0.4728 / 0.5376 | 0.4288 / 0.7571 |
| E1 | 4 | 0.6354 / 0.8675 | 0.3052 / 0.8285 | -0.0481 / 0.8275 | -0.1726 / 0.8689 |
| J1 | 1 | 0.7219 / 0.2645 | 0.6723 / 0.1480 | 0.6424 / 0.1126 | 0.2365 / 0.0407 |
| J1 | 2 | 0.6850 / 0.5040 | 0.6442 / 0.4059 | 0.6368 / 0.3987 | 0.2828 / 0.6491 |
| J1 | 3 | 0.4855 / 0.5540 | 0.4748 / 0.5734 | 0.4797 / 0.5670 | -0.0013 / 0.6887 |
| J1 | 4 | 0.6405 / 0.8851 | 0.2956 / 0.8628 | 0.0782 / 0.8721 | 0.0888 / 0.8858 |
| J2-R | 1 | 0.7192 / 0.2718 | 0.6400 / 0.1795 | 0.6052 / 0.1548 | 0.4269 / 0.0474 |
| J2-R | 2 | 0.6207 / 0.3982 | 0.5924 / 0.4128 | 0.5774 / 0.3797 | 0.7437 / 0.7320 |
| J2-R | 3 | 0.4437 / 0.5344 | 0.4318 / 0.5337 | 0.4483 / 0.5305 | 0.3921 / 0.7546 |
| J2-R | 4 | 0.5379 / 0.8569 | 0.0476 / 0.8081 | -0.0427 / 0.8565 | -0.1086 / 0.8936 |
| J2-D | 1 | 0.7266 / 0.2659 | 0.6828 / 0.1630 | 0.6610 / 0.1928 | 0.4386 / 0.0865 |
| J2-D | 2 | 0.6399 / 0.4275 | 0.6205 / 0.4459 | 0.6223 / 0.4484 | 0.5681 / 0.7148 |
| J2-D | 3 | 0.4428 / 0.5208 | 0.4203 / 0.5221 | 0.4316 / 0.5278 | 0.4455 / 0.7621 |
| J2-D | 4 | 0.6072 / 0.8752 | 0.0648 / 0.8253 | -0.0667 / 0.8744 | -0.0893 / 0.8740 |
| J3-F | 1 | 0.7234 / 0.2766 | 0.6769 / 0.1838 | 0.6415 / 0.1521 | 0.1045 / 0.0805 |
| J3-F | 2 | 0.6349 / 0.3963 | 0.6082 / 0.4257 | 0.6147 / 0.4239 | 0.5163 / 0.6783 |
| J3-F | 3 | 0.4340 / 0.5145 | 0.3624 / 0.5406 | 0.4186 / 0.5538 | 0.3710 / 0.7777 |
| J3-F | 4 | 0.6252 / 0.8711 | 0.1567 / 0.8374 | -0.0332 / 0.8602 | -0.1214 / 0.8944 |
| J3-R | 1 | 0.7018 / 0.2137 | 0.6632 / 0.1341 | 0.6446 / 0.0973 | 0.0445 / -0.0964 |
| J3-R | 2 | 0.6369 / 0.3978 | 0.5797 / 0.3770 | 0.5828 / 0.3769 | 0.6628 / 0.7107 |
| J3-R | 3 | 0.4143 / 0.5064 | 0.3510 / 0.4760 | 0.3933 / 0.4972 | 0.3664 / 0.7825 |
| J3-R | 4 | 0.6743 / 0.8760 | 0.3392 / 0.8563 | 0.0121 / 0.8522 | -0.1599 / 0.9108 |

#### 5.1.4 A1/A2汇总与解读

下表只用于快速总览，是对上面明细跨6个checkpoint和4个stage再取中位数：

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

明细揭示了汇总表隐藏的stage效应：NUAA的Stage 1通常更接近`C_square`，Stage 3/4明显转向`C_GR`；IRSTD的Raw/LFSS/Guided在Stage 1/2通常更接近`C_square`，Stage 4则明显转向`C_GR`，而Decoder low从Stage 2开始多数更接近`C_GR`。因此当前没有可跨数据集、跨stage直接复用的单一算子赢家。目标与困难背景区分上，IRSTD的`C_square`中位ROC-AUC较高（Raw/LFSS/Guided分别约`0.9244/0.9125/0.8685`）；NUAA最明显的是Decoder low的`C_square≈0.7440`。这些仍是描述/忠实度结果，不能替代A5处理效应判定。

A6每个数据集完成240组gap/AUC/分布比较，并记录NUAA `17,040`条、IRSTD `64,000`条逐样本`D_feat`。以下数值均来自正式`prior_distribution_drift.csv`、`feature_drift_summary.csv`和梯度审计，而不是文字快照。

#### 5.1.5 A6 prior分布漂移

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

#### 5.1.6 A6 特征漂移 `D_feat`

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

#### 5.1.7 A6 梯度审计

| 路径 | Low source梯度 | 输出是否可求导 | 解释 |
|---|---:|---|---|
| Square正常路径 | L2 norm约`0.00305` | 是 | Compactness会向low source及自身参数回传 |
| Square使用`low_source.detach()` | `None` | 是 | 输出梯度仅来自Square自身斜率/阈值参数，不回到low source |
| Gaussian-radial正式路径 | `None` | 否 | `C_GR`无可学习参数且source已detach |
| Experiment K正式实现 | `None` | 不适用 | 所有Compactness source统一stop-gradient |

A6因此支持K正式实现始终对Compactness source使用stop-gradient；原摘要中的`unexpected_grad_path`实际是把Square自身参数梯度误当作低频source梯度，已在审计代码中纠正。

#### 5.1.8 A5处理效应与A7 MAD/Gaussian扩散

A5首次尝试因J1-as-K没有开启`debug_tensors`失败，失败输出保存在`attempts/attempt_001_debug_tensors_disabled`，修复commit为`4b988e6`。第二次运行虽然完整写出CSV，但审计发现所有Gaussian on/off反事实差值均严格为0：

| 数据集 | J1 checkpoint | 图像数 | A5逐项记录 | 有效`C_GR`比较 | 结果 |
|---|---:|---:|---:|---:|---|
| NUAA | best epoch 572 | 213 | 38,880 | 0 | 无效，全部`delta_loss/delta_iou/delta_probability/delta_false_positive=0` |
| IRSTD | best epoch 441 | 800 | 171,552 | 0 | 无效，全部`delta_loss/delta_iou/delta_probability/delta_false_positive=0` |

这不是“Gaussian没有影响”的实验结论，而是反事实实现没有真正关闭局部Gaussian：`alpha_override=1`且`rho_override=0`时进入严格J1回归分支，该分支直接使用全局`noise`重建输出，覆盖了`spatial_dose_override`产生的局部dose。因此A5必须修复后重跑，当前`dataset_support=false`只表示没有有效处理效应证据。

A7本身的MAD与Gaussian空间描述统计有效。下表是逐候选、stage和H/V/D band记录的中位数；`N`表示噪声置信度，energy是每个ring内的平均绝对Gaussian残差：

| 数据集 | Stage | Aligned global MAD | Local-7 MAD | `N_aligned_global` | `N_local7` | Center energy | Ring-3 energy | Sign-flip rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | 1 | 0.03955 | 0.07361 | 0.1857 | 0.3344 | 0.05417 | 0.06751 | 0.1468 |
| NUAA | 2 | 0.14213 | 0.26552 | 0.4138 | 0.5422 | 0.27821 | 0.33578 | 0.2778 |
| NUAA | 3 | 0.28289 | 0.47214 | 0.4400 | 0.5514 | 0.61417 | 0.57487 | 0.3204 |
| NUAA | 4 | 0.42959 | 0.64689 | 0.4770 | 0.4878 | 0.92482 | 0.61439 | 0.3183 |
| IRSTD | 1 | 0.12961 | 0.57981 | 0.0691 | 0.4711 | 0.49203 | 0.27082 | 0.1176 |
| IRSTD | 2 | 0.75481 | 1.51021 | 0.2010 | 0.5174 | 2.56218 | 1.08190 | 0.3169 |
| IRSTD | 3 | 0.88409 | 1.64415 | 0.2556 | 0.4822 | 2.70157 | 1.13474 | 0.3457 |
| IRSTD | 4 | 1.06587 | 1.76399 | 0.5106 | 0.5261 | 2.32869 | 1.24363 | 0.3314 |

NUAA与IRSTD分别完成9,720和42,888条A7记录。局部MAD普遍高于全局aligned MAD，说明全局MAD会低估目标邻域的局部变化；IRSTD的差异尤其明显。由于A5的`delta_loss`是常数0，五种MAD/noise指标与处理效应的Spearman均为NaN，不能用A7判定Gaussian是否有利。`K_A_DECISION.json`继续保持No-Go。

A3/A4的NUAA旧任务运行约7小时后，在遍历到`decoder_low` source时失败：旧实现从尚未写入的source缓存读取，Stage 4首先触发`NoneType.detach()`。IRSTD旧任务使用相同实现，已在用户要求修正后安全停止；两者均未生成正式CSV。

#### 5.1.9 诊断修正、验证与重跑

2026-07-23 11:18完成以下修正：

1. `model/decoder_k_dose.py`：严格J1等价快捷路径新增`spatial_dose_override is None`条件。正常回归仍保持原J1算术顺序；局部反事实存在时改用实际dose，不再覆盖局部on/off干预。
2. `model/DWTFreqNet_SingleDecoder_LFSS_AWGM_K.py`：`_purify_stage_high`显式接收当前decoder low。Stage 4使用`E4`，Stage 3/2/1分别使用已经产生的`L3/L2/L1`，均位于对应IDWT之前，满足因果顺序。
3. `scripts/launch_experiment_k_diagnosis_queue.sh`：依赖顺序改为A5 treatment完成后先重算A7 MAD相关性，再启动耗时更长的A3/A4 counterfactual。

服务器GPU 5完整CUDA回归结果：

| 验证项 | 结果 |
|---|---|
| K0→E1、alpha=0→E1、K1→J1、alpha=1→J1、rho=0→K2 | 最大逐元素误差全部为`0` |
| A5局部override中心变化 | H/V/D分别为`0.828607/0.315892/0.454614`，确认干预实际生效 |
| A5局部override区域外误差 | H/V/D均为`2.38419e-7`，仅浮点重排误差 |
| A5 NUAA真实J1 checkpoint单图smoke | `dataset_support=true`，5组`C_GR`比较产生非零处理效应 |
| A3/A4单图全配置smoke | 224个配置全部完成 |
| `decoder_low`因果source | Stage 1/2/3/4形状分别为`1×64×32×32`、`1×128×16×16`、`1×256×8×8`、`1×256×4×4` |

旧输出没有删除或覆盖：

- A3/A4旧NUAA/IRSTD：`j1_counterfactual/<dataset>/attempts/attempt_001_decoder_low_unavailable/`
- A5第二次无效输出：`treatment_effect/<dataset>/attempts/attempt_002_spatial_override_bypassed/`
- 依赖无效A5的旧A7：`mad_gaussian_audit/<dataset>/attempts/attempt_001_invalid_a5_correlation/`

修正版正式队列PID为`1458403`，恢复全GPU动态空闲检测。11:18已启动A5–NUAA（GPU 0，wrapper `1456374`、python `1456390`）和A5–IRSTD（GPU 5，wrapper `1456396`、python `1456468`）。队列仅在显存低于1000 MiB、利用率低于10%且无compute PID时领取后续A7/A3/A4，不会停止或抢占J/K正式训练及其他用户任务。`K_A_DECISION.json`在修正版结果完成前继续保持No-Go。

11:22，修正版A5–NUAA已完成213张train图像和38,880条逐项记录，正式输出不再是全0。最高`C_GR`相关出现在Decoder low Stage 3：Spearman=`0.05927`、`p=0.09185`，target/hard-negative的`delta_loss`中位数分别为`1.00676e-6/0`；没有任何`C_GR`组合达到预注册的`Spearman≥0.10`门槛，因此NUAA暂为`dataset_support=false`，而不是实现无效。队列随后在GPU 0启动修正版A7–NUAA（wrapper `1460218`、python `1460237`），并利用K2–NUAA完成后释放的GPU 3启动修正版A3/A4–NUAA（wrapper `1460327`、python `1460340`）；A5–IRSTD继续在GPU 5运行。

截至13:49，修正版A5和依赖它重算的A7已在两个发现集全部完成：

| 数据集 | A5图像/记录数 | A5最强`C_GR`组合 | Spearman / p | target / hard `delta_loss`中位数 | A5结论 |
|---|---:|---|---:|---:|---|
| NUAA | 213 / 38,880 | Decoder low，Stage 3 | 0.05927 / 0.09185 | 1.00676e-6 / 0 | `dataset_support=false` |
| IRSTD | 800 / 171,552 | Decoder low，Stage 1 | 0.07789 / 3.132e-6 | 8.538e-6 / 0 | `dataset_support=false` |

两数据集均有非零处理效应，说明修正版反事实确实生效；但没有组合达到预注册的`Spearman≥0.10`支持门槛，故当前不能据A5放行Gaussian Compactness。

| 数据集 | `N_raw` | `N_aligned_global` | `N_aligned_per_channel` | `N_local7` | `N_local9` |
|---|---:|---:|---:|---:|---:|
| NUAA | -0.00293 | -0.02955 | -0.03043 | -0.03454 | -0.03067 |
| IRSTD | -0.08317 | -0.08802 | -0.08530 | -0.02149 | -0.03674 |

表中为修正版A7各MAD/noise指标与A5处理效应的Spearman相关；相关性均为负且绝对值小于0.10，没有支持“噪声分数越高，局部Gaussian处理收益越大”的预注册证据。A3/A4修正版仍在运行：NUAA使用GPU 3、Python PID `1460340`；IRSTD使用GPU 0、Python PID `1464694`。在A3/A4完成并重新聚合前，`K_A_DECISION.json`继续保持No-Go。

18:37复核：上述两个A3/A4进程均仍存活，已连续运行约7小时；正式输出目录当前只有启动锁与空日志，尚无结果CSV或完成标记。因此只能记录为“运行中”，不能据此更新A3/A4结论。诊断队列PID `1458403`正常，`K_A_DECISION.json`仍为`discovery_complete=false`、`compactness_treatment_go=false`，只放行K2。

### 5.2 NUDT两阶段K-A诊断

2026-07-23 19:17起，NUDT不再因J3尚未完成而整体排除。按可用checkpoint拆为两轮，且两轮输出目录彼此独立，不覆盖已有NUAA/IRSTD结果：

| 轮次 | Checkpoint | 执行诊断 | 解锁条件 | 输出根目录 |
|---|---|---|---|---|
| 立即执行 | E1、J1 | A1；A3/A4；A5；A6；A7 | E1、J1正式任务完成且best存在；A6等待A1，A7等待A5 | `analysis/experiment_k/nudt_staged/immediate/` |
| J任务完成后 | J2-R、J2-D、J3-F、J3-R | A1/A2；A6 | 四项J任务均完成1000 epoch且最终best存在；A6等待本轮A1/A2 | `analysis/experiment_k/nudt_staged/j2j3_completed/` |

`audit_compactness_fidelity.py`在一次遍历中同时输出A1算子忠实度和A2 source错配字段，所以第一轮E1/J1的A1运行也会自然保留对应A2字段；第二轮仍会按要求独立生成四个J2/J3 checkpoint的完整A1/A2结果。

启动前checkpoint状态：

| Checkpoint | 状态 | 当前/最终epoch | Best epoch | 是否允许本轮读取 |
|---|---|---:|---:|---|
| E1 | 完成 | 1000 | 由E1正式best固定 | 是，第一轮 |
| J1 | 完成 | 1000 | 646 | 是，第一轮 |
| J2-R | 完成 | 1000 | 642 | 等待第二轮统一解锁 |
| J2-D | 完成 | 1000 | 523 | 等待第二轮统一解锁 |
| J3-F | 运行 | 935/1000 | 当前best 683 | 否，不读取训练中best |
| J3-R | 运行 | 930/1000 | 当前best 490 | 否，不读取训练中best |

新增服务器常驻队列`scripts/launch_experiment_k_nudt_diagnosis_queue.sh`。该队列只在显存低于1000 MiB、利用率低于10%且无compute PID时领取GPU，不停止或抢占任何既有任务；检测到部分输出但无完成/失败标记时会写`ORPHANED`并保留现场，不自动覆盖。任务优先级为短任务A5、A1、长任务A3/A4；A6和A7由输出依赖严格门控，第一轮A6只在E1/J1的A1完成后启动，A7只在A5完成后启动；第二轮A6只在J2/J3的A1/A2完成后启动。

19:17:51，第一项`NUDT A5 × J1`在GPU 5启动并完成；正式输出为`nudt_staged/immediate/treatment_effect/NUDT-SIRST/train/`。19:39:59同卡启动A1并完成，19:52:30启动A3/A4（wrapper PID `1825325`、Python PID `1825335`），截至23:08仍在运行。第二轮由完成标记自动解锁，绝不使用J3训练过程中的临时best。

19:34按用户要求把K-A6/K-A7加入队列。为加载新增任务，仅终止旧外层调度器PID `1794346`，A5 wrapper/Python全程保持运行并转为PPID 1，没有中断或重启。旧A5子进程保留了第一版锁描述符，因此新版调度器使用`queue.v2.lock`，避免为了释放旧锁而停止A5。新版队列PID为`1808990`，已识别A5仍为running，不会重复写入；当前完整任务集为：

```text
立即执行：A5(J1)、A1(E1/J1)、A3/A4(J1)、A6(E1/J1)、A7(J1)
J3完成后：A1/A2(J2-R/J2-D/J3-F/J3-R)、A6(J2-R/J2-D/J3-F/J3-R)
```

#### 5.2.1 NUDT A1：E1/J1算子忠实度

A1完整覆盖NUDT的663张train图像，固定E1 best epoch 456和J1 best epoch 646，共写入264,384条候选记录。下表为跨Stage 1–4的Spearman中位数，衡量Dense算子与实例级`C_P2`的候选排序一致性：

| Checkpoint | Low source | `C_square` | `C_GR` | 更接近`C_P2` |
|---|---|---:|---:|---|
| E1 | Raw LL | 0.5948 | **0.6916** | `C_GR` |
| E1 | LFSS LL | 0.5588 | **0.6660** | `C_GR` |
| E1 | Guided LL | 0.5676 | **0.6488** | `C_GR` |
| E1 | Decoder low | -0.1492 | **0.5976** | `C_GR` |
| J1 | Raw LL | 0.5770 | **0.6914** | `C_GR` |
| J1 | LFSS LL | 0.5494 | **0.6468** | `C_GR` |
| J1 | Guided LL | 0.5742 | **0.6338** | `C_GR` |
| J1 | Decoder low | -0.1076 | **0.6742** | `C_GR` |

NUDT比NUAA/IRSTD更一致地支持`C_GR`作为`C_P2`的Dense近似，特别是Decoder low中`C_square`跨stage中位相关为负，而`C_GR`保持0.60–0.67。但这只说明算子忠实度，不能推出Gaussian处理有益或有害。

#### 5.2.2 NUDT A5：Gaussian处理效应

NUDT A5使用J1 best epoch 646，覆盖663张train图像，写入132,192条逐项处理效应记录。16个有效`C_GR`组合中，最强的是Decoder low Stage 3：

| 数据集 | 最强组合 | Spearman / p | target / hard `delta_loss`中位数 | 达到`≥0.10`组合数 | 结论 |
|---|---|---:|---:|---:|---|
| NUDT | Decoder low，Stage 3 | 0.07172 / 1.6545e-4 | 0 / 0 | 0/16 | `dataset_support=false` |

样本量使0.07172达到统计显著，但效应量没有达到预注册的0.10门槛，且target/hard-negative中位处理差均为0。因此不能把“小p值”解释为Gaussian剂量有实际支持。

结合已完成的三数据集修正版A5：

| 数据集 | 最强`C_GR` Spearman | 预注册门槛 | Dataset support |
|---|---:|---:|---|
| NUAA | 0.05927 | 0.10 | false |
| IRSTD | 0.07789 | 0.10 | false |
| NUDT | 0.07172 | 0.10 | false |

新增NUDT后，A5的方向没有推翻原结论，反而把“两数据集无支持”加强为“三数据集无支持”。因此当前不应解锁K3/K4/K5；K6同样不能事后设计。该No-Go仍标记为“强化但未最终锁定”，因为NUAA/IRSTD/NUDT的A3/A4、NUDT A6/A7及第二轮J2/J3诊断尚未全部完成。

#### 5.2.3 对“J1低于E1”的修正解读

NUDT正式best并不是J1在所有指标上都低于E1：

| Checkpoint | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 456 | **0.951625** | 0.948215 | **0.975213** | **0.994709** | 1.72351e-6 |
| J1 | 646 | 0.950554 | **0.950363** | 0.974650 | 0.992593 | **1.28689e-6** |

J1相对E1的mIoU/F1/Pd分别下降0.001071/0.000563/0.002116，但nIoU提高0.002148，Fa下降4.3662e-7。因此更准确的结论是J1产生了重叠、检测率与虚警之间的轻微权衡，而不是整体退化。A1又显示E1/J1的`C_GR`忠实度相近，故当前没有证据把这组指标权衡归因于compactness算子失真；A5也没有证据支持通过更强的局部Gaussian处理来修复它。

#### 5.2.4 当前调度状态

截至23:08，NUDT A1/A5完成，A3/A4在GPU 5运行；A6依赖A1、A7依赖A5，逻辑上均已解锁，但服务器7张GPU都有compute PID，队列按空闲规则等待，不抢占A3/A4、J3或K2。Experiment J共15项正式任务已完成13项，仅NUDT J3-F/J3-R仍为935/930 epoch；二者完成后，第二轮A1/A2和A6才会读取其最终best。

## 6. 正式实验安排

第一批固定：

```text
K2 × NUAA-SIRST × seed42 × 1000 epoch
K2 × IRSTD-1K × seed42 × 1000 epoch
K2 × NUDT-SIRST × seed42 × 1000 epoch（用户于2026-07-23 13:49明确授权提前启动）
```

第二批仅在K-A Go后：在NUAA/IRSTD运行`K_A_DECISION.json`放行的K3/K4/K5/K6。

原预注册第三批要求根据NUAA和IRSTD选择同一个全局配置的1–2个候选后再运行NUDT，且不得为三个数据集分别挑不同配置。用户随后明确要求立即训练NUDT，因此仅把无需K-A选择的固定对照`K2`提前到第一批；K3–K6仍严格受K-A和全局配置筛选约束。

GPU调度门槛：显存低于1000 MiB、利用率低于10%、无compute PID。队列不得停止、抢占或覆盖Experiment J及其他用户任务。

## 7. 正式任务状态

启动时间：`2026-07-23 00:19`（Asia/Shanghai）。队列PID `952620`，每60秒重新检查GPU，仅使用显存低于1000 MiB、利用率低于10%且无compute PID的卡。

| 变体 | 数据集 | GPU | Wrapper PID | Python PID | 状态 |
|---|---|---:|---:|---:|---|
| K2 | NUAA-SIRST | 3 | 952684 | 952691 | **完成1000，best 829** |
| K2 | IRSTD-1K | 4 | 952888 | 952895 | 运行，epoch 613/1000 |
| K2 | NUDT-SIRST | 6 | 1565171 | 1565177 | 运行，epoch 240/1000 |

状态更新时间：`2026-07-23 23:08`（Asia/Shanghai）。当前epoch指标来自`train.log`最后一条正式评估记录，历史最佳指标来自对应`best_metrics.json`，二者严格分开记录。K2–NUAA已完成；K2–IRSTD继续在GPU 4运行，K2–NUDT继续在GPU 6运行。NUDT配置为663张训练图像、664张测试图像、patch 256、batch size 4、seed 42、1000 epoch、epoch 100起每epoch评估。当前没有空闲GPU：GPU 0/3/5运行三数据集A3/A4，GPU 1/2运行J3，GPU 4/6运行K2。

### 7.1 当前epoch性能

| 数据集 | 变体 | 状态 | 当前epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | K2 | 完成 | 1000 | 0.798113 | 0.793542 | 0.887723 | 0.973282 | 2.43533e-05 |
| IRSTD-1K | K2 | 运行 | 613 | 0.663646 | 0.660801 | 0.797821 | 0.915825 | 1.13682e-05 |
| NUDT-SIRST | K2 | 运行 | 240 | 0.941994 | 0.940450 | 0.970130 | 0.990476 | 3.40106e-06 |

### 7.2 历史最佳性能

| 数据集 | 变体 | Best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | K2 | 829 | 0.802123 | 0.795360 | 0.890198 | 0.965649 | 2.41475e-05 |
| IRSTD-1K | K2 | 613 | 0.663646 | 0.660801 | 0.797821 | 0.915825 | 1.13682e-05 |
| NUDT-SIRST | K2 | 240 | 0.941994 | 0.940450 | 0.970130 | 0.990476 | 3.40106e-06 |

K3–K6没有启动，仍受K-A Go和NUAA/IRSTD同一全局配置筛选约束。K2–NUDT已按用户明确指令提前启动，并从epoch 100开始产生正式指标；后续继续分别记录当前epoch、best epoch及mIoU/nIoU/F1/Pd/Fa。

### 7.3 随数值变化更新的结论

- **NUAA已经完成，可作正式结论**：K2 best mIoU 0.802123，比J1的0.796152高0.005971，说明剂量校准在NUAA有效。
- **IRSTD结论有所改善但没有反转**：K2 best从原记录的0.658811提高到0.663646，增加0.004835；但仍比J1的0.671119低0.007473，也略低于J2-R的0.668504。不能再描述为“明显无效”，更准确的是“接近J系列但尚未超过”。
- **NUDT早期判断必须撤回**：best mIoU已从epoch 126的0.858815提高到epoch 240的0.941994，说明epoch 100附近的数值不具代表性；但当前仍比E1/J1分别低0.009632/0.008561，而且只训练到240/1000，因此只能记录为持续改善中的临时结果。
- **跨数据集结论**：K2目前在NUAA优于J1，在IRSTD和NUDT仍低于J1，收益方向不一致。结合三数据集A5均`dataset_support=false`，现阶段支持继续完整训练K2，但不支持解锁基于Compactness增强剂量的K3–K6。
