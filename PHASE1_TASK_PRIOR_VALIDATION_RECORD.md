# Phase 1：红外小目标任务先验验证记录

## 1. 代码与隔离

| 项目 | 实际值 |
|---|---|
| Repository | `RiyaoChan/DWTFreqNet` |
| Base branch | `codex/experiment-h-decoder-lfp-purification` |
| Base commit | `8cfd7a97bd460b07efbad28ca7b709d7277cdd1b` |
| Phase 1 branch | `codex/phase1-irst-task-prior-validation` |
| 本地独立 worktree | `G:\DWTFreqNet-main\DWTFreqNet-phase1` |
| 服务器独立目录 | `/DATA20T/bip/cry/code/DWTFreqNet_PHASE1_TASK_PRIOR_VALIDATION` |
| Draft PR | `https://github.com/RiyaoChan/DWTFreqNet/pull/8` |

本阶段只新增 `tools/phase1/*`、一键脚本和记录/报告；不修改 `model/*`、`train*.py`、`dataset.py`、`train_one.py`、训练配置或正式 checkpoint。

## 2. 环境与数据

- 服务器：`202.38.209.226`
- Python：`/DATA20T/bip/cry/anaconda3/envs/mirfd_mamba/bin/python`
- 数据路径：`/DATA20T/bip/cry/code/DWTFreqNet_DM_AWGM/datasets`
- 数据集：NUAA-SIRST、IRSTD-1K、NUDT-SIRST
- seed：42
- 统计依赖：NumPy 2.2.6、SciPy 1.15.3、scikit-image 0.25.2、PyTorch 2.8.0+cu128；脚本不依赖服务器中缺失的 pandas。

数据划分图像数：

| 数据集 | train | test |
|---|---:|---:|
| NUAA-SIRST | 213 | 214 |
| IRSTD-1K | 800 | 201 |
| NUDT-SIRST | 663 | 664 |

## 3. Checkpoint

E1：

```text
/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS/runs/experiment_e_lfss_awgm/E1_lfss_resblock/<DATASET>/seed42/best.pth.tar
```

Experiment H：

```text
/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_H_DECODER_LFP/runs/experiment_h_decoder_lfp/<VARIANT>/<DATASET>/seed42/best.pth.tar
```

所有正式运行元数据都会记录 checkpoint SHA256。

## 4. 实现内容

- `common.py`：8连通实例、困难/普通背景候选、背景估计、Gaussian拟合、效应量/CI/FDR、H/V/D校准、采样几何、公平性与E1只读特征hook。
- P1：两种背景估计 × 三种patch半径 × 两种强度版本，完整保存三类样本、全部预定义指标和统计。
- P2：从合成输入校准实际H/V方向、成对符号和D象限模板；比较raw LL、LFSS LL、guided LL、Decoder low及raw/aligned H/V/D。
- P3：Grid、Ring、Spiral、Random、Gaussian-radial在相同32点与最大半径下比较，覆盖Rmax 2/3/4/5和中心扰动。
- H交叉分析：比较raw LL/Decoder low注意力与P1/P2先验、逐样本IoU/Fa变化的相关性。
- 聚合：生成完整JSON/CSV、决策矩阵和中文报告；未完成项明确标记Pending。

## 5. 验收

- 本地单元/合成测试：11/11通过，覆盖8连通实例、困难背景排除、Gaussian参数恢复、径向单调性、H/V/D校准、成对一致性、D象限模板、五种采样32点/最大半径公平性、双线性采样、中心扰动和bootstrap复现。
- 服务器单元/合成测试：10项通过；Git commit级模型检查因隔离smoke副本没有`.git`而明确skip，随后使用目录diff独立检查源文件。
- `git diff 8cfd7a97 -- model/ train_experiment_h_decoder_lfp.py dataset.py train_one.py`：为空。
- H/V/D实测：`H=LH`对垂直结构响应，沿x轴成对采样、预期异号；`V=HL`对水平结构响应，沿y轴成对采样、预期异号；`D=HH`的中心Gaussian四象限模板为`[+,-,+,-]`；当前路由匹配实测方向。
- P1真实NUAA smoke（2张图）：成功生成12种稳健性设置、三类catalog、全部指标、FDR统计与summary；主设置target拟合成功率100%。样本数不足30，因此只允许描述性结果。
- P2真实NUAA/E1 smoke（1张图）：严格载入E1权重，四stage × 五类低频/高频来源全部输出，checkpoint SHA256和方向校准文件写出成功。
- P3真实NUAA/E1 train/test smoke（各1张图）：Grid/Ring/Spiral/Random/Gaussian-radial均为32点且最大半径一致；训练集完成四半径选择，测试集完成固定半径的完整中心扰动；优化后用时约18秒/22秒。
- H交叉分析真实NUAA smoke（1张图、6个H权重）：全部严格载入并输出attention target/hard-negative、逐样本delta IoU/Fa和P2相关性表。

## 6. 正式执行

一键入口：

```bash
bash scripts/run_phase1_task_prior_validation.sh
```

资源规则：P1仅CPU；P2/P3/H交叉分析每次自动选择 `memory.used < 1000 MiB`、`utilization < 10%` 且无compute PID的GPU，找不到时等待，不停止或抢占Experiment H。

正式任务矩阵共21项离线任务（不训练模型）：

| 分析 | NUAA | IRSTD | NUDT | 合计 |
|---|---:|---:|---:|---:|
| P1 train/test | 2 | 2 | 2 | 6 |
| P2 train/test | 2 | 2 | 2 | 6 |
| P3 train/test | 2 | 2 | 2 | 6 |
| H交叉分析test | 1 | 1 | 1（H完成后） | 3 |

P3遵循发现/确认隔离：训练集仅在零中心偏移下报告全部`Rmax=2/3/4/5`并选择半径；测试集仍报告全部半径的零偏移结果，同时只对训练集选定半径执行完整`0/±0.5/±1/±2`扰动。五种几何、32点、所有stage/source和Random 20次重复均保留。

正式运行于 `2026-07-21 11:33:46 CST` 启动，代码提交为 `cfc83b128fcf9128684dca7ae637893e2000b874`。运行元数据保存在 `analysis/phase1_task_prior_validation/metadata/formal_run_manifest.json`。GPU 0、5 启动前均为空闲；Experiment H 正在使用的 GPU 1/2/3/4/6 未被停止或抢占。

| 任务 | 状态 | PID/日志 | 结果 |
|---|---|---|---|
| P1（CPU，6项） | **全部完成** | `P1_gaussian_geometry/<DATASET>/<SPLIT>/summary.json` | 三数据集train/test均已生成正式summary |
| P2（GPU，6项） | **全部完成** | `P2_wavelet_consistency/<DATASET>/<SPLIT>/summary.json` | 三数据集train/test均已生成正式summary |
| P3（GPU，6项） | **全部完成** | `P3_sampling_geometry/<DATASET>/<SPLIT>/summary.json` | 三数据集train/test均已生成正式summary |
| H交叉分析（动态GPU，3项） | **全部完成** | `H_cross_analysis/<DATASET>/summary.json` | NUAA/IRSTD/NUDT均已完成 |
| 最终聚合 | **全部完成** | `logs/aggregate.log` | `complete=true`，`missing=[]` |

### 6.1 Experiment H GPU释放情况

截至 `2026-07-21 23:14`，Experiment H 的 18 项训练已经全部完成 1000 epoch并释放GPU。H权重齐备后，NUDT-SIRST 的H交叉分析已于19:03完成并发布正式summary；Phase 1没有停止或抢占任何H训练。

### 6.2 空闲GPU持续调度

`scripts/monitor_phase1_idle_gpus.sh` 于 `2026-07-21 11:59:33` 启动，每30秒检查全部GPU。只有同时满足显存低于1000 MiB、利用率低于10%且无compute PID时才认领任务。调度器补齐了P2、P3 train/test和三项H交叉分析；NUDT P3 test 于22:31发布正式summary，IRSTD P3 test 于 `2026-07-22 01:13` 发布正式summary，随后最终聚合完成。所有正式任务均已收尾，不再需要继续调度Phase 1任务。

为避免与原有P2/P3主进程冲突，监控器使用原子claim、每GPU slot PID、独立临时输出和完成后符号链接发布；如果正式目录已有运行进程或完整summary，则保留临时结果而不覆盖。监控日志位于 `analysis/phase1_task_prior_validation/idle_gpu_scheduler/logs/monitor.log`。此前的本地Codex心跳查询已按用户要求停止，服务器监控脚本独立运行，不依赖本地Codex。

## 7. 错误与重试

- 首次服务器单元测试只有Git基线检查失败：部署副本无`.git`，数值测试均通过。已改为无Git副本明确skip，并保留正式分支commit级检查及服务器源目录diff。
- 首次归档部署在服务器语法检查中发现启动脚本带CRLF，正式任务未启动。已为脚本固定LF行尾、保留原smoke目录至`DWTFreqNet_PHASE1_TASK_PRIOR_VALIDATION_SMOKE_20260721_1132`，并用干净Git克隆重新部署；随后`bash -n`、11/11单元测试和受保护代码diff检查全部通过。
- 首版P3把全部中心扰动与全部候选半径做笛卡尔积，1图约198秒/12MB；修正为训练集先选半径、测试集固定半径后做扰动，并将多通道特征先转为逐点幅值图。保持预定义统计不变后，train/test smoke分别约18/22秒，输出约0.6/1.7MB。
- P1少样本smoke最初显示No-Go；已按规范修正为任一类别少于30时只标记`Descriptive only`，禁止做正式No-Go结论。
- `2026-07-21 12:58:43`，GPU 0在P2主进程从IRSTD train切换到test的短暂空档满足空闲阈值，监控器启动了临时NUDT-P2 worker；主进程随后继续占用GPU 0，形成同卡两个不同输出任务。临时worker的正式目录从未发布，已于13:01:47仅终止该临时worker并保留日志/临时输出，原IRSTD test未中断。监控器已增加“主进程存活时保留GPU 0/5”规则并以PID `3598996`重启，复查两个轮询周期未再次误调度。

## 8. 正式阶段结果与Go/No-Go

结果更新时间：2026-07-22 10:49（Asia/Shanghai）。本节只记录正式全量输出；P1、P2、P3和H交叉分析均已完成，最终聚合显示 `complete=true`、`missing=[]`。

### 8.1 P1：Gaussian拟合与目标紧凑性先验

#### 8.1.1 比较设计与判定规则

P1不是单一Gaussian拟合，而是在12种配置下比较target、匹配hard-negative和easy-background：

```text
2种背景估计（local_plane / annulus_median）
× 3种patch半径比例（3 / 4 / 5）
× 2种强度输入（raw / linear_normalized）
= 12种稳健性配置
```

正式主配置固定为`local_plane + radius_scale=4 + raw`，主判定只比较target与hard-negative。三个主指标为：`R2`（椭圆Gaussian拟合优度）、`Compactness`（中心相对外围的能量紧凑性）和`Radial monotonicity`（径向能量是否单调衰减）。单个指标需要同时满足Cliff's delta≥0.20、ROC-AUC≥0.65、12种配置中至少75%保持target>hard-negative；支持至少两个主指标为Go，只支持一个为Partial Go。FDR、PR-AUC、平衡准确率、分层结果和其余20项拟合/形状指标用于诊断，不替代上述门槛。

#### 8.1.2 三个主指标的完整结果

表中`T/H`为target/hard-negative中位数，`稳健`为12种配置中Cliff's delta为正的比例。`✓`表示达到正式主指标门槛。

| 数据集 | split | 指标 | T / H中位数 | Cliff's Δ | ROC-AUC | 稳健 | 支持 |
|---|---|---|---:|---:|---:|---:|---|
| NUAA | train | R2 | 0.9085 / 0.4940 | 0.724 | 0.862 | 12/12 | ✓ |
| NUAA | train | Compactness | 14.3456 / 7.9812 | 0.700 | 0.850 | 12/12 | ✓ |
| NUAA | train | Radial monotonicity | 1.0000 / 1.0000 | 0.135 | 0.567 | 12/12 | — |
| NUAA | test | R2 | 0.9231 / 0.5282 | 0.707 | 0.854 | 12/12 | ✓ |
| NUAA | test | Compactness | 14.7250 / 8.0068 | 0.631 | 0.816 | 12/12 | ✓ |
| NUAA | test | Radial monotonicity | 1.0000 / 1.0000 | 0.107 | 0.553 | 12/12 | — |
| IRSTD | train | R2 | 0.8999 / 0.4412 | 0.813 | 0.907 | 12/12 | ✓ |
| IRSTD | train | Compactness | 15.1287 / 7.5722 | 0.662 | 0.831 | 12/12 | ✓ |
| IRSTD | train | Radial monotonicity | 1.0000 / 1.0000 | 0.160 | 0.580 | 11/12 | — |
| IRSTD | test | R2 | 0.9066 / 0.4499 | 0.782 | 0.891 | 12/12 | ✓ |
| IRSTD | test | Compactness | 15.5729 / 7.9744 | 0.635 | 0.818 | 12/12 | ✓ |
| IRSTD | test | Radial monotonicity | 1.0000 / 1.0000 | 0.161 | 0.580 | 12/12 | — |
| NUDT | train | R2 | 0.8195 / 0.5672 | 0.429 | 0.715 | 6/12 | — |
| NUDT | train | Compactness | 15.7733 / 8.5799 | 0.484 | 0.742 | 10/12 | ✓ |
| NUDT | train | Radial monotonicity | 1.0000 / 1.0000 | 0.050 | 0.525 | 12/12 | — |
| NUDT | test | R2 | 0.8356 / 0.5842 | 0.427 | 0.714 | 6/12 | — |
| NUDT | test | Compactness | 16.2238 / 8.4783 | 0.523 | 0.761 | 12/12 | ✓ |
| NUDT | test | Radial monotonicity | 1.0000 / 1.0000 | 0.025 | 0.512 | 12/12 | — |

Radial monotonicity在三数据集均出现中位数`1.0/1.0`的明显天花板效应，效应量和AUC都未达到门槛，因此不能作为区分目标与困难背景的有效先验。NUDT的R2虽然主配置效应量和AUC达标，但只有6/12配置保持预期方向，正式判定必须记为不支持，不能因主配置显著而写成Go。

#### 8.1.3 拟合成功率与数据质量

| 数据集 | split | target拟合成功 | hard-negative拟合成功 | easy-background拟合成功 | patch截断率 |
|---|---|---:|---:|---:|---:|
| NUAA | train | 99.63% | 99.26% | 100.00% | 23.98% |
| NUAA | test | 100.00% | 97.91% | 99.24% | 24.05% |
| IRSTD | train | 100.00% | 99.71% | 99.83% | 15.39% |
| IRSTD | test | 100.00% | 99.66% | 99.66% | 13.80% |
| NUDT | train | 100.00% | 98.42% | 99.89% | 41.91% |
| NUDT | test | 100.00% | 98.73% | 99.58% | 41.46% |

拟合求解本身在三数据集均稳定，但NUDT约41%的主配置patch触及图像边界，明显高于NUAA和IRSTD。NUDT的R2跨配置不稳可能同时受到目标尺度、背景结构和边界截断影响，不能简单解释为“NUDT目标不呈Gaussian”。

#### 8.1.4 大小/对比度分层与修正结论

- NUAA：R2和Compactness在test的tiny/medium/large及low/high contrast五个分层全部支持；train的high-contrast层未达到门槛，但test完成确认。
- IRSTD：R2和Compactness在train/test的五个分层全部支持，是P1最稳定的数据集。
- NUDT：test仅在tiny、medium及low/high contrast分层支持R2/Compactness，large层不支持；Radial monotonicity所有数据集、所有分层均不支持。

因此P1的正式数据集判定保持NUAA=`Go`、IRSTD=`Go`、NUDT=`Partial Go`，跨数据集聚合仍为`Go`；但科学结论应写成“Gaussian拟合优度与紧凑性在NUAA/IRSTD成立，NUDT只稳定支持紧凑性”，不能概括成所有Gaussian几何属性普遍成立。P1支持局部紧凑目标先验，不支持把径向单调性设为硬约束，也不足以单独证明某个固定Gaussian sigma、椭圆率或方向参数最优。

完整原始输出位于`P1_gaussian_geometry/<DATASET>/<SPLIT>/`下的`gaussian_instance_metrics.csv`、`statistical_comparisons.csv`、`stratified_comparisons.csv`和`summary.json`。

### 8.2 P2：低频与H/V/D方向一致性先验

#### 8.2.1 特征来源、指标与判定规则

P2在stage 1–4比较五个特征家族：

| 家族 | LL来源 | H/V/D来源 |
|---|---|---|
| `same_dwt_raw` | raw LL | raw H/V/D |
| `raw_ll_aligned_hvd` | raw LL | aligned H/V/D |
| `lfss_ll_aligned_hvd` | LFSS LL | aligned H/V/D |
| `guided_ll_aligned_hvd` | Guided LL | aligned H/V/D |
| `decoder_low_aligned_hvd` | Decoder low | aligned H/V/D |

`C_LL`衡量低频在目标中心相对外围的紧凑性；`C_H/C_V`按合成方向校准后的敏感轴做异号成对采样；`C_D`组合对角象限余弦与幅度平衡；`C_joint`为归一化LL/H/V/D的一致性几何均值。H/V/D校准确认代码`H=LH`响应垂直结构、`V=HL`响应水平结构、`D=HH`使用`[+,-,+,-]`象限模板，当前路由与实测方向一致。

单项支持门槛为Cliff's delta≥0.20且ROC-AUC≥0.62。方向判定只使用`same_dwt_raw`的`C_H/C_V/C_D`：一个stage至少两项支持算方向Go-stage，至少两个Go-stage为Go、一个为Partial Go、没有为No-Go。低频原规则把全部家族/stage的`C_LL`支持条目直接计数，至少两条为Go。

三个数据集的train/test判定一致：

| 数据集 | split | 低频自动判定 | 方向自动判定 | 平均坐标碰撞率 |
|---|---|---|---|---:|
| NUAA | train | **Go** | **Partial Go** | 2.51% |
| NUAA | test | **Go** | **Partial Go** | 2.93% |
| IRSTD | train | **Go** | **No-Go** | 3.40% |
| IRSTD | test | **Go** | **No-Go** | 3.84% |
| NUDT | train | **Go** | **Go** | 2.01% |
| NUDT | test | **Go** | **Go** | 2.01% |

#### 8.2.2 四种不重复低频来源的完整test结果

`same_dwt_raw`和`raw_ll_aligned_hvd`共享完全相同的raw LL，故其`C_LL`数值逐项相同；下表合并为一个`Raw LL`证据，避免重复计数。单元格为target/hard-negative中位数，`✓`表示达到低频门槛。

| 数据集 | stage | Raw LL | LFSS LL | Guided LL | Decoder low |
|---|---:|---:|---:|---:|---:|
| NUAA | 1 | 6.192/2.393 ✓ | 6.561/2.513 ✓ | 7.370/2.562 ✓ | 1.777/2.042 — |
| NUAA | 2 | 4.754/2.237 ✓ | 4.763/2.259 ✓ | 5.296/2.289 ✓ | 1.399/1.925 — |
| NUAA | 3 | 1.646/1.108 ✓ | 1.417/1.088 ✓ | 1.432/1.084 ✓ | 0.624/0.993 — |
| NUAA | 4 | 1.292/1.024 ✓ | 0.831/1.004 — | 0.754/1.004 — | 0.520/0.987 — |
| IRSTD | 1 | 4.724/2.550 ✓ | 6.270/2.764 ✓ | 7.466/2.906 ✓ | 1.723/2.054 — |
| IRSTD | 2 | 4.660/2.453 ✓ | 5.279/2.640 ✓ | 5.945/2.694 ✓ | 1.035/1.974 — |
| IRSTD | 3 | 1.574/1.157 ✓ | 1.609/1.211 ✓ | 1.632/1.230 ✓ | 0.448/1.015 — |
| IRSTD | 4 | 1.162/1.065 ✓ | 1.056/0.998 — | 0.891/0.989 — | 0.480/0.961 — |
| NUDT | 1 | 3.268/2.047 ✓ | 3.584/2.028 ✓ | 4.208/2.066 ✓ | 2.033/2.142 — |
| NUDT | 2 | 4.059/2.098 ✓ | 4.042/2.072 ✓ | 4.559/2.063 ✓ | 1.623/2.085 — |
| NUDT | 3 | 1.639/1.159 ✓ | 1.688/1.132 ✓ | 1.730/1.139 ✓ | 0.666/1.075 — |
| NUDT | 4 | 1.289/1.095 ✓ | 0.856/1.044 — | 0.716/1.046 — | 0.474/1.048 — |

三个test集在合并重复raw-LL别名后仍各有10个独立“stage×来源”支持项，因此低频Go不是单纯由重复计数造成。但“所有低频来源都支持目标先验”的说法是错误的：Raw LL在stage 1–4稳定，LFSS/Guided LL只在stage 1–3稳定；Decoder low在三个数据集、四个stage全部target<hard-negative且不支持。低频先验应放在encoder/raw/LFSS/guided的浅中层，不能直接施加到当前Decoder low或stage 4的LFSS/Guided分支。

#### 8.2.3 Raw H/V/D方向结果与联合证据

单元格为target/hard-negative中位数，`✓`表示对应单方向达到门槛；`C_joint`列给出Cliff's delta/ROC-AUC，它不参与自动方向判定。

| 数据集 | stage | C_H | C_V | C_D | C_joint Δ/AUC |
|---|---:|---:|---:|---:|---:|
| NUAA | 1 | 0.169/0.150 — | 0.210/0.160 ✓ | 0.01084/0.00275 ✓ | 0.867/0.933 |
| NUAA | 2 | 0.147/0.141 — | 0.163/0.140 — | 0.00726/0.00084 ✓ | 0.581/0.790 |
| NUAA | 3 | 0.139/0.146 — | 0.128/0.129 — | 0.00678/0.00014 ✓ | 0.612/0.806 |
| NUAA | 4 | 0.119/0.131 — | 0.110/0.113 — | 0.00183/0.00064 — | 0.448/0.724 |
| IRSTD | 1 | 0.153/0.147 — | 0.142/0.151 — | 0.00538/0.00147 ✓ | 0.614/0.807 |
| IRSTD | 2 | 0.162/0.140 — | 0.162/0.148 — | 0.00597/0.00117 ✓ | 0.633/0.816 |
| IRSTD | 3 | 0.140/0.135 — | 0.152/0.134 — | 0.00534/0.00073 ✓ | 0.535/0.768 |
| IRSTD | 4 | 0.142/0.147 — | 0.135/0.136 — | 0.00036/0.00002 — | 0.250/0.625 |
| NUDT | 1 | 0.201/0.140 ✓ | 0.209/0.131 ✓ | 0.00657/0.00211 — | 0.803/0.901 |
| NUDT | 2 | 0.158/0.128 ✓ | 0.165/0.130 ✓ | 0.00407/0.00096 ✓ | 0.779/0.889 |
| NUDT | 3 | 0.143/0.120 — | 0.140/0.114 ✓ | 0.00462/0.00180 ✓ | 0.630/0.815 |
| NUDT | 4 | 0.121/0.117 — | 0.116/0.114 — | 0.00186/-0.00085 ✓ | 0.514/0.757 |

NUAA只有stage 1同时支持V/D，因此为Partial Go；IRSTD各stage最多只支持D，因此为No-Go；NUDT在stage 1、2、3均至少支持两个方向，因此为Go。跨数据集最稳定的单方向是D而不是H/V；H/V具有明显数据集和stage依赖性。另一方面，`C_joint`在三数据集的大多数stage都具有中到很强的效应，说明“联合低频+高频一致性”比把H/V单方向设为硬先验更稳定。

#### 8.2.4 重复证据、适用边界与修正结论

- 原低频自动判定把`same_dwt_raw`和`raw_ll_aligned_hvd`的同一raw LL计为两条支持，统计口径存在重复；去重后每个test集仍有10项独立支持，因此数据集级低频Go保持不变，但后续判定代码应按唯一LL张量或唯一来源计数。
- Decoder low在12个“数据集×stage”组合中全部不支持并呈反向差异，表明当前decoder低频表征更响应困难背景，不能由Raw/LFSS/Guided低频结果外推。
- 显式H/V方向先验不是普适规律：NUDT支持、NUAA部分支持、IRSTD不支持；D在stage 1–3更稳定，stage 4则仅NUDT支持。
- `C_joint`虽然跨数据集稳定，但没有进入当前自动方向判定，因此只能作为补充证据；若后续据此设计模块，需要单独预注册联合指标实验，不能回填为本轮方向Go。
- 2.0%–3.8%的特征坐标碰撞率较低但非零，深stage结果应结合候选点映射碰撞理解。

P2的正式自动结论仍为：低频=`Go`，方向=NUDT `Go`、NUAA `Partial Go`、IRSTD `No-Go`。修正后的科学结论是：“浅中层Raw/LFSS/Guided低频紧凑性跨数据集成立，Decoder low不成立；联合HVD证据有潜力，但独立H/V方向不可作为通用硬约束。”

完整原始输出位于`P2_wavelet_consistency/<DATASET>/<SPLIT>/`下的`instance_consistency_metrics.csv`、`statistical_comparisons.csv`、`stage_collision.csv`、`HVD_DIRECTION_CALIBRATION.*`和`summary.json`。

### 8.3 P3：五种采样几何公平比较（以Spiral对Grid为主假设）

| 数据集 | split | 图像数 | 训练集选定Rmax（stage 0–4） | 公平性 | 判定/状态 |
|---|---|---:|---|---|---|
| NUAA-SIRST | train | 213 | 2 / 2 / 2 / 2 / 2 | 通过 | **Go** |
| NUAA-SIRST | test | 214 | 固定使用train选择 | 通过 | **Go** |
| NUDT-SIRST | train | 663 | 2 / 2 / 2 / 2 / 2 | 通过 | **Go** |
| NUDT-SIRST | test | 664 | 固定使用train选择 | 通过 | **Go** |
| IRSTD-1K | train | 800 | 2 / 2 / 2 / 2 / 2 | 通过 | **Go** |
| IRSTD-1K | test | 201 | 固定使用train选择 | 通过 | **Go** |

NUAA、NUDT和IRSTD均已完成发现/确认隔离并正式判定为Go。三个数据集的train均选择所有stage的`Rmax=2`，test固定使用train选择的半径；所有test任务均通过公平性检查。

#### 8.3.1 比较对象、口径与指标

P3实际比较五种几何，而不是只运行Spiral：`Grid`、`Ring`、`Spiral`、`Random`和`Gaussian-radial`。五种方法均固定32点，并归一化到相同最大半径；Random使用20次重复的均值。训练集在`Rmax=2/3/4/5`中按目标样本零偏移下的平均`useful_support_ratio - far_background_hit`选择每个stage的共享半径，测试集固定使用训练集选择，避免测试泄漏。

下表及后续逐stage表使用测试集、训练集选定半径、目标样本、零中心偏移，并与正式配对检验采用相同特征源：`same_dwt_raw_LL`、`same_dwt_raw_HVD`，以及stage 0的`input_dwt_LL`。表中：

- `USR`（useful support ratio）=`(目标内部命中点数 + 目标边界命中点数) / (远背景命中点数 + 1)`，越高越好；
- `FRR`（feature response ratio）=配对目标平均绝对特征响应/困难负样本平均绝对特征响应，越高越好；
- `IOC`（inner/outer contrast）=内圈特征响应/外圈特征响应，越高表示中心更集中；
- `Far hit`为远背景命中比例，越低越好。

#### 8.3.2 五种几何的测试集总体均值

这是对上述正式核心记录逐行等权汇总的均值，不是mIoU/F1等分割性能。

| 数据集 | 几何 | USR↑ | FRR↑ | IOC↑ | Far hit↓ |
|---|---|---:|---:|---:|---:|
| NUAA | Grid | 8.1818 | 3.2981 | 1.6993 | 0.4340 |
| NUAA | Ring | 9.1239 | 3.5720 | **1.7636** | 0.4143 |
| NUAA | Spiral | 9.1239 | 3.5720 | **1.7636** | 0.4143 |
| NUAA | Random | 9.4750 | 3.4531 | 1.5945 | 0.4164 |
| NUAA | **Gaussian-radial** | **10.8493** | **3.8203** | 1.4717 | **0.3958** |
| IRSTD | Grid | 9.2316 | 1.5627 | 1.5543 | 0.4010 |
| IRSTD | Ring | 10.0275 | 1.6484 | **1.6104** | 0.3810 |
| IRSTD | Spiral | 10.0275 | 1.6484 | **1.6104** | 0.3810 |
| IRSTD | Random | 10.4681 | 1.6074 | 1.4715 | 0.3835 |
| IRSTD | **Gaussian-radial** | **11.8126** | **1.7149** | 1.3768 | **0.3615** |
| NUDT | Grid | 7.2569 | 1.3071 | 1.5544 | 0.4673 |
| NUDT | Ring | 8.1010 | 1.4105 | **1.5831** | 0.4481 |
| NUDT | Spiral | 8.1010 | 1.4105 | **1.5831** | 0.4481 |
| NUDT | Random | 8.4866 | 1.3549 | 1.4725 | 0.4510 |
| NUDT | **Gaussian-radial** | **9.8552** | **1.4772** | 1.3857 | **0.4303** |

完整五方法结果改变了只看Spiral/Grid时的解释：Gaussian-radial在三个数据集上均取得最高USR、最高FRR和最低Far hit；Ring/Spiral则取得最高IOC。Random的USR均高于Ring/Spiral，但FRR通常低于Ring/Spiral。因此不存在“Spiral在所有指标上整体第一”的证据。

#### 8.3.3 逐stage主指标（USR / FRR）

| 数据集 | stage | Grid | Ring | Spiral | Random | Gaussian-radial |
|---|---:|---:|---:|---:|---:|---:|
| NUAA | 0 | 20.921 / 1.795 | 22.268 / 1.943 | 22.268 / 1.943 | 23.113 / 1.865 | **25.222 / 2.045** |
| NUAA | 1 | 20.706 / 3.979 | 22.165 / 4.021 | 22.165 / 4.021 | 22.918 / 4.039 | **25.164 / 4.144** |
| NUAA | 2 | 4.926 / 3.724 | 6.534 / 4.030 | 6.534 / 4.030 | 7.001 / 3.894 | **9.220 / 4.282** |
| NUAA | 3 | 0.638 / 3.742 | 1.059 / 4.210 | 1.059 / 4.210 | 1.015 / 3.967 | **1.546 / 4.541** |
| NUAA | 4 | 0.087 / 2.499 | 0.165 / 2.842 | 0.165 / 2.842 | 0.146 / 2.706 | **0.281 / 3.202** |
| IRSTD | 0 | 22.213 / 1.746 | 23.241 / 1.881 | 23.241 / 1.881 | 24.073 / 1.804 | **25.999 / 1.993** |
| IRSTD | 1 | 22.213 / 1.691 | 23.241 / 1.715 | 23.241 / 1.715 | 24.073 / 1.712 | **25.999 / 1.749** |
| IRSTD | 2 | 6.670 / 1.550 | 8.177 / 1.630 | 8.177 / 1.630 | 8.758 / 1.594 | **10.856 / 1.692** |
| IRSTD | 3 | 1.357 / 1.600 | 1.849 / 1.741 | 1.849 / 1.741 | 1.960 / 1.668 | **2.864 / 1.833** |
| IRSTD | 4 | 0.196 / 1.318 | 0.237 / 1.392 | 0.237 / 1.392 | 0.279 / 1.357 | **0.438 / 1.446** |
| NUDT | 0 | 18.017 / 1.312 | 19.567 / 1.326 | 19.567 / 1.326 | 20.439 / 1.321 | **22.868 / 1.347** |
| NUDT | 1 | 18.017 / 1.074 | 19.567 / 1.153 | 19.567 / 1.153 | 20.439 / 1.111 | **22.868 / 1.190** |
| NUDT | 2 | 4.972 / 1.280 | 6.029 / 1.399 | 6.029 / 1.399 | 6.451 / 1.335 | **8.294 / 1.476** |
| NUDT | 3 | 0.551 / 1.594 | 0.885 / 1.767 | 0.885 / 1.767 | 0.906 / 1.675 | **1.449 / 1.883** |
| NUDT | 4 | 0.108 / 1.278 | 0.189 / 1.364 | 0.189 / 1.364 | 0.176 / 1.315 | **0.303 / 1.425** |

Gaussian-radial在15个“数据集×stage”组合中均同时取得最高USR和最高FRR。随着stage加深，所有方法的USR明显下降且Far hit上升，说明深层特征坐标中的固定2点邻域仍会更频繁覆盖远背景；深stage的巨大相对提升需要结合很小的绝对基线理解。

#### 8.3.4 中心扰动下的五方法USR

下表覆盖正式测试目标记录的全部特征源，并固定使用训练集选定的`Rmax=2`。偏移单位是对应stage的特征图坐标。

| 数据集 | 偏移幅度 | Grid | Ring | Spiral | Random | Gaussian-radial |
|---|---:|---:|---:|---:|---:|---:|
| NUAA | 0.0 | 9.3992 | 10.3041 | 10.3041 | 10.6360 | **11.9475** |
| NUAA | 0.5 | 8.1145 | 8.9221 | 8.9221 | 9.3185 | **10.5133** |
| NUAA | 1.0 | 5.9656 | 6.6147 | 6.6147 | 6.9476 | **7.8352** |
| NUAA | 2.0 | 2.5492 | 2.7813 | 2.7813 | 2.9269 | **3.3798** |
| IRSTD | 0.0 | 10.3714 | 11.1350 | 11.1350 | 11.5555 | **12.8425** |
| IRSTD | 0.5 | 9.0866 | 9.8716 | 9.8716 | 10.2705 | **11.4241** |
| IRSTD | 1.0 | 7.1063 | 7.7211 | 7.7211 | 8.0685 | **8.9360** |
| IRSTD | 2.0 | 3.4936 | 3.7713 | 3.7713 | 3.9211 | **4.3746** |
| NUDT | 0.0 | 8.5176 | 9.2888 | 9.2888 | 9.6774 | **10.9926** |
| NUDT | 0.5 | 7.3327 | 8.0701 | 8.0701 | 8.4257 | **9.6628** |
| NUDT | 1.0 | 5.6070 | 6.1533 | 6.1533 | 6.4171 | **7.3233** |
| NUDT | 2.0 | 2.4866 | 2.7164 | 2.7164 | 2.8332 | **3.2503** |

五种方法均随中心误差增大而明显退化，但排序基本稳定：Gaussian-radial最高、Random第二、Ring/Spiral相同、Grid最低。因此当前数据更支持“径向非均匀覆盖优于规则Grid”，而不仅是Spiral相对Grid的二元结论。

#### 8.3.5 Spiral与四种基线的配对比较摘要

表中“更高stage/过门槛stage”分别表示Spiral均值更高的stage，以及同时满足样本数不少于30、rank-biserial不低于0.15、相对提升不低于5%的stage。只有Spiral对Grid用于预注册Go判定；其它比较使用相同门槛作补充解读。

| 数据集 | 对照 | USR：更高/过门槛 | FRR：更高/过门槛 |
|---|---|---|---|
| NUAA | Grid | 0–4 / 0–4 | 0–4 / 0、2–4 |
| NUAA | Ring | 相同 / 无 | 相同 / 无 |
| NUAA | Random | 3、4 / 4 | 0、2–4 / 3、4 |
| NUAA | Gaussian-radial | 无 / 无 | 无 / 无 |
| IRSTD | Grid | 0–4 / 2–4 | 0–4 / 0、2–4 |
| IRSTD | Ring | 相同 / 无 | 相同 / 无 |
| IRSTD | Random | 无 / 无 | 0–4 / 无 |
| IRSTD | Gaussian-radial | 无 / 无 | 无 / 无 |
| NUDT | Grid | 0–4 / 0–4 | 0–4 / 1–4 |
| NUDT | Ring | 相同 / 无 | 相同 / 无 |
| NUDT | Random | 4 / 4 | 0–4 / 3 |
| NUDT | Gaussian-radial | 无 / 无 | 无 / 无 |

Spiral稳定优于Grid，因而按原协议三个数据集均为Go；但Spiral从未优于Gaussian-radial，且对Random的优势不稳定。这是完整比较后必须保留的边界。

#### 8.3.6 五方法Friedman总体检验

每个数据集执行5个stage×4个指标，共20项Friedman检验并做FDR校正。三数据集均为20/20显著；下表给出各指标5个stage中的最大`p_fdr`，因此每个stage的校正后p值均不大于该值。

| 数据集 | 显著/总数 | USR最大p_fdr | FRR最大p_fdr | IOC最大p_fdr | Far hit最大p_fdr |
|---|---:|---:|---:|---:|---:|
| NUAA | 20/20 | 2.604e-31 | 1.005e-48 | 2.517e-50 | 3.270e-29 |
| IRSTD | 20/20 | 5.253e-44 | 4.786e-17 | 1.243e-53 | 1.602e-28 |
| NUDT | 20/20 | 6.647e-103 | 1.078e-5 | 4.159e-60 | 1.439e-106 |

总体检验说明五种几何的分布确实不同，但不单独说明哪一种最好；方向和实际差异仍应结合前述均值与配对检验。

#### 8.3.7 Ring/Spiral退化与修正后的科学解读

代码方向检查发现，当前`Ring`由4个半径×8个角度组成；当前`Spiral`由8个head×4个半径组成。对`Rmax=2/3/4/5`逐一排序坐标后，两者点集最大绝对差仅为`1.55e-15/2.22e-15/3.11e-15/4.00e-15`，即数值精度范围内完全相同，只是点的排列顺序不同。P3的命中率、均值、对比度和成对距离指标均对点顺序不敏感，因此Ring与Spiral所有正式汇总完全相同。

据此，原始自动判定仍按预注册代码保留为P3=`Go`，但科学解释必须收窄：

- 已证实：共享的Ring/Spiral空间点集相对Grid具有跨数据集优势；紧凑的`Rmax=2`在所有stage被训练集选择；
- 未证实：Spiral的遍历顺序或“螺旋序列”优于Ring，因为当前无顺序敏感算子参与P3；
- 新发现：Gaussian-radial在三个数据集、全部stage的USR与FRR上均为最佳，并在所有中心偏移幅度下保持最高USR；
- 后续若要验证真正的Spiral优势，需要构造与Ring点集不同的螺旋几何，或者把点序输入顺序敏感模块，再重新执行发现/确认隔离；
- 因而更准确的Phase 1结论是“非Grid的径向结构化采样值得进入后续实验，Gaussian-radial优先级高；当前Spiral特异性证据不足”。

完整原始输出位于服务器：

```text
analysis/phase1_task_prior_validation/P3_sampling_geometry/<DATASET>/<SPLIT>/sampling_instance_metrics.csv
analysis/phase1_task_prior_validation/P3_sampling_geometry/<DATASET>/<SPLIT>/paired_statistical_comparisons.csv
analysis/phase1_task_prior_validation/P3_sampling_geometry/<DATASET>/<SPLIT>/friedman_tests.csv
analysis/phase1_task_prior_validation/P3_sampling_geometry/<DATASET>/<SPLIT>/geometry_fairness.csv
analysis/phase1_task_prior_validation/P3_sampling_geometry/<DATASET>/<SPLIT>/center_perturbation_summary.csv
analysis/phase1_task_prior_validation/P3_sampling_geometry/<DATASET>/<SPLIT>/summary.json
```

### 8.4 Experiment H交叉分析

| 数据集 | test图像数 | attention分离统计 | 强相关项数 | 绝对值最强相关 |
|---|---:|---:|---:|---|
| NUAA-SIRST | 214 | 6变体×4stage，共24项；target-hard gap全部为负 | 76 | H3-D stage1 与 `C_LL`：rho=-0.8071 |
| IRSTD-1K | 201 | 6变体×4stage，共24项；target-hard gap全部为负 | 56 | H1-D stage1 与 `C_LL`：rho=-0.7375 |
| NUDT-SIRST | 664 | 6变体×4stage，共24项；target-hard gap全部为负 | 93 | H3-D stage1 与 `C_LL`：rho=-0.8123 |

三个数据集的24项attention分离统计均为负gap，即当前attention数值在target区域低于配对hard-negative区域；强相关项按脚本预定义阈值筛选。该现象已跨三数据集复现，但它是相关性诊断，不足以单独证明因果机制或直接给出模型Go/No-Go。

### 8.5 最终结论

- P1：NUAA/IRSTD凭R2与Compactness为Go，NUDT仅凭Compactness为Partial Go；Radial monotonicity三数据集均不支持。跨数据集自动结论为Go，但只能支持“拟合优度/紧凑性”的部分Gaussian先验，不能支持全部径向几何属性。
- P2：三个数据集的低频自动判定均为Go；去除重复raw-LL别名后，Raw LL在stage 1–4、LFSS/Guided LL在stage 1–3仍稳定支持，而Decoder low在所有数据集/stage均不支持。方向先验为NUDT Go、NUAA Partial Go、IRSTD No-Go；D与联合HVD证据比独立H/V更稳定，因此P2综合为Partial Go。
- P3：NUAA、IRSTD和NUDT的train/test按预注册Spiral-vs-Grid规则全部为Go，五个stage均选择`Rmax=2`；完整五方法比较显示Gaussian-radial在全部数据集和stage的USR/FRR上最佳，且当前Ring/Spiral点集相同，因此只能确认径向结构化采样优于Grid，不能确认Spiral序列本身最优。
- H交叉分析：三个数据集均完成；24项attention分离统计均为负gap，这是稳定复现的相关性诊断，不单独作为模型Go/No-Go证据。
- 最终聚合：P1=`Go`、P2=`Partial Go`、P2 directional=`Partial Go`、P2 low-frequency=`Go`、P3=`Go`；综合case为`Mixed/Partial`，无缺失任务。
