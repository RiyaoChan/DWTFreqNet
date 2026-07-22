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

### 8.1 P1：Gaussian目标几何先验

表中的R2和Compactness分别为target与hard-negative的中位数；判定使用预注册的效应量、FDR和12种稳健性设置，不只看中位数差异。

| 数据集 | split | R2：target / hard-negative | Compactness：target / hard-negative | 支持的主指标 | 判定 |
|---|---|---:|---:|---|---|
| NUAA-SIRST | train | 0.9085 / 0.4940 | 14.3456 / 7.9812 | R2、Compactness | **Go** |
| NUAA-SIRST | test | 0.9231 / 0.5282 | 14.7250 / 8.0068 | R2、Compactness | **Go** |
| IRSTD-1K | train | 0.8999 / 0.4412 | 15.1287 / 7.5722 | R2、Compactness | **Go** |
| IRSTD-1K | test | 0.9066 / 0.4499 | 15.5729 / 7.9744 | R2、Compactness | **Go** |
| NUDT-SIRST | train | 0.8195 / 0.5672 | 15.7733 / 8.5799 | Compactness | **Partial Go** |
| NUDT-SIRST | test | 0.8356 / 0.5842 | 16.2238 / 8.4783 | Compactness | **Partial Go** |

NUAA和IRSTD在发现集与确认集上均稳定支持R2和Compactness，Gaussian目标几何先验为Go。NUDT train/test均只稳定支持Compactness；R2虽有显著差异，但都只有一半稳健性设置保持预期方向，因此确认结论为Partial Go。

### 8.2 P2：低频与方向一致性先验

三个数据集的train/test判定完全一致，P2六项正式任务已经全部完成。

| 数据集 | split | 低频先验 | 方向先验 |
|---|---|---|---|
| NUAA-SIRST | train | **Go** | **Partial Go** |
| NUAA-SIRST | test | **Go** | **Partial Go** |
| IRSTD-1K | train | **Go** | **No-Go** |
| IRSTD-1K | test | **Go** | **No-Go** |
| NUDT-SIRST | train | **Go** | **Go** |
| NUDT-SIRST | test | **Go** | **Go** |

当前最稳定的跨数据集结论是低频先验：三个数据集均为Go。显式方向先验具有明显数据集依赖性，不能作为普适假设：NUDT支持，NUAA仅部分支持，IRSTD不支持。

### 8.3 P3：Spiral采样几何

| 数据集 | split | 图像数 | 训练集选定Rmax（stage 0–4） | 公平性 | 判定/状态 |
|---|---|---:|---|---|---|
| NUAA-SIRST | train | 213 | 2 / 2 / 2 / 2 / 2 | 通过 | **Go** |
| NUAA-SIRST | test | 214 | 固定使用train选择 | 通过 | **Go** |
| NUDT-SIRST | train | 663 | 2 / 2 / 2 / 2 / 2 | 通过 | **Go** |
| NUDT-SIRST | test | 664 | 固定使用train选择 | 通过 | **Go** |
| IRSTD-1K | train | 800 | 2 / 2 / 2 / 2 / 2 | 通过 | **Go** |
| IRSTD-1K | test | 201 | 固定使用train选择 | 通过 | **Go** |

NUAA、NUDT和IRSTD均已完成发现/确认隔离并正式判定为Go。三个数据集的train均选择所有stage的`Rmax=2`，test固定使用train选择的半径；所有test任务均通过公平性检查。

### 8.4 Experiment H交叉分析

| 数据集 | test图像数 | attention分离统计 | 强相关项数 | 绝对值最强相关 |
|---|---:|---:|---:|---|
| NUAA-SIRST | 214 | 6变体×4stage，共24项；target-hard gap全部为负 | 76 | H3-D stage1 与 `C_LL`：rho=-0.8071 |
| IRSTD-1K | 201 | 6变体×4stage，共24项；target-hard gap全部为负 | 56 | H1-D stage1 与 `C_LL`：rho=-0.7375 |
| NUDT-SIRST | 664 | 6变体×4stage，共24项；target-hard gap全部为负 | 93 | H3-D stage1 与 `C_LL`：rho=-0.8123 |

三个数据集的24项attention分离统计均为负gap，即当前attention数值在target区域低于配对hard-negative区域；强相关项按脚本预定义阈值筛选。该现象已跨三数据集复现，但它是相关性诊断，不足以单独证明因果机制或直接给出模型Go/No-Go。

### 8.5 最终结论

- P1：NUAA/IRSTD为Go，NUDT为Partial Go；跨数据集结论为Go。
- P2：低频先验在三个数据集均为Go；方向先验为NUDT Go、NUAA Partial Go、IRSTD No-Go，因此P2综合为Partial Go。
- P3：NUAA、IRSTD和NUDT的train/test全部为Go，五个stage均选择`Rmax=2`。
- H交叉分析：三个数据集均完成；24项attention分离统计均为负gap，这是稳定复现的相关性诊断，不单独作为模型Go/No-Go证据。
- 最终聚合：P1=`Go`、P2=`Partial Go`、P2 directional=`Partial Go`、P2 low-frequency=`Go`、P3=`Go`；综合case为`Mixed/Partial`，无缺失任务。
