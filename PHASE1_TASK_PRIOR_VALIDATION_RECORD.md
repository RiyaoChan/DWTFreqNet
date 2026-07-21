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
| Draft PR | 待代码与首轮验收完成后创建 |

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

| 任务 | 状态 | PID/日志 | 结果 |
|---|---|---|---|
| P1 | 待部署 | — | — |
| P2 | 待部署 | — | — |
| P3 | 待部署 | — | — |
| H交叉分析 | 待部署 | — | NUDT须等H六项完成 |

## 7. 错误与重试

- 首次服务器单元测试只有Git基线检查失败：部署副本无`.git`，数值测试均通过。已改为无Git副本明确skip，并保留正式分支commit级检查及服务器源目录diff。
- 首版P3把全部中心扰动与全部候选半径做笛卡尔积，1图约198秒/12MB；修正为训练集先选半径、测试集固定半径后做扰动，并将多通道特征先转为逐点幅值图。保持预定义统计不变后，train/test smoke分别约18/22秒，输出约0.6/1.7MB。
- P1少样本smoke最初显示No-Go；已按规范修正为任一类别少于30时只标记`Descriptive only`，禁止做正式No-Go结论。

## 8. 最终结果与Go/No-Go

待P1/P2/P3 confirmation split及H交叉分析完成后填写。Phase 1完成前不启动I1/I2/I3/I4/I5/GCSWR或任何新模型训练。
