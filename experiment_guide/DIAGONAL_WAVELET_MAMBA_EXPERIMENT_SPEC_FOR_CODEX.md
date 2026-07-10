# Wavelet-Aligned Eight-Directional Mamba：斜对角扫描与参数共享实验方案

## 0. 任务背景

当前仓库：

- Repository: `RiyaoChan/DWTFreqNet`
- Branch: `main`
- Base commit: `bbaab94`
- Existing module: `DirectionMatchedAWGM`
- Existing variants:
  - `awgm_original`
  - `dm_awgm_full`
  - `dm_awgm_no_mamba`
  - `dm_awgm_no_dcn`
  - `dm_awgm_conv_only`

当前 `dm_awgm_full` 的结构是：

```text
H → LR/RL 两个独立 Mamba
V → TB/BT 两个独立 Mamba
D → DeformConv2d
          ↓
DirectionFusionGate
          ↓
A' = A * attention + A
```

本实验计划在保持 AWGM 输入输出接口和后续 HGE/LDRC 不变的前提下，将 D 分支的可变形卷积替换为斜对角 Mamba 扫描，并系统比较不同方向参数共享方式。

---

# 1. 研究问题

本实验需要回答四个核心问题：

1. 对角高频分量 `D` 使用斜对角 Mamba 是否优于普通卷积和可变形卷积？
2. 对角分量只使用一组双向斜对角扫描，还是使用两组四向斜对角扫描更有效？
3. 多个扫描方向是否需要独立的 Mamba 参数？
4. 轴向四方向共用一个 Mamba、对角四方向共用另一个 Mamba，是否能在降低参数量的同时保持性能？

目标不是简单地把“四方向扫描”扩展为“八方向扫描”，而是实现：

> **Wavelet-aligned directional state-space modeling**：根据 H、V、D 小波子带的方向属性，分别匹配水平、垂直和斜对角扫描拓扑。

---

# 2. 概念澄清：原始 Mamba 与四方向视觉 Mamba

## 2.1 原始 Mamba

原始 `mamba_ssm.Mamba` 是一维序列模块：

```text
[B, L, C] → Mamba → [B, L, C]
```

它本身没有“左、右、上、下”四方向概念。四方向来自视觉模型在 Mamba 外部对二维特征进行不同顺序的序列化。

因此不要在论文或代码注释中写：

> 原始 Mamba 自带四方向扫描。

推荐写法：

> Vision Mamba methods extend the one-dimensional selective state-space model to images through multi-directional scan routes.

## 2.2 VMamba/SS2D 是否四方向共用一套参数？

典型 VMamba SS2D 并不是简单地创建一个 `Mamba` 对四个序列重复调用。其实现通常具有：

- 四个扫描序列；
- 共享的外部输入投影、局部卷积和输出投影；
- 在 SSM 核心中保留 `K=4` 的方向参数组，例如方向独立的 `x_proj`、`dt_proj`、`A` 和 `D` 参数切片。

因此更准确的结论是：

> **典型 VMamba SS2D 对外层部分参数存在共享，但四条扫描路径的核心 SSM 参数并非完全共享。**

当前 DWTFreqNet 的 DM-AWGM 更直接：

```python
self.mamba_lr = Mamba(...)
self.mamba_rl = Mamba(...)
self.mamba_tb = Mamba(...)
self.mamba_bt = Mamba(...)
```

即 LR、RL、TB、BT 是四个独立 Mamba 实例，参数不共享。

---

# 3. 新模块命名

建议模块总名：

## `WaveletAlignedEightDirectionalMamba`

缩写：

## `W8M`

用于替换 AWGM 时，可命名为：

## `W8M-AWGM`

完整含义：

> Wavelet-Aligned Eight-Directional Mamba Guided Adaptive Wavelet Guidance Module

代码类建议：

```python
class WaveletEightDirectionAWGM(nn.Module):
    def forward(self, A, H, V, D):
        ...
        return A_enhanced
```

保持原接口：

```text
A, H, V, D → A_enhanced
```

不得直接把 H/V/D 拼接后输出，仍需保持高频引导低频的稳定设计：

```python
A_enhanced = A * attention + A
```

---

# 4. 八方向与小波子带的对应关系

## 4.1 H 分支

```text
H → Left-to-Right
H → Right-to-Left
```

记作：

```text
LR, RL
```

## 4.2 V 分支

```text
V → Top-to-Bottom
V → Bottom-to-Top
```

记作：

```text
TB, BT
```

## 4.3 D 分支

覆盖两类斜对角方向：

```text
NW → SE
SE → NW
NE → SW
SW → NE
```

分别记作：

```text
NWSE, SENW, NESW, SWNE
```

总计：

```text
2 horizontal + 2 vertical + 4 diagonal = 8 scan routes
```

---

# 5. 对角扫描实现

## 5.1 基础输入

输入：

```python
x.shape == [B, C, H, W]
```

转换为按行展平的序列：

```python
x_flat = x.flatten(2).transpose(1, 2)
# [B, H*W, C]
```

然后使用对角排列索引：

```python
seq_nwse = x_flat[:, idx_nwse, :]
seq_senw = x_flat[:, idx_senw, :]
seq_nesw = x_flat[:, idx_nesw, :]
seq_swne = x_flat[:, idx_swne, :]
```

四个序列形状均为：

```text
[B, H*W, C]
```

## 5.2 3×3 单元测试

对于：

```text
1 2 3
4 5 6
7 8 9
```

必须至少验证以下一种固定定义：

```text
NW → SE:
[1, 5, 9, 2, 6, 3, 4, 8, 7]

SE → NW:
[7, 8, 4, 3, 6, 2, 9, 5, 1]

NE → SW:
[3, 5, 7, 2, 4, 1, 6, 8, 9]

SW → NE:
[9, 8, 6, 1, 4, 2, 7, 5, 3]
```

对应索引：

```python
idx_nwse = [0, 4, 8, 1, 5, 2, 3, 7, 6]
idx_senw = list(reversed(idx_nwse))

idx_nesw = [2, 4, 6, 1, 3, 0, 5, 7, 8]
idx_swne = list(reversed(idx_nesw))
```

注意：

- 对角排列顺序不是唯一的；
- 一旦选择一种顺序，必须保证训练、测试和逆恢复完全一致；
- 所有位置必须恰好出现一次，不能遗漏或重复。

## 5.3 逆排列

```python
def inverse_permutation(index: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(index)
    inverse[index] = torch.arange(index.numel(), device=index.device)
    return inverse
```

Mamba 输出恢复：

```python
out_seq = mamba(seq)
out_flat = out_seq[:, inverse_idx, :]
out_2d = out_flat.transpose(1, 2).reshape(B, C, H, W)
```

## 5.4 对角线边界问题

方法一把多条对角线连接为一个长度为 `H*W` 的序列，Mamba 状态会跨越相邻对角线边界传播。

第一版实现允许这一行为，但需要：

1. 明确写入实验记录；
2. 优先采用 diagonal snake 顺序，减小相邻对角线之间的空间跳跃；
3. 后续可增加“每条对角线独立序列”的严格版本，但不作为第一阶段必做项。

建议提供参数：

```python
diag_order = "concat" | "snake"
```

第一轮正式实验默认：

```python
diag_order = "snake"
```

---

# 6. 参数共享方案

## 6.1 方案一：八方向完全独立

名称：

```text
share_mode = independent_8
```

模块数：

```text
mamba_lr
mamba_rl
mamba_tb
mamba_bt
mamba_nwse
mamba_senw
mamba_nesw
mamba_swne
```

总计 8 个 Mamba。

优点：

- 表达能力最高；
- 每个方向可以学习独立状态转移规律。

缺点：

- 参数量最大；
- 红外小目标数据集较小时过拟合风险高；
- 不利于轻量化。

---

## 6.2 方案二：正反方向成对共享

名称：

```text
share_mode = pair_shared_4
```

模块数：

```text
horizontal_mamba: LR/RL 共享
vertical_mamba: TB/BT 共享
diag_backslash_mamba: NWSE/SENW 共享
diag_slash_mamba: NESW/SWNE 共享
```

总计 4 个 Mamba。

意义：

- 同一几何轴的正反扫描共享状态空间动力学；
- `\` 与 `/` 两类斜向结构仍保留不同参数。

这是结构上最细致的共享方案。

---

## 6.3 方案三：按小波子带共享

名称：

```text
share_mode = subband_shared_3
```

模块数：

```text
h_mamba: LR/RL 共享
v_mamba: TB/BT 共享
diag_mamba: 四个对角方向全部共享
```

总计 3 个 Mamba。

意义：

- H、V、D 三个小波子带的数据分布不同，因此分别使用一套参数；
- 同一个子带内部只改变扫描路径，不改变状态空间动力学。

这是推荐的默认主方案。

---

## 6.4 方案四：轴向共享 + 对角共享

名称：

```text
share_mode = axial_diag_shared_2
```

模块数：

```text
axial_mamba: LR/RL/TB/BT 全部共享
diag_mamba: NWSE/SENW/NESW/SWNE 全部共享
```

总计 2 个 Mamba。

### 是否有意义？

有明确意义。

该方案用于验证：

> 性能提升主要来自新的扫描拓扑，还是来自给不同方向配置更多独立参数？

如果方案四接近或优于方案三，说明：

- 水平和垂直方向可共享统一的轴向状态空间动力学；
- 方向差异主要由输入序列排列决定；
- 不需要为 H/V 分支分别维护独立 Mamba；
- 模块具有更好的参数效率和正则化效果。

如果方案四明显低于方案三，说明：

- H 与 V 小波子带存在实质性分布差异；
- 水平和垂直方向需要方向专用状态空间参数；
- 不能只依靠排列顺序表达方向差异。

### 重要说明

参数共享只减少参数量，**不会自动减少扫描次数和理论 FLOPs**。

八条序列仍然都要经过 Mamba：

```text
8 directions = 8 sequence-processing routes
```

可以通过把相同长度的方向序列堆叠进 batch 维度来减少 Python 调用和提升 GPU 利用率，但总计算量不会消失。

---

## 6.5 方案五：八方向全部共享

名称：

```text
share_mode = all_shared_1
```

模块数：

```text
shared_mamba: 所有八方向共用
```

总计 1 个 Mamba。

仅作为压力测试，不建议默认作为主模型。

作用：

- 验证单一状态空间动力学能否仅通过扫描顺序适应所有方向；
- 得到参数共享的下界；
- 若性能下降明显，可以说明轴向与斜向至少需要两类参数。

---

# 7. 方向条件编码

共享参数后，同一个 Mamba 并不会显式知道当前输入属于哪个方向。虽然序列顺序不同会产生不同状态演化，但可以进一步加入轻量方向条件。

建议增加可选版本：

```text
use_direction_embedding = True
```

例如：

```python
sequence = sequence + direction_embedding[direction_id]
```

方向 embedding 形状：

```text
[8, 1, C]
```

或者按组使用：

```text
axial direction embedding: 4
diagonal direction embedding: 4
```

建议将其作为独立消融，不要在所有共享实验中默认开启。

实验名称：

```text
axial_diag_shared_2
axial_diag_shared_2_dir_embed
```

这样可以回答：

> 共享 Mamba 是否需要显式方向标识？

---

# 8. 推荐代码结构

## 8.1 对角索引构建器

```python
class DiagonalIndexCache:
    @staticmethod
    def build(height, width, order="snake", device=None):
        # return:
        # idx_nwse, idx_senw, idx_nesw, idx_swne
        # inv_nwse, inv_senw, inv_nesw, inv_swne
        ...
```

要求：

- 支持任意 `H, W`；
- 输出 `torch.long`；
- 缓存键至少包含 `(H, W, order, device)`；
- 不允许每个 iteration 在 CPU 重新循环生成并复制到 GPU。

## 8.2 对角 Mamba 分支

```python
class DiagonalFourDirectionMamba(nn.Module):
    def __init__(
        self,
        dim,
        share_mode="subband_shared_3",
        diag_order="snake",
        use_direction_embedding=False,
        allow_fallback=False,
    ):
        ...
```

前向流程：

```python
def forward(self, x):
    B, C, H, W = x.shape
    x_flat = x.flatten(2).transpose(1, 2)

    seqs = {
        "nwse": x_flat[:, idx_nwse, :],
        "senw": x_flat[:, idx_senw, :],
        "nesw": x_flat[:, idx_nesw, :],
        "swne": x_flat[:, idx_swne, :],
    }

    outputs = {}
    for direction, seq in seqs.items():
        seq = self.norm(seq)
        seq = self.add_direction_embedding(seq, direction)
        outputs[direction] = self.get_mamba(direction)(seq)
        outputs[direction] = outputs[direction][:, inverse_idx[direction], :]

    output = self.direction_fuse(outputs)
    output = output.transpose(1, 2).reshape(B, C, H, W)
    return x + self.proj(output)
```

## 8.3 轴向共享分支

不要继续把水平与垂直写成两个完全独立类。建议增加统一类：

```python
class AxialFourDirectionMamba(nn.Module):
    def __init__(
        self,
        dim,
        share_mode,
        use_direction_embedding=False,
        allow_fallback=False,
    ):
        ...
```

需要支持：

```text
independent_8:
  LR/RL/TB/BT 各自独立

pair_shared_4 或 subband_shared_3:
  LR/RL 共用 h_mamba
  TB/BT 共用 v_mamba

axial_diag_shared_2:
  LR/RL/TB/BT 共用 axial_mamba

all_shared_1:
  由上层传入 shared_mamba
```

## 8.4 W8M-AWGM

```python
class WaveletEightDirectionAWGM(nn.Module):
    def __init__(
        self,
        in_channels,
        share_mode="subband_shared_3",
        diag_directions=4,
        diag_order="snake",
        use_direction_embedding=False,
        fusion="spatial_softmax",
        allow_fallback=False,
    ):
        super().__init__()

        self.pre_h = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_v = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_d = nn.Conv2d(in_channels, in_channels, 1)

        self.axial_branch = AxialFourDirectionMamba(...)
        self.diagonal_branch = DiagonalFourDirectionMamba(...)

        self.fusion = DirectionFusionGate(in_channels)

    def forward(self, A, H, V, D):
        FH, FV = self.axial_branch(
            self.pre_h(A + H),
            self.pre_v(A + V),
        )
        FD = self.diagonal_branch(self.pre_d(A + D))

        attention, direction_weights = self.fusion(A, FH, FV, FD)
        return A * attention + A
```

保留现有 `DirectionFusionGate`，第一轮不要同时修改方向融合方法，否则无法判断收益来自扫描还是融合。

---

# 9. 新增配置名称

在 `AWGM_VARIANTS` 增加：

```python
AWGM_VARIANTS = (
    "awgm_original",
    "dm_awgm_full",
    "dm_awgm_no_mamba",
    "dm_awgm_no_dcn",
    "dm_awgm_conv_only",

    "w8m_diag2_subband_shared",
    "w8m_diag4_independent",
    "w8m_diag4_pair_shared",
    "w8m_diag4_subband_shared",
    "w8m_diag4_axial_diag_shared",
    "w8m_diag4_axial_diag_shared_dir_embed",
    "w8m_diag4_all_shared",
)
```

含义：

| Variant | Axial branch | Diagonal branch | Mamba instances |
|---|---|---|---:|
| `dm_awgm_full` | 4 个独立轴向 Mamba | DCN | 4 Mamba + DCN |
| `dm_awgm_no_dcn` | 4 个独立轴向 Mamba | Conv | 4 Mamba |
| `w8m_diag2_subband_shared` | H/V 子带内共享 | 仅 NWSE/SENW | 3 |
| `w8m_diag4_independent` | 四轴向独立 | 四对角独立 | 8 |
| `w8m_diag4_pair_shared` | H、V 各自共享 | `\`、`/` 各自共享 | 4 |
| `w8m_diag4_subband_shared` | H、V 各自共享 | 四对角共享 | 3 |
| `w8m_diag4_axial_diag_shared` | 四轴向共享 | 四对角共享 | 2 |
| `w8m_diag4_axial_diag_shared_dir_embed` | 四轴向共享 + direction embedding | 四对角共享 + direction embedding | 2 |
| `w8m_diag4_all_shared` | 八方向全部共享 | 八方向全部共享 | 1 |

---

# 10. 必须完成的单元测试

## 10.1 索引正确性

对 `3×3` 输入验证四个预期序列。

## 10.2 索引完整性

对多种尺寸：

```text
3×3
4×4
3×5
5×3
16×16
32×32
64×64
```

验证：

```python
sorted(idx.tolist()) == list(range(H * W))
```

所有索引无重复、无遗漏。

## 10.3 逆排列

```python
restored = permuted[:, inverse_idx, :]
assert torch.equal(restored, original)
```

## 10.4 共享参数验证

方案四必须验证只有两个 Mamba 参数集合：

```python
assert model.axial_mamba is model.get_mamba("lr")
assert model.axial_mamba is model.get_mamba("rl")
assert model.axial_mamba is model.get_mamba("tb")
assert model.axial_mamba is model.get_mamba("bt")

assert model.diag_mamba is model.get_mamba("nwse")
assert model.diag_mamba is model.get_mamba("senw")
assert model.diag_mamba is model.get_mamba("nesw")
assert model.diag_mamba is model.get_mamba("swne")
```

## 10.5 梯度累积

同一共享 Mamba 被多次调用后：

- loss backward 成功；
- 参数梯度有限；
- 梯度不为全零；
- 四条路线的梯度共同累积到同一参数。

## 10.6 输出差异

共享参数不等于输出相同。必须验证：

```python
not torch.allclose(out_lr, out_rl)
not torch.allclose(out_nwse, out_nesw)
```

## 10.7 模型检查

每个 variant 完成：

- forward；
- backward；
- shape；
- finite value；
- parameters；
- FLOPs；
- GPU memory；
- speed；
- checkpoint save/load；
- train/test CLI。

---

# 11. 实验阶段

## Stage 0：功能验证

数据：

```text
随机输入 + 3×3 索引测试
```

目标：

- 所有 variant 可运行；
- 无 NaN/Inf；
- 共享关系正确；
- 参数量符合预期顺序。

预期参数量：

```text
independent_8
  >
pair_shared_4
  >
subband_shared_3
  >
axial_diag_shared_2
  >
all_shared_1
```

注意：FLOPs 不一定按相同比例下降，因为扫描次数仍为 8。

---

## Stage 1：NUDT-SIRST 初筛

比较：

```text
awgm_original
dm_awgm_full
dm_awgm_no_dcn
w8m_diag2_subband_shared
w8m_diag4_independent
w8m_diag4_pair_shared
w8m_diag4_subband_shared
w8m_diag4_axial_diag_shared
w8m_diag4_axial_diag_shared_dir_embed
w8m_diag4_all_shared
```

建议：

- 相同 seed；
- 相同训练配置；
- 相同评估频率；
- 第一轮可运行到 400 epochs 做故障和趋势筛选；
- 不得仅凭 400 epochs 结果作为最终结论；
- 选出前 3 个 W8M 变体继续到 1000 epochs。

重点回答：

1. Diagonal Mamba 是否优于 D-Conv 和 D-DCN？
2. 2 对角方向与 4 对角方向谁更好？
3. 参数共享是否影响性能？
4. 方向 embedding 是否能补偿共享约束？

---

## Stage 2：三数据集完整实验

选择 Stage 1 最好的两个 W8M 版本，加上：

```text
awgm_original
dm_awgm_full
dm_awgm_no_dcn
```

在以下数据集完整训练 1000 epochs：

```text
NUAA-SIRST
NUDT-SIRST
IRSTD-1K
```

保持现有统一设置：

- patch size: 256；
- batch size: 4；
- seed: 42；
- max epochs: 1000；
- 100 epoch 开始评估；
- 每 5 epoch 评估；
- 每 20 epoch 保存；
- mIoU、nIoU、F1、Pd、Fa。

---

## Stage 3：多随机种子复验

最终最佳方案至少运行：

```text
seed = 42, 3407, 2026
```

报告：

```text
mean ± standard deviation
```

至少包括：

```text
mIoU
nIoU
F1
Pd
Fa
```

---

# 12. 关键消融表

## 12.1 对角建模方式

| H/V | D | 目的 |
|---|---|---|
| Mamba | Conv | 当前 `no_dcn` |
| Mamba | DCN | 当前 `full` |
| Mamba | 2-direction diagonal Mamba | 验证单一斜轴 |
| Mamba | 4-direction diagonal Mamba | 验证完整斜向覆盖 |

## 12.2 参数共享方式

| Sharing | Mamba 数量 | 目的 |
|---|---:|---|
| independent | 8 | 最大表达能力 |
| pair shared | 4 | 正反方向共享 |
| subband shared | 3 | H/V/D 子带级共享 |
| axial + diagonal shared | 2 | 用户提出的方案四 |
| all shared | 1 | 共享下界 |

## 12.3 方向条件

| Shared Mamba | Direction embedding | 目的 |
|---|---|---|
| Yes | No | 仅依靠排列顺序 |
| Yes | Yes | 共享参数但显式编码方向 |

## 12.4 对角序列组织

| Order | 说明 |
|---|---|
| concat | 同方向对角线依次拼接 |
| snake | 相邻对角线交替方向，减少边界跳跃 |

---

# 13. 结果记录要求

更新 `EXPERIMENT_RECORD.md`，为每个实验记录：

```text
variant
dataset
seed
status
latest epoch
best epoch
mIoU
nIoU
F1
Pd
Fa
parameter count
FLOPs
inference latency
FPS
peak GPU memory
Mamba backend
diagonal order
sharing mode
direction embedding
checkpoint path
log path
```

同时保存每个方向权重的统计：

```text
mean G_H
mean G_V
mean G_D
```

建议进一步保存：

```text
axial/diagonal branch feature norm
attention map mean/std
```

用于判断模型是否过度依赖某一分支。

---

# 14. 可视化要求

至少选取以下场景：

1. 孤立点目标；
2. 斜向云边缘；
3. 建筑斜边；
4. 海天线；
5. 多目标；
6. 弱目标；
7. 高噪声背景。

输出：

```text
Input
GT
AWGM original prediction
DM-AWGM DCN prediction
W8M prediction
H/V/D wavelet maps
H/V/D direction weights
final attention map
```

重点观察：

- 斜向背景边缘是否被误报；
- D 分支是否增强孤立目标；
- 对角 Mamba 是否比 DCN 更少捕获背景纹理；
- 参数共享后方向权重是否退化为近似常数。

---

# 15. 判定标准

## 支持对角 Mamba 的条件

至少满足：

1. 在两个及以上数据集上优于 `dm_awgm_no_dcn`；
2. mIoU/nIoU 提升不是以 Fa 明显上升为代价；
3. 多 seed 结果稳定；
4. 对斜向复杂背景具有可解释的假警抑制；
5. 性能增益能够覆盖额外延迟。

## 支持方案四的条件

`w8m_diag4_axial_diag_shared` 相比 `w8m_diag4_subband_shared`：

- 参数量显著降低；
- mIoU 下降不超过约 0.2–0.3 个百分点，或反而提升；
- Fa 不恶化；
- 多 seed 方差不增大。

如果满足，可将方案四作为主轻量版本。

如果方案四明显下降，则保留方案三：

```text
H 使用 h_mamba
V 使用 v_mamba
D 使用 diag_mamba
```

---

# 16. 推荐执行优先级

第一优先级：

```text
w8m_diag4_subband_shared
w8m_diag4_axial_diag_shared
w8m_diag4_axial_diag_shared_dir_embed
```

第二优先级：

```text
w8m_diag2_subband_shared
w8m_diag4_pair_shared
```

第三优先级：

```text
w8m_diag4_independent
w8m_diag4_all_shared
```

原因：

- 方案三和方案四最符合性能—参数效率平衡；
- 八个独立 Mamba 代价较高；
- 全共享主要是边界消融。

---

# 17. Codex 交付要求

Codex 完成后必须：

1. 修改 `model/DWTFreqNet.py`；
2. 增加 W8M variants；
3. 保持 `awgm_original` 默认行为不变；
4. 不破坏现有 checkpoint 推理；
5. 增加对角索引单元测试；
6. 扩展 `tools/smoke_test_dm_awgm.py`；
7. 增加运行脚本；
8. 更新 README；
9. 更新 `EXPERIMENT_RECORD.md`；
10. 提交并推送 GitHub；
11. 报告 commit SHA、文件清单、测试结果和正在运行的实验。

建议提交信息：

```text
Add wavelet-aligned diagonal Mamba scan variants
```

---

# 18. 最终建议

用户提出的方案四：

```text
LR/RL/TB/BT 共用 axial_mamba
NWSE/SENW/NESW/SWNE 共用 diag_mamba
```

具有明确的研究意义，不应只作为“省参数技巧”处理。它实际上检验了一个重要假设：

> 水平和垂直小波子带是否可以共享统一的轴向状态空间动力学，而轴向与斜向之间是否需要不同的状态空间模型。

推荐将：

```text
w8m_diag4_subband_shared
```

作为高表达版本，将：

```text
w8m_diag4_axial_diag_shared
```

作为参数高效版本，同时开展对比。
