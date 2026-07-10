# DM-AWGM：用方向 Mamba + 可变形卷积替换 DWTFreqNet 中 AWGM 的代码修改与实验方案

## 0. 背景与目标

当前仓库：`RiyaoChan/DWTFreqNet`

当前分支：`main`

当前提交：`cd908ac`，提交信息：`Publish DWTFreqNet code`

现有 AWGM 在代码中主要对应 `WaveDownattention`。它的接口是：

```python
forward(self, A, H, V, D) -> o
```

其中：

- `A`：对输入特征做 Haar/DWT 后得到的低频近似分量；
- `H`：水平/横向高频分量；
- `V`：垂直/纵向高频分量；
- `D`：对角高频分量；
- `o = A * att_map + A`：最终输出的是被高频引导增强后的低频分量。

本方案目标是设计一个新的 AWGM 替代模块：

> **DM-AWGM：Direction-Matched Adaptive Wavelet Guidance Module**

核心思想：

> 小波高频子带具有方向差异。横向高频和纵向高频更适合用方向性 Mamba 扫描建模长程连续结构；对角高频包含斜向边缘、角点和不规则局部突变，更适合用可变形卷积建模局部几何变化。最终仍然生成 attention/gate 去调制低频 `A`，避免高频噪声直接进入后续 Transformer/LDRC。

---

## 1. 原始 AWGM / WaveDownattention 的逻辑

当前 `WaveDownattention` 的主要逻辑可以概括为：

```python
AH = conv_A_H(A + H)
AV = conv_A_V(A + V)
AD = conv_A_D(A + D)

AH_att = concat(channel_max(AH), channel_mean(AH))
AV_att = concat(channel_max(AV), channel_mean(AV))
AD_att = concat(channel_max(AD), channel_mean(AD))

wave_att = w_H * AH_att + w_V * AV_att + w_D * AD_att
att_map = sigmoid(conv1x1(wave_att))
o = A * att_map + A
```

其本质是：

1. 先把低频 `A` 分别和三个高频方向 `H/V/D` 相加；
2. 对每个方向做 depthwise convolution；
3. 通过 channel max pooling 和 channel mean pooling 压缩成 2 通道空间注意力描述；
4. 使用三个可学习标量权重融合方向信息；
5. 得到单通道 spatial attention map；
6. 用该 attention map 调制 `A`。

---

## 2. 原始 AWGM 为什么使用 max/mean pooling？

### 2.1 合理性

原始设计中的 max/mean pooling 实际上类似 CBAM 的 spatial attention 思路：

- **channel max pooling**：突出某个空间位置上最强的通道响应。红外小目标通常是稀疏强响应，max pooling 有利于保留这种局部峰值。
- **channel mean pooling**：描述该空间位置的整体平均激活，能够反映更稳定的背景/上下文响应。
- **concat(max, mean)**：用两个统计量同时描述“最强响应”和“平均响应”，然后用轻量 `Conv1x1 + Sigmoid` 得到空间 attention。

这种设计的优点是：

- 参数少；
- 计算量低；
- 对小目标稀疏响应比较敏感；
- 不容易因为训练样本有限而过拟合。

### 2.2 局限性

但是它也有明显不足：

1. **方向建模弱**  
   `H/V/D` 都用相同形式的 depthwise conv + max/mean pooling，不能显式利用水平、垂直、对角子带的方向差异。

2. **通道信息被过早压缩**  
   `max/mean` 直接把 `C` 个通道压成 2 个通道，可能丢失有价值的高频通道差异。

3. **方向权重是全局静态的**  
   `att_weights` 是 3 个全局可学习参数，对所有图像、所有空间位置基本共享，不能根据不同场景动态调整方向权重。

4. **缺少长程方向上下文**  
   普通卷积主要看局部，难以判断高频响应是孤立小目标，还是海天线、云边缘、建筑边缘等连续背景结构。

---

## 3. 新模块 DM-AWGM 与原 AWGM 的区别

| 对比项 | 原 AWGM / WaveDownattention | 新 DM-AWGM |
|---|---|---|
| 输入输出接口 | `A,H,V,D -> A'` | 保持不变：`A,H,V,D -> A'` |
| 高频处理方式 | H/V/D 都用 depthwise conv | H 用水平双向 Mamba，V 用垂直双向 Mamba，D 用可变形卷积 |
| 方向建模 | 只用三个静态可学习标量融合方向 | 显式按照小波子带方向属性匹配不同结构 |
| 空间权重生成 | channel max/mean pooling + Conv1x1 | 动态方向门控 / spatial-direction gate |
| 长程建模 | 弱，主要是局部卷积 | H/V 分支具备轴向长程建模能力 |
| 对角结构建模 | 普通局部卷积 | 可变形卷积自适应处理斜向边缘、不规则突变 |
| 高频噪声控制 | 输出仍作用在 A 上，较稳 | 保留 `A * gate + A`，避免高频直接失控 |

---

## 4. 新模块总体设计

模块命名建议：

```python
class DirectionMatchedAWGM(nn.Module):
    def forward(self, A, H, V, D):
        ...
        return A_enhanced
```

整体流程：

```text
A, H, V, D
   │
   ├── H branch: A + H → Conv1x1 → horizontal bidirectional Mamba → F_H
   │
   ├── V branch: A + V → Conv1x1 → vertical bidirectional Mamba   → F_V
   │
   └── D branch: A + D → Conv1x1 → deformable convolution          → F_D

F_H, F_V, F_D, A
   ↓
Dynamic Direction Fusion / Spatial-Direction Gate
   ↓
M ∈ [B,1,H,W]
   ↓
A_enhanced = A * M + A
```

推荐公式：

```text
F_H = Mamba_LR_RL(Conv(A + H))
F_V = Mamba_TB_BT(Conv(A + V))
F_D = DCN(Conv(A + D))

G = Softmax(Conv([F_H, F_V, F_D, A]), dim=direction)
F_dir = G_H * F_H + G_V * F_V + G_D * F_D
M = Sigmoid(Conv([F_dir, A]))
A' = A * M + A
```

这里的关键约束是：

> 不直接输出 `concat(F_H,F_V,F_D)`，而是生成 attention/gate 去调制 `A`。

这样既替换了 AWGM 的注意力生成方式，又保留了 AWGM “高频引导低频”的稳定性。

---

## 5. 方向 Mamba 分支设计

### 5.1 H branch：横向双向 Mamba

用于处理 `H` 子带。

输入：`X_H = Conv1x1(A + H)`，形状 `[B,C,Hs,Ws]`。

处理：

1. 按行展开：`[B,C,Hs,Ws] -> [B*Hs, Ws, C]`
2. 左到右 Mamba：`LR`
3. 右到左 Mamba：`RL`，通过 `flip` 实现
4. 两个方向融合后还原为 `[B,C,Hs,Ws]`

伪代码：

```python
class HorizontalBiMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_lr = Mamba(d_model=dim)
        self.mamba_rl = Mamba(d_model=dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        seq_n = self.norm(seq)
        y_lr = self.mamba_lr(seq_n)
        y_rl = torch.flip(self.mamba_rl(torch.flip(seq_n, dims=[1])), dims=[1])
        y = self.proj(y_lr + y_rl)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x + y
```

### 5.2 V branch：纵向双向 Mamba

用于处理 `V` 子带。

输入：`X_V = Conv1x1(A + V)`，形状 `[B,C,Hs,Ws]`。

处理：

1. 按列展开：`[B,C,Hs,Ws] -> [B*Ws, Hs, C]`
2. 上到下 Mamba：`TB`
3. 下到上 Mamba：`BT`
4. 两个方向融合后还原为 `[B,C,Hs,Ws]`

伪代码：

```python
class VerticalBiMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_tb = Mamba(d_model=dim)
        self.mamba_bt = Mamba(d_model=dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        seq_n = self.norm(seq)
        y_tb = self.mamba_tb(seq_n)
        y_bt = torch.flip(self.mamba_bt(torch.flip(seq_n, dims=[1])), dims=[1])
        y = self.proj(y_tb + y_bt)
        y = y.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        return x + y
```

### 5.3 Mamba 依赖处理

优先使用真实 `mamba_ssm`：

```python
try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None
```

如果环境不能安装 `mamba_ssm`，可以临时提供 fallback 版本用于 smoke test，但最终论文实验必须使用真实 Mamba 版本，不能把 fallback 结果当成 Mamba 结果。

建议 fallback：

```python
class FallbackSequenceMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    def forward(self, x):
        return self.ffn(x)
```

---

## 6. 对角分支：Deformable Convolution

用于处理 `D` 子带。

设计动机：

- `D` 子带主要包含斜向边缘、角点、局部突变、不规则背景纹理；
- 这些结构不一定沿水平/垂直方向连续；
- 直接用 Mamba 进行轴向扫描可能导致对角噪声沿行列方向传播；
- 可变形卷积可以自适应调整采样位置，更适合处理斜向和不规则局部结构。

推荐实现：

```python
class DeformableDiagonalBranch(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.offset = nn.Conv2d(dim, 2 * kernel_size * kernel_size, kernel_size, padding=padding)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.dcn = torchvision.ops.DeformConv2d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        off = self.offset(x)
        y = self.dcn(x, off)
        y = self.pw(self.act(self.norm(y)))
        return x + y
```

如果 `torchvision.ops.DeformConv2d` 不可用，使用普通 depthwise separable conv 作为 fallback 以保证程序能跑通，但记录日志：

```text
[Warning] torchvision.ops.DeformConv2d is unavailable. Falling back to depthwise separable conv for smoke test only.
```

---

## 7. 动态方向融合设计

### 7.1 推荐主方案：spatial-direction softmax gate

不要继续使用原始 AWGM 中的三个全局静态标量权重。建议改成对每个空间位置动态预测三个方向权重：

```python
class DirectionFusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 3, 1)
        )
        self.to_att = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, A, FH, FV, FD):
        logits = self.gate(torch.cat([A, FH, FV, FD], dim=1))
        weights = torch.softmax(logits, dim=1)  # [B,3,H,W]
        GH, GV, GD = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        Fdir = GH * FH + GV * FV + GD * FD
        M = self.to_att(torch.cat([A, Fdir], dim=1))
        return M, weights
```

优点：

- 每张图、每个空间位置都能动态选择 H/V/D 方向；
- 比原始 `att_weights` 三个全局参数更灵活；
- 仍然输出单通道 attention map，接口稳定。

### 7.2 可选替代方案

为了验证 max/mean pooling 是否是瓶颈，可以增加下面几种融合方式作为 ablation：

1. **原始 max/mean pooling 方案**  
   保留 `channel max + channel mean`，但 H/V/D 分支换成 Mamba/DCN。

2. **learned channel squeeze**  
   用 `Conv1x1(C -> C/r -> 1)` 替代 max/mean pooling。

3. **SE/ECA-style direction weighting**  
   用全局池化生成每个方向的 image-level 权重，但不做 spatially dynamic gate。

4. **cross-attention gate**  
   用 `A` 作为 query，`[FH,FV,FD]` 作为 key/value 生成 attention，计算量较大，不建议第一版使用。

5. **variance/high-frequency energy pooling**  
   用局部方差或局部能量描述高频强度，但实现复杂，建议作为后续探索。

本轮主实验建议使用 `spatial-direction softmax gate`。

---

## 8. 完整 DM-AWGM 伪代码

```python
class DirectionMatchedAWGM(nn.Module):
    def __init__(self, in_channels, use_mamba=True, use_dcn=True, fusion='spatial_softmax'):
        super().__init__()
        self.pre_h = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_v = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_d = nn.Conv2d(in_channels, in_channels, 1)

        self.h_branch = HorizontalBiMamba(in_channels) if use_mamba else ConvBranch(in_channels)
        self.v_branch = VerticalBiMamba(in_channels) if use_mamba else ConvBranch(in_channels)
        self.d_branch = DeformableDiagonalBranch(in_channels) if use_dcn else ConvBranch(in_channels)

        self.fusion = DirectionFusionGate(in_channels)

    def forward(self, A, H, V, D):
        XH = self.pre_h(A + H)
        XV = self.pre_v(A + V)
        XD = self.pre_d(A + D)

        FH = self.h_branch(XH)
        FV = self.v_branch(XV)
        FD = self.d_branch(XD)

        M, dir_weights = self.fusion(A, FH, FV, FD)
        out = A * M + A
        return out
```

注意：

- `return out` 的 shape 必须和 `A` 一致；
- 不要直接返回 `[FH,FV,FD]` 的 concat；
- 如果需要可视化，可以在 `vis=True` 或 debug 模式下额外返回 `dir_weights`，但默认 forward 保持单输出，避免破坏原训练流程。

---

## 9. 代码修改任务清单

### 9.1 新建或修改模块文件

Codex 请先搜索：

```bash
grep -R "class WaveDownattention" -n .
grep -R "wave_att_f1" -n .
grep -R "self.har" -n .
```

找到当前模型文件后，做以下修改：

1. 保留原始 `WaveDownattention`，不要删除，作为 baseline。
2. 新增：
   - `HorizontalBiMamba`
   - `VerticalBiMamba`
   - `DeformableDiagonalBranch`
   - `ConvBranch`
   - `DirectionFusionGate`
   - `DirectionMatchedAWGM`
3. 新增工厂函数：

```python
def build_wave_guidance(name, in_channels):
    if name == 'awgm_original':
        return WaveDownattention(in_channels)
    elif name == 'dm_awgm_full':
        return DirectionMatchedAWGM(in_channels, use_mamba=True, use_dcn=True, fusion='spatial_softmax')
    elif name == 'dm_awgm_no_mamba':
        return DirectionMatchedAWGM(in_channels, use_mamba=False, use_dcn=True, fusion='spatial_softmax')
    elif name == 'dm_awgm_no_dcn':
        return DirectionMatchedAWGM(in_channels, use_mamba=True, use_dcn=False, fusion='spatial_softmax')
    elif name == 'dm_awgm_conv_only':
        return DirectionMatchedAWGM(in_channels, use_mamba=False, use_dcn=False, fusion='spatial_softmax')
    else:
        raise ValueError(f'Unknown wave guidance variant: {name}')
```

### 9.2 模型参数接入

在 `DWTFreqNet.__init__` 中增加参数，例如：

```python
awgm_variant='awgm_original'
```

并将：

```python
self.wave_att_input_t = WaveDownattention(32)
self.wave_att_f1 = WaveDownattention(in_channels * 2)
self.wave_att_f2 = WaveDownattention(in_channels * 4)
self.wave_att_f3 = WaveDownattention(in_channels * 8)
```

替换为：

```python
self.wave_att_input_t = build_wave_guidance(awgm_variant, 32)
self.wave_att_f1 = build_wave_guidance(awgm_variant, in_channels * 2)
self.wave_att_f2 = build_wave_guidance(awgm_variant, in_channels * 4)
self.wave_att_f3 = build_wave_guidance(awgm_variant, in_channels * 8)
```

默认值必须是 `awgm_original`，以保证原始模型结果可复现。

### 9.3 命令行参数接入

在 `train.py`、`test.py` 或 config parser 中加入：

```python
parser.add_argument('--awgm_variant', type=str, default='awgm_original',
                    choices=['awgm_original', 'dm_awgm_full', 'dm_awgm_no_mamba',
                             'dm_awgm_no_dcn', 'dm_awgm_conv_only'])
```

创建模型时把该参数传入 `DWTFreqNet`。

### 9.4 依赖处理

如果使用 `mamba_ssm`：

- 不要让 import 失败直接导致原始模型不能跑；
- 只有当 `awgm_variant` 需要 Mamba 且 `mamba_ssm` 不存在时才报错，或者使用 fallback 并打印清晰警告；
- 最终正式实验必须确认日志中使用的是 `mamba_ssm.Mamba`，不是 fallback。

如果使用 `torchvision.ops.DeformConv2d`：

- 检查当前环境是否支持；
- 不支持时 fallback 到普通卷积，仅用于 smoke test；
- 正式实验需要记录是否真实使用 DCN。

---

## 10. 实验设计

### 10.1 快速 smoke test

新增脚本：

```bash
python tools/smoke_test_dm_awgm.py --awgm_variant awgm_original
python tools/smoke_test_dm_awgm.py --awgm_variant dm_awgm_full
python tools/smoke_test_dm_awgm.py --awgm_variant dm_awgm_no_mamba
python tools/smoke_test_dm_awgm.py --awgm_variant dm_awgm_no_dcn
python tools/smoke_test_dm_awgm.py --awgm_variant dm_awgm_conv_only
```

smoke test 要检查：

- forward 能跑通；
- 输出 shape 和原始模型一致；
- loss 能反传；
- 没有 NaN；
- 参数量和 FLOPs 能统计。

### 10.2 主实验第一阶段：NUDT-SIRST

先只在 NUDT-SIRST 上跑完整训练，减少试错成本：

| 实验名 | `awgm_variant` | 目的 |
|---|---|---|
| baseline | `awgm_original` | 原始 AWGM 复现 |
| conv-only | `dm_awgm_conv_only` | 验证是否只是增加参数带来收益 |
| no-mamba | `dm_awgm_no_mamba` | 只有 D 使用 DCN，H/V 不用 Mamba |
| no-dcn | `dm_awgm_no_dcn` | H/V 使用 Mamba，D 不用 DCN |
| full | `dm_awgm_full` | H/V Mamba + D DCN 完整方案 |

重点关注：

- IoU
- nIoU
- F-measure
- Pd
- Fa
- 参数量
- FLOPs
- 推理速度

### 10.3 主实验第二阶段：三数据集完整实验

如果 NUDT-SIRST 上 `dm_awgm_full` 优于 baseline，继续跑：

- NUDT-SIRST
- NUAA-SIRST
- IRSTD-1K

每个数据集至少跑：

```text
awgm_original
best_dm_awgm_variant
```

如果算力允许，补充：

```text
dm_awgm_no_mamba
dm_awgm_no_dcn
dm_awgm_conv_only
```

### 10.4 可视化分析

输出以下可视化：

1. 原始图像；
2. GT；
3. 原 AWGM 的 attention map；
4. DM-AWGM 的 attention map；
5. DM-AWGM 的方向权重图：`G_H, G_V, G_D`；
6. 预测结果；
7. False alarm 对比图。

重点看：

- 海天线/云边缘等水平背景是否被 H-Mamba 抑制；
- 垂直建筑边缘/条带噪声是否被 V-Mamba 抑制；
- 斜向边缘或角点背景是否由 D-DCN 更好区分；
- `Fa` 是否下降，而不是只提升 `Pd`。

---

## 11. 结果判定标准

这个模块是否值得保留，不能只看 IoU。建议按下面标准判断：

### 强正向结果

满足以下多数条件：

- `dm_awgm_full` 的 IoU/nIoU/F-measure 高于 `awgm_original`；
- `Fa` 明显下降或不升高；
- `Pd` 不明显下降；
- 可视化中 false alarm 被抑制；
- 参数量/FLOPs 增加可接受。

### 中性结果

- IoU 略升，但 Fa 明显升高；
- 或 Fa 下降但 Pd 明显下降；
- 或只在一个数据集提升。

这种情况下保留模块作为 ablation，不作为主贡献。

### 负向结果

- 三个数据集大部分指标低于原 AWGM；
- Fa 明显升高；
- 训练不稳定；
- 对 Mamba/DCN 依赖导致复杂度过高。

这种情况下建议退回轻量方向 gate，而不使用完整 Mamba/DCN。

---

## 12. 论文写作时的创新点表述

可以写成：

> Unlike existing wavelet-guided attention modules that fuse horizontal, vertical, and diagonal high-frequency subbands in a uniform manner, the proposed DM-AWGM performs direction-matched high-frequency modeling. Specifically, horizontal and vertical wavelet subbands are processed by paired directional Mamba scans to capture long-range axial continuity, while the diagonal subband is modeled by deformable convolution to adaptively capture oblique and irregular local structures. The resulting direction-aware representation is then used to generate a spatial guidance map for enhancing the low-frequency approximation component, thereby preserving stable target saliency while suppressing high-frequency clutter.

中文理解：

> 原 AWGM 把三个高频方向基本当作同质信息进行注意力融合；新模块根据高频子带的方向属性进行结构匹配：H/V 用方向 Mamba 建模长程轴向连续性，D 用可变形卷积建模斜向和不规则局部变化，最后仍然用方向感知权重调制低频 A，从而避免高频噪声直接放大。

---

## 13. 推荐提交信息

```bash
git checkout -b codex/dm-awgm-directional-mamba-dcn
# 修改代码并完成 smoke test / 初步实验后
git add .
git commit -m "Add direction-matched Mamba deformable AWGM variants"
git push origin codex/dm-awgm-directional-mamba-dcn
```

---

## 14. Codex 最终需要汇报的内容

请 Codex 完成后输出：

1. 修改了哪些文件；
2. 新增了哪些类和参数；
3. 是否成功使用真实 `mamba_ssm.Mamba`；
4. 是否成功使用真实 `torchvision.ops.DeformConv2d`；
5. smoke test 结果；
6. 每个 variant 的参数量/FLOPs；
7. NUDT-SIRST 初步实验结果；
8. 如果失败，明确报错位置和失败原因。
