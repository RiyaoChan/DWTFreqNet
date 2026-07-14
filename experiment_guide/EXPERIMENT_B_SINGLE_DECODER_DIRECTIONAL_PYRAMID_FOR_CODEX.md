# Experiment B：单一 Wavelet Decoder + Stage-wise AWGM + Directional Top-down Pyramid

## 1. 实验目标

Experiment A 表明，简化嵌套 Low-Frequency Local Feature Encoder 后，模型精度整体变化不大，部分数据集还有提升。因此下一步进一步消除结构重复：

- 删除 WULLE 内部 `E4 → D3 → D2 → D1` 解码路径；
- 只保留一套最终 Wavelet Decoder；
- AWGM 前移到第一次 DWT 后；
- 用同一次 DWT 的方向高频 `H/V/D` 调制同源低频 `A`；
- 用 Directional Top-down Pyramid 聚合多尺度方向高频；
- Pyramid 输出用于校准唯一 decoder 各级 IDWT 的高频系数；
- 删除第二次 DWT、后置 AWGM 和当前 LDRC。

本实验回答：

1. 单 decoder 是否足够？
2. 同源高频调制同源低频是否有效？
3. 方向高频跨尺度融合是否能改善 decoder？

---

## 2. 代码隔离

基于：

```text
Repository: RiyaoChan/DWTFreqNet
Base branch: codex/experiment-a-wulle-v2
```

Codex 开始前执行：

```bash
git checkout codex/experiment-a-wulle-v2
git pull
git rev-parse HEAD
```

新建分支：

```text
codex/experiment-b-single-decoder-directional-pyramid
```

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
```

新建：

```text
model/DWTFreqNet_SingleDecoder.py
```

模型类：

```python
class DWTFreqNet_SingleDecoder(nn.Module):
    ...
```

不要继承完整的 `DWTFreqNet` 或 `DWTFreqNet_WULLE` 再删除大量模块。新模型只构造实际使用的组件。

可复用：

```python
from model.DWTFreqNet import (
    Res_block,
    HaarWaveletTransform,
    InverseHaarWaveletTransform,
    check_haar_direction_correspondence,
)
```

---

## 3. 四个结构变体

在同一个新模型文件实现：

```python
SINGLE_DECODER_VARIANTS = (
    "sd_raw",
    "sd_awgm",
    "sd_pyramid",
    "sd_full",
)
```

| Variant | Stage-wise AWGM | Directional Pyramid | Decoder 高频系数 |
|---|---:|---:|---|
| `sd_raw` | No | No | 原始 H/V/D 经通道对齐 |
| `sd_awgm` | Yes | No | 原始 H/V/D 经通道对齐 |
| `sd_pyramid` | No | Yes | 原始系数 + Pyramid 残差 |
| `sd_full` | Yes | Yes | 原始系数 + Pyramid 残差 |

比较关系：

```text
sd_raw vs sd_awgm       → AWGM贡献
sd_raw vs sd_pyramid    → Pyramid贡献
sd_awgm vs sd_full      → 在AWGM基础上增加Pyramid
sd_pyramid vs sd_full   → 在Pyramid基础上增加AWGM
```

---

## 4. 总体结构

```text
Input
  │
  ▼
Stem X0
  │
  ▼ DWT-1
A1 / H1 / V1 / D1
 │        │
 │        ▼
 │   Direction Encoder-1
 │   F1H / F1V / F1D
 │        │
 └─ AWGM-1
      │
     A1'
      │
     E1
      │
      ▼ DWT-2
A2 / H2 / V2 / D2
 │        │
 └─ AWGM-2
      │
     E2
      │
      ▼ DWT-3
A3 / H3 / V3 / D3
 │        │
 └─ AWGM-3
      │
     E3
      │
      ▼ DWT-4
A4 / H4 / V4 / D4
 │        │
 └─ AWGM-4
      │
     E4

F1–F4
  │
  ▼
Directional Top-down Pyramid
  │
P1H/P1V/P1D
P2H/P2V/P2D
P3H/P3V/P3D
P4H/P4V/P4D
  │
  ▼
Coefficient Refinement
  │
Ĉ1H/V/D ... Ĉ4H/V/D
  │
  ▼
Single Wavelet Decoder
E4 → L3 → L2 → L1 → L0
 ↑     ↑     ↑     ↑
E3    E2    E1    X0
  │
  ▼
Prediction
```

---

## 5. 编码器

通道：

```text
X0: 32
E1: 64
E2: 128
E3: 256
E4: 256
```

### Stage 1

```python
X0 = self.inc(x)
A1, H1, V1, D1 = self.har(X0)
F1H, F1V, F1D = self.dir_encoder1(H1, V1, D1)
A1g = self.stage_awgm1(A1, F1H, F1V, F1D)
E1 = self.local_encoder1(A1g)
```

形状：

```text
A1/H1/V1/D1: [B, 32, H/2, W/2]
E1:          [B, 64, H/2, W/2]
```

### Stage 2

```python
A2, H2, V2, D2 = self.har(E1)
F2H, F2V, F2D = self.dir_encoder2(H2, V2, D2)
A2g = self.stage_awgm2(A2, F2H, F2V, F2D)
E2 = self.local_encoder2(A2g)
```

```text
A2/H2/V2/D2: [B, 64, H/4, W/4]
E2:          [B, 128, H/4, W/4]
```

### Stage 3

```python
A3, H3, V3, D3 = self.har(E2)
F3H, F3V, F3D = self.dir_encoder3(H3, V3, D3)
A3g = self.stage_awgm3(A3, F3H, F3V, F3D)
E3 = self.local_encoder3(A3g)
```

```text
A3/H3/V3/D3: [B, 128, H/8, W/8]
E3:          [B, 256, H/8, W/8]
```

### Stage 4

```python
A4, H4, V4, D4 = self.har(E3)
F4H, F4V, F4D = self.dir_encoder4(H4, V4, D4)
A4g = self.stage_awgm4(A4, F4H, F4V, F4D)
E4 = self.local_encoder4(A4g)
```

```text
A4/H4/V4/D4: [B, 256, H/16, W/16]
E4:          [B, 256, H/16, W/16]
```

---

## 6. AWGM调制的低频定义

本实验 AWGM 只调制：

\[
A_s=\operatorname{DWT}(X_{s-1})_{\mathrm{LL}}
\]

即第一次 DWT 直接得到的 LL 系数。

不得调制：

```text
E_s：卷积后的低频分支语义特征
DWT(G_s)_LL：高频全局特征再次DWT后的低频
```

核心关系：

```text
同一次DWT产生的 H_s/V_s/D_s
            ↓
         方向编码
            ↓
         调制 A_s
```

---

## 7. Directional Band Encoder

新建：

```python
class DirectionalBandEncoder(nn.Module):
    ...
```

三个方向必须显式独立，不允许先拼接后统一卷积。

当前仓库约定：

```text
H = LH，响应垂直结构
V = HL，响应水平结构
D = HH，对角结构
```

第一版不用 Mamba，先采用轻量方向卷积：

```text
H分支：Depthwise Conv 5×1
V分支：Depthwise Conv 1×5
D分支：Depthwise Conv 3×3
```

每个分支后接：

```text
Pointwise Conv 1×1
BatchNorm
GELU
Residual
```

示例：

```python
F_H = H + self.h_branch(H)
F_V = V + self.v_branch(V)
F_D = D + self.d_branch(D)
```

---

## 8. Stage-wise AWGM

新建：

```python
class StageWiseAWGM(nn.Module):
    ...
```

输入：

```text
A, F_H, F_V, F_D
```

### 方向权重

```python
logits = self.direction_gate(
    torch.cat([A, F_H, F_V, F_D], dim=1)
)
weights = torch.softmax(logits, dim=1)
w_H, w_V, w_D = torch.chunk(weights, 3, dim=1)
```

### 高频聚合

\[
F_{\mathrm{HF}}
=
w_HF_H+w_VF_V+w_DF_D
\]

### 双向调制

```python
gate = torch.tanh(
    self.modulation_gate(
        torch.cat([A, F_HF], dim=1)
    )
)
```

\[
A' = A\odot(1+\alpha\odot gate)
\]

其中：

```python
self.alpha = nn.Parameter(
    torch.full((1, channels, 1, 1), 0.1)
)
```

`tanh`允许增强和抑制，而不是只放大。

变体开关：

```text
sd_raw、sd_pyramid：A_guided = A
sd_awgm、sd_full：启用Stage-wise AWGM
```

---

## 9. Directional Top-down Pyramid

新建：

```python
class DirectionalTopDownPyramid(nn.Module):
    ...
```

输入：

```text
F1H/F1V/F1D
F2H/F2V/F2D
F3H/F3V/F3D
F4H/F4V/F4D
```

输出：

```text
P1: 64 channels,  H/2
P2: 128 channels, H/4
P3: 256 channels, H/8
P4: 256 channels, H/16
```

每个方向独立 top-down：

```python
P4_b = lateral4_b(F4_b)

P3_b = fuse3_b(torch.cat([
    lateral3_b(F3_b),
    F.interpolate(P4_b, scale_factor=2, mode="bilinear"),
], dim=1))

P2_b = fuse2_b(torch.cat([
    lateral2_b(F2_b),
    F.interpolate(reduce3to2_b(P3_b), scale_factor=2, mode="bilinear"),
], dim=1))

P1_b = fuse1_b(torch.cat([
    lateral1_b(F1_b),
    F.interpolate(reduce2to1_b(P2_b), scale_factor=2, mode="bilinear"),
], dim=1))
```

其中 \(b\in\{H,V,D\}\)。

禁止：

```python
torch.cat([P_H, P_V, P_D], dim=1)
```

再统一卷积。H/V/D三条路径必须独立到系数生成阶段。

变体：

```text
sd_raw、sd_awgm：不启用Pyramid
sd_pyramid、sd_full：启用Pyramid
```

---

## 10. 高频系数残差校准

Pyramid 特征不是严格 Haar 系数，不能直接送入 IDWT。

每个尺度、每个方向设置独立 head。

### 原始系数对齐

```python
base_H_s = self.align_H_s(H_s)
base_V_s = self.align_V_s(V_s)
base_D_s = self.align_D_s(D_s)
```

原始通道：

```text
Stage1: 32
Stage2: 64
Stage3: 128
Stage4: 256
```

decoder系数通道：

```text
Stage1: 64
Stage2: 128
Stage3: 256
Stage4: 256
```

### Pyramid残差

```python
delta_H_s = torch.tanh(self.delta_H_s(P_s_H))
delta_V_s = torch.tanh(self.delta_V_s(P_s_V))
delta_D_s = torch.tanh(self.delta_D_s(P_s_D))
```

\[
\widehat H_s=\operatorname{Align}(H_s)+\beta_s^H\Delta H_s
\]

\[
\widehat V_s=\operatorname{Align}(V_s)+\beta_s^V\Delta V_s
\]

\[
\widehat D_s=\operatorname{Align}(D_s)+\beta_s^D\Delta D_s
\]

```python
beta = nn.Parameter(
    torch.full((1, channels, 1, 1), 0.1)
)
```

无 Pyramid 时：

```python
coef_H_s = align_H_s(H_s)
coef_V_s = align_V_s(V_s)
coef_D_s = align_D_s(D_s)
```

Pyramid 的功能是校准原始小波系数，而不是凭空生成系数。

---

## 11. 唯一 Wavelet Decoder

### H/16 → H/8

```python
U3 = self.inversehar(E4, coef_H4, coef_V4, coef_D4)
L3 = self.decoder_fuse3(torch.cat([U3, E3], dim=1))
```

```text
decoder_fuse3: ResBlock(512, 256)
L3: [B, 256, H/8, W/8]
```

### H/8 → H/4

```python
U2 = self.inversehar(L3, coef_H3, coef_V3, coef_D3)
L2 = self.decoder_fuse2(torch.cat([U2, E2], dim=1))
```

```text
decoder_fuse2: ResBlock(384, 128)
L2: [B, 128, H/4, W/4]
```

### H/4 → H/2

```python
U1 = self.inversehar(L2, coef_H2, coef_V2, coef_D2)
L1 = self.decoder_fuse1(torch.cat([U1, E1], dim=1))
```

```text
decoder_fuse1: ResBlock(192, 64)
L1: [B, 64, H/2, W/2]
```

### H/2 → H

```python
U0 = self.inversehar(L1, coef_H1, coef_V1, coef_D1)
L0 = self.decoder_fuse0(torch.cat([U0, X0], dim=1))
out = self.out_head(L0)
```

```text
decoder_fuse0: ResBlock(96, 32)
L0: [B, 32, H, W]
```

---

## 12. Deep Supervision

保持6输出兼容当前训练逻辑：

```text
gt5：E4
gt4：L3
gt3：L2
gt2：L1
d0：多尺度融合
out：最终输出
```

模块：

```python
self.gt_conv5 = nn.Conv2d(256, 1, 1)
self.gt_conv4 = nn.Conv2d(256, 1, 1)
self.gt_conv3 = nn.Conv2d(128, 1, 1)
self.gt_conv2 = nn.Conv2d(64, 1, 1)
self.out_head = nn.Conv2d(32, 1, 1)
self.outconv = nn.Conv2d(5, 1, 1)
```

训练返回：

```python
(
    sigmoid(gt5),
    sigmoid(gt4),
    sigmoid(gt3),
    sigmoid(gt2),
    sigmoid(d0),
    sigmoid(out),
)
```

测试只返回：

```python
sigmoid(out)
```

---

## 13. 新模型中禁止出现的旧模块

不得实例化：

```text
所有嵌套local节点
wulle_decoder1/2/3
所有global dense节点
所有后置wave_att
所有TransTo/LDRC
```

具体包括：

```text
local_encoder1_2
local_encoder2_2
local_encoder3_2
local_encoder1_3
local_encoder2_3
local_encoder1_4

global_encoder1_2
global_encoder2_2
global_encoder3_2
global_encoder1_3
global_encoder2_3
global_encoder1_4

wave_att_input_t
wave_att_f1
wave_att_f2
wave_att_f3

TransTo_input
TransTo1e
TransTo2e
TransTo3e
```

forward中预期：

```text
DWT调用4次
IDWT调用4次
```

不存在第二次 DWT。

---

## 14. 独立训练入口

为避免继续扩大当前 `train_one.py`，新建：

```text
train_experiment_b.py
```

参数：

```python
parser.add_argument(
    "--sd-variant",
    required=True,
    choices=(
        "sd_raw",
        "sd_awgm",
        "sd_pyramid",
        "sd_full",
    ),
)
```

构建：

```python
model = DWTFreqNet_SingleDecoder(
    get_DWTFreqNet_config(),
    mode="train",
    deepsuper=True,
    sd_variant=args.sd_variant,
)
```

`run_config.json`记录：

```json
{
  "model_variant": "dwtfreqnet_single_decoder",
  "sd_variant": "sd_full",
  "model_base_commit": "<actual commit>",
  "single_decoder": true,
  "stage_wise_awgm": true,
  "directional_pyramid": true,
  "second_dwt": false,
  "ldrc": false,
  "mamba": false,
  "coefficient_mode": "raw_plus_directional_residual"
}
```

---

## 15. 单元测试

新增：

```text
tools/test_single_decoder_experiment_b.py
```

### 输出

输入：

```python
x = torch.randn(2, 1, 256, 256)
```

训练：

```text
6个输出，每个[2,1,256,256]
```

测试：

```text
[2,1,256,256]
```

### 中间形状

```text
X0: [2,32,256,256]
E1: [2,64,128,128]
E2: [2,128,64,64]
E3: [2,256,32,32]
E4: [2,256,16,16]

L3: [2,256,32,32]
L2: [2,128,64,64]
L1: [2,64,128,128]
L0: [2,32,256,256]
```

Pyramid变体：

```text
P1H/V/D: [2,64,128,128]
P2H/V/D: [2,128,64,64]
P3H/V/D: [2,256,32,32]
P4H/V/D: [2,256,16,16]
```

### 调用计数

```text
DWT = 4
IDWT = 4
```

### 梯度

检查：

```text
stem
local_encoder1–4
decoder_fuse0–3
coefficient align heads
output heads
```

AWGM变体额外检查：

```text
dir_encoder1–4
stage_awgm1–4
alpha参数
```

Pyramid变体额外检查：

```text
Directional Pyramid
delta heads
beta参数
```

### 方向检查

运行：

```bash
python tools/check_haar_direction_mapping.py
```

确认：

```text
H/LH → vertical
V/HL → horizontal
routing_aligned = True
```

### 旁路检查

```text
sd_raw：A_guided == A，coef == aligned raw
sd_awgm：无Pyramid残差
sd_pyramid：AWGM严格旁路
```

---

## 16. 复杂度

新增：

```text
tools/profile_single_decoder_experiment_b.py
```

统一输入：

```text
[1,1,256,256]
```

比较：

```text
Original
WULLE-A
sd_raw
sd_awgm
sd_pyramid
sd_full
```

报告：

```text
Parameters
FLOPs
Latency
FPS
Inference peak memory
Training peak memory
DWT/IDWT count
```

要求：

```text
sd_raw应低于WULLE-A
sd_full不应超过原DWTFreqNet复杂度
```

---

## 17. 正式实验

### Phase I：NUAA和NUDT 2×2消融

运行：

```text
B0: sd_raw
B1: sd_awgm
B2: sd_pyramid
B3: sd_full
```

共：

```text
2 datasets × 4 variants = 8 runs
```

设置：

```text
seed: 42
patch: 256
batch: 4
epochs: 1000
optimizer: Adam
lr: 1e-3
scheduler: CosineAnnealingLR
eta_min: 1e-5
eval start: 100
eval every: 1
save every: 20
threshold: 0.5
```

禁止：

```text
预训练权重
修改数据划分
修改loss
修改增强
```

### Phase II：IRSTD

至少运行：

```text
sd_raw
sd_full
```

若 `sd_awgm` 或 `sd_pyramid` 在Phase I优于 `sd_full`，则一并运行。

### Phase III：多seed

满足以下任一条件时：

```text
至少两个数据集不低于WULLE-A
或平均mIoU下降≤0.3个百分点且复杂度明显下降
```

运行：

```text
42, 3407, 2026
```

---

## 18. GPU调度

新增：

```text
scripts/run_experiment_b.sh
scripts/launch_experiment_b_queue.sh
```

要求：

1. 不终止当前 Experiment A 任务；
2. 只使用空闲 GPU；
3. 输出目录隔离；
4. 记录 PID、GPU、启动时间、epoch；
5. 不覆盖任何已有结果。

目录：

```text
runs/experiment_b/<dataset>/<sd_variant>/seed42
```

优先顺序：

```text
NUAA: sd_raw, sd_awgm, sd_pyramid, sd_full
NUDT: sd_raw, sd_full
NUDT: sd_awgm, sd_pyramid
IRSTD: Phase II
```

---

## 19. 实验记录

新增：

```text
EXPERIMENT_B_SINGLE_DECODER_RECORD.md
```

记录：

```text
base commit
branch
commit
测试结果
复杂度
GPU/PID
best epoch
mIoU/nIoU/F1/Pd/Fa
```

结果表：

| Dataset | Variant | AWGM | Pyramid | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | FLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | sd_raw | No | No | | | | | | | | |
| NUAA | sd_awgm | Yes | No | | | | | | | | |
| NUAA | sd_pyramid | No | Yes | | | | | | | | |
| NUAA | sd_full | Yes | Yes | | | | | | | | |

AWGM额外记录：

```text
H/V/D方向权重
gate mean/std
alpha
A_guided/A norm
```

Pyramid额外记录：

```text
P1–P4 norm
raw coefficient norm
delta norm
beta
final/raw coefficient norm
```

---

## 20. 判定逻辑

### 单decoder

```text
sd_raw vs WULLE-A
```

精度基本持平且复杂度下降，则删除WULLE内部decoder成立。

### AWGM

```text
sd_awgm vs sd_raw
sd_full vs sd_pyramid
```

两组一致提升，则同源高频调制同源低频有效。

### Pyramid

```text
sd_pyramid vs sd_raw
sd_full vs sd_awgm
```

Pd提升且Fa不恶化，则方向高频跨尺度融合有效。

### 模块冲突

若：

```text
sd_awgm > sd_raw
sd_pyramid > sd_raw
但sd_full下降
```

说明高频信息重复增强，应降低 `alpha/beta` 或增加约束。

---

## 21. 建议新增文件

```text
model/DWTFreqNet_SingleDecoder.py
train_experiment_b.py
tools/test_single_decoder_experiment_b.py
tools/profile_single_decoder_experiment_b.py
scripts/run_experiment_b.sh
scripts/launch_experiment_b_queue.sh
EXPERIMENT_B_SINGLE_DECODER_RECORD.md
```

允许最小修改：

```text
README.md
```

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
```

---

## 22. Codex交付

返回：

1. base commit；
2. 分支；
3. commit SHA；
4. 新增文件；
5. 四个variant结构；
6. 前向、反向、形状测试；
7. DWT/IDWT计数；
8. Haar方向检查；
9. 参数、FLOPs、延迟、显存；
10. 启动命令；
11. GPU/PID/输出目录；
12. 当前epoch；
13. `EXPERIMENT_B_SINGLE_DECODER_RECORD.md`。

建议 commit：

```text
Add single-decoder directional frequency experiment
```

---

## 23. 最终设计主线

```text
第一次DWT的方向高频
        │
        ├── AWGM调制同源低频A
        │
        └── Directional Pyramid跨尺度融合
                         │
                         ▼
                校准decoder高频系数
                         │
                         ▼
                 唯一Wavelet Decoder
```

AWGM调制对象：

\[
\boxed{A_s=\operatorname{DWT}(X_{s-1})_{\mathrm{LL}}}
\]

Pyramid输出用途：

\[
\boxed{
P_s^{H/V/D}
\rightarrow
\text{高频系数残差校准}
\rightarrow
\text{对应尺度IDWT}
}
\]

新模型不再存在双 decoder，也不再对 global feature 进行第二次 DWT。
