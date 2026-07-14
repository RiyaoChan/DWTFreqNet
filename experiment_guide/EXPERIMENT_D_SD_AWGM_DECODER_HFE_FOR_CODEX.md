# Experiment D：SD-AWGM + Decoder-side HFE

## 0. 实验定位

本实验以 Experiment B 中表现较好的 `sd_awgm` 为唯一基础结构，在唯一的 Wavelet Decoder 中加入受 Wave-Mamba 启发的 Decoder-side High-Frequency Enhancement（Decoder HFE）。

核心目标：

> 编码阶段由同源高频引导低频后，解码阶段是否还需要利用已经形成的低频语义，对原始 H/V/D 高频系数进行同尺度匹配与校正。

双向但职责分离的频率交互：

```text
Encoder：H/V/D ──Stage-wise AWGM──> 同源低频 A
Decoder：当前低频语义 ──Decoder HFE──> 校正 H/V/D
```

首轮只实现和训练：

```text
sd_awgm_hfe
```

本轮不加入：

```text
LDRC
Directional Pyramid
第二次 DWT
后置 AWGM
W8M
LFSSBlock
新的损失函数
新的数据增强
```

---

## 1. 参考来源与适配原则

参考：

```text
Paper:
Wave-Mamba: Wavelet State Space Model for Ultra-High-Definition
Low-Light Image Enhancement

Repository:
https://github.com/AlexZou14/Wave-Mamba
```

Wave-Mamba 的 HFEBlock 本质是：

```text
增强后的低频 → 匹配和校正高频
```

本项目不直接复制 Wave-Mamba 的完整代码文件，而是重新实现其核心思想，并进行以下适配：

1. 保留当前项目的 Haar DWT/IWT；
2. 保留 `H=LH、V=HL、D=HH` 约定；
3. 保留 `sd_awgm` 的原始高频系数主路径；
4. HFE 只预测 H/V/D 的残差校正，不完全替换原始系数；
5. H/V/D 使用三个显式方向头；
6. 不增加跨尺度高频传播；
7. 每一级只进行同尺度低频—高频匹配；
8. 在代码注释和实验记录中注明设计参考 Wave-Mamba。

Wave-Mamba 仓库包含 CC BY-NC-SA 4.0 许可信息。若直接复用代码，应遵守署名、非商业和相同方式共享要求。本实验优先采用独立重写。

---

## 2. 基础分支与代码隔离

为隔离 HFE 与 LDRC 的影响，本实验应基于 Experiment B，而不是 Experiment C：

```text
Repository:
RiyaoChan/DWTFreqNet

Base branch:
codex/experiment-b-single-decoder-directional-pyramid

Known record commit:
435ab1827ecee4c6b83b669789bb9833a5fd5320
```

Codex 开始前执行：

```bash
git checkout codex/experiment-b-single-decoder-directional-pyramid
git pull
git rev-parse HEAD
```

以实际最新 commit 为准，并写入实验记录。

建议新分支：

```text
codex/experiment-d-sd-awgm-decoder-hfe
```

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
model/DWTFreqNet_SingleDecoder.py
model/DWTFreqNet_SingleDecoder_LDRC.py
```

新建：

```text
model/DWTFreqNet_SingleDecoder_HFE.py
```

建议类名：

```python
class DWTFreqNet_SingleDecoder_HFE(DWTFreqNet_SingleDecoder):
    ...
```

父类固定：

```python
sd_variant="sd_awgm"
```

---

## 3. 总体结构

```text
Input
  │
  ▼
Stem X0
  │
  ▼
4-stage DWT Encoder
  │
  ├── H/V/D → DirectionalBandEncoder
  │                 │
  └── A ← Stage-wise AWGM
            │
            ▼
      E1 / E2 / E3 / E4
            │
            ▼
      Single Wavelet Decoder

Stage 4:
E4 + HFE(E4, H4/V4/D4) → IDWT → L3

Stage 3:
L3 + HFE(L3, H3/V3/D3) → IDWT → L2

Stage 2:
L2 + HFE(L2, H2/V2/D2) → IDWT → L1

Stage 1:
L1 + HFE(L1, H1/V1/D1) → IDWT → L0
```

保持：

```text
DWT = 4
IDWT = 4
第二次DWT = 0
单decoder = True
```

---

## 4. 每一级 HFE 的输入与形状

```text
X0: [B,  32, H,    W]
E1: [B,  64, H/2,  W/2]
E2: [B, 128, H/4,  W/4]
E3: [B, 256, H/8,  W/8]
E4: [B, 256, H/16, W/16]
```

原始 DWT 高频通道：

```text
Stage 1: H1/V1/D1 = 32
Stage 2: H2/V2/D2 = 64
Stage 3: H3/V3/D3 = 128
Stage 4: H4/V4/D4 = 256
```

IDWT 所需通道：

```text
Stage 1: 64
Stage 2: 128
Stage 3: 256
Stage 4: 256
```

| Stage | Decoder低频输入 | 原始高频 | 对齐后通道 | 空间尺寸 |
|---|---|---|---:|---:|
| 4 | `E4` | `H4/V4/D4` | 256 | H/16 |
| 3 | `L3` | `H3/V3/D3` | 256 | H/8 |
| 2 | `L2` | `H2/V2/D2` | 128 | H/4 |
| 1 | `L1` | `H1/V1/D1` | 64 | H/2 |

注意：

```text
Stage 3 用 L3 引导，不用 E3；
Stage 2 用 L2 引导，不用 E2；
Stage 1 用 L1 引导，不用 E1。
```

---

## 5. 新增模块

在 `model/DWTFreqNet_SingleDecoder_HFE.py` 中独立实现：

```text
LayerNorm2d
SubbandSelectiveFusion
ChannelMatching
MatchingTransformation
ChannelMatchedAttention
ChannelMatchedFFN
DecoderHFEBlock
DirectionalResidualHead
DecoderHFERefiner
DWTFreqNet_SingleDecoder_HFE
```

不要从 Wave-Mamba 仓库整文件复制。

---

## 6. LayerNorm2d

```python
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (
            self.weight.view(1, -1, 1, 1) * x
            + self.bias.view(1, -1, 1, 1)
        )
```

要求支持 AMP，不使用自定义 autograd Function。

---

## 7. SubbandSelectiveFusion

参考 SKFF，将对齐后的 H/V/D 融合成共享高频表示。

输入：

```text
base_H/base_V/base_D: [B,C,h,w]
```

输出：

```text
shared_HF: [B,C,h,w]
weights:   [B,3,C,1,1]
```

公式：

\[
F_{HF}=w_H\odot H+w_V\odot V+w_D\odot D
\]

建议实现：

```python
class SubbandSelectiveFusion(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.PReLU(),
        )

        self.to_h = nn.Conv2d(hidden, channels, 1, bias=False)
        self.to_v = nn.Conv2d(hidden, channels, 1, bias=False)
        self.to_d = nn.Conv2d(hidden, channels, 1, bias=False)

        self.last_weights = None

    def forward(self, band_h, band_v, band_d):
        stacked = torch.stack([band_h, band_v, band_d], dim=1)
        descriptor = self.reduce(band_h + band_v + band_d)

        logits = torch.stack(
            [
                self.to_h(descriptor),
                self.to_v(descriptor),
                self.to_d(descriptor),
            ],
            dim=1,
        )

        weights = torch.softmax(logits, dim=1)
        fused = (stacked * weights).sum(dim=1)

        self.last_weights = weights.detach()
        return fused
```

方向约定不得交换：

```text
H = LH
V = HL
D = HH
```

---

## 8. ChannelMatching

该 matching 是通道匹配，不是像素匹配。

输入：

```text
query_feature:     [B,C,H,W]，高频
candidate_feature: [B,C,H,W]，当前decoder低频
```

展平：

```python
query = query_feature.flatten(2)
candidate = candidate_feature.flatten(2)
```

L2距离：

```python
distance = torch.cdist(
    query.float(),
    candidate.float(),
)
```

得到：

```text
[B,C,C]
```

每个高频通道选择最相似低频通道：

```python
indices = distance.argmin(dim=-1)
```

返回：

```text
selected: [B,C,H,W]
indices:  [B,C]
```

要求：

1. `match_factor=1`；
2. 每个高频通道选择一个低频通道；
3. 不降低通道数；
4. indices 不参与梯度；
5. selected 保持梯度；
6. `torch.cdist` 使用 float32；
7. 记录 selected index unique ratio。

---

## 9. MatchingTransformation

```text
高频 x
  │
  ├── 从低频 perception 中选择匹配通道
  │
  ▼
selected_low
  │
concat[x, selected_low]
  ├── 1×1 Conv + Sigmoid → gate
  └── depthwise 3×3 Conv → value
  ▼
gate × value
  ▼
1×1 Conv：2C→C
```

建议接口：

```python
class MatchingTransformation(nn.Module):
    def __init__(self, channels):
        ...

    def forward(self, x, perception):
        ...
        return output
```

保存：

```text
last_indices
last_unique_ratio
```

---

## 10. ChannelMatchedAttention

高频生成 Q/K/V：

```text
1×1 Conv → depthwise 3×3 Conv → Q/K/V
```

低频修改 Q：

```text
Q = MatchingTransformation(Q, low)
```

再做通道注意力：

```text
q/k/v: [B,heads,C_head,HW]
attention: [B,heads,C_head,C_head]
```

公式：

\[
A=\operatorname{Softmax}(QK^T\cdot	au)
\]

\[
Y=AV
\]

首轮 heads：

```text
Stage1: 1
Stage2: 2
Stage3: 4
Stage4: 4
```

必须断言：

```python
channels % num_heads == 0
```

---

## 11. ChannelMatchedFFN

结构：

```text
高频
  ↓ 1×1 Conv
  ↓ depthwise 3×3 Conv
  ↓ MatchingTransformation(高频, 低频)
  ↓ depthwise 3×3 Conv
  ↓ GELU
  ↓ 1×1 Conv
```

首轮：

```text
expansion = 1.0
```

---

## 12. DecoderHFEBlock

\[
F_H'
=
F_H+
\operatorname{CMTA}
(\operatorname{LN}(F_H),\operatorname{LN}(F_L))
\]

\[
F_H^{out}
=
F_H'+
\operatorname{CMFFN}
(\operatorname{LN}(F_H'),\operatorname{LN}(F_L))
\]

建议：

```python
class DecoderHFEBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.high_norm1 = LayerNorm2d(channels)
        self.high_norm2 = LayerNorm2d(channels)
        self.low_norm = LayerNorm2d(channels)

        self.attn = ChannelMatchedAttention(
            channels,
            num_heads=num_heads,
        )

        self.ffn = ChannelMatchedFFN(
            channels,
            expansion=1.0,
        )

    def forward(self, high_feature, low_feature):
        low = self.low_norm(low_feature)

        high_feature = high_feature + self.attn(
            self.high_norm1(high_feature),
            low,
        )

        high_feature = high_feature + self.ffn(
            self.high_norm2(high_feature),
            low,
        )

        return high_feature
```

---

## 13. DirectionalResidualHead

SKFF + HFE 得到共享高频，但最终 IDWT 需要 H/V/D 三组系数。

```python
class DirectionalResidualHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, base_direction, refined_shared):
        return self.net(
            torch.cat(
                [base_direction, refined_shared],
                dim=1,
            )
        )
```

分别实例化：

```text
head_H
head_V
head_D
```

禁止使用：

```python
single_conv(shared).chunk(3, dim=1)
```

---

## 14. DecoderHFERefiner

每个 decoder stage 使用一个独立模块。

```python
class DecoderHFERefiner(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()

        self.subband_fusion = SubbandSelectiveFusion(
            channels,
            reduction=8,
        )

        self.hfe = DecoderHFEBlock(
            channels,
            num_heads=num_heads,
        )

        self.head_h = DirectionalResidualHead(channels)
        self.head_v = DirectionalResidualHead(channels)
        self.head_d = DirectionalResidualHead(channels)

        self.beta_h = nn.Parameter(
            torch.full((1, channels, 1, 1), 1e-3)
        )
        self.beta_v = nn.Parameter(
            torch.full((1, channels, 1, 1), 1e-3)
        )
        self.beta_d = nn.Parameter(
            torch.full((1, channels, 1, 1), 1e-3)
        )

    def forward(self, low_feature, base_h, base_v, base_d):
        shared_hf = self.subband_fusion(
            base_h,
            base_v,
            base_d,
        )

        refined_hf = self.hfe(
            shared_hf,
            low_feature,
        )

        delta_h = self.head_h(base_h, refined_hf)
        delta_v = self.head_v(base_v, refined_hf)
        delta_d = self.head_d(base_d, refined_hf)

        coef_h = base_h + self.beta_h * delta_h
        coef_v = base_v + self.beta_v * delta_v
        coef_d = base_d + self.beta_d * delta_d

        return (
            coef_h,
            coef_v,
            coef_d,
            {
                "shared_hf": shared_hf,
                "refined_hf": refined_hf,
                "delta_h": delta_h,
                "delta_v": delta_v,
                "delta_d": delta_d,
            },
        )
```

不对 delta 使用 `tanh`。beta 已控制残差强度。

---

## 15. 模型构造

```python
class DWTFreqNet_SingleDecoder_HFE(
    DWTFreqNet_SingleDecoder
):
    def __init__(...):
        super().__init__(
            ...,
            sd_variant="sd_awgm",
        )

        self.decoder_hfe1 = DecoderHFERefiner(64, 1)
        self.decoder_hfe2 = DecoderHFERefiner(128, 2)
        self.decoder_hfe3 = DecoderHFERefiner(256, 4)
        self.decoder_hfe4 = DecoderHFERefiner(256, 4)
```

元数据：

```python
self.model_variant = "dwtfreqnet_single_decoder_hfe"
self.sd_variant = "sd_awgm_hfe"
self.single_decoder = True
self.stage_wise_awgm = True
self.decoder_hfe = True
self.decoder_hfe_source = "wave_mamba_inspired"
self.decoder_hfe_matching = "hard_l2_channel_top1"
self.decoder_hfe_subband_fusion = "channel_skff"
self.decoder_hfe_residual = True
self.directional_pyramid = False
self.second_dwt = False
self.ldrc = False
self.mamba = False
self.coefficient_mode = (
    "aligned_raw_plus_hfe_directional_residual"
)
```

---

## 16. 原始系数对齐

不要调用父类 `_refine_coefficients()` 生成最终系数，因为 HFE 需要逐级使用 `E4/L3/L2/L1`。

```python
def _align_stage_coefficients(
    self,
    stage,
    raw_bands,
):
    _, raw_h, raw_v, raw_d = raw_bands

    base_h = getattr(self, f"align_H{stage}")(raw_h)
    base_v = getattr(self, f"align_V{stage}")(raw_v)
    base_d = getattr(self, f"align_D{stage}")(raw_d)

    return base_h, base_v, base_d
```

---

## 17. 新模型 forward

编码部分完全复用 `sd_awgm`：

```python
x0 = self.stem(x)

encoded = {}
raw_bands = {}
directional = {}
guided = {}

current = x0

for stage in range(1, 5):
    (
        encoded[stage],
        raw_bands[stage],
        directional[stage],
        guided[stage],
    ) = self._encode_stage(stage, current)

    current = encoded[stage]
```

对齐：

```python
base1 = self._align_stage_coefficients(1, raw_bands[1])
base2 = self._align_stage_coefficients(2, raw_bands[2])
base3 = self._align_stage_coefficients(3, raw_bands[3])
base4 = self._align_stage_coefficients(4, raw_bands[4])
```

Stage 4：

```python
coef_h4, coef_v4, coef_d4, debug4 = (
    self.decoder_hfe4(
        encoded[4],
        *base4,
    )
)

u3 = self._idwt(
    encoded[4],
    coef_h4,
    coef_v4,
    coef_d4,
)

l3 = self.decoder_fuse3(
    torch.cat([u3, encoded[3]], dim=1)
)
```

Stage 3：

```python
coef_h3, coef_v3, coef_d3, debug3 = (
    self.decoder_hfe3(
        l3,
        *base3,
    )
)

u2 = self._idwt(
    l3,
    coef_h3,
    coef_v3,
    coef_d3,
)

l2 = self.decoder_fuse2(
    torch.cat([u2, encoded[2]], dim=1)
)
```

Stage 2：

```python
coef_h2, coef_v2, coef_d2, debug2 = (
    self.decoder_hfe2(
        l2,
        *base2,
    )
)

u1 = self._idwt(
    l2,
    coef_h2,
    coef_v2,
    coef_d2,
)

l1 = self.decoder_fuse1(
    torch.cat([u1, encoded[1]], dim=1)
)
```

Stage 1：

```python
coef_h1, coef_v1, coef_d1, debug1 = (
    self.decoder_hfe1(
        l1,
        *base1,
    )
)

u0 = self._idwt(
    l1,
    coef_h1,
    coef_v1,
    coef_d1,
)

l0 = self.decoder_fuse0(
    torch.cat([u0, x0], dim=1)
)

out = self.out_head(l0)
```

---

## 18. Deep Supervision

保持当前 `sd_awgm` 的6输出形式：

```text
gt5: E4
gt4: L3
gt3: L2
gt2: L1
d0
out
```

不要修改现有 loss。

---

## 19. 输入断言

```python
def assert_hfe_inputs(
    low,
    band_h,
    band_v,
    band_d,
    stage,
):
    expected = low.shape

    for name, tensor in (
        ("H", band_h),
        ("V", band_v),
        ("D", band_d),
    ):
        if tensor.shape != expected:
            raise RuntimeError(
                f"Decoder HFE stage {stage}: "
                f"{name} shape {tuple(tensor.shape)} "
                f"!= low shape {tuple(expected)}"
            )
```

输入 H/W 必须能被16整除。

---

## 20. 独立训练入口

新建：

```text
train_experiment_d.py
```

模型固定：

```python
model = DWTFreqNet_SingleDecoder_HFE(
    get_DWTFreqNet_config(),
    mode=mode,
    deepsuper=True,
)
```

`run_config.json` 至少记录：

```json
{
  "model_variant": "dwtfreqnet_single_decoder_hfe",
  "sd_variant": "sd_awgm_hfe",
  "single_decoder": true,
  "stage_wise_awgm": true,
  "decoder_hfe": true,
  "decoder_hfe_matching": "hard_l2_channel_top1",
  "decoder_hfe_subband_fusion": "channel_skff",
  "decoder_hfe_direction_heads": true,
  "decoder_hfe_beta_init": 0.001,
  "directional_pyramid": false,
  "second_dwt": false,
  "ldrc": false,
  "mamba": false,
  "coefficient_mode": "aligned_raw_plus_hfe_directional_residual"
}
```

---

## 21. Checkpoint 隔离

输出目录：

```text
runs/experiment_d/<dataset>/sd_awgm_hfe/seed42
```

正式实验随机初始化，不加载：

```text
sd_awgm checkpoint
sd_awgm_ldrc checkpoint
原始模型 checkpoint
WULLE checkpoint
```

---

## 22. 单元测试

新建：

```text
tools/test_sd_awgm_hfe_experiment_d.py
```

必须包括：

### 22.1 输出形状

输入：

```python
x = torch.randn(2, 1, 256, 256)
```

训练：

```text
6 × [2,1,256,256]
```

测试：

```text
[2,1,256,256]
```

### 22.2 中间形状

```text
E1 [2,64,128,128]
E2 [2,128,64,64]
E3 [2,256,32,32]
E4 [2,256,16,16]

HFE4 shared/refined [2,256,16,16]
HFE3 shared/refined [2,256,32,32]
HFE2 shared/refined [2,128,64,64]
HFE1 shared/refined [2,64,128,128]

L3 [2,256,32,32]
L2 [2,128,64,64]
L1 [2,64,128,128]
L0 [2,32,256,256]
```

### 22.3 DWT/IDWT计数

```text
DWT = 4
IDWT = 4
```

### 22.4 近似baseline回归

构建：

```text
baseline = DWTFreqNet_SingleDecoder(sd_awgm)
hfe_model = DWTFreqNet_SingleDecoder_HFE()
```

复制共同参数，临时将全部 beta 置零，eval模式检查：

```python
torch.testing.assert_close(
    baseline_output,
    hfe_output,
    rtol=1e-5,
    atol=1e-6,
)
```

### 22.5 Matching测试

```text
distance: [B,C,C]
indices:  [B,C]
selected: [B,C,H,W]
indices范围为[0,C-1]
selected保留梯度
```

### 22.6 梯度测试

检查非零梯度：

```text
stem
local_encoder1–4
dir_encoder1–4
stage_awgm1–4
align_H/V/D 1–4

decoder_hfe1–4.subband_fusion
decoder_hfe1–4.hfe.attn
decoder_hfe1–4.hfe.ffn
decoder_hfe1–4.head_h/v/d
decoder_hfe1–4.beta_h/v/d

decoder_fuse0–3
output heads
```

### 22.7 Haar方向

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

---

## 23. 复杂度统计

新建：

```text
tools/profile_sd_awgm_hfe_experiment_d.py
```

比较：

```text
Original DWTFreqNet
WULLE-A
sd_awgm
sd_awgm_hfe
```

统一输入：

```text
[1,1,256,256]
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
HFE parameter count
```

注意：

```text
THOP/FVCore可能不能准确统计torch.cdist；
必须额外报告实测latency和显存。
```

---

## 24. 诊断统计

每一级记录：

### SKFF

```text
mean_weight_H
mean_weight_V
mean_weight_D
weight variance
```

### Matching

```text
matching_unique_ratio
most_selected_channel_frequency
mean_channel_reuse
```

### HFE

```text
shared_HF_norm
refined_HF_norm
refined/shared norm ratio
```

### 高频残差

```text
beta_H/V/D mean
delta/base norm ratio
beta*delta/base norm ratio
final/base norm ratio
```

重点监控：

\[
r_H=
rac{\|eta_H\Delta H\|}{\|H_{base}\|}
\]

V、D同理。

---

## 25. 正式实验

首轮只训练：

```text
sd_awgm_hfe
```

对照使用已有：

```text
sd_awgm
```

数据集顺序：

```text
1. NUDT-SIRST
2. NUAA-SIRST
3. IRSTD-1K
```

统一设置：

```text
seed: 42
patch size: 256
batch size: 4
epochs: 1000
optimizer: Adam
initial lr: 1e-3
scheduler: CosineAnnealingLR
eta_min: 1e-5
eval start: 100
eval every: 1
save every: 20
threshold: 0.5
```

必须与 `sd_awgm` 保持一致的数据划分、loss、增强、优化器和评价流程。

---

## 26. GPU调度

新建：

```text
scripts/run_experiment_d.sh
scripts/launch_experiment_d_queue.sh
```

要求：

1. 不终止 Experiment A/B/C 或 W8M 正式任务；
2. 只使用空闲 GPU；
3. 每个数据集独立目录；
4. 记录 GPU、PID、启动时间和 epoch；
5. OOM 时先记录，再调整 batch size；
6. 不覆盖已有结果。

目录：

```text
runs/experiment_d/NUDT-SIRST/sd_awgm_hfe/seed42
runs/experiment_d/NUAA-SIRST/sd_awgm_hfe/seed42
runs/experiment_d/IRSTD-1K/sd_awgm_hfe/seed42
```

---

## 27. 实验记录

新增：

```text
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
```

结果表：

| Dataset | Model | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | FLOPs | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | sd_awgm | existing | | | | | | | | |
| NUAA | sd_awgm_hfe | | | | | | | | | |
| NUDT | sd_awgm | existing | | | | | | | | |
| NUDT | sd_awgm_hfe | | | | | | | | | |
| IRSTD | sd_awgm | existing | | | | | | | | |
| IRSTD | sd_awgm_hfe | | | | | | | | | |

---

## 28. 结果判定

### HFE有效

若：

```text
sd_awgm_hfe > sd_awgm
```

且 Pd提高、Fa下降或mIoU明显提高，说明低频语义校正高频有价值。

### mIoU下降但Fa改善

说明 HFE 可能更偏向抑制背景边缘，但削弱目标边界。后续可考虑仅在 Stage 3/4 使用，但首轮不要立即修改。

### HFE未被使用

若 beta 长期接近初值、`beta*delta/base` 接近0，说明模型主动忽略 HFE。

### HFE扰动过强

若 `beta*delta/base > 1` 且性能下降，说明 HFE 正在覆盖原始系数。

---

## 29. 首轮不做的消融

首轮完成前不新增：

```text
HFE_stage34_only
HFE_without_matching
HFE_without_SKFF
soft_matching
cosine_matching
directional_matching
HFE + LDRC
HFE + Pyramid
LFSSBlock
```

只有 `sd_awgm_hfe` 显示正向趋势后，再设计内部消融。

---

## 30. 建议新增文件

```text
model/DWTFreqNet_SingleDecoder_HFE.py
train_experiment_d.py
tools/test_sd_awgm_hfe_experiment_d.py
tools/profile_sd_awgm_hfe_experiment_d.py
scripts/run_experiment_d.sh
scripts/launch_experiment_d_queue.sh
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
```

---

## 31. Codex最终交付

完成后返回：

1. 实际 base commit；
2. 新分支；
3. 最终 commit SHA；
4. 新增和修改文件；
5. Decoder HFE结构；
6. 四级输入输出形状；
7. Matching矩阵和索引形状；
8. 近似baseline回归测试；
9. 前向、反向和形状测试；
10. DWT/IDWT调用计数；
11. Haar方向检查；
12. 参数、FLOPs、延迟和显存；
13. 三数据集启动命令；
14. GPU、PID和输出目录；
15. 当前epoch；
16. `EXPERIMENT_D_SD_AWGM_HFE_RECORD.md`。

建议 commit：

```text
Add decoder-side HFE to SD-AWGM
```

---

## 32. 最终设计主线

\[
F_{HF,s}
=
\operatorname{SKFF}
(ar H_s,ar V_s,ar D_s)
\]

\[
F_{HF,s}^{e}
=
\operatorname{HFE}
(F_{HF,s},L_s)
\]

\[
\widehat H_s
=
ar H_s+eta_s^H\Delta H_s
\]

\[
\widehat V_s
=
ar V_s+eta_s^V\Delta V_s
\]

\[
\widehat D_s
=
ar D_s+eta_s^D\Delta D_s
\]

\[
L_{s-1}
=
\operatorname{IDWT}
(L_s,\widehat H_s,\widehat V_s,\widehat D_s)
\]

最终职责：

\[
oxed{	ext{AWGM负责高频引导低频编码}}
\]

\[
oxed{	ext{Decoder HFE负责低频语义校正高频重构}}
\]
