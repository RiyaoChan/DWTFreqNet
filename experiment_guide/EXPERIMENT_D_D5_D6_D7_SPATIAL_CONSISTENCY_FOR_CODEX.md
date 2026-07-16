# Experiment D：D5/D6/D7 Spatial Cross-Frequency Consistency Ablation

## 0. 实验定位

本方案属于 **Experiment D：SD-AWGM + Decoder-side HFE** 的内部消融，不建立新的主实验。

现有序列：

| ID | 模型 | 高频—低频关系 |
|---|---|---|
| D0 | `sd_awgm` | 无 Decoder HFE |
| D1 | `sd_awgm_hfe` | Hard L2 Top-1 通道匹配 |
| D2 | `sd_awgm_hfe_softcos` | Soft Cosine Top-k 通道匹配 |
| D3 | `sd_awgm_hfe_scaleaware` | 浅层 Correlation Gate + 深层 Soft Cosine |
| D4 | `sd_awgm_hfe_nomatch` | 无显式匹配，直接低频条件融合 |

新增空间一致性消融：

| ID | 模型 | 空间关系 | 位置偏移 | 目标先验 |
|---|---|---|---:|---:|
| D5 | `sd_awgm_hfe_samepos` | 同位置高低频一致性 + 低频局部对比 | 否 | 否 |
| D6 | `sd_awgm_hfe_neighborhood` | 3×3 邻域高低频局部注意力 | 是 | 否 |
| D7 | `sd_awgm_hfe_targetlocal` | 目标先验引导的 3×3 邻域局部注意力 | 是 | 是 |

D5、D6、D7 不再计算全图高低频通道相似度矩阵，不使用：

```text
torch.cdist
全图通道 L2 距离
C×C Cosine Matching
Top-1
Top-k
argmin
候选通道索引
```

研究问题从：

```text
哪个低频通道与哪个高频通道最相似？
```

转为：

```text
某个空间位置的低频目标响应，
是否与同一位置或附近位置的高频变化一致？
```

---

# 1. 核心实验假设

## 1.1 D5：同位置空间一致性

验证：

> 同一空间位置上的低频语义和高频变化是否一致，以及该位置是否具有低频局部对比。

D5 不搜索邻域，只产生：

```text
[B,1,H,W]
```

的空间调制图。

## 1.2 D6：邻域空间一致性

验证：

> 高频位置与低频位置之间是否存在 1 像素左右的局部偏移。

D6 在每个位置的 3×3 低频邻域中计算局部注意力：

```text
[B,9,H,W]
```

而不是构造：

```text
[B,C,C]
```

的全图通道匹配矩阵。

## 1.3 D7：目标感知邻域一致性

验证：

> 在局部高低频一致的基础上，利用现有 Deep Supervision Side Head 提供的目标置信度，能否抑制背景亮点、云边缘和建筑角点造成的错误一致性。

D7 不增加新的监督损失，复用现有：

```text
gt_conv5
gt_conv4
gt_conv3
gt_conv2
```

生成各 decoder stage 的 targetness prior。

---

# 2. 基础分支与代码状态

继续基于当前 Experiment D 分支和 Draft PR：

```text
Repository:
RiyaoChan/DWTFreqNet

Working branch:
codex/experiment-d-hfe-matching-ablation-d2-d3

Draft PR:
PR #3

Known HEAD after D4:
e5747b7f35fed3ecd3702a5f45332a8c35be8bd3
```

当前 D4 已完成：

```text
Direct Low Fusion
无 cdist
无 topk
无候选通道索引
DWT/IDWT = 4/4
```

Codex 开始前执行：

```bash
git checkout codex/experiment-d-hfe-matching-ablation-d2-d3
git pull
git rev-parse HEAD
git status
```

以实际最新 HEAD 为准。

D5/D6/D7 继续提交到同一分支并更新 PR #3，不创建 Experiment E。

---

# 3. 代码隔离原则

禁止修改以下原始模型文件：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
model/DWTFreqNet_SingleDecoder.py
model/DWTFreqNet_SingleDecoder_LDRC.py
model/DWTFreqNet_SingleDecoder_HFE.py
```

尽量不要修改 D2/D3/D4 已有实现：

```text
SoftCosineTopKMatching
SoftMatchingTransformation
LocalCorrelationGate
DirectFusionTransformation
D2_STAGE_CONFIG
D3_STAGE_CONFIG
D4_STAGE_CONFIG
```

新建模型文件：

```text
model/DWTFreqNet_SingleDecoder_HFE_SpatialAblation.py
```

建议类名：

```python
class DWTFreqNet_SingleDecoder_HFE_SpatialAblation(
    DWTFreqNet_SingleDecoder_HFE_Ablation
):
    ...
```

新文件从当前 D4 消融模型导入并复用：

```python
from model.DWTFreqNet_SingleDecoder_HFE_Ablation import (
    AblationChannelAttention,
    AblationFFN,
    DirectFusionTransformation,
    DWTFreqNet_SingleDecoder_HFE_Ablation,
)
```

同时从 HFE 主模型导入：

```python
from model.DWTFreqNet_SingleDecoder_HFE import (
    DirectionalResidualHead,
    LayerNorm2d,
    SubbandSelectiveFusion,
    assert_hfe_inputs,
)
```

---

# 4. 共同结构

D4、D5、D6、D7 共同保留：

```text
Stage-wise AWGM Encoder
原始H/V/D系数对齐
SubbandSelectiveFusion
HFE Channel Attention主体
HFE FFN主体
H/V/D三个独立残差头
beta=1e-3
四级逐步IDWT
Single Decoder
Deep Supervision
```

各级低频输入保持：

| Stage | 当前低频 |
|---:|---|
| 4 | `E4` |
| 3 | `L3` |
| 2 | `L2` |
| 1 | `L1` |

各级通道：

| Stage | Channels | Heads | 空间尺寸（256输入） |
|---:|---:|---:|---:|
| 1 | 64 | 1 | 128×128 |
| 2 | 128 | 2 | 64×64 |
| 3 | 256 | 4 | 32×32 |
| 4 | 256 | 4 | 16×16 |

---

# 5. 变体定义

新增：

```python
SPATIAL_HFE_ABLATION_VARIANTS = (
    "d5_same_position",
    "d6_neighborhood",
    "d7_target_neighborhood",
)
```

固定配置：

```python
SPATIAL_STAGE_CONFIG = {
    1: {
        "channels": 64,
        "num_heads": 1,
        "embed_channels": 16,
        "kernel_size": 3,
    },
    2: {
        "channels": 128,
        "num_heads": 2,
        "embed_channels": 32,
        "kernel_size": 3,
    },
    3: {
        "channels": 256,
        "num_heads": 4,
        "embed_channels": 64,
        "kernel_size": 3,
    },
    4: {
        "channels": 256,
        "num_heads": 4,
        "embed_channels": 64,
        "kernel_size": 3,
    },
}
```

首轮固定：

```text
D5：四尺度同位置一致性
D6：四尺度3×3邻域一致性
D7：四尺度目标先验3×3邻域一致性
```

不要进行：

```text
不同window搜索
不同embed_channels搜索
不同temperature搜索
不同stage组合搜索
```

---

# 6. 公共低频融合头

D5/D6/D7 应复用 D4 的 `DirectFusionTransformation` 作为最终融合头。

D4 的核心形式：

\[
Z=[X,L]
\]

\[
Y=
\operatorname{Project}
\left(
\sigma(\operatorname{Gate}(Z))
\odot
\operatorname{Value}(Z)
\right)
\]

因此 D5/D6/D7 的 relation 模块建议继承：

```python
DirectFusionTransformation
```

只负责先生成不同形式的：

```text
conditioned_low
或
matched_low
```

再调用：

```python
super().forward(
    high_feature,
    conditioned_low,
)
```

这样 D4–D7 的最终：

```text
gate/value/project
```

保持一致。

---

# 7. D5：Same-position Spatial Consistency Fusion

## 7.1 输入输出

输入：

```text
high_feature: [B,C,H,W]
low_feature:  [B,C,H,W]
```

输出：

```text
output:       [B,C,H,W]
spatial_scale:[B,1,H,W]
```

## 7.2 降维投影

```python
self.q_proj = nn.Conv2d(
    channels,
    embed_channels,
    1,
    bias=False,
)

self.k_proj = nn.Conv2d(
    channels,
    embed_channels,
    1,
    bias=False,
)

self.low_response = nn.Conv2d(
    channels,
    1,
    1,
    bias=False,
)
```

## 7.3 同位置高低频余弦一致性

\[
Q(p)=\operatorname{Conv}_{1\times1}(F_H)(p)
\]

\[
K(p)=\operatorname{Conv}_{1\times1}(F_L)(p)
\]

\[
S(p)
=
\frac{
Q(p)^TK(p)
}{
\|Q(p)\|_2\|K(p)\|_2+\epsilon
}
\]

代码：

```python
q = self.q_proj(high_feature)
k = self.k_proj(low_feature)

with torch.autocast(
    device_type=q.device.type,
    enabled=False,
):
    q_norm = F.normalize(
        q.float(),
        dim=1,
    )
    k_norm = F.normalize(
        k.float(),
        dim=1,
    )

    similarity = (
        q_norm * k_norm
    ).sum(
        dim=1,
        keepdim=True,
    )
```

输出：

```text
similarity: [B,1,H,W]
```

## 7.4 低频局部对比

\[
R_L=\operatorname{Conv}_{1\times1}(F_L)
\]

\[
C_L(p)
=
|R_L(p)-\operatorname{AvgPool}_{3\times3}(R_L)(p)|
\]

```python
low_response = self.low_response(
    low_feature
)

local_mean = F.avg_pool2d(
    low_response,
    kernel_size=3,
    stride=1,
    padding=1,
)

local_contrast = torch.abs(
    low_response - local_mean
)
```

对每个样本归一化：

```python
contrast_mean = local_contrast.mean(
    dim=(2, 3),
    keepdim=True,
)

contrast_normalized = (
    local_contrast
    / (contrast_mean + 1e-6)
)

contrast_normalized = (
    contrast_normalized
    / (1.0 + contrast_normalized)
)
```

输出范围约为：

```text
[0,1)
```

## 7.5 空间调制图

输入：

```text
[similarity, contrast_normalized]
```

结构：

```python
self.spatial_gate = nn.Sequential(
    nn.Conv2d(
        2,
        8,
        3,
        padding=1,
        bias=True,
    ),
    nn.GELU(),
    nn.Conv2d(
        8,
        1,
        3,
        padding=1,
        bias=True,
    ),
)
```

必须将最后一层初始化为零：

```python
nn.init.zeros_(
    self.spatial_gate[-1].weight
)

nn.init.zeros_(
    self.spatial_gate[-1].bias
)
```

使用：

\[
A_s=2\sigma(\operatorname{SpatialGate}([S,C_L]))
\]

```python
gate_logits = self.spatial_gate(
    torch.cat(
        [
            similarity.to(
                low_feature.dtype
            ),
            contrast_normalized,
        ],
        dim=1,
    )
)

spatial_scale = (
    2.0 * torch.sigmoid(
        gate_logits
    )
)
```

范围：

```text
(0,2)
```

初始值：

```text
exactly 1
```

因此 D5 初始时不抑制也不增强低频，近似 D4。

## 7.6 条件低频

\[
\widetilde F_L=A_s\odot F_L
\]

```python
conditioned_low = (
    spatial_scale
    * low_feature
)
```

随后：

```python
output, base_info = super().forward(
    high_feature,
    conditioned_low,
)
```

## 7.7 建议类

```python
class SamePositionConsistencyFusion(
    DirectFusionTransformation
):
    def __init__(
        self,
        channels,
        embed_channels,
    ):
        ...

    def forward(
        self,
        high_feature,
        low_feature,
    ):
        ...
        return output, info
```

## 7.8 D5统计

记录：

```text
similarity_shape
similarity_mean/std/min/max
local_contrast_mean/std
spatial_scale_shape
spatial_scale_mean/std/min/max
conditioned_low_norm
raw_low_norm
output_norm
```

不得在训练阶段执行逐batch GPU→CPU标量同步。

---

# 8. D6：Neighborhood Cross-frequency Fusion

## 8.1 核心定义

对于高频位置 \(p\)，在同尺度低频的 3×3 邻域中计算局部注意力：

\[
\mathcal N(p)
=
\{
p+(-1,-1),\ldots,p+(1,1)
\}
\]

\[
a(p,q)
=
\frac{
Q_H(p)^TK_L(q)
}{
\tau
},
\quad q\in\mathcal N(p)
\]

\[
w(p,q)
=
\operatorname{Softmax}_{q\in\mathcal N(p)}
a(p,q)
\]

\[
F_L^{match}(p)
=
\sum_{q\in\mathcal N(p)}
w(p,q)F_L(q)
\]

## 8.2 投影

```python
self.q_proj = nn.Conv2d(
    channels,
    embed_channels,
    1,
    bias=False,
)

self.k_proj = nn.Conv2d(
    channels,
    embed_channels,
    1,
    bias=False,
)
```

温度：

```python
self.log_temperature = nn.Parameter(
    torch.tensor(0.1).log()
)
```

实际：

```python
temperature = (
    self.log_temperature.exp()
    .clamp(min=0.03, max=1.0)
)
```

## 8.3 固定偏移

```python
OFFSETS_3X3 = (
    (-1, -1),
    (-1,  0),
    (-1,  1),
    ( 0, -1),
    ( 0,  0),
    ( 0,  1),
    ( 1, -1),
    ( 1,  0),
    ( 1,  1),
)
```

中心索引：

```text
4
```

## 8.4 内存友好的shift实现

不建议 Stage 1 直接构造：

```text
[B,C,9,H,W]
```

的完整 value tensor。

使用 `replicate` padding 和逐offset切片。

建议辅助函数：

```python
def shift_feature(
    feature,
    dy,
    dx,
):
    height, width = (
        feature.shape[-2:]
    )

    padded = F.pad(
        feature,
        (1, 1, 1, 1),
        mode="replicate",
    )

    y0 = 1 + dy
    x0 = 1 + dx

    return padded[
        :,
        :,
        y0:y0 + height,
        x0:x0 + width,
    ]
```

必须用合成坐标图测试 offset 方向是否正确。

## 8.5 局部相似度

```python
q = self.q_proj(high_feature)
k = self.k_proj(low_feature)

with torch.autocast(
    device_type=q.device.type,
    enabled=False,
):
    q_norm = F.normalize(
        q.float(),
        dim=1,
    )

    k_norm = F.normalize(
        k.float(),
        dim=1,
    )

    logits = []

    for dy, dx in OFFSETS_3X3:
        shifted_k = shift_feature(
            k_norm,
            dy,
            dx,
        )

        score = (
            q_norm * shifted_k
        ).sum(
            dim=1,
            keepdim=True,
        )

        logits.append(score)

    logits = torch.cat(
        logits,
        dim=1,
    )

    temperature = (
        self.log_temperature
        .exp()
        .clamp(
            min=0.03,
            max=1.0,
        )
    )

    attention = torch.softmax(
        logits / temperature,
        dim=1,
    )
```

形状：

```text
logits:    [B,9,H,W]
attention: [B,9,H,W]
```

## 8.6 邻域低频聚合

为避免保存 9 份 C 通道 low：

```python
matched_low = torch.zeros_like(
    low_feature
)

for index, (dy, dx) in enumerate(
    OFFSETS_3X3
):
    shifted_low = shift_feature(
        low_feature,
        dy,
        dx,
    )

    matched_low = (
        matched_low
        + attention[
            :,
            index:index + 1,
        ].to(
            low_feature.dtype
        ) * shifted_low
    )
```

随后调用 D4 融合头：

```python
output, base_info = super().forward(
    high_feature,
    matched_low,
)
```

## 8.7 建议类

```python
class NeighborhoodCrossFrequencyFusion(
    DirectFusionTransformation
):
    def __init__(
        self,
        channels,
        embed_channels,
        temperature=0.1,
    ):
        ...

    def forward(
        self,
        high_feature,
        low_feature,
    ):
        ...
        return output, info
```

## 8.8 D6统计

记录：

```text
attention_shape
temperature
attention_entropy
normalized_attention_entropy
center_weight_mean
center_selection_ratio
neighbor_selection_ratio
mean_abs_dy
mean_abs_dx
mean_offset_distance
matched_low_norm
raw_low_norm
```

其中：

\[
H=-\sum_{i=1}^{9}w_i\log(w_i+\epsilon)
\]

\[
H_{norm}=\frac{H}{\log 9}
\]

最大权重offset：

```python
argmax_index = attention.argmax(
    dim=1
)
```

统计9个offset选择比例。

---

# 9. D7：Targetness-prior Neighborhood Fusion

## 9.1 目标先验来源

D7复用现有 Deep Supervision Side Head，不增加新head和新loss。

各stage：

| Stage | Low feature | Raw side logit |
|---:|---|---|
| 4 | `E4` | `gt_conv5(E4)` |
| 3 | `L3` | `gt_conv4(L3)` |
| 2 | `L2` | `gt_conv3(L2)` |
| 1 | `L1` | `gt_conv2(L1)` |

目标先验：

\[
T_s=\sigma(\operatorname{SideHead}_s(L_s))
\]

使用：

```python
targetness = torch.sigmoid(
    side_logit
).detach()
```

必须 `detach()`。

原因：

1. Side Head 仍由现有 Deep Supervision Loss 监督；
2. HFE 不反向操纵 targetness predictor；
3. 避免 HFE 和 prior 形成自强化退化；
4. 不增加额外loss变量。

## 9.2 Targetness邻域先验

从低频 targetness 生成9个邻域先验：

```python
target_neighbors = []

for dy, dx in OFFSETS_3X3:
    shifted_target = shift_feature(
        targetness,
        dy,
        dx,
    )

    target_neighbors.append(
        shifted_target
    )

target_neighbors = torch.cat(
    target_neighbors,
    dim=1,
)
```

形状：

```text
[B,9,H,W]
```

## 9.3 融合到局部attention logit

D6基础相似度：

\[
a_{sim}(p,q)
=
\frac{
Q_H(p)^TK_L(q)
}{
\tau
}
\]

目标先验：

\[
a_{target}(p,q)
=
\eta\log(T_s(q)+\epsilon)
\]

最终：

\[
a'(p,q)
=
a_{sim}(p,q)
+
a_{target}(p,q)
\]

实现：

```python
targetness_scale = F.softplus(
    self.raw_targetness_scale
)

target_prior_logits = torch.log(
    target_neighbors.float()
    .clamp_min(1e-6)
)

combined_logits = (
    similarity_logits
    / temperature
    + targetness_scale
    * target_prior_logits
)

attention = torch.softmax(
    combined_logits,
    dim=1,
)
```

## 9.4 Targetness scale初始化

要求初始：

```text
targetness_scale = 1.0
```

逆 softplus：

```python
initial_raw = math.log(
    math.exp(1.0) - 1.0
)

self.raw_targetness_scale = (
    nn.Parameter(
        torch.tensor(initial_raw)
    )
)
```

实际值：

```python
targetness_scale = F.softplus(
    self.raw_targetness_scale
)
```

## 9.5 建议类

```python
class TargetAwareNeighborhoodFusion(
    NeighborhoodCrossFrequencyFusion
):
    def forward(
        self,
        high_feature,
        low_feature,
        targetness,
    ):
        ...
        return output, info
```

D7不能直接使用 D5/D6 的两输入 relation 接口，因此需要独立 Target-aware HFE Block。

---

# 10. D5/D6通用HFE Block

D5/D6可复用已有：

```text
AblationChannelAttention
AblationFFN
```

新建一个接收 relation module 实例的轻量Block：

```python
class SpatialDecoderHFEBlock(
    nn.Module
):
    def __init__(
        self,
        channels,
        num_heads,
        attn_relation,
        ffn_relation,
    ):
        super().__init__()

        self.high_norm1 = LayerNorm2d(
            channels
        )

        self.high_norm2 = LayerNorm2d(
            channels
        )

        self.low_norm = LayerNorm2d(
            channels
        )

        self.attn = AblationChannelAttention(
            channels=channels,
            num_heads=num_heads,
            relation_module=attn_relation,
        )

        self.ffn = AblationFFN(
            channels=channels,
            relation_module=ffn_relation,
        )

    def forward(
        self,
        high_feature,
        low_feature,
    ):
        low = self.low_norm(
            low_feature
        )

        attn_out, attn_info = self.attn(
            self.high_norm1(
                high_feature
            ),
            low,
        )

        high_feature = (
            high_feature + attn_out
        )

        ffn_out, ffn_info = self.ffn(
            self.high_norm2(
                high_feature
            ),
            low,
        )

        high_feature = (
            high_feature + ffn_out
        )

        return high_feature, {
            "attn_relation": attn_info,
            "ffn_relation": ffn_info,
        }
```

Attention relation 和 FFN relation 必须为独立参数实例。

---

# 11. D5/D6通用Refiner

```python
class SpatialDecoderHFERefiner(
    nn.Module
):
    def __init__(
        self,
        channels,
        num_heads,
        stage,
        relation_factory,
    ):
        super().__init__()

        self.stage = stage

        self.subband_fusion = (
            SubbandSelectiveFusion(
                channels,
                reduction=8,
            )
        )

        self.hfe = SpatialDecoderHFEBlock(
            channels=channels,
            num_heads=num_heads,
            attn_relation=relation_factory(),
            ffn_relation=relation_factory(),
        )

        self.head_h = (
            DirectionalResidualHead(
                channels
            )
        )

        self.head_v = (
            DirectionalResidualHead(
                channels
            )
        )

        self.head_d = (
            DirectionalResidualHead(
                channels
            )
        )

        self.beta_h = nn.Parameter(
            torch.full(
                (1, channels, 1, 1),
                1e-3,
            )
        )

        self.beta_v = nn.Parameter(
            torch.full(
                (1, channels, 1, 1),
                1e-3,
            )
        )

        self.beta_d = nn.Parameter(
            torch.full(
                (1, channels, 1, 1),
                1e-3,
            )
        )
```

forward接口与 D4 refiner完全一致：

```python
def forward(
    self,
    low_feature,
    base_h,
    base_v,
    base_d,
):
    ...
    return (
        coef_h,
        coef_v,
        coef_d,
        debug,
    )
```

因此 D5/D6可直接继承父类 forward。

---

# 12. D7 Target-aware HFE Block

## 12.1 TargetAwareChannelAttention

结构与 `AblationChannelAttention` 相同，但relation调用为：

```python
q, relation_info = self.relation(
    q,
    low_feature,
    targetness,
)
```

建议：

```python
class TargetAwareChannelAttention(
    nn.Module
):
    def __init__(
        self,
        channels,
        num_heads,
        relation_module,
    ):
        ...

    def forward(
        self,
        high_feature,
        low_feature,
        targetness,
    ):
        ...
        return output, relation_info
```

通道自注意力主体保持：

```text
Q/K/V
channel attention
temperature
project_out
```

与 D1–D6 一致。

## 12.2 TargetAwareFFN

relation调用：

```python
x, relation_info = self.relation(
    x,
    low_feature,
    targetness,
)
```

其他结构保持：

```text
project_in
depthwise conv
relation
depthwise conv
GELU
project_out
```

## 12.3 TargetAwareDecoderHFEBlock

```python
class TargetAwareDecoderHFEBlock(
    nn.Module
):
    def forward(
        self,
        high_feature,
        low_feature,
        targetness,
    ):
        ...
```

Attention和FFN均使用同一个 targetness tensor，但relation参数独立。

## 12.4 TargetAwareDecoderHFERefiner

```python
class TargetAwareDecoderHFERefiner(
    nn.Module
):
    def forward(
        self,
        low_feature,
        base_h,
        base_v,
        base_d,
        targetness,
    ):
        ...
```

其他：

```text
SKFF
directional residual heads
beta
statistics
```

与D5/D6一致。

---

# 13. D7专用forward

父类 forward 在所有 decoder 完成后才计算 side logits。D7必须重写 forward，使 side logits 在各级 HFE 前计算。

## 13.1 编码

保持父类一致：

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
    ) = self._encode_stage(
        stage,
        current,
    )

    current = encoded[stage]
```

对齐：

```python
bases = {
    stage:
    self._align_stage_coefficients(
        stage,
        raw_bands[stage],
    )
    for stage in range(1, 5)
}
```

## 13.2 Stage 4

```python
side4_raw = self.gt_conv5(
    encoded[4]
)

target4 = torch.sigmoid(
    side4_raw
).detach()

coef_h4, coef_v4, coef_d4, debug4 = (
    self.decoder_hfe4(
        encoded[4],
        *bases[4],
        targetness=target4,
    )
)

u3 = self._idwt(
    encoded[4],
    coef_h4,
    coef_v4,
    coef_d4,
)

l3 = self.decoder_fuse3(
    torch.cat(
        [u3, encoded[3]],
        dim=1,
    )
)
```

## 13.3 Stage 3

```python
side3_raw = self.gt_conv4(
    l3
)

target3 = torch.sigmoid(
    side3_raw
).detach()

coef_h3, coef_v3, coef_d3, debug3 = (
    self.decoder_hfe3(
        l3,
        *bases[3],
        targetness=target3,
    )
)
```

然后 IDWT → `l2`。

## 13.4 Stage 2

```python
side2_raw = self.gt_conv3(
    l2
)

target2 = torch.sigmoid(
    side2_raw
).detach()
```

然后 HFE → IDWT → `l1`。

## 13.5 Stage 1

```python
side1_raw = self.gt_conv2(
    l1
)

target1 = torch.sigmoid(
    side1_raw
).detach()
```

然后 HFE → IDWT → `l0` → `out`。

## 13.6 Deep Supervision复用

不要重复调用side head。

```python
target_size = x.shape[-2:]

gt5 = F.interpolate(
    side4_raw,
    target_size,
    mode="bilinear",
    align_corners=False,
)

gt4 = F.interpolate(
    side3_raw,
    target_size,
    mode="bilinear",
    align_corners=False,
)

gt3 = F.interpolate(
    side2_raw,
    target_size,
    mode="bilinear",
    align_corners=False,
)

gt2 = F.interpolate(
    side1_raw,
    target_size,
    mode="bilinear",
    align_corners=False,
)
```

后续 `d0` 和输出格式与父类完全一致。

## 13.7 模型forward分派

```python
def forward(self, x):
    if (
        self.spatial_hfe_ablation
        == "d7_target_neighborhood"
    ):
        return self._forward_d7(x)

    return super().forward(x)
```

D5/D6直接使用父类forward。

---

# 14. 模型构造

建议：

```python
class DWTFreqNet_SingleDecoder_HFE_SpatialAblation(
    DWTFreqNet_SingleDecoder_HFE_Ablation
):
    def __init__(
        self,
        config,
        spatial_hfe_ablation,
        ...,
    ):
        if spatial_hfe_ablation not in (
            SPATIAL_HFE_ABLATION_VARIANTS
        ):
            raise ValueError(...)

        super().__init__(
            config=config,
            hfe_ablation="d4_no_matching",
            ...,
        )

        self.spatial_hfe_ablation = (
            spatial_hfe_ablation
        )

        for stage in range(1, 5):
            cfg = SPATIAL_STAGE_CONFIG[
                stage
            ]

            if spatial_hfe_ablation == (
                "d5_same_position"
            ):
                relation_factory = (
                    lambda cfg=cfg:
                    SamePositionConsistencyFusion(
                        channels=cfg["channels"],
                        embed_channels=cfg[
                            "embed_channels"
                        ],
                    )
                )

                refiner = (
                    SpatialDecoderHFERefiner(
                        channels=cfg["channels"],
                        num_heads=cfg["num_heads"],
                        stage=stage,
                        relation_factory=(
                            relation_factory
                        ),
                    )
                )

            elif spatial_hfe_ablation == (
                "d6_neighborhood"
            ):
                relation_factory = (
                    lambda cfg=cfg:
                    NeighborhoodCrossFrequencyFusion(
                        channels=cfg["channels"],
                        embed_channels=cfg[
                            "embed_channels"
                        ],
                        temperature=0.1,
                    )
                )

                refiner = (
                    SpatialDecoderHFERefiner(
                        channels=cfg["channels"],
                        num_heads=cfg["num_heads"],
                        stage=stage,
                        relation_factory=(
                            relation_factory
                        ),
                    )
                )

            else:
                refiner = (
                    TargetAwareDecoderHFERefiner(
                        channels=cfg["channels"],
                        num_heads=cfg["num_heads"],
                        stage=stage,
                        embed_channels=cfg[
                            "embed_channels"
                        ],
                        temperature=0.1,
                        targetness_scale=1.0,
                    )
                )

            setattr(
                self,
                f"decoder_hfe{stage}",
                refiner,
            )
```

调用父类 D4 只用于复用 encoder/decoder/forward，再将四个 D4 refiner 完全替换。

---

# 15. 元数据

共同：

```python
self.experiment_group = "experiment_d"
self.experiment_type = "ablation"
self.ablation_axis = (
    "decoder_hfe_spatial_relation"
)

self.explicit_channel_matching = False
self.channel_similarity_matrix = False
self.channel_candidate_selection = False

self.directional_pyramid = False
self.second_dwt = False
self.ldrc = False
self.mamba = False
```

## 15.1 D5

```python
self.ablation_id = "D5"
self.model_variant = (
    "dwtfreqnet_single_decoder_hfe_samepos"
)
self.sd_variant = (
    "sd_awgm_hfe_samepos"
)
self.decoder_hfe_relation = (
    "same_position_consistency_local_contrast"
)
self.spatial_offset_search = False
self.targetness_prior = False
```

## 15.2 D6

```python
self.ablation_id = "D6"
self.model_variant = (
    "dwtfreqnet_single_decoder_hfe_neighborhood"
)
self.sd_variant = (
    "sd_awgm_hfe_neighborhood"
)
self.decoder_hfe_relation = (
    "local_3x3_cross_frequency_attention"
)
self.spatial_offset_search = True
self.neighborhood_size = 3
self.targetness_prior = False
```

## 15.3 D7

```python
self.ablation_id = "D7"
self.model_variant = (
    "dwtfreqnet_single_decoder_hfe_targetlocal"
)
self.sd_variant = (
    "sd_awgm_hfe_targetlocal"
)
self.decoder_hfe_relation = (
    "targetness_prior_local_3x3_cross_frequency_attention"
)
self.spatial_offset_search = True
self.neighborhood_size = 3
self.targetness_prior = True
self.targetness_prior_source = (
    "existing_deep_supervision_side_heads"
)
self.targetness_prior_detached = True
self.targetness_scale_init = 1.0
```

---

# 16. 新训练入口

新建：

```text
train_experiment_d_hfe_spatial_ablation.py
```

参数只允许：

```text
--spatial-hfe-ablation d5_same_position
--spatial-hfe-ablation d6_neighborhood
--spatial-hfe-ablation d7_target_neighborhood
```

模型：

```python
model = (
    DWTFreqNet_SingleDecoder_HFE_SpatialAblation(
        get_DWTFreqNet_config(),
        spatial_hfe_ablation=(
            args.spatial_hfe_ablation
        ),
        mode=mode,
        deepsuper=True,
    )
)
```

不要修改现有 D2/D3/D4 训练入口。

---

# 17. 输出目录

D5：

```text
runs/experiment_d_spatial_ablation/
    D5_same_position/
    <dataset>/
    seed42/
```

D6：

```text
runs/experiment_d_spatial_ablation/
    D6_neighborhood/
    <dataset>/
    seed42/
```

D7：

```text
runs/experiment_d_spatial_ablation/
    D7_target_neighborhood/
    <dataset>/
    seed42/
```

不得覆盖：

```text
runs/experiment_d_ablation/D2_softcos_all
runs/experiment_d_ablation/D3_scaleaware
runs/experiment_d_ablation/D4_no_matching
```

---

# 18. Checkpoint与run_config

所有正式实验随机初始化，不加载 D0–D4 权重。

必须记录：

```text
experiment_group
experiment_type
ablation_axis
ablation_id
model_variant
sd_variant
spatial_hfe_ablation
relation_mode
embed_channels
kernel_size
temperature_init
targetness_prior
targetness_prior_detached
targetness_scale_init
DWT/IDWT count
```

D7还记录：

```text
side_head_mapping:
stage4=gt_conv5
stage3=gt_conv4
stage2=gt_conv3
stage1=gt_conv2
```

---

# 19. 单元测试

新建：

```text
tools/test_experiment_d_hfe_spatial_ablation.py
```

## 19.1 原模型回归

确认以下模型仍可构建和前向：

```text
D0
D1
D2
D3
D4
```

## 19.2 模块类型

D5 Stage1–4：

```text
SamePositionConsistencyFusion
```

D6 Stage1–4：

```text
NeighborhoodCrossFrequencyFusion
```

D7 Stage1–4：

```text
TargetAwareNeighborhoodFusion
```

## 19.3 输出形状

输入：

```python
x = torch.randn(
    2,
    1,
    256,
    256,
)
```

训练：

```text
6 × [2,1,256,256]
```

测试：

```text
[2,1,256,256]
```

## 19.4 D5形状

```text
Stage1 spatial_scale:
[2,1,128,128]

Stage2:
[2,1,64,64]

Stage3:
[2,1,32,32]

Stage4:
[2,1,16,16]
```

初始化检查：

```python
torch.testing.assert_close(
    spatial_scale,
    torch.ones_like(
        spatial_scale
    ),
    atol=1e-6,
    rtol=0,
)
```

仅对刚初始化且未训练的模型执行。

## 19.5 D5与D4初始化公平性

复制 D4 Direct Fusion Head 的：

```text
gate
value
project
```

到 D5 relation。

D5 spatial gate最后一层为零初始化。

同一 high/low 输入下：

```python
torch.testing.assert_close(
    d4_relation_output,
    d5_relation_output,
    atol=1e-6,
    rtol=1e-5,
)
```

证明 D5 初始退化为 D4。

## 19.6 D6形状与权重

```text
attention:
Stage1 [2,9,128,128]
Stage2 [2,9,64,64]
Stage3 [2,9,32,32]
Stage4 [2,9,16,16]
```

检查：

```python
torch.testing.assert_close(
    attention.sum(dim=1),
    torch.ones_like(
        attention[:, 0]
    ),
    atol=1e-6,
    rtol=1e-5,
)
```

## 19.7 offset方向测试

构造单通道坐标图或单脉冲图，分别检查：

```text
(-1,-1)
(-1,0)
...
(1,1)
```

的 shift 输出位置正确。

该测试必须通过后才能启动正式D6/D7。

## 19.8 D6中心聚合退化测试

构造人工 attention：

```text
中心offset权重=1
其他=0
```

验证：

```text
matched_low == raw_low
```

将 D4和D6 fusion head参数复制一致后：

```text
D6 relation output == D4 relation output
```

## 19.9 D7 uniform-prior退化测试

D6和D7使用相同：

```text
q_proj
k_proj
temperature
fusion head
```

输入恒定 targetness：

```python
targetness = torch.full(
    (B,1,H,W),
    0.5,
)
```

因为每个邻域位置的 log prior相同，softmax不变。

检查：

```python
torch.testing.assert_close(
    d6_output,
    d7_output,
    atol=1e-5,
    rtol=1e-5,
)
```

## 19.10 D7 targetness detach测试

检查：

```python
targetness.requires_grad is False
```

或对 side logit 保留梯度后：

```python
targetness = sigmoid(
    side_logit
).detach()
```

HFE loss不得通过 targetness prior回传到 side logit。

Side head仍必须通过原有 Deep Supervision Loss 获得梯度。

## 19.11 禁止全图通道Matching

Monkeypatch：

```python
torch.cdist = forbidden
torch.topk = forbidden
```

D5/D6/D7完整forward均必须通过。

检查模型中不存在：

```text
SoftCosineTopKMatching
SoftMatchingTransformation
ChannelMatching
MatchingTransformation
LocalCorrelationGate
```

D5/D6/D7允许：

```text
torch.matmul
channel self-attention
local softmax
argmax仅用于评估统计
```

## 19.12 beta置零退化

D5/D6/D7分别将全部：

```text
beta_h
beta_v
beta_d
```

置零。

复制共同参数后，与 D0 `sd_awgm` 输出一致。

## 19.13 DWT/IDWT

```text
DWT=4
IDWT=4
```

## 19.14 梯度

检查非零梯度：

D5：

```text
q_proj
k_proj
low_response
spatial_gate
Direct Fusion gate/value/project
Attention/FFN
方向头
beta
```

D6：

```text
q_proj
k_proj
log_temperature
Direct Fusion gate/value/project
Attention/FFN
方向头
beta
```

D7：

```text
q_proj
k_proj
log_temperature
raw_targetness_scale
Direct Fusion gate/value/project
Attention/FFN
方向头
beta
side heads通过deep supervision获得梯度
```

## 19.15 AMP与真实数据

RTX 3090：

```text
2×1×256×256
forward
backward
AMP
```

真实 NUAA：

```text
batch=4
单步训练
optimizer.step
checkpoint metadata
```

D5/D6/D7均必须通过。

---

# 20. 复杂度与速度测试

新建：

```text
tools/profile_experiment_d_hfe_spatial_ablation.py
```

比较：

```text
D0
D1
D2
D3
D4
D5
D6
D7
```

统一：

```text
RTX 3090
1×1×256×256
warmup=5
repeat>=20
```

报告：

```text
Parameters
THOP FLOPs
Latency
FPS
Inference peak memory
Training peak memory
Relation参数量
DWT/IDWT
```

注意：

```text
THOP可能不统计shift、局部相似度和attention加权；
必须报告实测latency和显存。
```

D6/D7重点检查 Stage 1 的显存。

禁止长期保存：

```text
9份完整C通道邻域tensor
训练阶段完整attention历史
逐batchGPU→CPU统计
```

---

# 21. 目标区域与背景区域诊断

新建：

```text
tools/analyze_experiment_d_spatial_consistency.py
```

仅用于验证集离线诊断，不参与训练和loss。

## 21.1 GT尺度对齐

不能简单使用 bilinear。

为避免深层小目标消失，使用：

```python
mask_stage = F.adaptive_max_pool2d(
    gt_mask.float(),
    output_size=relation_map.shape[-2:],
)
```

目标：

```python
target_mask = (
    mask_stage > 0.5
)
```

背景：

```python
background_mask = (
    ~target_mask
)
```

## 21.2 D5诊断

计算：

\[
G_{target}
=
\operatorname{Mean}
(A_s\mid Y=1)
\]

\[
G_{background}
=
\operatorname{Mean}
(A_s\mid Y=0)
\]

\[
R_G
=
\frac{
G_{target}
}{
G_{background}+\epsilon
}
\]

记录：

```text
target_spatial_scale_mean
background_spatial_scale_mean
target_background_scale_ratio
target_similarity_mean
background_similarity_mean
target_background_similarity_ratio
```

## 21.3 D6/D7诊断

记录：

```text
target_attention_entropy
background_attention_entropy
target_center_weight
background_center_weight
target_neighbor_selection_ratio
background_neighbor_selection_ratio
target_mean_offset_distance
background_mean_offset_distance
```

D7额外：

```text
target_targetness_mean
background_targetness_mean
targetness_separation_ratio
targetness_scale
```

## 21.4 高频残差区域集中度

对H/V/D分别计算：

\[
R_{\Delta}
=
\frac{
\operatorname{Mean}
(|\beta\Delta|\mid Y=1)
}{
\operatorname{Mean}
(|\beta\Delta|\mid Y=0)+\epsilon
}
\]

记录：

```text
H_target_background_residual_ratio
V_target_background_residual_ratio
D_target_background_residual_ratio
```

若比值长期接近1或小于1，说明 HFE 校正没有集中在目标区域。

---

# 22. 正式实验顺序

D4 当前正在运行或等待最终结果，不停止D4。

D5、D6、D7完成代码测试后按以下优先级启动：

```text
第一优先：D5
第二优先：D6
第三优先：D7
```

每个变体内部的数据集优先级：

```text
1. NUAA-SIRST
2. IRSTD-1K
3. NUDT-SIRST
```

原因：

```text
NUAA和IRSTD用于验证复杂背景误警；
NUDT用于验证是否保留Decoder HFE的显著收益。
```

---

# 23. GPU调度方案

## 23.1 至少9张空闲GPU

D5/D6/D7三个数据集全部并行。

## 23.2 6张空闲GPU

第一批：

| GPU | Variant | Dataset |
|---:|---|---|
| 0 | D5 | NUAA |
| 1 | D5 | IRSTD |
| 2 | D5 | NUDT |
| 3 | D6 | NUAA |
| 4 | D6 | IRSTD |
| 5 | D6 | NUDT |

D7进入队列，任一任务完成或出现空闲GPU后启动。

## 23.3 3张空闲GPU

第一批：

```text
D5-NUAA
D5-IRSTD
D5-NUDT
```

第二批：

```text
D6-NUAA
D6-IRSTD
D6-NUDT
```

第三批：

```text
D7-NUAA
D7-IRSTD
D7-NUDT
```

## 23.4 动态检查

启动脚本必须检查：

```text
GPU已分配显存
GPU utilization
已有PID
输出目录锁
完成标记
失败标记
```

不得停止或抢占：

```text
D4
Experiment C
WULLE/W8M
其他正式任务
```

---

# 24. 统一训练设置

D5/D6/D7与D0–D4完全一致：

```text
seed=42
patch size=256
batch size=4
epochs=1000
optimizer=Adam
initial lr=1e-3
scheduler=CosineAnnealingLR
eta_min=1e-5
eval start=100
eval every=1
save every=20
threshold=0.5
```

禁止修改：

```text
数据划分
loss
数据增强
学习率
scheduler
评价阈值
```

D7不得增加额外 targetness loss。

---

# 25. 启动脚本

新建：

```text
scripts/run_experiment_d_hfe_spatial_ablation.sh
scripts/launch_experiment_d_hfe_spatial_ablation_queue.sh
```

示例：

```bash
bash scripts/run_experiment_d_hfe_spatial_ablation.sh \
  d5_same_position \
  NUAA-SIRST \
  0
```

```bash
bash scripts/run_experiment_d_hfe_spatial_ablation.sh \
  d6_neighborhood \
  IRSTD-1K \
  1
```

```bash
bash scripts/run_experiment_d_hfe_spatial_ablation.sh \
  d7_target_neighborhood \
  NUDT-SIRST \
  2
```

---

# 26. 实验记录

新建：

```text
EXPERIMENT_D_HFE_SPATIAL_CONSISTENCY_RECORD.md
```

并更新：

```text
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
EXPERIMENT_RECORD.md
PR #3 body
```

声明：

```text
D5/D6/D7均为Experiment D内部空间关系消融；
不属于新的主实验。
```

---

# 27. 结果表

| Dataset | ID | Relation | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA | D0 | None | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 | |
| NUAA | D1 | Hard L2 Top-1 | 286 | 0.7747 | 0.7809 | 0.8731 | 0.9695 | 2.394e-5 | |
| NUAA | D2 | Soft Cosine Top-k | 350 | 0.773369 | 0.782203 | 0.872203 | 0.973282 | 3.739e-5 | |
| NUAA | D3 | Gate + Soft Cosine | 549 | 0.776066 | 0.785939 | 0.873916 | 0.973282 | 3.211e-5 | |
| NUAA | D4 | Direct Low Fusion | pending | | | | | | |
| NUAA | D5 | Same-position Consistency | | | | | | | |
| NUAA | D6 | 3×3 Neighborhood | | | | | | | |
| NUAA | D7 | Target-aware Neighborhood | | | | | | | |
| NUDT | D0 | None | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 | |
| NUDT | D1 | Hard L2 Top-1 | 694 | 0.943166 | 0.949027 | 0.970752 | 0.991534 | 4.343e-6 | |
| NUDT | D2 | Soft Cosine Top-k | 419 | 0.946951 | 0.947689 | 0.972753 | 0.990476 | 1.953e-6 | |
| NUDT | D3 | Gate + Soft Cosine | 513 | 0.947825 | 0.949940 | 0.973214 | 0.995767 | 1.540e-6 | |
| NUDT | D4 | Direct Low Fusion | pending | | | | | | |
| NUDT | D5 | Same-position Consistency | | | | | | | |
| NUDT | D6 | 3×3 Neighborhood | | | | | | | |
| NUDT | D7 | Target-aware Neighborhood | | | | | | | |
| IRSTD | D0 | None | 894 | 0.6561 | 0.6477 | 0.7924 | 0.9091 | 1.537e-5 | |
| IRSTD | D1 | Hard L2 Top-1 | 556 | 0.657358 | 0.658863 | 0.793260 | 0.922559 | 1.395e-5 | |
| IRSTD | D2 | Soft Cosine Top-k | 464 | 0.658223 | 0.659029 | 0.793890 | 0.915825 | 1.704e-5 | |
| IRSTD | D3 | Gate + Soft Cosine | 735 | 0.657163 | 0.660729 | 0.793119 | 0.936027 | 1.782e-5 | |
| IRSTD | D4 | Direct Low Fusion | pending | | | | | | |
| IRSTD | D5 | Same-position Consistency | | | | | | | |
| IRSTD | D6 | 3×3 Neighborhood | | | | | | | |
| IRSTD | D7 | Target-aware Neighborhood | | | | | | | |

---

# 28. 结论判定

## 28.1 D5 > D4

说明：

> 低频信息不应在所有位置无差别注入；同位置高低频一致性和低频局部对比能够筛选有效位置。

重点观察：

```text
NUAA/IRSTD Fa是否下降
target/background spatial scale ratio是否>1
```

## 28.2 D6 > D5

说明：

> 高频与低频之间存在局部空间偏移，严格同位置对齐假设过强。

同时应看到：

```text
neighbor_selection_ratio不接近0
mean_offset_distance > 0
```

## 28.3 D5 ≈ D6

说明：

> 同位置一致性已经足够，没有必要承担3×3邻域注意力的额外复杂度。

优先选择D5。

## 28.4 D7 > D6

说明：

> 局部高低频一致性仍会响应背景亮点，目标置信度先验能够提高背景选择性。

理想表现：

```text
Fa下降
mIoU提高
targetness target/background separation > 1
```

## 28.5 D7提高Pd但Fa同步增加

说明：

> targetness prior强化了候选响应，但side head对复杂背景仍不够可靠，不能称为背景抑制成功。

## 28.6 D4–D7差异均小

若三数据集平均 mIoU 差异均小于约0.002：

> Decoder HFE relation细节不是性能主要因素；后续不应继续扩展更复杂的高低频对齐模块。

---

# 29. 随机波动说明

当前统一为 seed 42。

不得将：

```text
<0.002 的平均mIoU差异
```

直接表述为确定性优势。

记录中区分：

```text
结构性趋势
轻微数值波动
```

本轮不自动启动多seed。

---

# 30. 本轮不做

D5/D6/D7完成前不新增：

```text
可变形卷积offset
5×5或7×7邻域
dilation
不同stage混合
不同targetness scale搜索
方向独立空间匹配
HFE + LDRC
新loss
多seed
```

---

# 31. 建议新增文件

```text
model/DWTFreqNet_SingleDecoder_HFE_SpatialAblation.py
train_experiment_d_hfe_spatial_ablation.py
tools/test_experiment_d_hfe_spatial_ablation.py
tools/profile_experiment_d_hfe_spatial_ablation.py
tools/analyze_experiment_d_spatial_consistency.py
scripts/run_experiment_d_hfe_spatial_ablation.sh
scripts/launch_experiment_d_hfe_spatial_ablation_queue.sh
EXPERIMENT_D_HFE_SPATIAL_CONSISTENCY_RECORD.md
```

允许最小修改：

```text
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
EXPERIMENT_RECORD.md
README.md
PR #3 body
```

---

# 32. Codex最终交付

完成后返回：

1. 开始时实际HEAD；
2. 最终commit SHA；
3. PR #3更新状态；
4. 新增和修改文件；
5. D5/D6/D7模型定义；
6. D5 spatial scale结构和初始化；
7. D6 3×3 offset顺序与shift方向测试；
8. D7 side-head mapping与detach测试；
9. D5初始退化为D4测试；
10. D6中心权重退化为D4测试；
11. D7 uniform-prior退化为D6测试；
12. 禁止cdist/topk测试；
13. beta置零退化为D0测试；
14. CPU/CUDA前向、反向和AMP测试；
15. 真实NUAA batch=4 smoke test；
16. DWT/IDWT计数；
17. 参数、FLOPs、延迟和显存；
18. D5/D6/D7三数据集启动命令；
19. GPU、PID和输出目录；
20. 当前epoch；
21. 最终1000 epoch结果；
22. target/background空间诊断；
23. D0–D7完整对比；
24. `EXPERIMENT_D_HFE_SPATIAL_CONSISTENCY_RECORD.md`。

建议commit：

```text
Add D5-D7 spatial consistency ablations for decoder HFE
```

---

# 33. 最终公式

D5：

\[
S_s(p)
=
\operatorname{Cosine}
(
F_{HF,s}(p),
L_s(p)
)
\]

\[
C_s(p)
=
\left|
R_s(p)
-
\operatorname{AvgPool}_{3\times3}(R_s)(p)
\right|
\]

\[
A_s(p)
=
2\sigma(
\operatorname{Conv}
[
S_s(p),C_s(p)
]
)
\]

\[
\widetilde L_s(p)
=
A_s(p)L_s(p)
\]

D6：

\[
w_s(p,q)
=
\operatorname{Softmax}_{q\in\mathcal N(p)}
\left(
\frac{
Q_s(p)^TK_s(q)
}{
\tau_s
}
\right)
\]

\[
\widetilde L_s(p)
=
\sum_{q\in\mathcal N(p)}
w_s(p,q)L_s(q)
\]

D7：

\[
w_s^{T}(p,q)
=
\operatorname{Softmax}_{q\in\mathcal N(p)}
\left(
\frac{
Q_s(p)^TK_s(q)
}{
\tau_s
}
+
\eta_s\log(T_s(q)+\epsilon)
\right)
\]

\[
T_s
=
\sigma(
\operatorname{SideHead}_s(L_s)
)
\]

统一：

\[
R_s
=
\operatorname{DirectFusion}
(
F_{HF,s},
\widetilde L_s
)
\]

\[
\widehat H_s
=
\bar H_s
+
\beta_s^H\Delta H_s
\]

\[
\widehat V_s
=
\bar V_s
+
\beta_s^V\Delta V_s
\]

\[
\widehat D_s
=
\bar D_s
+
\beta_s^D\Delta D_s
\]

核心判断：

\[
\boxed{
\text{红外小目标更需要空间局部高低频一致性，而非全图通道对应}
}
\]
