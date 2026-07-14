# Experiment D：Decoder HFE Matching Ablation（D2 → D3）

## 0. 实验归属与编号

本方案属于 **Experiment D：SD-AWGM + Decoder-side HFE** 的内部消融，不建立新的 Experiment E。

完整消融序列统一定义为：

| 编号 | 模型 | Stage 1/2 | Stage 3/4 | 状态 |
|---|---|---|---|---|
| D0 | `sd_awgm` | 无 HFE | 无 HFE | 已有基础对照 |
| D1 | `sd_awgm_hfe` | Hard L2 Top-1 | Hard L2 Top-1 | Experiment D 主模型，已有结果 |
| D2 | `sd_awgm_hfe_softcos` | Soft Cosine Top-k | Soft Cosine Top-k | 本轮第一顺序 |
| D3 | `sd_awgm_hfe_scaleaware` | Correlation Gate | Soft Cosine Top-k | 本轮第二顺序 |

本轮严格按照：

```text
先 D2
后 D3
```

执行。

其中：

- D1 → D2：验证将 Hard L2 Top-1 改为 Soft Cosine Top-k 是否更稳定、更高效；
- D2 → D3：在深层 Soft Cosine 不变的条件下，仅将浅层匹配改为 Correlation Gate，验证浅层全局通道匹配是否导致复杂背景误增强。

---

# 1. 当前问题与消融逻辑

Experiment D 的阶段性结果显示：

```text
NUDT-SIRST：
sd_awgm      mIoU ≈ 0.9058
sd_awgm_hfe  mIoU ≈ 0.9429

NUAA-SIRST：
sd_awgm      mIoU ≈ 0.7799
sd_awgm_hfe  mIoU ≈ 0.7747

IRSTD-1K：
sd_awgm      mIoU ≈ 0.6561
sd_awgm_hfe  mIoU ≈ 0.6488
```

D1 表现说明：

1. Decoder HFE 在 NUDT 上有效；
2. NUAA 和 IRSTD 的 Pd 有所提高，但 Fa 同时增加；
3. HFE 能提高目标敏感性，但可能增强复杂背景高频；
4. 四尺度统一使用全图 L2 Top-1 可能不适合浅层高分辨率特征；
5. `torch.cdist` 带来明显实际延迟；
6. Top-1 硬选择存在不连续和通道选择抖动。

因此消融顺序必须分两步。

## 1.1 D2：先验证匹配度量和硬选择问题

D2 四尺度全部替换为：

```text
Soft Cosine Top-k
```

D1 与 D2 的主要差异：

```text
D1：
L2距离 + Top-1硬选择

D2：
Cosine相似度 + Top-k软聚合
```

用于回答：

> HFE 在 NUAA/IRSTD 上的问题，是否主要来自 L2 幅值敏感和 Top-1 硬选择？

## 1.2 D3：再验证浅层匹配机制问题

D3 保持：

```text
Stage 3/4 = 与 D2 完全相同的 Soft Cosine Top-k
```

仅修改：

```text
Stage 1/2：
Soft Cosine Top-k → Local Correlation Gate
```

用于回答：

> 即使采用 Soft Cosine，浅层是否仍不适合全图通道匹配？

这样 D2 → D3 只有浅层 relation 模块发生变化，消融关系清楚。

---

# 2. 基础分支

基于：

```text
Repository:
RiyaoChan/DWTFreqNet

Base branch:
codex/experiment-d-sd-awgm-decoder-hfe

Known base commit:
6fb19768dd7013aff536447b39652a44c1538912
```

Codex 开始前执行：

```bash
git checkout codex/experiment-d-sd-awgm-decoder-hfe
git pull
git rev-parse HEAD
```

若 Experiment D 分支已有更新，以实际最新 commit 为准，并写入记录。

建议新建消融分支：

```text
codex/experiment-d-hfe-matching-ablation-d2-d3
```

分支名称必须保留 `experiment-d`。

---

# 3. 代码隔离

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
model/DWTFreqNet_SingleDecoder.py
model/DWTFreqNet_SingleDecoder_LDRC.py
model/DWTFreqNet_SingleDecoder_HFE.py
```

D0 和 D1 的模型代码、checkpoint 和实验记录必须保持不变。

新建：

```text
model/DWTFreqNet_SingleDecoder_HFE_Ablation.py
```

建议只实现一个可控消融类：

```python
class DWTFreqNet_SingleDecoder_HFE_Ablation(
    DWTFreqNet_SingleDecoder_HFE
):
    ...
```

支持两个固定变体：

```text
d2_softcos_all
d3_scaleaware
```

禁止加入其他未计划变体。

---

# 4. 共同结构保持不变

D1、D2、D3 共同保留：

```text
四级 DWT Encoder
DirectionalBandEncoder
Stage-wise AWGM
SubbandSelectiveFusion（SKFF式 H/V/D 融合）
Decoder HFE 所在位置
Channel Attention 主体
FFN 主体
H/V/D 三个独立方向残差头
beta=1e-3 的系数残差校正
四级 IDWT
Single Wavelet Decoder
Deep Supervision
训练损失和评价流程
```

唯一消融轴：

```text
Decoder HFE 中 high-frequency 与 decoder low-frequency 的 relation/matching 方式
```

---

# 5. D2 与 D3 固定配置

## 5.1 D2：四尺度 Soft Cosine Top-k

```python
D2_STAGE_CONFIG = {
    1: {
        "mode": "soft_cosine_topk",
        "channels": 64,
        "num_heads": 1,
        "topk": 8,
        "temperature": 0.1,
    },
    2: {
        "mode": "soft_cosine_topk",
        "channels": 128,
        "num_heads": 2,
        "topk": 8,
        "temperature": 0.1,
    },
    3: {
        "mode": "soft_cosine_topk",
        "channels": 256,
        "num_heads": 4,
        "topk": 8,
        "temperature": 0.1,
    },
    4: {
        "mode": "soft_cosine_topk",
        "channels": 256,
        "num_heads": 4,
        "topk": 8,
        "temperature": 0.1,
    },
}
```

## 5.2 D3：浅层 Gate，深层 Soft Cosine

```python
D3_STAGE_CONFIG = {
    1: {
        "mode": "local_correlation_gate",
        "channels": 64,
        "num_heads": 1,
    },
    2: {
        "mode": "local_correlation_gate",
        "channels": 128,
        "num_heads": 2,
    },
    3: {
        "mode": "soft_cosine_topk",
        "channels": 256,
        "num_heads": 4,
        "topk": 8,
        "temperature": 0.1,
    },
    4: {
        "mode": "soft_cosine_topk",
        "channels": 256,
        "num_heads": 4,
        "topk": 8,
        "temperature": 0.1,
    },
}
```

关键约束：

```text
D2 与 D3 的 Stage 3/4 必须完全相同；
D2 与 D3 的唯一结构差异是 Stage 1/2 relation 模块。
```

---

# 6. Soft Cosine Top-k Matching

## 6.1 输入

```text
query_feature:     [B,C,H,W]，HFE中的高频特征
candidate_feature: [B,C,H,W]，当前decoder低频特征
```

D2 四个尺度均使用该模块。

D3 仅 Stage 3/4 使用该模块。

## 6.2 展平与余弦归一化

```python
query = query_feature.flatten(2)
candidate = candidate_feature.flatten(2)

query_norm = F.normalize(
    query.float(),
    dim=-1,
)

candidate_norm = F.normalize(
    candidate.float(),
    dim=-1,
)
```

形状：

```text
query_norm:     [B,C,HW]
candidate_norm: [B,C,HW]
```

## 6.3 通道余弦相似度

```python
similarity = torch.matmul(
    query_norm,
    candidate_norm.transpose(-2, -1),
)
```

形状：

```text
Stage 1: [B, 64, 64]
Stage 2: [B,128,128]
Stage 3: [B,256,256]
Stage 4: [B,256,256]
```

公式：

\[
S_{ij}
=
\frac{
Q_i^TK_j
}{
\|Q_i\|_2\|K_j\|_2+\epsilon
}
\]

禁止使用：

```text
torch.cdist
L2距离
argmin
Hard Top-1
```

## 6.4 Top-k软权重

```python
topk_values, topk_indices = torch.topk(
    similarity,
    k=self.topk,
    dim=-1,
)
```

温度参数：

```python
temperature = (
    self.log_temperature.exp()
    .clamp(min=0.03, max=1.0)
)
```

权重：

```python
topk_weights = torch.softmax(
    topk_values / temperature,
    dim=-1,
)
```

聚合时不要强制构造完整 `C×C` 稀疏权重矩阵，优先直接 gather 候选通道。

推荐实现：

```python
candidate_expanded = candidate.unsqueeze(1).expand(
    -1,
    candidate.shape[1],
    -1,
    -1,
)

gather_index = topk_indices.unsqueeze(-1).expand(
    -1,
    -1,
    -1,
    candidate.shape[-1],
)

selected = torch.gather(
    candidate_expanded,
    dim=2,
    index=gather_index,
)

matched = (
    selected
    * topk_weights.unsqueeze(-1)
).sum(dim=2)

matched = matched.reshape_as(
    candidate_feature
).to(candidate_feature.dtype)
```

但如果 `expand + gather` 的显存或速度不理想，也可构造稀疏权重后用矩阵乘法。两种实现必须通过数值一致性测试。

## 6.5 模块接口

```python
class SoftCosineTopKMatching(nn.Module):
    def __init__(
        self,
        channels,
        topk=8,
        temperature=0.1,
    ):
        super().__init__()

        if topk > channels:
            raise ValueError(...)

        self.channels = channels
        self.topk = topk

        self.log_temperature = nn.Parameter(
            torch.tensor(
                float(temperature)
            ).log()
        )

    def forward(
        self,
        query_feature,
        candidate_feature,
    ):
        ...
        return matched, info
```

---

# 7. Soft Cosine 诊断统计

`info` 至少包含：

```text
similarity_shape
topk_indices_shape
topk_weights_shape
temperature
matching_entropy
normalized_matching_entropy
effective_candidate_count
candidate_usage_ratio
most_used_candidate_frequency
topk_weight_max
topk_weight_min
```

匹配熵：

\[
H_i
=
-\sum_{j=1}^{k}
w_{ij}\log(w_{ij}+\epsilon)
\]

归一化熵：

\[
H_i^{norm}
=
\frac{H_i}{\log k}
\]

有效候选数：

\[
N_{eff}
=
\exp(H_i)
\]

解释：

```text
entropy接近0：
退化为近似Top-1

entropy接近1：
权重可能过度平均
```

正式训练中不得长期保存完整 similarity tensor，只记录形状、必要索引和标量统计。

---

# 8. SoftMatchingTransformation

```python
class SoftMatchingTransformation(nn.Module):
    def __init__(
        self,
        channels,
        topk=8,
        temperature=0.1,
    ):
        super().__init__()

        self.matching = SoftCosineTopKMatching(
            channels=channels,
            topk=topk,
            temperature=temperature,
        )

        combined = channels * 2

        self.gate = nn.Conv2d(
            combined,
            combined,
            1,
        )

        self.value = nn.Conv2d(
            combined,
            combined,
            3,
            padding=1,
            groups=combined,
            bias=False,
        )

        self.project = nn.Conv2d(
            combined,
            channels,
            1,
            bias=False,
        )

    def forward(self, x, perception):
        matched, info = self.matching(
            x,
            perception,
        )

        combined = torch.cat(
            [x, matched],
            dim=1,
        )

        output = self.project(
            torch.sigmoid(
                self.gate(combined)
            )
            * self.value(combined)
        )

        return output, info
```

该模块在 D2 的 Stage 1–4 使用，在 D3 的 Stage 3–4 使用。

---

# 9. Local Correlation Gate

仅 D3 的 Stage 1/2 使用。

## 9.1 输入

```text
high_feature: [B,C,H,W]
low_feature:  [B,C,H,W]
```

## 9.2 关系特征

\[
R
=
[
F_H,
F_L,
F_H\odot F_L,
|F_H-F_L|
]
\]

```python
relation = torch.cat(
    [
        high_feature,
        low_feature,
        high_feature * low_feature,
        torch.abs(
            high_feature - low_feature
        ),
    ],
    dim=1,
)
```

## 9.3 Gate结构

```python
self.gate = nn.Sequential(
    nn.Conv2d(
        channels * 4,
        channels,
        1,
        bias=False,
    ),
    nn.GELU(),
    nn.Conv2d(
        channels,
        channels,
        3,
        padding=1,
        groups=channels,
        bias=False,
    ),
    nn.Conv2d(
        channels,
        channels,
        1,
    ),
    nn.Sigmoid(),
)
```

高低频值分支：

```python
self.high_value = nn.Sequential(
    nn.Conv2d(
        channels,
        channels,
        1,
        bias=False,
    ),
    nn.Conv2d(
        channels,
        channels,
        3,
        padding=1,
        groups=channels,
        bias=False,
    ),
)

self.low_value = nn.Sequential(
    nn.Conv2d(
        channels,
        channels,
        1,
        bias=False,
    ),
    nn.Conv2d(
        channels,
        channels,
        3,
        padding=1,
        groups=channels,
        bias=False,
    ),
)
```

输出：

\[
F_{out}
=
(1-G)\odot V_H
+
G\odot V_L
\]

```python
gate = self.gate(relation)

output = (
    (1.0 - gate)
    * self.high_value(high_feature)
    + gate
    * self.low_value(low_feature)
)
```

## 9.4 接口

```python
class LocalCorrelationGate(nn.Module):
    def __init__(self, channels):
        ...

    def forward(
        self,
        high_feature,
        low_feature,
    ):
        ...
        return output, info
```

`info`记录：

```text
gate_mean
gate_std
gate_min
gate_max
low_selected_ratio
high_selected_ratio
```

---

# 10. Relation工厂

```python
def build_relation_module(
    mode,
    channels,
    topk=8,
    temperature=0.1,
):
    if mode == "soft_cosine_topk":
        return SoftMatchingTransformation(
            channels=channels,
            topk=topk,
            temperature=temperature,
        )

    if mode == "local_correlation_gate":
        return LocalCorrelationGate(
            channels=channels,
        )

    raise ValueError(
        f"Unknown HFE relation mode: {mode}"
    )
```

Attention 和 FFN 必须分别创建独立 relation module，不能共享参数实例。

---

# 11. Ablation Channel Attention

保留 D1 的通道注意力主体，只替换 Q 与低频交互模块。

```python
class AblationChannelAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        relation_module,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(...)

        self.channels = channels
        self.num_heads = num_heads
        self.relation = relation_module

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            1,
            bias=False,
        )

        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )

        self.temperature = nn.Parameter(
            torch.ones(
                num_heads,
                1,
                1,
            )
        )

        self.project_out = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False,
        )
```

forward：

```python
q, k, v = self.qkv_dw(
    self.qkv(high_feature)
).chunk(3, dim=1)

q, relation_info = self.relation(
    q,
    low_feature,
)

q = q.reshape(
    B, heads, C_head, H * W
)
k = k.reshape(
    B, heads, C_head, H * W
)
v = v.reshape(
    B, heads, C_head, H * W
)

q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)

attention = torch.softmax(
    (
        q @ k.transpose(-2, -1)
    ) * self.temperature,
    dim=-1,
)

output = (
    attention @ v
).reshape(B, C, H, W)

return (
    self.project_out(output),
    relation_info,
)
```

---

# 12. Ablation FFN

```python
class AblationFFN(nn.Module):
    def __init__(
        self,
        channels,
        relation_module,
    ):
        super().__init__()

        self.project_in = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False,
        )

        self.dwconv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.relation = relation_module

        self.dwconv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.project_out = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False,
        )

    def forward(
        self,
        high_feature,
        low_feature,
    ):
        x = self.dwconv1(
            self.project_in(high_feature)
        )

        x, relation_info = self.relation(
            x,
            low_feature,
        )

        x = F.gelu(
            self.dwconv2(x)
        )

        return (
            self.project_out(x),
            relation_info,
        )
```

---

# 13. Ablation HFE Block

```python
class AblationDecoderHFEBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        relation_mode,
        topk=8,
        temperature=0.1,
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
            relation_module=build_relation_module(
                relation_mode,
                channels,
                topk,
                temperature,
            ),
        )

        self.ffn = AblationFFN(
            channels=channels,
            relation_module=build_relation_module(
                relation_mode,
                channels,
                topk,
                temperature,
            ),
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

---

# 14. Ablation HFE Refiner

继续复用 D1 的：

```text
SubbandSelectiveFusion
DirectionalResidualHead
beta=1e-3
```

```python
class AblationDecoderHFERefiner(
    nn.Module
):
    def __init__(
        self,
        channels,
        num_heads,
        stage,
        relation_mode,
        topk=8,
        temperature=0.1,
    ):
        super().__init__()

        self.stage = stage
        self.relation_mode = relation_mode

        self.subband_fusion = (
            SubbandSelectiveFusion(
                channels,
                reduction=8,
            )
        )

        self.hfe = (
            AblationDecoderHFEBlock(
                channels=channels,
                num_heads=num_heads,
                relation_mode=relation_mode,
                topk=topk,
                temperature=temperature,
            )
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

forward：

```python
assert_hfe_inputs(
    low_feature,
    base_h,
    base_v,
    base_d,
    self.stage,
)

shared_hf = self.subband_fusion(
    base_h,
    base_v,
    base_d,
)

refined_hf, relation_info = (
    self.hfe(
        shared_hf,
        low_feature,
    )
)

delta_h = self.head_h(
    base_h,
    refined_hf,
)
delta_v = self.head_v(
    base_v,
    refined_hf,
)
delta_d = self.head_d(
    base_d,
    refined_hf,
)

coef_h = (
    base_h
    + self.beta_h * delta_h
)
coef_v = (
    base_v
    + self.beta_v * delta_v
)
coef_d = (
    base_d
    + self.beta_d * delta_d
)

return (
    coef_h,
    coef_v,
    coef_d,
    debug,
)
```

返回接口必须与 D1 的 `DecoderHFERefiner` 完全一致，直接复用父类 forward。

---

# 15. 消融模型构造

```python
HFE_ABLATION_VARIANTS = (
    "d2_softcos_all",
    "d3_scaleaware",
)
```

```python
class DWTFreqNet_SingleDecoder_HFE_Ablation(
    DWTFreqNet_SingleDecoder_HFE
):
    def __init__(
        self,
        config,
        hfe_ablation,
        ...,
    ):
        if hfe_ablation not in (
            HFE_ABLATION_VARIANTS
        ):
            raise ValueError(...)

        super().__init__(...)

        stage_config = (
            D2_STAGE_CONFIG
            if hfe_ablation
            == "d2_softcos_all"
            else D3_STAGE_CONFIG
        )

        for stage in range(1, 5):
            cfg = stage_config[stage]

            setattr(
                self,
                f"decoder_hfe{stage}",
                AblationDecoderHFERefiner(
                    channels=cfg["channels"],
                    num_heads=cfg["num_heads"],
                    stage=stage,
                    relation_mode=cfg["mode"],
                    topk=cfg.get(
                        "topk",
                        8,
                    ),
                    temperature=cfg.get(
                        "temperature",
                        0.1,
                    ),
                ),
            )
```

父类创建的旧 `decoder_hfe1–4` 必须被完全替换。

---

# 16. 模型命名与元数据

## 16.1 D2

```python
self.experiment_group = "experiment_d"
self.experiment_type = "ablation"
self.ablation_axis = "decoder_hfe_matching"
self.ablation_id = "D2"

self.model_variant = (
    "dwtfreqnet_single_decoder_hfe_softcos"
)

self.sd_variant = (
    "sd_awgm_hfe_softcos"
)

self.hfe_ablation = (
    "d2_softcos_all"
)
```

## 16.2 D3

```python
self.experiment_group = "experiment_d"
self.experiment_type = "ablation"
self.ablation_axis = "decoder_hfe_matching"
self.ablation_id = "D3"

self.model_variant = (
    "dwtfreqnet_single_decoder_hfe_scaleaware"
)

self.sd_variant = (
    "sd_awgm_hfe_scaleaware"
)

self.hfe_ablation = (
    "d3_scaleaware"
)
```

共同：

```python
self.directional_pyramid = False
self.second_dwt = False
self.ldrc = False
self.mamba = False
```

---

# 17. 训练入口

新建：

```text
train_experiment_d_hfe_ablation.py
```

参数：

```text
--hfe-ablation d2_softcos_all
--hfe-ablation d3_scaleaware
```

不要提供其他值。

模型：

```python
model = (
    DWTFreqNet_SingleDecoder_HFE_Ablation(
        get_DWTFreqNet_config(),
        hfe_ablation=args.hfe_ablation,
        mode=mode,
        deepsuper=True,
    )
)
```

---

# 18. 输出目录

D2：

```text
runs/experiment_d_ablation/
    D2_softcos_all/
    <dataset>/
    seed42/
```

D3：

```text
runs/experiment_d_ablation/
    D3_scaleaware/
    <dataset>/
    seed42/
```

不得覆盖 D0/D1。

---

# 19. 单元测试

新建：

```text
tools/test_experiment_d_hfe_matching_ablation.py
```

## 19.1 D2模块配置

检查：

```text
Stage 1–4：
SoftMatchingTransformation
```

## 19.2 D3模块配置

检查：

```text
Stage 1–2：
LocalCorrelationGate

Stage 3–4：
SoftMatchingTransformation
```

## 19.3 D2/D3深层严格一致

复制相同随机种子构造后，检查 D2 与 D3 的 Stage 3/4：

```text
模块类型一致
超参数一致
参数形状一致
```

## 19.4 输出形状

```text
训练：
6 × [2,1,256,256]

测试：
[2,1,256,256]
```

## 19.5 DWT/IDWT

```text
DWT=4
IDWT=4
```

## 19.6 Soft Cosine形状

```text
Stage1：
similarity [B,64,64]
topk_indices [B,64,8]

Stage2：
similarity [B,128,128]
topk_indices [B,128,8]

Stage3/4：
similarity [B,256,256]
topk_indices [B,256,8]
```

Top-k权重和为1。

## 19.7 Correlation Gate形状

D3：

```text
Stage1 gate:
[B,64,128,128]

Stage2 gate:
[B,128,64,64]
```

范围为 `[0,1]`。

## 19.8 禁止torch.cdist

Monkeypatch：

```python
torch.cdist = forbidden_function
```

D2 与 D3 forward 均必须通过。

D1不执行该断言。

## 19.9 beta置零退化测试

D2、D3分别与：

```text
sd_awgm
```

比较。

复制共同参数，将全部 beta 置零，eval输出应一致。

## 19.10 梯度与AMP

D2/D3均检查：

```text
relation模块
temperature
attention
FFN
方向残差头
beta
encoder/decoder主路径
```

CUDA autocast前向和反向不得出现 NaN/Inf。

---

# 20. 复杂度统计

新建：

```text
tools/profile_experiment_d_hfe_matching_ablation.py
```

比较：

```text
D0：sd_awgm
D1：sd_awgm_hfe
D2：sd_awgm_hfe_softcos
D3：sd_awgm_hfe_scaleaware
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

重点比较：

```text
D1 → D2：
Hard L2 vs Soft Cosine

D2 → D3：
全尺度Soft Cosine vs 浅层Correlation Gate
```

---

# 21. 实验执行顺序

## Phase 0：代码和测试

1. 实现共用 Soft Cosine 和 Correlation Gate；
2. 实现 D2、D3 配置；
3. 完成所有单元测试；
4. 完成复杂度和延迟预检；
5. 确认 D0/D1 不受影响。

## Phase 1：先执行 D2

正式训练：

```text
D2：sd_awgm_hfe_softcos
```

数据集优先顺序：

```text
1. NUAA-SIRST
2. IRSTD-1K
3. NUDT-SIRST
```

原因：

```text
D2首先验证Soft Cosine是否能降低复杂背景误增强；
NUAA和IRSTD是当前主要问题数据集；
NUDT用于检查是否保留D1收益。
```

若有3张空闲GPU，可三数据集并行；否则按上述队列。

D2未完成基本可用性验证前，不启动D3正式训练。

基本可用性要求：

```text
无NaN/Inf
无OOM
评估流程正常
best_metrics.json正常更新
至少完成epoch 100后的首次正式评估
```

## Phase 2：再执行 D3

D2通过基本可用性验证后，启动：

```text
D3：sd_awgm_hfe_scaleaware
```

数据集顺序同样为：

```text
1. NUAA-SIRST
2. IRSTD-1K
3. NUDT-SIRST
```

D3必须使用与D2完全相同的训练配置。

## Phase 3：最终比较

比较顺序：

```text
D0 → D1：
是否需要Decoder HFE

D1 → D2：
Soft Cosine是否优于Hard L2 Top-1

D2 → D3：
浅层Correlation Gate是否优于浅层Soft Cosine
```

---

# 22. 统一训练设置

D2和D3与D1完全一致：

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

禁止修改：

```text
数据划分
loss
数据增强
优化器
学习率
评价阈值
```

---

# 23. 启动脚本

新建：

```text
scripts/run_experiment_d_hfe_ablation.sh
scripts/launch_experiment_d_hfe_ablation_queue.sh
```

脚本支持：

```bash
bash scripts/run_experiment_d_hfe_ablation.sh \
  d2_softcos_all \
  NUAA-SIRST \
  0
```

```bash
bash scripts/run_experiment_d_hfe_ablation.sh \
  d3_scaleaware \
  NUAA-SIRST \
  0
```

队列脚本必须保证：

```text
D2优先
D3后启动
```

不得因D3排队而抢占或停止D2。

---

# 24. 记录文件

新建辅助记录：

```text
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
```

并在：

```text
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
```

增加链接和摘要。

必须明确：

```text
D2和D3均为Experiment D内部消融，不是新的主实验。
```

---

# 25. 结果表

| Dataset | ID | Model | Stage1/2 | Stage3/4 | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Latency |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA | D0 | sd_awgm | None | None | | | | | | | |
| NUAA | D1 | sd_awgm_hfe | Hard L2 | Hard L2 | | | | | | | |
| NUAA | D2 | sd_awgm_hfe_softcos | Soft Cosine | Soft Cosine | | | | | | | |
| NUAA | D3 | sd_awgm_hfe_scaleaware | Correlation Gate | Soft Cosine | | | | | | | |
| NUDT | D0 | sd_awgm | None | None | | | | | | | |
| NUDT | D1 | sd_awgm_hfe | Hard L2 | Hard L2 | | | | | | | |
| NUDT | D2 | sd_awgm_hfe_softcos | Soft Cosine | Soft Cosine | | | | | | | |
| NUDT | D3 | sd_awgm_hfe_scaleaware | Correlation Gate | Soft Cosine | | | | | | | |
| IRSTD | D0 | sd_awgm | None | None | | | | | | | |
| IRSTD | D1 | sd_awgm_hfe | Hard L2 | Hard L2 | | | | | | | |
| IRSTD | D2 | sd_awgm_hfe_softcos | Soft Cosine | Soft Cosine | | | | | | | |
| IRSTD | D3 | sd_awgm_hfe_scaleaware | Correlation Gate | Soft Cosine | | | | | | | |

---

# 26. 结果判定

## 26.1 D2 > D1

说明：

```text
HFE思路有效；
Hard L2 Top-1限制了稳定性或泛化；
Soft Cosine Top-k更适合作为高低频通道关系。
```

## 26.2 D2降低Fa但NUDT略降

说明：

```text
软匹配更保守、泛化更好；
Hard Top-1对规则目标数据具有更强激进增强。
```

## 26.3 D3 > D2

说明：

```text
浅层不适合全局通道匹配；
局部空间相关门控更能抑制背景高频误增强。
```

## 26.4 D2 > D3

说明：

```text
浅层仍需要显式通道语义匹配；
Correlation Gate可能过度混入低频或削弱细节。
```

## 26.5 D1 > D2和D3

说明：

```text
Hard L2 Top-1可能是D1在NUDT上有效的关键；
或Soft Cosine的temperature/top-k导致过度平均。
```

本轮不立即调整top-k和temperature，先完成消融结论。

---

# 27. 本轮不继续扩展

D2和D3完成前，不新增：

```text
D4不同topk
D5不同temperature
D6分方向Matching
D7仅深层HFE
D8Window Cross-Attention
HFE + LDRC
```

---

# 28. 建议新增文件

```text
model/DWTFreqNet_SingleDecoder_HFE_Ablation.py
train_experiment_d_hfe_ablation.py
tools/test_experiment_d_hfe_matching_ablation.py
tools/profile_experiment_d_hfe_matching_ablation.py
scripts/run_experiment_d_hfe_ablation.sh
scripts/launch_experiment_d_hfe_ablation_queue.sh
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
```

允许最小修改：

```text
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
EXPERIMENT_RECORD.md
README.md
```

---

# 29. Codex最终交付

返回：

1. 实际base commit；
2. 新分支；
3. commit SHA；
4. D0/D1/D2/D3定义；
5. 新增和修改文件；
6. D2四尺度配置；
7. D3四尺度配置；
8. D2与D3深层一致性检查；
9. Soft Cosine shape和权重和测试；
10. Correlation Gate shape和范围测试；
11. 禁止torch.cdist测试；
12. beta置零退化测试；
13. 前向、反向、AMP测试；
14. DWT/IDWT计数；
15. 参数/FLOPs/延迟/显存；
16. D2三数据集启动状态；
17. D3队列或启动状态；
18. GPU/PID/输出目录；
19. Experiment D消融记录。

建议commit：

```text
Add D2 and D3 matching ablations for decoder HFE
```

---

# 30. 最终实验顺序

\[
\boxed{
D0:
\text{无HFE}
}
\]

\[
\boxed{
D1:
\text{Hard L2 Top-1 / Hard L2 Top-1}
}
\]

\[
\boxed{
D2:
\text{Soft Cosine Top-k / Soft Cosine Top-k}
}
\]

\[
\boxed{
D3:
\text{Correlation Gate / Soft Cosine Top-k}
}
\]

执行顺序：

\[
\boxed{
D2\rightarrow D3
}
\]

其中 D2 先验证匹配度量与硬选择，D3 再验证浅层匹配策略。
