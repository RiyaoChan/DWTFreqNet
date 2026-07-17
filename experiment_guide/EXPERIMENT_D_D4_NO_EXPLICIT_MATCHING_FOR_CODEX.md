# Experiment D：D4 No-Explicit-Matching Decoder HFE 消融方案

## 0. 实验归属与核心问题

本方案属于 **Experiment D：SD-AWGM + Decoder-side HFE** 的内部消融，不建立新的 Experiment E。

现有消融序列：

| ID | 模型 | Stage 1/2 | Stage 3/4 |
|---|---|---|---|
| D0 | `sd_awgm` | 无 HFE | 无 HFE |
| D1 | `sd_awgm_hfe` | Hard L2 Top-1 | Hard L2 Top-1 |
| D2 | `sd_awgm_hfe_softcos` | Soft Cosine Top-k | Soft Cosine Top-k |
| D3 | `sd_awgm_hfe_scaleaware` | Local Correlation Gate | Soft Cosine Top-k |

新增：

| ID | 模型 | Stage 1/2 | Stage 3/4 |
|---|---|---|---|
| D4 | `sd_awgm_hfe_nomatch` | Direct Low-conditioned Fusion | Direct Low-conditioned Fusion |

D4 需要回答的关键问题是：

> Decoder HFE 是否真的需要显式计算高频通道与低频通道之间的相似性、距离、Top-k或最近邻对应关系？

D4 不计算：

```text
L2距离
Cosine相似度矩阵
Top-1
Top-k
argmin
通道索引
C×C Matching矩阵
```

但 D4 仍然保留：

```text
当前decoder低频对高频的条件引导
```

具体做法是：

```text
高频特征 x
当前decoder低频 low
       │
       ▼
直接 concat[x, low]
       │
       ▼
与D1/D2相同形式的 gate/value/project
       │
       ▼
HFE Attention和FFN
```

因此，D4 是：

```text
No Explicit Matching
```

而不是：

```text
No Low-frequency Guidance
```

---

# 1. 为什么需要D4

D1、D2、D3 的最终 mIoU 差异很小：

```text
NUAA：
D1 0.774700
D2 0.773369
D3 0.776066

NUDT：
D1 0.943166
D2 0.946951
D3 0.947825

IRSTD：
D1 0.657358
D2 0.658223
D3 0.657163
```

三数据集平均：

```text
D1 ≈ 0.791741
D2 ≈ 0.792848
D3 ≈ 0.793685
```

D1、D2、D3 的主要区别是 relation/matching 方式，但性能差异有限。这可能说明：

1. 显式通道匹配不是 Decoder HFE 的关键因素；
2. 模型真正需要的只是低频语义参与高频重构；
3. 高频与低频通道之间不存在稳定的一一或Top-k语义对应；
4. 全图通道相似度主要受到大面积背景影响；
5. Matching 机制增加复杂度，但没有形成稳定跨数据集收益。

D4 用最直接的低频条件融合替代所有显式 Matching，可对上述判断进行关键验证。

---

# 2. 消融逻辑

完整比较顺序：

```text
D0 → D1：
加入带Hard L2 Matching的Decoder HFE是否有效

D1 → D2：
Hard L2 Top-1改为Soft Cosine Top-k是否有效

D2 → D3：
浅层Soft Cosine改为Correlation Gate是否有效

D1/D2/D3 → D4：
显式通道匹配是否有必要
```

特别需要比较：

## 2.1 D0 vs D4

回答：

> 不使用显式Matching时，低频条件引导的Decoder HFE本身是否有价值？

## 2.2 D1/D2/D3 vs D4

回答：

> 显式通道距离、相似度和候选选择是否提供了额外收益？

---

# 3. 基础分支与PR

继续使用当前 Experiment D 消融分支和 Draft PR，不新建新的实验分支。

```text
Repository:
RiyaoChan/DWTFreqNet

Working branch:
codex/experiment-d-hfe-matching-ablation-d2-d3

Draft PR:
PR #3

Current known HEAD:
14cfa930ec234aeda1b1d432390d49001794f52c
```

Codex 开始前执行：

```bash
git checkout codex/experiment-d-hfe-matching-ablation-d2-d3
git pull
git rev-parse HEAD
git status
```

以实际最新 HEAD 为准。

D4 完成后继续推送到同一分支，并更新 PR #3，不新建 Experiment E PR。

---

# 4. 代码隔离要求

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
model/DWTFreqNet_SingleDecoder.py
model/DWTFreqNet_SingleDecoder_LDRC.py
model/DWTFreqNet_SingleDecoder_HFE.py
```

不得改变 D0、D1、D2、D3 的模型行为、参数、训练入口和已有结果。

主要修改现有文件：

```text
model/DWTFreqNet_SingleDecoder_HFE_Ablation.py
train_experiment_d_hfe_ablation.py
tools/test_experiment_d_hfe_matching_ablation.py
tools/profile_experiment_d_hfe_matching_ablation.py
scripts/run_experiment_d_hfe_ablation.sh
scripts/launch_experiment_d_hfe_ablation_queue.sh
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
```

可按需要新增：

```text
scripts/launch_experiment_d_d4.sh
```

但优先复用现有 Experiment D 消融脚本。

---

# 5. D4唯一变量

D4 保持 D1 的以下结构不变：

```text
SubbandSelectiveFusion
HFE Channel Attention主体
HFE FFN主体
LayerNorm2d
三个DirectionalResidualHead
beta=1e-3
四级逐步Decoder HFE位置
四级IDWT
Single Decoder
Deep Supervision
```

唯一变化：

```text
D1：
selected_low = HardL2Top1Match(x, low)

D4：
selected_low = low
```

随后两者均执行：

```text
combined = concat[x, selected_low]
output = project(sigmoid(gate(combined)) * value(combined))
```

因此 D4 是最干净的“No Explicit Matching”对照。

---

# 6. D4四尺度配置

新增：

```python
D4_STAGE_CONFIG = {
    1: {
        "mode": "direct_low_fusion",
        "channels": 64,
        "num_heads": 1,
    },
    2: {
        "mode": "direct_low_fusion",
        "channels": 128,
        "num_heads": 2,
    },
    3: {
        "mode": "direct_low_fusion",
        "channels": 256,
        "num_heads": 4,
    },
    4: {
        "mode": "direct_low_fusion",
        "channels": 256,
        "num_heads": 4,
    },
}
```

四个尺度均不得使用：

```text
SoftCosineTopKMatching
LocalCorrelationGate
ChannelMatching
torch.cdist
torch.topk
argmin
```

---

# 7. DirectFusionTransformation

## 7.1 功能

直接使用当前 decoder 低频，不进行通道重排或候选选择。

输入：

```text
x:          [B,C,H,W]
perception: [B,C,H,W]
```

输出：

```text
output:     [B,C,H,W]
```

## 7.2 结构

```text
x ───────────────┐
                 ├─ concat ─ gate/value/project ─ output
perception ──────┘
```

数学形式：

\[
Z=[X,L]
\]

\[
G=\sigma(\operatorname{Conv}_{1\times1}(Z))
\]

\[
V=\operatorname{DWConv}_{3\times3}(Z)
\]

\[
Y=\operatorname{Conv}_{1\times1}(G\odot V)
\]

其中：

```text
X = 当前高频特征
L = 当前decoder低频特征
```

## 7.3 建议实现

```python
class DirectFusionTransformation(nn.Module):
    # Low-conditioned fusion without explicit channel matching.

    def __init__(self, channels):
        super().__init__()

        self.channels = int(channels)
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

        self.record_statistics = True
        self.last_info = None

    def forward(self, x, perception):
        if x.shape != perception.shape:
            raise RuntimeError(
                "DirectFusionTransformation requires "
                f"equal shapes, got {tuple(x.shape)} "
                f"and {tuple(perception.shape)}"
            )

        if x.shape[1] != self.channels:
            raise RuntimeError(
                f"Expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )

        combined = torch.cat(
            [x, perception],
            dim=1,
        )

        gate = torch.sigmoid(
            self.gate(combined)
        )

        value = self.value(combined)

        output = self.project(
            gate * value
        )

        info = {
            "input_shape": tuple(x.shape),
            "relation_mode": "direct_low_fusion",
        }

        if self.record_statistics:
            detached_gate = gate.detach().float()

            def rms(tensor):
                return float(
                    tensor.detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt()
                    .cpu()
                )

            high_norm = rms(x)
            low_norm = rms(perception)

            info.update(
                {
                    "gate_mean": float(
                        detached_gate.mean().cpu()
                    ),
                    "gate_std": float(
                        detached_gate.std().cpu()
                    ),
                    "gate_min": float(
                        detached_gate.min().cpu()
                    ),
                    "gate_max": float(
                        detached_gate.max().cpu()
                    ),
                    "high_norm": high_norm,
                    "low_norm": low_norm,
                    "output_norm": rms(output),
                    "low_high_norm_ratio": (
                        low_norm
                        / (high_norm + 1e-12)
                    ),
                }
            )

        self.last_info = info
        return output, info
```

---

# 8. 为什么D4使用这种Direct Fusion

该结构具有以下控制优势：

1. D1、D2、D4 的 `gate/value/project` 形式一致；
2. D4 只删除通道Matching步骤；
3. D4 不引入额外空间注意力；
4. D4 不引入新的Top-k、温度或窗口超参数；
5. D4 参数量应与 D1 基本一致；
6. D4 仍保留低频对高频的条件引导；
7. 可直接判断显式通道对应是否必要。

不要将 D4 实现为：

```text
q = q
```

因为这会同时删除低频引导，无法区分：

```text
显式Matching是否必要
```

和：

```text
低频条件信息是否必要
```

---

# 9. Relation工厂修改

现有：

```python
def build_relation_module(
    mode,
    channels,
    topk=8,
    temperature=0.1,
):
    ...
```

增加：

```python
if mode == "direct_low_fusion":
    return DirectFusionTransformation(
        channels=channels,
    )
```

完整逻辑：

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

    if mode == "direct_low_fusion":
        return DirectFusionTransformation(
            channels=channels,
        )

    raise ValueError(
        f"Unknown HFE relation mode: {mode}"
    )
```

---

# 10. 复用AblationChannelAttention与AblationFFN

D4 继续使用现有：

```text
AblationChannelAttention
AblationFFN
AblationDecoderHFEBlock
AblationDecoderHFERefiner
```

不新增另一套 Attention 或 FFN。

现有接口：

```python
q, relation_info = self.relation(
    q,
    low_feature,
)
```

D4 的 `DirectFusionTransformation` 返回相同接口，因此应直接兼容。

FFN 同理：

```python
x, relation_info = self.relation(
    x,
    low_feature,
)
```

---

# 11. 扩展消融变体

修改：

```python
HFE_ABLATION_VARIANTS = (
    "d2_softcos_all",
    "d3_scaleaware",
    "d4_no_matching",
)
```

注意：

```text
d4_no_matching
```

的准确含义是：

```text
No Explicit Channel Matching
+
Direct Low-conditioned Fusion
```

不是完全忽略低频。

---

# 12. Stage Config选择逻辑

不要继续使用简单二选一：

```python
D2 if ... else D3
```

改为明确映射：

```python
HFE_ABLATION_STAGE_CONFIGS = {
    "d2_softcos_all": D2_STAGE_CONFIG,
    "d3_scaleaware": D3_STAGE_CONFIG,
    "d4_no_matching": D4_STAGE_CONFIG,
}
```

构造：

```python
self.stage_config = (
    HFE_ABLATION_STAGE_CONFIGS[
        hfe_ablation
    ]
)
```

避免 D4 被错误映射到 D3。

---

# 13. D4模型命名与元数据

D4：

```python
self.experiment_group = "experiment_d"
self.experiment_type = "ablation"
self.ablation_axis = "decoder_hfe_matching"
self.ablation_id = "D4"

self.model_variant = (
    "dwtfreqnet_single_decoder_hfe_nomatch"
)

self.sd_variant = (
    "sd_awgm_hfe_nomatch"
)

self.hfe_ablation = (
    "d4_no_matching"
)

self.decoder_hfe_matching = (
    "direct_low_fusion_no_explicit_matching"
)

self.coefficient_mode = (
    "aligned_raw_plus_nomatch_hfe_directional_residual"
)
```

共同元数据：

```python
self.directional_pyramid = False
self.second_dwt = False
self.ldrc = False
self.mamba = False
```

---

# 14. experiment_metadata修正

当前 D2/D3 元数据统一写入：

```text
hfe_topk=8
hfe_initial_temperature=0.1
```

D4 不应错误记录这些字段。

建议：

```python
metadata.update(
    {
        "experiment_group": self.experiment_group,
        "experiment_type": self.experiment_type,
        "ablation_axis": self.ablation_axis,
        "ablation_id": self.ablation_id,
        "hfe_ablation": self.hfe_ablation,
        "model_base_commit": self.model_base_commit,
        "hfe_stage_modes": {
            str(stage): cfg["mode"]
            for stage, cfg
            in self.stage_config.items()
        },
        "explicit_channel_matching": (
            self.hfe_ablation
            != "d4_no_matching"
        ),
    }
)

if self.hfe_ablation in (
    "d2_softcos_all",
    "d3_scaleaware",
):
    metadata.update(
        {
            "hfe_topk": 8,
            "hfe_initial_temperature": 0.1,
        }
    )
else:
    metadata.update(
        {
            "hfe_topk": None,
            "hfe_initial_temperature": None,
            "direct_fusion_uses_raw_low": True,
            "channel_similarity_matrix": False,
            "channel_candidate_selection": False,
        }
    )
```

---

# 15. 参数公平性检查

D4 的 DirectFusionTransformation 使用：

```text
gate:    Conv2d(2C,2C,1)
value:   DWConv2d(2C,2C,3)
project: Conv2d(2C,C,1)
```

这与 D1/D2 MatchingTransformation 在完成 matching 后的融合头完全一致。

因此：

```text
D1 relation可训练参数
≈
D2 relation可训练参数
≈
D4 relation可训练参数
```

D2仅额外包含可学习温度标量。

必须增加测试：

```python
def count_trainable(module):
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )
```

比较每一级：

```text
D1 MatchingTransformation
D2 SoftMatchingTransformation
D4 DirectFusionTransformation
```

预期：

```text
D1与D4参数完全一致
D2比D4仅多1个temperature标量/每个relation实例
```

Attention 和 FFN 各一个 relation，因此 D2 每个stage比D4多2个标量。

---

# 16. 训练入口修改

现有：

```text
train_experiment_d_hfe_ablation.py
```

增加允许值：

```text
d4_no_matching
```

命令：

```bash
python train_experiment_d_hfe_ablation.py \
  --hfe-ablation d4_no_matching \
  ...
```

不得新建 `train_experiment_e.py`。

---

# 17. 输出目录

D4：

```text
runs/experiment_d_ablation/
    D4_no_matching/
    <dataset>/
    seed42/
```

完整示例：

```text
runs/experiment_d_ablation/D4_no_matching/NUAA-SIRST/seed42
runs/experiment_d_ablation/D4_no_matching/NUDT-SIRST/seed42
runs/experiment_d_ablation/D4_no_matching/IRSTD-1K/seed42
```

禁止覆盖：

```text
D1
D2
D3
```

已有目录。

---

# 18. Checkpoint要求

正式实验随机初始化，不加载：

```text
D0 checkpoint
D1 checkpoint
D2 checkpoint
D3 checkpoint
原始DWTFreqNet checkpoint
```

Checkpoint和`run_config.json`必须记录：

```json
{
  "experiment_group": "experiment_d",
  "experiment_type": "ablation",
  "ablation_axis": "decoder_hfe_matching",
  "ablation_id": "D4",
  "hfe_ablation": "d4_no_matching",
  "model_variant": "dwtfreqnet_single_decoder_hfe_nomatch",
  "sd_variant": "sd_awgm_hfe_nomatch",
  "explicit_channel_matching": false,
  "channel_similarity_matrix": false,
  "channel_candidate_selection": false,
  "direct_fusion_uses_raw_low": true,
  "stage1_relation": "direct_low_fusion",
  "stage2_relation": "direct_low_fusion",
  "stage3_relation": "direct_low_fusion",
  "stage4_relation": "direct_low_fusion",
  "directional_pyramid": false,
  "second_dwt": false,
  "ldrc": false,
  "mamba": false
}
```

---

# 19. 单元测试

更新：

```text
tools/test_experiment_d_hfe_matching_ablation.py
```

## 19.1 D4模块类型

检查：

```text
Stage 1–4：
DirectFusionTransformation
```

不得出现：

```text
SoftMatchingTransformation
SoftCosineTopKMatching
LocalCorrelationGate
ChannelMatching
MatchingTransformation
```

## 19.2 输出形状

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

## 19.3 中间形状

```text
Stage1 relation:
input/high/low/output = [2,64,128,128]

Stage2:
[2,128,64,64]

Stage3:
[2,256,32,32]

Stage4:
[2,256,16,16]
```

## 19.4 DWT/IDWT

```text
DWT=4
IDWT=4
```

## 19.5 禁止cdist和topk

D4前向测试中 monkeypatch：

```python
original_cdist = torch.cdist
original_topk = torch.topk

def forbidden_cdist(*args, **kwargs):
    raise AssertionError(
        "D4 must not call torch.cdist"
    )

def forbidden_topk(*args, **kwargs):
    raise AssertionError(
        "D4 must not call torch.topk"
    )

torch.cdist = forbidden_cdist
torch.topk = forbidden_topk

try:
    model(x)
finally:
    torch.cdist = original_cdist
    torch.topk = original_topk
```

D4完整forward必须通过。

不要禁止：

```text
torch.matmul
```

因为 HFE 内部 Channel Attention 本身仍包含通道自注意力矩阵，这不属于高低频显式 Matching。

## 19.6 不存在Matching模块

```python
for module in model.modules():
    assert not isinstance(
        module,
        SoftCosineTopKMatching,
    )

    assert not isinstance(
        module,
        SoftMatchingTransformation,
    )

    assert not isinstance(
        module,
        LocalCorrelationGate,
    )
```

并检查类名不包含：

```text
ChannelMatching
MatchingTransformation
```

## 19.7 beta置零退化测试

构建：

```text
baseline:
DWTFreqNet_SingleDecoder(sd_awgm)

D4:
DWTFreqNet_SingleDecoder_HFE_Ablation(
    hfe_ablation="d4_no_matching"
)
```

复制共同参数。

将 D4 全部：

```text
beta_h
beta_v
beta_d
```

置零。

eval模式：

```python
torch.testing.assert_close(
    baseline_output,
    d4_output,
    rtol=1e-5,
    atol=1e-6,
)
```

最大绝对误差应接近0。

## 19.8 relation参数公平性

检查 D1 与 D4 的 relation融合头参数量完全一致。

分别检查：

```text
Attention relation
FFN relation
Stage1–4
```

## 19.9 梯度

执行前向、loss和backward，检查：

```text
DirectFusionTransformation.gate
DirectFusionTransformation.value
DirectFusionTransformation.project
AblationChannelAttention
AblationFFN
DirectionalResidualHead
beta_h/v/d
Encoder Stage-wise AWGM
decoder_fuse0–3
output heads
```

均有非零梯度。

## 19.10 AMP

CUDA autocast前后向：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    outputs = model(x)
```

不得出现：

```text
NaN
Inf
OOM
```

## 19.11 真实数据smoke test

使用 NUAA：

```text
batch size=4
256×256
单个训练step
```

检查：

```text
前向
loss
反向
optimizer.step
显存
```

---

# 20. 诊断统计

D4不记录：

```text
similarity_shape
topk_indices
topk_weights
matching_entropy
candidate_usage_ratio
temperature
```

D4记录：

## 20.1 Direct Fusion

```text
gate_mean
gate_std
gate_min
gate_max
high_norm
low_norm
output_norm
low_high_norm_ratio
```

## 20.2 HFE共同统计

```text
SKFF mean_weight_H/V/D
SKFF weight_variance
shared_HF_norm
refined_HF_norm
refined/shared norm ratio
```

## 20.3 方向残差

```text
beta_H/V/D
delta/base norm ratio
beta*delta/base norm ratio
final/base norm ratio
```

---

# 21. 复杂度和速度测试

更新：

```text
tools/profile_experiment_d_hfe_matching_ablation.py
```

比较：

```text
D0：sd_awgm
D1：sd_awgm_hfe
D2：sd_awgm_hfe_softcos
D3：sd_awgm_hfe_scaleaware
D4：sd_awgm_hfe_nomatch
```

统一：

```text
RTX 3090
输入 [1,1,256,256]
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
DWT/IDWT count
Relation参数量
```

重点比较：

```text
D1 vs D4：
相同融合头下，删除Hard L2 Matching的影响

D2 vs D4：
删除Soft Cosine C×C匹配的影响

D3 vs D4：
删除尺度自适应relation的影响
```

预期：

```text
D4参数量约等于D1
D4不包含temperature
D4不产生高低频C×C Matching矩阵
```

不要预设 D4 一定显著更快，必须报告实测结果。

---

# 22. 正式实验顺序

D1、D2、D3 已完成。

D4直接进入正式训练：

```text
1. NUAA-SIRST
2. IRSTD-1K
3. NUDT-SIRST
```

原因：

```text
NUAA和IRSTD用于优先判断删除显式Matching后能否减少复杂背景误增强；
NUDT用于判断是否能保留HFE的大幅收益。
```

若有三张空闲GPU，可同时启动三个数据集。

不得停止、覆盖或抢占其他正式任务。

---

# 23. 统一训练设置

与 D1、D2、D3 完全一致：

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
优化器
学习率
scheduler
评价阈值
```

---

# 24. 启动脚本

更新：

```text
scripts/run_experiment_d_hfe_ablation.sh
scripts/launch_experiment_d_hfe_ablation_queue.sh
```

支持：

```bash
bash scripts/run_experiment_d_hfe_ablation.sh \
  d4_no_matching \
  NUAA-SIRST \
  0
```

若新增独立脚本：

```text
scripts/launch_experiment_d_d4.sh
```

必须只调用现有训练入口，不复制训练逻辑。

---

# 25. 结果记录

更新：

```text
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
```

新增章节：

```text
## D4：No Explicit Matching
```

同时更新：

```text
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
EXPERIMENT_RECORD.md
PR #3 body
```

明确：

```text
D4是Experiment D内部消融；
它保留低频条件融合，但删除显式通道相似性和候选选择。
```

---

# 26. 完整结果表

| Dataset | ID | Model | Relation | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Latency |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA | D0 | sd_awgm | None | 489 | 0.7799 | 0.7848 | 0.8764 | 0.9466 | 1.935e-5 | |
| NUAA | D1 | sd_awgm_hfe | Hard L2 Top-1 | 286 | 0.7747 | 0.7809 | 0.8731 | 0.9695 | 2.394e-5 | |
| NUAA | D2 | sd_awgm_hfe_softcos | Soft Cosine Top-k | 350 | 0.773369 | 0.782203 | 0.872203 | 0.973282 | 3.739e-5 | |
| NUAA | D3 | sd_awgm_hfe_scaleaware | Shallow Gate + Deep Soft Cosine | 549 | 0.776066 | 0.785939 | 0.873916 | 0.973282 | 3.211e-5 | |
| NUAA | D4 | sd_awgm_hfe_nomatch | Direct Low Fusion | | | | | | | |
| NUDT | D0 | sd_awgm | None | 556 | 0.9058 | 0.9019 | 0.9505 | 0.9852 | 4.182e-6 | |
| NUDT | D1 | sd_awgm_hfe | Hard L2 Top-1 | 694 | 0.943166 | 0.949027 | 0.970752 | 0.991534 | 4.343e-6 | |
| NUDT | D2 | sd_awgm_hfe_softcos | Soft Cosine Top-k | 419 | 0.946951 | 0.947689 | 0.972753 | 0.990476 | 1.953e-6 | |
| NUDT | D3 | sd_awgm_hfe_scaleaware | Shallow Gate + Deep Soft Cosine | 513 | 0.947825 | 0.949940 | 0.973214 | 0.995767 | 1.540e-6 | |
| NUDT | D4 | sd_awgm_hfe_nomatch | Direct Low Fusion | | | | | | | |
| IRSTD | D0 | sd_awgm | None | 894 | 0.6561 | 0.6477 | 0.7924 | 0.9091 | 1.537e-5 | |
| IRSTD | D1 | sd_awgm_hfe | Hard L2 Top-1 | 556 | 0.657358 | 0.658863 | 0.793260 | 0.922559 | 1.395e-5 | |
| IRSTD | D2 | sd_awgm_hfe_softcos | Soft Cosine Top-k | 464 | 0.658223 | 0.659029 | 0.793890 | 0.915825 | 1.704e-5 | |
| IRSTD | D3 | sd_awgm_hfe_scaleaware | Shallow Gate + Deep Soft Cosine | 735 | 0.657163 | 0.660729 | 0.793119 | 0.936027 | 1.782e-5 | |
| IRSTD | D4 | sd_awgm_hfe_nomatch | Direct Low Fusion | | | | | | | |

---

# 27. 结论判定规则

## 27.1 D4与D1/D2/D3相近或更好

若 D4 在三数据集平均 mIoU 上：

```text
不低于最佳D1/D2/D3超过0.002
```

或达到更好结果，则可得出：

> 显式通道相似性不是 Decoder HFE 的必要组成。模型主要需要低频条件融合，而非高低频通道之间的最近邻或Top-k对应。

此时建议后续主模型删除：

```text
L2/Cosine Matching
Top-k
Channel indices
C×C高低频相似度
```

## 27.2 D4明显优于D1/D2/D3

说明：

> 显式通道匹配不仅没有帮助，还可能受背景主导并干扰红外小目标高频重构。

D4可作为更简洁的 Decoder HFE 实现。

## 27.3 D4低于D1/D2/D3超过0.005

若至少两个数据集出现：

```text
mIoU下降 > 0.005
```

说明显式通道关系可能确实提供有效信息。

但仍需判断：

```text
是否只有NUDT受益
是否跨数据集稳定
是否值得复杂度代价
```

## 27.4 D4接近D0但低于D1/D2/D3

说明：

> Decoder HFE 的收益可能主要来自显式关系建模，而不是一般的低频条件融合。

## 27.5 D4高于D0但接近D1/D2/D3

这是最支持当前假设的结果：

> 低频引导高频是有效的，但显式通道相似性没有必要。

---

# 28. 平均指标

完成后计算：

```text
三数据集平均mIoU
三数据集平均nIoU
三数据集平均F1
三数据集平均Pd
三数据集平均Fa
```

同时计算：

```text
D4 - D0
D4 - D1
D4 - D2
D4 - D3
```

不要只报告单个数据集。

---

# 29. 随机波动说明

D1、D2、D3 的差异较小，D4同样只采用 seed 42 时，不应把小于约0.2–0.3个百分点的差异表述为确定性优势。

结果记录中明确区分：

```text
显著结构趋势
小幅数值波动
```

本轮先完成与现有实验一致的 seed 42 全训练。

不要自动启动多seed实验。

若 D4 与 D1–D3 的平均 mIoU 差异小于0.002，记录：

```text
需要多seed确认，但当前结果支持“显式Matching贡献有限”
```

---

# 30. 本轮不做

D4完成前不新增：

```text
D5完全忽略低频
D6仅低频加法
D7不同gate结构
D8方向独立Direct Fusion
D9HFE + LDRC
多seed扩展
```

先完成最关键的 No Explicit Matching 对照。

---

# 31. 建议修改文件

```text
model/DWTFreqNet_SingleDecoder_HFE_Ablation.py
train_experiment_d_hfe_ablation.py
tools/test_experiment_d_hfe_matching_ablation.py
tools/profile_experiment_d_hfe_matching_ablation.py
scripts/run_experiment_d_hfe_ablation.sh
scripts/launch_experiment_d_hfe_ablation_queue.sh
EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md
EXPERIMENT_D_SD_AWGM_HFE_RECORD.md
EXPERIMENT_RECORD.md
README.md
```

可新增：

```text
scripts/launch_experiment_d_d4.sh
```

---

# 32. Codex最终交付

完成后返回：

1. 开始时实际HEAD；
2. 最终commit SHA；
3. PR #3更新状态；
4. 修改文件列表；
5. D4模型定义；
6. `DirectFusionTransformation`结构；
7. D4四尺度relation配置；
8. D1与D4 relation参数公平性检查；
9. 禁止`torch.cdist`和`torch.topk`测试；
10. D4不存在Matching模块的检查；
11. beta置零退化测试；
12. CPU/CUDA前向、反向和AMP测试；
13. 真实NUAA batch=4 smoke test；
14. DWT/IDWT计数；
15. 参数、FLOPs、延迟和显存；
16. 三数据集训练启动命令；
17. GPU、PID和输出目录；
18. 当前epoch；
19. 三数据集1000 epoch最终结果；
20. D0–D4完整对比；
21. `EXPERIMENT_D_HFE_MATCHING_ABLATION_RECORD.md`更新。

建议commit message：

```text
Add D4 no-matching ablation for decoder HFE
```

---

# 33. 最终结构定义

D1：

\[
M_s=
\operatorname{HardL2Top1}
(F_{HF,s},L_s)
\]

D2：

\[
M_s=
\operatorname{SoftCosineTopK}
(F_{HF,s},L_s)
\]

D3：

\[
M_s=
\begin{cases}
\operatorname{LocalGate}(F_{HF,s},L_s),
& s=1,2\\
\operatorname{SoftCosineTopK}(F_{HF,s},L_s),
& s=3,4
\end{cases}
\]

D4：

\[
M_s=L_s
\]

\[
Z_s=[F_{HF,s},M_s]
\]

\[
R_s=
\operatorname{Project}
\left(
\sigma(\operatorname{Gate}(Z_s))
\odot
\operatorname{Value}(Z_s)
\right)
\]

最终系数：

\[
\widehat H_s
=
\bar H_s+\beta_s^H\Delta H_s
\]

\[
\widehat V_s
=
\bar V_s+\beta_s^V\Delta V_s
\]

\[
\widehat D_s
=
\bar D_s+\beta_s^D\Delta D_s
\]

核心判断：

\[
\boxed{
\text{低频条件融合是否足够，而无需显式通道相似性}
}
\]
