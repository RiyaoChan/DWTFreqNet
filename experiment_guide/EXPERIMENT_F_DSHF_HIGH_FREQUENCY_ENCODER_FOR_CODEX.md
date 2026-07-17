# Experiment F：DSHFBlock 高频方向稀疏特征提取实验

## 0. 实验定位

本实验建立新的 **Experiment F**，研究 DWT 后高频子带 `H/V/D` 的特征提取问题。

Experiment F 不属于 Experiment D，也不是继续扩展 Experiment E 的低频消融。其基础模型固定为 Experiment E 中表现更稳定的 E1：

```text
F0 = Experiment E1

LL → Wave-Mamba LFSSBlock → AWGM → 原 Res_block
H/V/D → 当前 DirectionalBandEncoder → AWGM
Decoder 高频系数仍使用原始 Haar H/V/D 经 1×1 Conv 对齐
```

Experiment F 只替换 Encoder 中用于 AWGM 的：

```text
DirectionalBandEncoder
```

为新设计的：

```text
DSHFBlock
Directional Sparse High-Frequency Block
方向稀疏高频块
```

保持不变：

```text
LFSSBlock
StageWiseAWGM
Encoder Res_block
原始 Haar H/V/D 保存方式
Decoder 原始高频系数路径
IDWT
Single Decoder
Deep Supervision
Loss
数据划分
训练设置
评价代码
```

---

## 1. 研究问题

当前 `DirectionalBandEncoder` 对每个高频子带只执行一次单尺度方向卷积：

```text
H：5×1 Depthwise Conv
V：1×5 Depthwise Conv
D：3×3 Depthwise Conv
```

随后：

```text
1×1 Conv → BN → GELU → 与原始子带相加
```

该模块没有显式处理：

```text
不同尺寸小目标的多尺度高频响应
高频噪声与背景杂波的稀疏筛选
H/V/D 多方向共同响应
单方向连续背景边缘
LFSS 低频目标位置对高频筛选的空间约束
```

核心假设：

> 红外小目标高频特征提取应同时建模方向多尺度响应、高频稀疏支持、跨方向局部一致性，并在完整版本中利用 LFSS 低频局部对比提供空间引导。

---

## 2. 实验变体

### F0：E1参考基线

```text
LL → LFSS → AWGM → 原 Res_block

H/V/D
→ 当前 DirectionalBandEncoder
→ AWGM
```

F0 不重新训练，优先复用已完成的 E1 seed42 结果，但必须先完成代码、数据、训练参数和评价逻辑一致性核验。

### F1：多尺度方向提取

```text
H/V/D
→ Directional Multi-scale Extractor
→ raw band + learned feature
→ AWGM
```

研究问题：

> 当前单尺度方向卷积是否限制了高频小目标特征提取？

### F2：多尺度方向提取 + 稀疏支持门控

```text
H/V/D
→ Directional Multi-scale Extractor
→ Adaptive Sparse Support Gate
→ raw band + learned feature
→ AWGM
```

研究问题：

> 对高频特征进行自适应稀疏筛选，是否能抑制噪声与低置信杂波？

### F3：多尺度 + 稀疏支持 + 跨方向一致性

```text
H/V/D
→ Directional Multi-scale Extractor
→ Adaptive Sparse Support Gate
→ Cross-direction Local Consistency Gate
→ raw band + learned feature
→ AWGM
```

研究问题：

> H/V/D 的局部共同响应是否能保留点状小目标，并抑制单方向连续背景边缘？

### F4：完整低频引导 DSHFBlock

```text
LFSS LL
   │
   └─→ Local Contrast Prior
                │
H/V/D           │
→ Multi-scale Directional Extractor
→ Adaptive Sparse Support Gate
→ Cross-direction Local Consistency Gate
   ◀──────── LFSS Low-frequency Local Contrast
→ raw band + learned feature
→ AWGM
```

研究问题：

> LFSS 提取后的低频局部目标线索，能否进一步约束高频筛选位置，降低高频背景误增强？

---

## 3. 正式执行优先级

按照用户指定顺序：

```text
F1 → F4 → F2 → F3
```

含义：

```text
F1：先验证最基础的多尺度方向提取
F4：尽快获得完整模块的性能上界
F2：拆解稀疏筛选的独立贡献
F3：拆解跨方向一致性的增量贡献
```

架构消融关系仍为：

```text
F1
 └─ + Sparse Gate = F2
      └─ + Cross-direction Gate = F3
           └─ + LFSS Low Guidance = F4
```

最终分析必须按：

```text
F1 vs F0
F2 vs F1
F3 vs F2
F4 vs F3
```

解释，不能因为启动顺序是 F1、F4、F2、F3 而改变消融逻辑。

---

## 4. 基础分支与新分支

### 4.1 基础分支

从 Experiment E 分支创建：

```text
Repository:
RiyaoChan/DWTFreqNet

Base branch:
codex/experiment-e-lfss-before-awgm

Known latest commit:
68ede894be748c8842427e140898f007dbe67953
```

Codex 开始前执行：

```bash
git fetch --all --prune
git checkout codex/experiment-e-lfss-before-awgm
git pull
git rev-parse HEAD
git status
```

记录实际 HEAD。若远端分支已更新，以实际最新 HEAD 为准。

### 4.2 新分支

```text
codex/experiment-f-dshf-high-frequency-encoder
```

```bash
git checkout -b codex/experiment-f-dshf-high-frequency-encoder
```

### 4.3 独立工作区

建议：

```text
/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_F_DSHF
```

不得复用：

```text
DWTFreqNet_EXPERIMENT_D_ABLATION
DWTFreqNet_EXPERIMENT_E_LFSS
```

不得停止、修改或覆盖现有 Experiment D/E 任务与结果。

### 4.4 新 Draft PR

Experiment F 创建独立 Draft PR，不继续更新 PR #3 或 PR #4。

---

## 5. 代码隔离

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_SingleDecoder.py
model/DWTFreqNet_SingleDecoder_LFSS_AWGM.py
model/DWTFreqNet_SingleDecoder_LDRC.py
model/DWTFreqNet_SingleDecoder_HFE.py
model/DWTFreqNet_SingleDecoder_HFE_Ablation.py
model/DWTFreqNet_SingleDecoder_HFE_SpatialAblation.py
model/third_party/wavemamba_lfss.py
```

新建：

```text
model/DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM.py
train_experiment_f_dshf.py
tools/test_experiment_f_dshf.py
tools/profile_experiment_f_dshf.py
tools/analyze_experiment_f_high_frequency.py
scripts/run_experiment_f_dshf.sh
scripts/launch_experiment_f_dshf_queue.sh
EXPERIMENT_F_DSHF_RECORD.md
```

允许最小修改：

```text
EXPERIMENT_RECORD.md
README.md
```

---

## 6. DSHFBlock总体定义

全称：

```text
Directional Sparse High-Frequency Block
```

输入：

```text
band_h: [B,C,H,W]
band_v: [B,C,H,W]
band_d: [B,C,H,W]
low_feature: optional [B,C,H,W]
```

输出：

```text
feature_h: [B,C,H,W]
feature_v: [B,C,H,W]
feature_d: [B,C,H,W]
```

固定处理顺序：

```text
1. Directional Multi-scale Extraction
2. Adaptive Sparse Support Gate（F2/F3/F4）
3. Cross-direction Local Consistency Gate（F3/F4）
4. LFSS Low-frequency Local Contrast Guidance（仅F4）
5. raw Haar band + learned residual feature
```

禁止加入：

```text
Mamba/SS2D 到 H/V/D
Transformer
全图 C×C 通道匹配
torch.cdist
top-k 通道选择
第二次 DWT
跨尺度高频金字塔
Decoder HFE
额外 loss
可学习外部 beta/gamma
```


## 7. Directional Multi-scale Extractor

### 7.1 H方向

```text
Branch 1：3×1 Depthwise Conv
Branch 2：5×1 Depthwise Conv
Concat
1×1 Conv
BatchNorm
GELU
```

\[
U_H=
\operatorname{GELU}
\left[
\operatorname{BN}
\left(
\operatorname{Conv}_{1\times1}
[
\operatorname{DWConv}_{3\times1}(H),
\operatorname{DWConv}_{5\times1}(H)
]
\right)
\right]
\]

### 7.2 V方向

```text
Branch 1：1×3 Depthwise Conv
Branch 2：1×5 Depthwise Conv
Concat
1×1 Conv
BatchNorm
GELU
```

### 7.3 D方向

```text
Branch 1：3×3 Depthwise Conv，dilation=1
Branch 2：3×3 Depthwise Conv，dilation=2，padding=2
Concat
1×1 Conv
BatchNorm
GELU
```

### 7.4 固定实现

```python
class DirectionalMultiScaleExtractor(nn.Module):
    def __init__(self, channels, direction):
        super().__init__()

        if direction == "H":
            kernel1, padding1, dilation1 = (3, 1), (1, 0), 1
            kernel2, padding2, dilation2 = (5, 1), (2, 0), 1
        elif direction == "V":
            kernel1, padding1, dilation1 = (1, 3), (0, 1), 1
            kernel2, padding2, dilation2 = (1, 5), (0, 2), 1
        elif direction == "D":
            kernel1, padding1, dilation1 = (3, 3), (1, 1), 1
            kernel2, padding2, dilation2 = (3, 3), (2, 2), 2
        else:
            raise ValueError(...)

        self.branch1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel1,
            padding=padding1,
            dilation=dilation1,
            groups=channels,
            bias=False,
        )

        self.branch2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel2,
            padding=padding2,
            dilation=dilation2,
            groups=channels,
            bias=False,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fuse(torch.cat(
            [self.branch1(x), self.branch2(x)],
            dim=1,
        ))
```

三个方向参数独立，不共享权重。

---

## 8. Adaptive Sparse Support Gate

用于 F2、F3、F4。

### 8.1 阈值估计

对于方向特征 \(U_d\)：

\[
M_d=\operatorname{GAP}(|U_d|)
\]

\[
Q_d=\sigma(\operatorname{MLP}(M_d))
\]

\[
\tau_d=M_d\odot Q_d
\]

### 8.2 稀疏支持

\[
G_d^{sparse}
=
\sigma
\left(
\frac{|U_d|-\tau_d}{\tau_d+\epsilon}
\right)
\]

\[
S_d=U_d\odot G_d^{sparse}
\]

`G_sparse` 范围为 `(0,1)`。不采用硬阈值，避免直接截断弱小目标。

### 8.3 固定实现

```python
class AdaptiveSparseSupportGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden_channels = max(
            channels // reduction,
            8,
        )

        self.threshold_predictor = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                channels,
                1,
                bias=True,
            ),
        )

        self.reset_control_parameters()

    def reset_control_parameters(self):
        last_conv = self.threshold_predictor[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, feature):
        magnitude = feature.abs()

        mean_magnitude = F.adaptive_avg_pool2d(
            magnitude,
            output_size=1,
        )

        threshold_ratio = torch.sigmoid(
            self.threshold_predictor(
                mean_magnitude
            )
        )

        threshold = (
            mean_magnitude
            * threshold_ratio
        )

        support = torch.sigmoid(
            (magnitude - threshold)
            / (threshold + 1e-6)
        )

        output = feature * support

        return output, {
            "mean_magnitude": mean_magnitude,
            "threshold_ratio": threshold_ratio,
            "threshold": threshold,
            "support": support,
        }
```

最后一层零初始化，使初始：

```text
threshold_ratio = 0.5
```

该控制初始化在通用模型初始化之后必须恢复。

---

## 9. Cross-direction Local Consistency Gate

用于 F3、F4。

### 9.1 方向能量

\[
E_H=\operatorname{Mean}_C(|S_H|),\quad
E_V=\operatorname{Mean}_C(|S_V|),\quad
E_D=\operatorname{Mean}_C(|S_D|)
\]

### 9.2 有界归一化

\[
\bar E_d=
\frac{E_d}{\operatorname{Mean}_{HW}(E_d)+\epsilon}
\]

\[
\widetilde E_d=
\frac{\bar E_d}{1+\bar E_d}
\]

联合能量：

\[
E_J=
\sqrt{
\widetilde E_H^2+
\widetilde E_V^2+
\widetilde E_D^2+
\epsilon
}
\]

### 9.3 方向空间调制

F3输入：

```text
[E_H_norm, E_V_norm, E_D_norm, E_J]
```

F4输入：

```text
[E_H_norm, E_V_norm, E_D_norm, E_J, C_L]
```

输出：

\[
[A_H,A_V,A_D]
=
2\sigma(\operatorname{LocalGate}(\cdot))
\]

\[
C_H=S_H\odot A_H,\quad
C_V=S_V\odot A_V,\quad
C_D=S_D\odot A_D
\]

最后一层零初始化，使初始：

```text
A_H = A_V = A_D = 1
```

因此：

```text
F3初始化严格退化为F2
F4在复制共享参数后初始化严格退化为F3
```

### 9.4 固定实现

```python
class CrossDirectionLocalConsistencyGate(nn.Module):
    def __init__(self, use_low_guidance):
        super().__init__()

        input_channels = (
            5 if use_low_guidance else 4
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                input_channels,
                12,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                12,
                3,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        self.reset_control_parameters()

    def reset_control_parameters(self):
        last_conv = self.gate[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    @staticmethod
    def normalize_energy(energy):
        normalized = energy / (
            energy.mean(
                dim=(2, 3),
                keepdim=True,
            )
            + 1e-6
        )
        return normalized / (
            1.0 + normalized
        )

    def forward(
        self,
        feature_h,
        feature_v,
        feature_d,
        low_contrast=None,
    ):
        energy_h = self.normalize_energy(
            feature_h.abs().mean(
                dim=1,
                keepdim=True,
            )
        )
        energy_v = self.normalize_energy(
            feature_v.abs().mean(
                dim=1,
                keepdim=True,
            )
        )
        energy_d = self.normalize_energy(
            feature_d.abs().mean(
                dim=1,
                keepdim=True,
            )
        )

        joint_energy = torch.sqrt(
            energy_h.square()
            + energy_v.square()
            + energy_d.square()
            + 1e-6
        )

        relation = [
            energy_h,
            energy_v,
            energy_d,
            joint_energy,
        ]

        if low_contrast is not None:
            relation.append(low_contrast)

        scales = 2.0 * torch.sigmoid(
            self.gate(torch.cat(
                relation,
                dim=1,
            ))
        )

        scale_h, scale_v, scale_d = (
            scales.chunk(3, dim=1)
        )

        return (
            feature_h * scale_h,
            feature_v * scale_v,
            feature_d * scale_d,
            {
                "energy_h": energy_h,
                "energy_v": energy_v,
                "energy_d": energy_d,
                "joint_energy": joint_energy,
                "scales": scales,
            },
        )
```


## 10. LFSS Low-frequency Local Contrast Prior

仅用于 F4。

输入必须是本级：

```text
refined_a = LFSS(raw_LL)
```

不得使用：

```text
raw_LL
AWGM输出
Encoder Res_block输出
GT mask
side head预测
```

### 10.1 低频响应与局部对比

\[
P_L=
\operatorname{Conv}_{1\times1}(L_{LFSS})
\]

\[
C_L=
\left|
P_L-
\operatorname{AvgPool}_{3\times3}(P_L)
\right|
\]

有界归一化：

\[
\bar C_L=
\frac{C_L}{\operatorname{Mean}_{HW}(C_L)+\epsilon}
\]

\[
\widetilde C_L=
\frac{\bar C_L}{1+\bar C_L}
\]

输出：

```text
[B,1,H,W]
```

### 10.2 固定实现

```python
class LFSSLowFrequencyLocalContrast(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.response = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=False,
        )

    def forward(self, low_feature):
        response = self.response(
            low_feature
        )

        local_mean = F.avg_pool2d(
            response,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        contrast = (
            response - local_mean
        ).abs()

        normalized = contrast / (
            contrast.mean(
                dim=(2, 3),
                keepdim=True,
            )
            + 1e-6
        )

        bounded = normalized / (
            1.0 + normalized
        )

        return bounded, {
            "low_response": response,
            "low_contrast_raw": contrast,
            "low_contrast": bounded,
        }
```

---

## 11. DSHFBlock完整实现

### 11.1 变体配置

```python
DSHF_VARIANT_CONFIGS = {
    "f1_multiscale": {
        "use_sparse_gate": False,
        "use_cross_direction": False,
        "use_low_guidance": False,
    },
    "f2_sparse": {
        "use_sparse_gate": True,
        "use_cross_direction": False,
        "use_low_guidance": False,
    },
    "f3_cross_direction": {
        "use_sparse_gate": True,
        "use_cross_direction": True,
        "use_low_guidance": False,
    },
    "f4_low_guided_full": {
        "use_sparse_gate": True,
        "use_cross_direction": True,
        "use_low_guidance": True,
    },
}
```

### 11.2 类定义

```python
class DSHFBlock(nn.Module):
    def __init__(self, channels, variant):
        super().__init__()

        if variant not in DSHF_VARIANT_CONFIGS:
            raise ValueError(...)

        config = DSHF_VARIANT_CONFIGS[
            variant
        ]

        self.channels = int(channels)
        self.variant = variant

        self.use_sparse_gate = config[
            "use_sparse_gate"
        ]
        self.use_cross_direction = config[
            "use_cross_direction"
        ]
        self.use_low_guidance = config[
            "use_low_guidance"
        ]

        self.extract_h = DirectionalMultiScaleExtractor(
            channels, "H"
        )
        self.extract_v = DirectionalMultiScaleExtractor(
            channels, "V"
        )
        self.extract_d = DirectionalMultiScaleExtractor(
            channels, "D"
        )

        if self.use_sparse_gate:
            self.sparse_h = AdaptiveSparseSupportGate(
                channels
            )
            self.sparse_v = AdaptiveSparseSupportGate(
                channels
            )
            self.sparse_d = AdaptiveSparseSupportGate(
                channels
            )

        if self.use_low_guidance:
            self.low_contrast = (
                LFSSLowFrequencyLocalContrast(
                    channels
                )
            )

        if self.use_cross_direction:
            self.cross_direction = (
                CrossDirectionLocalConsistencyGate(
                    use_low_guidance=(
                        self.use_low_guidance
                    )
                )
            )

    def reset_control_parameters(self):
        if self.use_sparse_gate:
            self.sparse_h.reset_control_parameters()
            self.sparse_v.reset_control_parameters()
            self.sparse_d.reset_control_parameters()

        if self.use_cross_direction:
            self.cross_direction.reset_control_parameters()

    def forward(
        self,
        band_h,
        band_v,
        band_d,
        low_feature=None,
    ):
        assert_hf_inputs(
            band_h,
            band_v,
            band_d,
            expected_channels=self.channels,
        )

        if self.use_low_guidance:
            if low_feature is None:
                raise RuntimeError(
                    "F4 requires LFSS low feature"
                )
            if low_feature.shape != band_h.shape:
                raise RuntimeError(...)

        feature_h = self.extract_h(band_h)
        feature_v = self.extract_v(band_v)
        feature_d = self.extract_d(band_d)

        debug = {
            "multiscale_h": feature_h,
            "multiscale_v": feature_v,
            "multiscale_d": feature_d,
        }

        if self.use_sparse_gate:
            feature_h, sparse_h = self.sparse_h(
                feature_h
            )
            feature_v, sparse_v = self.sparse_v(
                feature_v
            )
            feature_d, sparse_d = self.sparse_d(
                feature_d
            )

            debug.update({
                "sparse_h": sparse_h,
                "sparse_v": sparse_v,
                "sparse_d": sparse_d,
            })

        low_contrast = None

        if self.use_low_guidance:
            low_contrast, low_info = (
                self.low_contrast(
                    low_feature
                )
            )
            debug["low_guidance"] = low_info

        if self.use_cross_direction:
            (
                feature_h,
                feature_v,
                feature_d,
                cross_info,
            ) = self.cross_direction(
                feature_h,
                feature_v,
                feature_d,
                low_contrast=low_contrast,
            )
            debug["cross_direction"] = (
                cross_info
            )

        output_h = band_h + feature_h
        output_v = band_v + feature_v
        output_d = band_d + feature_d

        debug.update({
            "output_h": output_h,
            "output_v": output_v,
            "output_d": output_d,
        })

        return (
            output_h,
            output_v,
            output_d,
            debug,
        )
```

### 11.3 高频残差约束

固定：

\[
H^{out}=H+R_H,\quad
V^{out}=V+R_V,\quad
D^{out}=D+R_D
\]

不增加：

```text
beta
gamma
LayerScale
额外 raw/output blend
```

这里保留一次 `raw band + learned feature`，与原 `DirectionalBandEncoder` 的残差职责一致。

---

## 12. Stage配置

```python
DSHF_STAGE_CONFIG = {
    1: {"channels": 32},
    2: {"channels": 64},
    3: {"channels": 128},
    4: {"channels": 256},
}
```

每个 stage 使用独立 DSHFBlock。

首轮不搜索：

```text
卷积核组合
dilation
sparse reduction
cross-direction hidden channels
stage选择
模块数量
```

---

## 13. Experiment F模型

新建：

```text
model/DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM.py
```

### 13.1 变体名

```python
EXPERIMENT_F_VARIANTS = (
    "f1_multiscale",
    "f2_sparse",
    "f3_cross_direction",
    "f4_low_guided_full",
)
```

训练优先级：

```python
EXPERIMENT_F_LAUNCH_ORDER = (
    "f1_multiscale",
    "f4_low_guided_full",
    "f2_sparse",
    "f3_cross_direction",
)
```

### 13.2 继承E1

```python
class DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM(
    DWTFreqNet_SingleDecoder_LFSS_AWGM
):
    def __init__(
        self,
        config,
        hf_variant,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if hf_variant not in (
            EXPERIMENT_F_VARIANTS
        ):
            raise ValueError(...)

        super().__init__(
            config=config,
            encoder_variant=(
                "e1_lfss_resblock"
            ),
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
        )

        self.hf_variant = hf_variant

        for stage in range(1, 5):
            channels = DSHF_STAGE_CONFIG[
                stage
            ]["channels"]

            setattr(
                self,
                f"dir_encoder{stage}",
                DSHFBlock(
                    channels=channels,
                    variant=hf_variant,
                ),
            )
```

通过 `setattr` 直接替换父类原有 `DirectionalBandEncoder`，不得保留未使用的旧高频编码器参数。

---

## 14. `_encode_stage`顺序

```python
def _encode_stage(self, stage, tensor):
    (
        band_a,
        band_h,
        band_v,
        band_d,
    ) = self._dwt(tensor)

    refined_a = self.lfss_blocks[
        str(stage)
    ](band_a)

    (
        feature_h,
        feature_v,
        feature_d,
        hf_debug,
    ) = getattr(
        self,
        f"dir_encoder{stage}",
    )(
        band_h,
        band_v,
        band_d,
        low_feature=(
            refined_a
            if self.hf_variant
            == "f4_low_guided_full"
            else None
        ),
    )

    guided_a = self._apply_stage_awgm(
        stage,
        refined_a,
        feature_h,
        feature_v,
        feature_d,
    )

    encoded = getattr(
        self,
        f"local_encoder{stage}",
    )(guided_a)

    if self.debug_tensors:
        self._experiment_f_debug[stage] = {
            "raw_ll": band_a.detach(),
            "lfss_ll": refined_a.detach(),
            "raw_h": band_h.detach(),
            "raw_v": band_v.detach(),
            "raw_d": band_d.detach(),
            "dshf": detach_nested(
                hf_debug
            ),
            "guided_ll": guided_a.detach(),
            "encoded": encoded.detach(),
        }

    return (
        encoded,
        (
            band_a,
            band_h,
            band_v,
            band_d,
        ),
        {
            "H": feature_h,
            "V": feature_v,
            "D": feature_d,
        },
        guided_a,
    )
```

严格顺序：

```text
DWT
→ LFSS(raw LL)
→ DSHF(raw H/V/D, optional LFSS LL)
→ AWGM(LFSS LL, DSHF H/V/D)
→ 原 Res_block
```


## 15. 原始高频Decoder路径必须保持不变

`raw_bands` 必须保存：

```text
band_a
band_h
band_v
band_d
```

不得把 DSHF 输出保存为 raw bands。

Decoder仍使用：

```text
raw Haar H/V/D
→ align_H/V/D 1×1 Conv
→ IDWT
```

DSHF输出只允许用于：

```text
StageWiseAWGM
离线诊断
```

不得进入：

```text
IDWT coefficient path
Decoder HFE
Directional Pyramid
```

---

## 16. 元数据

共同：

```python
self.experiment_group = "experiment_f"
self.experiment_type = (
    "encoder_high_frequency_ablation"
)
self.ablation_axis = (
    "pre_awgm_high_frequency_extractor"
)
self.base_low_frequency_variant = (
    "experiment_e_e1_lfss_resblock"
)
self.encoder_lfss = True
self.post_awgm_encoder = (
    "original_res_block"
)
self.high_frequency_encoder = (
    "dshf_block"
)
self.high_frequency_decoder_source = (
    "raw_haar_aligned"
)
self.dshf_multiscale = True
self.decoder_hfe = False
self.directional_pyramid = False
self.second_dwt = False
self.ldrc = False
self.coefficient_mode = "aligned_raw"
```

F1：

```python
self.ablation_id = "F1"
self.model_variant = (
    "dwtfreqnet_e1_dshf_multiscale"
)
self.sd_variant = (
    "e1_dshf_multiscale"
)
self.dshf_sparse_gate = False
self.dshf_cross_direction = False
self.dshf_low_guidance = False
```

F2：

```python
self.ablation_id = "F2"
self.model_variant = (
    "dwtfreqnet_e1_dshf_sparse"
)
self.sd_variant = "e1_dshf_sparse"
self.dshf_sparse_gate = True
self.dshf_cross_direction = False
self.dshf_low_guidance = False
```

F3：

```python
self.ablation_id = "F3"
self.model_variant = (
    "dwtfreqnet_e1_dshf_cross_direction"
)
self.sd_variant = (
    "e1_dshf_cross_direction"
)
self.dshf_sparse_gate = True
self.dshf_cross_direction = True
self.dshf_low_guidance = False
```

F4：

```python
self.ablation_id = "F4"
self.model_variant = (
    "dwtfreqnet_e1_dshf_low_guided_full"
)
self.sd_variant = (
    "e1_dshf_low_guided_full"
)
self.dshf_sparse_gate = True
self.dshf_cross_direction = True
self.dshf_low_guidance = True
self.dshf_low_guidance_source = (
    "same_stage_lfss_output"
)
```

---

## 17. 初始化

继续保护 Wave-Mamba LFSS 特殊初始化：

```text
dt_projs_weight
dt_projs_bias
A_logs
Ds
skip_scale
skip_scale2
```

同时在通用初始化之后恢复 DSHF 控制层初始化：

```text
Sparse threshold predictor最后一层 = 0
Cross-direction gate最后一层 = 0
```

建议：

```python
def initialize_experiment_f_model(
    model,
    baseline_init_fn,
):
    initialize_experiment_e_model(
        model,
        baseline_init_fn,
    )

    for stage in range(1, 5):
        getattr(
            model,
            f"dir_encoder{stage}",
        ).reset_control_parameters()
```

强制验证：

```text
LFSS特殊参数初始化前后最大差 = 0
Sparse最后一层权重/偏置 = 0
Cross gate最后一层权重/偏置 = 0
```

---

## 18. 训练入口

新建：

```text
train_experiment_f_dshf.py
```

参数：

```text
--hf-variant f1_multiscale
--hf-variant f4_low_guided_full
--hf-variant f2_sparse
--hf-variant f3_cross_direction
```

模型：

```python
model = (
    DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM(
        get_DWTFreqNet_config(),
        hf_variant=args.hf_variant,
        mode=mode,
        deepsuper=True,
    )
)
```

可以复用 Experiment E 的：

```text
数据读取
loss
optimizer
scheduler
evaluation
checkpoint格式
```

---

## 19. 输出目录

```text
runs/experiment_f_dshf/
    F1_multiscale/<dataset>/seed42/
    F4_low_guided_full/<dataset>/seed42/
    F2_sparse/<dataset>/seed42/
    F3_cross_direction/<dataset>/seed42/
```

不得写入 Experiment E 目录。

---

## 20. 单元测试

新建：

```text
tools/test_experiment_f_dshf.py
```

### 20.1 E1基线回归

构建原 E1，确认其：

```text
模型文件
LFSS配置
AWGM
Res_block
Decoder
输出格式
```

均未被 Experiment F 修改。

### 20.2 模块替换

F1–F4四级 `dir_encoder1–4` 必须均为 `DSHFBlock`，模型中不得保留未使用的 `DirectionalBandEncoder` 参数。

### 20.3 变体结构

```text
F1：Multi-scale only
F2：Multi-scale + Sparse
F3：Multi-scale + Sparse + Cross-direction
F4：全部四部分
```

### 20.4 Hook执行顺序

每级必须：

```text
LFSS → DSHF → AWGM → 原 Res_block
```

### 20.5 AWGM输入来源

要求：

```python
awgm_low_input == lfss_output
awgm_h_input == dshf_h
awgm_v_input == dshf_v
awgm_d_input == dshf_d
```

### 20.6 raw bands不被覆盖

要求：

```text
DWT raw H/V/D
raw_bands中保存的H/V/D
_refine_coefficients读取的H/V/D
```

完全一致，DSHF输出不得进入 raw coefficient path。

### 20.7 Stage形状

```text
Stage1：
LL/H/V/D      [B,32,128,128]
DSHF H/V/D     [B,32,128,128]
E1             [B,64,128,128]

Stage2：
[B,64,64,64] → [B,128,64,64]

Stage3：
[B,128,32,32] → [B,256,32,32]

Stage4：
[B,256,16,16] → [B,256,16,16]
```

### 20.8 F1残差退化

将F1三个方向 extractor 的融合卷积输出置零，要求：

```text
output_h == raw_h
output_v == raw_v
output_d == raw_d
```

### 20.9 Sparse Gate

检查：

```text
support shape == feature shape
0 < support < 1
threshold shape == [B,C,1,1]
threshold_ratio初始为0.5
```

非零位置上：

```text
sign(output) == sign(input)
```

### 20.10 F3初始化退化为F2

复制F2/F3共有参数，F3 cross gate保持零初始化，要求输出一致。

### 20.11 F4初始化退化为F3

复制F3/F4共有参数，F4 cross gate保持零初始化，要求输出一致。

### 20.12 F4低频来源

```text
F4 low_feature == LFSS_LL
F4 low_feature != raw_LL
```

### 20.13 F1/F2/F3不依赖低频

相同H/V/D、不同low tensor时，F1/F2/F3输出必须一致；F4允许不同。

### 20.14 DWT/IDWT

```text
DWT=4
IDWT=4
```

### 20.15 输出

```text
训练：6 × [2,1,256,256]
测试：[2,1,256,256]
```

### 20.16 梯度

F1检查多尺度分支；F2额外检查threshold predictor；F3额外检查cross-direction gate；F4额外检查low response Conv。LFSS、AWGM、Res_block、Decoder和side heads均须有梯度。

### 20.17 禁止错误模块

Monkeypatch：

```python
torch.cdist = forbidden
torch.topk = forbidden
```

F1–F4完整forward必须通过。

模型中不得包含：

```text
ChannelMatching
SoftCosineTopKMatching
DecoderHFE
DirectionalPyramid
```

### 20.18 CUDA与AMP

四变体均完成：

```text
FP32 forward/backward
AMP forward/backward
2×1×256×256
```

### 20.19 真实NUAA smoke test

四变体分别：

```text
batch=4
patch=256
forward
loss
backward
optimizer.step
```

全部通过后才允许正式训练。

---

## 21. 复杂度测试

新建：

```text
tools/profile_experiment_f_dshf.py
```

比较：

```text
F0/E1
F1
F4
F2
F3
```

统一：

```text
RTX 3090
输入 1×1×256×256
FP32
eval
warmup=5
repeat>=20
torch.cuda.synchronize()
```

报告：

```text
Parameters
DSHF parameters
Encoder parameters
THOP FLOPs
Latency
FPS
Inference peak memory
Training peak memory
DWT/IDWT
```

---

## 22. 高频离线诊断

新建：

```text
tools/analyze_experiment_f_high_frequency.py
```

只在 best checkpoint 上离线运行，不参与训练或模型选择。

### 22.1 GT对齐

```python
stage_mask = F.adaptive_max_pool2d(
    gt_mask.float(),
    output_size=feature.shape[-2:],
)
```

### 22.2 目标/背景响应比

对每个stage、每个方向记录：

```text
raw H/V/D
multi-scale H/V/D
sparse H/V/D
cross-gated H/V/D
final DSHF H/V/D
```

\[
R(F)=
\frac{
\operatorname{Mean}(|F|\mid Y=1)
}{
\operatorname{Mean}(|F|\mid Y=0)+\epsilon
}
\]

### 22.3 Sparse Gate诊断

```text
target_support_mean
background_support_mean
target_background_support_ratio
target_active_fraction
background_active_fraction
threshold_mean/std/min/max
threshold_ratio_mean
```

active定义：

```text
support > 0.5
```

### 22.4 跨方向一致性

```text
target_scale_H/V/D
background_scale_H/V/D
target_background_scale_ratio_H/V/D
target_joint_energy
background_joint_energy
target_background_joint_energy_ratio
```

### 22.5 方向熵

\[
p_d=
\frac{E_d}{E_H+E_V+E_D+\epsilon}
\]

\[
H_{dir}=
-\frac{\sum_d p_d\log(p_d+\epsilon)}{\log 3}
\]

记录：

```text
target_direction_entropy
background_direction_entropy
```

### 22.6 F4低频引导

```text
target_low_contrast_mean
background_low_contrast_mean
target_background_low_contrast_ratio
low_contrast与joint_energy的目标区相关性
low_contrast与joint_energy的背景区相关性
```

### 22.7 DSHF残差强度

\[
R_{\Delta,d}=
\frac{\|F_d^{out}-B_d\|_2}{\|B_d\|_2+\epsilon}
\]

### 22.8 AWGM联动

```text
AWGM direction weight H/V/D
AWGM target/background gate
LFSS LL target/background ratio
guided LL target/background ratio
```


## 23. 正式训练设置

与 Experiment E1 完全一致：

```text
seed=42
patch size=256
batch size=4
epochs=1000
optimizer=Adam
initial lr=1e-3
10 epoch warmup
CosineAnnealingLR
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
optimizer
lr
scheduler
threshold
```

全部随机初始化，不加载 E1/E2/D0-D7/Wave-Mamba pretrained checkpoint。

---

## 24. F0复用规则

F0就是已完成的E1。

当前参考结果：

| Dataset | best epoch | mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | 595 | 0.7842 | 0.7960 | 0.8790 | 0.9656 | 2.717e-5 |
| IRSTD-1K | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 |
| NUDT-SIRST | 456 | 0.9516 | 0.9482 | 0.9752 | 0.9947 | 1.724e-6 |

复用前必须验证：

```text
model/DWTFreqNet_SingleDecoder_LFSS_AWGM.py
model/third_party/wavemamba_lfss.py
dataset.py
train_one.py
loss
evaluation
训练参数
数据划分
```

与 Experiment F 基础完全一致。

---

## 25. 动态GPU队列

新建：

```text
scripts/launch_experiment_f_dshf_queue.sh
```

### 25.1 全局优先级

```text
1.  F1 NUAA-SIRST
2.  F4 NUAA-SIRST
3.  F2 NUAA-SIRST
4.  F3 NUAA-SIRST

5.  F1 IRSTD-1K
6.  F4 IRSTD-1K
7.  F2 IRSTD-1K
8.  F3 IRSTD-1K

9.  F1 NUDT-SIRST
10. F4 NUDT-SIRST
11. F2 NUDT-SIRST
12. F3 NUDT-SIRST
```

目的：

```text
优先在NUAA比较四种结构
随后验证复杂背景IRSTD
最后验证NUDT强基线场景
```

### 25.2 动态并发

队列每60秒检查空闲GPU，按优先级把下一个未启动任务分配到任意空闲GPU。

支持：

```text
GPU_ALLOWLIST
MAX_CONCURRENT
POLL_SECONDS=60
```

### 25.3 空闲条件

建议：

```text
memory.used < 1000 MiB
utilization.gpu < 10%
无其他compute PID
```

不得停止或抢占：

```text
Experiment D
Experiment E
其他用户任务
```

### 25.4 防竞态

沿用 Experiment E 修复后的：

```text
任务级GPU预留
队列单实例flock
子进程不继承队列锁
输出目录RUNNING.lock
PID存活检查
```

### 25.5 防重复启动

每个任务目录：

```text
RUNNING.lock
FAILED
TRAINING_COMPLETE
launcher.pid
python.pid
```

存在有效checkpoint时不得覆盖。FAILED任务不自动无限重启。

---

## 26. 单任务启动脚本

新建：

```text
scripts/run_experiment_f_dshf.sh
```

示例：

```bash
bash scripts/run_experiment_f_dshf.sh \
  f1_multiscale \
  NUAA-SIRST \
  0
```

```bash
bash scripts/run_experiment_f_dshf.sh \
  f4_low_guided_full \
  NUAA-SIRST \
  1
```

```bash
bash scripts/run_experiment_f_dshf.sh \
  f2_sparse \
  NUAA-SIRST \
  2
```

```bash
bash scripts/run_experiment_f_dshf.sh \
  f3_cross_direction \
  NUAA-SIRST \
  3
```

---

## 27. 结果表

| Dataset | ID | 高频模块 | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | F0/E1 | 原DirectionalBandEncoder | 595 | 0.7842 | 0.7960 | 0.8790 | 0.9656 | 2.717e-5 | | |
| NUAA | F1 | Multi-scale | | | | | | | | |
| NUAA | F2 | Multi-scale + Sparse | | | | | | | | |
| NUAA | F3 | F2 + Cross-direction | | | | | | | | |
| NUAA | F4 | F3 + LFSS Low Guidance | | | | | | | | |
| IRSTD | F0/E1 | 原DirectionalBandEncoder | 551 | 0.6647 | 0.6630 | 0.7985 | 0.9158 | 1.230e-5 | | |
| IRSTD | F1 | Multi-scale | | | | | | | | |
| IRSTD | F2 | Multi-scale + Sparse | | | | | | | | |
| IRSTD | F3 | F2 + Cross-direction | | | | | | | | |
| IRSTD | F4 | F3 + LFSS Low Guidance | | | | | | | | |
| NUDT | F0/E1 | 原DirectionalBandEncoder | 456 | 0.9516 | 0.9482 | 0.9752 | 0.9947 | 1.724e-6 | | |
| NUDT | F1 | Multi-scale | | | | | | | | |
| NUDT | F2 | Multi-scale + Sparse | | | | | | | | |
| NUDT | F3 | F2 + Cross-direction | | | | | | | | |
| NUDT | F4 | F3 + LFSS Low Guidance | | | | | | | | |

---

## 28. 结论判定

### F1 > F0

说明单尺度 DirectionalBandEncoder 限制了高频小目标特征提取，多尺度方向感受野有效。

### F2 > F1

说明高频自适应稀疏筛选有效。重点观察：

```text
Fa是否下降
target/background support ratio是否>1
```

### F2提高Pd但Fa同步增加

说明 Sparse Gate 只扩大了高频敏感性，没有形成背景抑制。

### F3 > F2

说明 H/V/D 多方向共同响应有助于识别点状目标并抑制单方向背景结构。

### F3 ≈ F2

说明跨方向一致性不是主要贡献，优先保留F2。

### F4 > F3

说明 LFSS 低频局部对比提供了有效空间先验。

理想现象：

```text
mIoU提高
Fa下降
target/background low contrast ratio > 1
高频残差更集中于目标区域
```

### F4提高Pd但Fa明显增加

说明低频局部对比同时响应了背景亮点，不能称为有效目标引导。

### F4最好，但F2/F3不单调

说明完整模块存在交互效应，不能把F4收益简单分解为各子模块独立相加。

### F1–F4均不优于F0

说明当前轻量 DirectionalBandEncoder 已足够，复杂高频处理可能放大背景高频或破坏原始 Haar 方向结构。

---

## 29. 多数据集判断

完成后计算：

```text
三数据集平均mIoU
平均nIoU
平均F1
平均Pd
平均Fa

F1-F0
F2-F1
F3-F2
F4-F3
F4-F0
```

小于约 `0.002` 的平均mIoU差异不直接表述为确定性优势。

同时综合：

```text
Fa
Pd
复杂度
延迟
显存
```

首轮仅seed42，不自动扩展多seed。

---

## 30. 本轮不做

F1/F4/F2/F3全部完成前不新增：

```text
不同卷积核
不同dilation
不同稀疏阈值公式
只在部分stage使用DSHF
高频Mamba
高频跨尺度金字塔
与Decoder HFE组合
不同loss
多seed
```

---

## 31. 实验记录

新建：

```text
EXPERIMENT_F_DSHF_RECORD.md
```

记录：

```text
base branch
base commit
new branch
Draft PR
文件列表
模型结构
初始化
单元测试
复杂度
GPU队列
每个任务PID
当前epoch
best指标
最终结果
高频离线诊断
结论
```

---

## 32. Codex最终交付

完成代码、测试与启动后返回：

1. 实际 Experiment E base HEAD；
2. Experiment F 新分支；
3. Draft PR；
4. 最终 commit SHA；
5. 新增和修改文件；
6. F0一致性核验；
7. F1/F2/F3/F4模块配置；
8. 四级 DirectionalMultiScaleExtractor 配置；
9. Sparse Gate 阈值和初始化；
10. Cross-direction Gate 零初始化；
11. F4 LFSS低频来源；
12. LFSS→DSHF→AWGM→ResBlock hook顺序；
13. AWGM输入来源测试；
14. raw H/V/D Decoder路径未改变测试；
15. F1残差退化测试；
16. F3初始化退化为F2测试；
17. F4初始化退化为F3测试；
18. DWT/IDWT计数；
19. 输出形状；
20. 梯度测试；
21. CUDA FP32/AMP测试；
22. 四变体NUAA batch4 smoke test；
23. 参数/FLOPs/延迟/显存；
24. 动态队列PID；
25. 实际空闲GPU；
26. F1、F4、F2、F3启动状态；
27. 12项任务输出目录和PID；
28. 当前epoch和best结果；
29. 1000 epoch最终结果；
30. F0–F4完整对比；
31. 高频target/background诊断；
32. `EXPERIMENT_F_DSHF_RECORD.md`更新。

建议commit：

```text
Add Experiment F DSHF high-frequency encoder ablations
```

---

## 33. 最终公式

F1：

\[
U_d=
\operatorname{DirectionalMultiScale}_d(B_d)
\]

\[
F_d^{out}=B_d+U_d
\]

F2：

\[
\tau_d=
\operatorname{GAP}(|U_d|)
\odot
\sigma(
\operatorname{MLP}(
\operatorname{GAP}(|U_d|)
)
)
\]

\[
S_d=
U_d\odot
\sigma
\left(
\frac{|U_d|-\tau_d}{\tau_d+\epsilon}
\right)
\]

\[
F_d^{out}=B_d+S_d
\]

F3：

\[
A_H,A_V,A_D=
2\sigma(
\operatorname{LocalGate}
[
E_H,E_V,E_D,E_J
]
)
\]

\[
F_H^{out}=H+S_H\odot A_H
\]

\[
F_V^{out}=V+S_V\odot A_V
\]

\[
F_D^{out}=D+S_D\odot A_D
\]

F4：

\[
C_L=
\operatorname{Normalize}
\left(
\left|
\operatorname{Conv}_{1\times1}(L_{LFSS})
-
\operatorname{AvgPool}_{3\times3}
(
\operatorname{Conv}_{1\times1}(L_{LFSS})
)
\right|
\right)
\]

\[
A_H,A_V,A_D=
2\sigma(
\operatorname{LocalGate}
[
E_H,E_V,E_D,E_J,C_L
]
)
\]

统一：

\[
L_s^{LFSS}=
\operatorname{LFSS}(LL_s)
\]

\[
(H_s^e,V_s^e,D_s^e)=
\operatorname{DSHF}
(
H_s,V_s,D_s,L_s^{LFSS}
)
\]

\[
L_s^g=
\operatorname{AWGM}
(
L_s^{LFSS},
H_s^e,V_s^e,D_s^e
)
\]

\[
E_s=
\operatorname{ResBlock}
(
L_s^g
)
\]

核心判断：

\[
\boxed{
\text{高频小目标特征需要方向多尺度、稀疏筛选、跨方向一致性和低频空间约束}
}
\]
