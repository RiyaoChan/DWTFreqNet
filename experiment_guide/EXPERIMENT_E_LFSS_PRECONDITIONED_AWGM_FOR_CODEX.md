# Experiment E：LFSS-Preconditioned AWGM Encoder

## 0. 实验定位

本实验建立新的 **Experiment E**，研究 Encoder 低频主路径的处理顺序与特征提取器。

本实验不是 Experiment D 的继续扩展，不使用 D1–D7 的 Decoder HFE，也不在
`codex/experiment-d-hfe-matching-ablation-d2-d3` 分支上继续开发。

核心问题：

> DWT 得到的本级低频 LL 是否应先经过专门的低频特征提取，再接受高频 AWGM 引导？

现有单解码器 AWGM 基线的顺序是：

```text
DWT
 ├── LL ───────────────┐
 └── H/V/D → DirectionalBandEncoder
                       │
LL + H/V/D → AWGM
                       │
                       ▼
                   Res_block
                       │
                       ▼
                      E_s
```

Experiment E 改为：

```text
DWT
 ├── LL → 原始 Wave-Mamba LFSSBlock ──┐
 └── H/V/D → DirectionalBandEncoder   │
                                      ▼
                                     AWGM
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  E1：原 Res_block        E2：轻量 Transition
                         │                         │
                         ▼                         ▼
                        E_s                       E_s
```

正式实验只包含两个变体：

| ID | 低频路径 | 目的 |
|---|---|---|
| E1 | `LL → LFSSBlock → AWGM → 原 Res_block` | 验证 AWGM 前进行 LFSS 低频建模是否有效 |
| E2 | `LL → LFSSBlock → AWGM → 轻量 Transition` | 验证 LFSSBlock 能否替代原 Res_block 的低频特征提取职责 |

参考基线：

| ID | 模型 | 低频路径 |
|---|---|---|
| E0 | `sd_awgm` | `LL → AWGM → 原 Res_block` |

---

# 1. 实验边界

Experiment E 只改变 Encoder 中每一级 DWT 后的低频路径。

保持不变：

```text
Haar DWT / IDWT
四级 Encoder / Decoder
DirectionalBandEncoder
StageWiseAWGM 的内部实现
原始 H/V/D 高频系数对齐
Single Wavelet Decoder
Decoder skip fusion
Deep Supervision
Loss
数据增强
数据划分
优化器
学习率策略
评价指标
```

不加入：

```text
Decoder HFE
D1–D7 relation modules
LDRC
Directional Pyramid
第二次 DWT
额外 targetness prior
额外 loss
额外外层残差
额外 gamma / beta 控制 LFSS 输出
```

特别约束：

> 原始 LFSSBlock 内部已有两条残差路径，本实验不得在 LFSSBlock 外再包装任何残差、LayerScale、gamma 或 residual blend。

禁止：

```python
refined = low + gamma * (lfss(low) - low)
```

正式结构必须直接使用：

```python
refined = lfss(low)
```

---

# 2. 基础分支与新分支

## 2.1 基础分支

从单解码器 AWGM 基线分支创建，不从 Experiment D 分支创建。

```text
Repository:
RiyaoChan/DWTFreqNet

Base branch:
codex/experiment-b-single-decoder-directional-pyramid

Known base commit:
435ab1827ecee4c6b83b669789bb9833a5fd5320
```

Codex 开始前执行：

```bash
git fetch --all --prune
git checkout codex/experiment-b-single-decoder-directional-pyramid
git pull
git rev-parse HEAD
git status
```

记录实际 HEAD。

若实际 HEAD 与上述 known commit 不同，以远端实际 HEAD 为准，并在记录中说明差异。

## 2.2 新分支

创建：

```text
codex/experiment-e-lfss-before-awgm
```

命令：

```bash
git checkout -b codex/experiment-e-lfss-before-awgm
```

## 2.3 独立工作区

建议服务器使用独立 worktree：

```text
/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_E_LFSS
```

不得复用或修改：

```text
/DATA20T/bip/cry/code/DWTFreqNet_EXPERIMENT_D_ABLATION
```

不得停止当前 D4–D7 正式任务。

## 2.4 新 PR

Experiment E 单独创建 Draft PR。

不得继续更新：

```text
RiyaoChan/DWTFreqNet#3
```

---

# 3. 新模型文件与代码隔离

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
model/DWTFreqNet_SingleDecoder.py
model/DWTFreqNet_SingleDecoder_LDRC.py
model/DWTFreqNet_SingleDecoder_HFE.py
model/DWTFreqNet_SingleDecoder_HFE_Ablation.py
model/DWTFreqNet_SingleDecoder_HFE_SpatialAblation.py
```

新建：

```text
model/DWTFreqNet_SingleDecoder_LFSS_AWGM.py
```

Wave-Mamba 原始 LFSS 最小依赖单独放置：

```text
model/third_party/wavemamba_lfss.py
model/third_party/WAVE_MAMBA_NOTICE.md
model/third_party/WAVE_MAMBA_LICENSE
```

训练、测试和记录新建：

```text
train_experiment_e_lfss_awgm.py
tools/test_experiment_e_lfss_awgm.py
tools/profile_experiment_e_lfss_awgm.py
tools/analyze_experiment_e_low_frequency.py
scripts/run_experiment_e_lfss_awgm.sh
scripts/launch_experiment_e_lfss_awgm_queue.sh
EXPERIMENT_E_LFSS_AWGM_RECORD.md
```

允许最小修改：

```text
EXPERIMENT_RECORD.md
README.md
requirements.txt
```

只有在当前环境确实缺少必要依赖时才修改 `requirements.txt`。

---

# 4. Wave-Mamba LFSSBlock 来源

使用官方实现：

```text
Repository:
https://github.com/AlexZou14/Wave-Mamba

Source file:
basicsr/archs/wavemamba_arch.py

Source classes/functions:
SimpleGate
ffn
SS2D
LFSSBlock
```

Codex 必须在实现时获取并记录官方仓库的实际 commit SHA：

```bash
git ls-remote \
  https://github.com/AlexZou14/Wave-Mamba.git \
  refs/heads/main
```

或：

```bash
git clone --depth 1 \
  https://github.com/AlexZou14/Wave-Mamba.git \
  /tmp/Wave-Mamba

git -C /tmp/Wave-Mamba rev-parse HEAD
```

记录：

```text
wave_mamba_source_commit
wave_mamba_source_file
wave_mamba_source_url
```

不得编造 source commit。

---

# 5. 原始实现保真要求

## 5.1 直接提取最小必要实现

从 Wave-Mamba 官方文件中提取：

```text
SimpleGate
ffn
SS2D
LFSSBlock
```

只允许做以下适配：

```text
整理文件格式
删除未使用 import
添加类型标注
添加来源注释
添加 NCHW adapter
修复与当前 PyTorch 版本直接相关的兼容性问题
```

不得改变：

```text
SS2D 四方向扫描逻辑
in_proj / conv2d / SiLU / out_norm / out_proj
dt projection 初始化
A_logs 初始化
Ds 初始化
selective_scan_fn 调用
LFSSBlock 的 skip_scale
LFSSBlock 的 skip_scale2
LFSSBlock 的两个内部残差
原始 ffn 的 gated depthwise-conv 结构
```

## 5.2 原始 LFSSBlock 核心结构

保持：

```python
x = input * skip_scale \
    + drop_path(
        self_attention(
            ln_1(input)
        )
    )

x = x * skip_scale2 \
    + conv_blk(
        ln_2(x)
    )
```

其中 `conv_blk` 内部保持原始：

```text
Conv1×1
Depthwise Conv3×3
GELU-gated multiplication
Conv1×1
```

## 5.3 不增加外部残差

禁止在 adapter 或 Experiment E model 中出现：

```text
lfss_residual
lfss_gamma
lfss_beta
outer_skip
residual_blend
low + lfss(low)
low + gamma * ...
```

`WaveMambaLFSSNCHWAdapter` 只能执行形状转换和原始 LFSS 调用。

---

# 6. 许可证与来源记录

Wave-Mamba 官方仓库采用：

```text
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
CC BY-NC-SA 4.0
```

`model/third_party/WAVE_MAMBA_NOTICE.md` 至少记录：

```text
项目名称
论文名称
作者
官方仓库
源文件路径
实际 source commit SHA
提取的类
本项目中的修改范围
许可证
```

`wavemamba_lfss.py` 文件头加入：

```python
# Extracted and adapted from:
# Wave-Mamba: Wavelet State Space Model for
# Ultra-High-Definition Low-Light Image Enhancement
# Official repository:
# https://github.com/AlexZou14/Wave-Mamba
# Original source:
# basicsr/archs/wavemamba_arch.py
# Source commit: <ACTUAL_SHA>
# License: CC BY-NC-SA 4.0
```

不得复制 Wave-Mamba 的完整网络、DWT、HFE、BasicSR 注册器或低光增强训练代码。

只提取 LFSS 所需最小依赖。

---

# 7. 依赖与运行环境

LFSS 原始 SS2D 使用：

```python
from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
)
```

并依赖：

```text
torch
mamba_ssm
einops
timm
```

正式实现不得静默替换为：

```text
自写简化SSM
Transformer
普通卷积近似
纯PyTorch伪selective scan
selective_scan_ref正式训练
```

## 7.1 环境预检

记录：

```bash
python - <<'PY'
import torch
import mamba_ssm
import einops
import timm

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("mamba_ssm:", getattr(mamba_ssm, "__version__", "unknown"))
print("einops:", einops.__version__)
print("timm:", timm.__version__)
PY
```

并执行：

```python
from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
)
```

## 7.2 不允许静默降级

若 `mamba_ssm` 或 CUDA kernel 不可用：

```text
停止正式训练
记录错误
修复环境
重新运行测试
```

不得自动改成其他模块后仍命名为 LFSSBlock。

---

# 8. NCHW Adapter

当前 DWTFreqNet 特征格式：

```text
[B,C,H,W]
```

原始 LFSSBlock 接口：

```text
input: [B,HW,C]
x_size: (H,W)
output: [B,HW,C]
```

新建无参数 adapter：

```python
class WaveMambaLFSSNCHWAdapter(
    nn.Module
):
    def __init__(
        self,
        channels,
        d_state=16,
        expand=2.0,
        drop_path=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()

        self.channels = int(channels)

        self.block = LFSSBlock(
            hidden_dim=channels,
            d_state=d_state,
            expand=expand,
            drop_path=drop_path,
            attn_drop_rate=attn_drop_rate,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise RuntimeError(...)

        batch, channels, height, width = (
            x.shape
        )

        if channels != self.channels:
            raise RuntimeError(...)

        tokens = (
            x.permute(0, 2, 3, 1)
            .reshape(
                batch,
                height * width,
                channels,
            )
            .contiguous()
        )

        tokens = self.block(
            tokens,
            (height, width),
        )

        output = (
            tokens.reshape(
                batch,
                height,
                width,
                channels,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return output
```

该 adapter：

```text
无Conv
无Norm
无残差
无scale
无gamma
无额外参数
```

---

# 9. LFSS固定配置

四个 stage 各使用一个原始 LFSSBlock。

```python
LFSS_STAGE_CONFIG = {
    1: {
        "channels": 32,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
    2: {
        "channels": 64,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
    3: {
        "channels": 128,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
    4: {
        "channels": 256,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
}
```

首轮不搜索：

```text
d_state
expand
drop_path
LFSS block数量
扫描方向
stage选择
```

每个 stage 只使用一个 LFSSBlock。

---

# 10. E1结构

E1：

```text
LL
→ 原始 LFSSBlock
→ AWGM
→ 原 Res_block
→ E_s
```

四级：

```text
Stage1:
LL [B,32,128,128]
→ LFSS(32)
→ AWGM(32)
→ Res_block(32,64)
→ E1 [B,64,128,128]

Stage2:
LL [B,64,64,64]
→ LFSS(64)
→ AWGM(64)
→ Res_block(64,128)
→ E2 [B,128,64,64]

Stage3:
LL [B,128,32,32]
→ LFSS(128)
→ AWGM(128)
→ Res_block(128,256)
→ E3 [B,256,32,32]

Stage4:
LL [B,256,16,16]
→ LFSS(256)
→ AWGM(256)
→ Res_block(256,256)
→ E4 [B,256,16,16]
```

E1 保留父类已有：

```text
local_encoder1
local_encoder2
local_encoder3
local_encoder4
```

不得替换它们。

---

# 11. E2结构

E2：

```text
LL
→ 原始 LFSSBlock
→ AWGM
→ 轻量 Transition
→ E_s
```

LFSSBlock 已包含局部 depthwise 3×3 gated FFN，因此 E2 的 Transition 只负责：

```text
通道转换
线性投影
归一化
激活
```

固定实现：

```python
class LowFrequencyTransition(
    nn.Module
):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(
            self.norm(
                self.proj(x)
            )
        )
```

不得增加：

```text
Res_block
第二个3×3卷积
外部残差
SE/CBAM
额外门控
```

四级：

```text
32  → 64
64  → 128
128 → 256
256 → 256
```

E2 在模型构造后，将：

```text
local_encoder1–4
```

替换为：

```text
LowFrequencyTransition
```

替换后不得保留未使用的原 Res_block 参数。

---

# 12. 模型类

新文件：

```text
model/DWTFreqNet_SingleDecoder_LFSS_AWGM.py
```

建议：

```python
EXPERIMENT_E_VARIANTS = (
    "e1_lfss_resblock",
    "e2_lfss_transition",
)
```

```python
class DWTFreqNet_SingleDecoder_LFSS_AWGM(
    DWTFreqNet_SingleDecoder
):
    def __init__(
        self,
        config,
        encoder_variant,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if encoder_variant not in (
            EXPERIMENT_E_VARIANTS
        ):
            raise ValueError(...)

        super().__init__(
            config=config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
            sd_variant="sd_awgm",
        )

        self.encoder_variant = (
            encoder_variant
        )

        self.lfss_blocks = nn.ModuleDict(
            {
                str(stage):
                WaveMambaLFSSNCHWAdapter(
                    **LFSS_STAGE_CONFIG[
                        stage
                    ]
                )
                for stage in range(1, 5)
            }
        )

        if encoder_variant == (
            "e2_lfss_transition"
        ):
            transitions = (
                (32, 64),
                (64, 128),
                (128, 256),
                (256, 256),
            )

            for stage, (
                in_channels,
                out_channels,
            ) in enumerate(
                transitions,
                start=1,
            ):
                setattr(
                    self,
                    f"local_encoder{stage}",
                    LowFrequencyTransition(
                        in_channels,
                        out_channels,
                    ),
                )
```

---

# 13. `_encode_stage`顺序

必须重写：

```python
def _encode_stage(
    self,
    stage,
    tensor,
):
    (
        band_a,
        band_h,
        band_v,
        band_d,
    ) = self._dwt(tensor)

    (
        feature_h,
        feature_v,
        feature_d,
    ) = getattr(
        self,
        f"dir_encoder{stage}",
    )(
        band_h,
        band_v,
        band_d,
    )

    refined_a = self.lfss_blocks[
        str(stage)
    ](
        band_a
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
    )(
        guided_a
    )

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

严格执行顺序：

```text
DWT
→ DirectionalBandEncoder(H/V/D)
→ LFSS(LL)
→ AWGM(refined_LL, encoded_H/V/D)
→ ResBlock或Transition
```

不得写成：

```text
LL → AWGM → LFSS
LL → ResBlock → LFSS → AWGM
LL → LFSS → ResBlock → AWGM
```

---

# 14. 原始小波系数保存规则

`raw_bands` 必须继续保存原始：

```text
band_a
band_h
band_v
band_d
```

不得用 LFSS 输出覆盖：

```python
raw_bands[stage][0]
```

原因：

```text
保留真实DWT系数用于调试与可追溯性
decoder高频系数路径保持与E0一致
```

AWGM使用：

```text
refined_a
```

而不是：

```text
raw band_a
```

---

# 15. 模型元数据

共同：

```python
self.experiment_group = (
    "experiment_e"
)

self.experiment_type = (
    "encoder_ablation"
)

self.ablation_axis = (
    "pre_awgm_low_frequency_extractor"
)

self.encoder_lfss = True
self.lfss_source = "wave_mamba"
self.lfss_outer_residual = False
self.lfss_blocks_per_stage = 1
self.awgm_input_low = (
    "lfss_refined_ll"
)

self.decoder_hfe = False
self.directional_pyramid = False
self.second_dwt = False
self.ldrc = False
self.mamba = True
self.coefficient_mode = "aligned_raw"
```

E1：

```python
self.ablation_id = "E1"
self.model_variant = (
    "dwtfreqnet_single_decoder_lfss_awgm_resblock"
)
self.sd_variant = (
    "sd_awgm_lfss_resblock"
)
self.post_awgm_encoder = (
    "original_res_block"
)
```

E2：

```python
self.ablation_id = "E2"
self.model_variant = (
    "dwtfreqnet_single_decoder_lfss_awgm_transition"
)
self.sd_variant = (
    "sd_awgm_lfss_transition"
)
self.post_awgm_encoder = (
    "conv1x1_bn_gelu_transition"
)
```

额外记录：

```text
wave_mamba_source_commit
mamba_ssm_version
torch_version
cuda_version
selective_scan_backend
LFSS d_state
LFSS expand
LFSS drop_path
```

---

# 16. LFSS特殊初始化保护

原始 SS2D 包含特殊初始化：

```text
dt_projs_weight
dt_projs_bias
A_logs
Ds
```

训练脚本不得使用通用初始化器覆盖这些参数。

Codex 必须检查现有训练流程是否执行：

```python
model.apply(init_weights)
```

或其他递归初始化。

## 16.1 推荐实现

仅对非 LFSS 模块使用原基线初始化器：

```python
def initialize_experiment_e_model(
    model,
    baseline_init_fn,
):
    lfss_prefixes = tuple(
        f"lfss_blocks.{stage}."
        for stage in range(1, 5)
    )

    for name, module in (
        model.named_modules()
    ):
        if any(
            name.startswith(prefix)
            for prefix in lfss_prefixes
        ):
            continue

        baseline_init_fn(module)
```

如果原训练脚本不是逐module初始化，Codex应实现等价的跳过逻辑。

## 16.2 强制测试

在初始化前后保存并比较：

```text
dt_projs_weight
dt_projs_bias
A_logs
Ds
skip_scale
skip_scale2
```

要求：

```text
max_abs_difference = 0
```

不得依赖 `_no_reinit` 单独保护全部 LFSS 参数，因为它只覆盖部分参数。

---

# 17. 训练入口

新建：

```text
train_experiment_e_lfss_awgm.py
```

参数：

```text
--encoder-variant e1_lfss_resblock
--encoder-variant e2_lfss_transition
```

模型：

```python
model = (
    DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(),
        encoder_variant=(
            args.encoder_variant
        ),
        mode=mode,
        deepsuper=True,
    )
)
```

不得复用 Experiment D 训练入口。

---

# 18. 输出目录

E1：

```text
runs/experiment_e_lfss_awgm/
    E1_lfss_resblock/
    <dataset>/
    seed42/
```

E2：

```text
runs/experiment_e_lfss_awgm/
    E2_lfss_transition/
    <dataset>/
    seed42/
```

完整：

```text
runs/experiment_e_lfss_awgm/E1_lfss_resblock/NUAA-SIRST/seed42
runs/experiment_e_lfss_awgm/E2_lfss_transition/NUAA-SIRST/seed42

runs/experiment_e_lfss_awgm/E1_lfss_resblock/IRSTD-1K/seed42
runs/experiment_e_lfss_awgm/E2_lfss_transition/IRSTD-1K/seed42

runs/experiment_e_lfss_awgm/E1_lfss_resblock/NUDT-SIRST/seed42
runs/experiment_e_lfss_awgm/E2_lfss_transition/NUDT-SIRST/seed42
```

不得写入 Experiment D 目录。

---

# 19. 单元测试

新建：

```text
tools/test_experiment_e_lfss_awgm.py
```

## 19.1 源实现检查

检查：

```text
LFSSBlock
SS2D
ffn
SimpleGate
```

均来自：

```text
model.third_party.wavemamba_lfss
```

检查 source commit 和 license 文件存在。

## 19.2 无外层残差

模型源码和模块属性中不得出现：

```text
lfss_gamma
outer_residual
residual_blend
low_scale
```

adapter参数量必须为：

```text
仅原始LFSSBlock参数
```

## 19.3 LFSS内部初始化

检查：

```text
skip_scale == 1
skip_scale2 == 1
A_logs为float32
Ds为float32
dt_projs_bias有限
```

## 19.4 初始化保护

执行 Experiment E 正式初始化函数后：

```text
LFSS关键参数完全不变
```

## 19.5 模块配置

E1：

```text
local_encoder1–4均为原Res_block
```

E2：

```text
local_encoder1–4均为LowFrequencyTransition
```

E2的四个低频主路径中不得保留原 Res_block 参数。

注意：

```text
Stem和Decoder Fuse仍允许使用Res_block
```

## 19.6 执行顺序hook测试

每个stage注册hook，记录：

```text
LFSS
AWGM
Post-AWGM encoder
```

要求：

```text
LFSS → AWGM → ResBlock/Transition
```

四级全部通过。

## 19.7 AWGM输入来源测试

捕获：

```text
raw_LL
LFSS_output
AWGM第一个输入
```

要求：

```python
torch.testing.assert_close(
    awgm_low_input,
    lfss_output,
)
```

并确认不是误传 raw_LL。

## 19.8 Stage形状

256×256输入：

```text
Stage1:
raw_LL       [B,32,128,128]
LFSS_LL      [B,32,128,128]
guided_LL    [B,32,128,128]
encoded      [B,64,128,128]

Stage2:
raw_LL       [B,64,64,64]
LFSS_LL      [B,64,64,64]
guided_LL    [B,64,64,64]
encoded      [B,128,64,64]

Stage3:
raw_LL       [B,128,32,32]
LFSS_LL      [B,128,32,32]
guided_LL    [B,128,32,32]
encoded      [B,256,32,32]

Stage4:
raw_LL       [B,256,16,16]
LFSS_LL      [B,256,16,16]
guided_LL    [B,256,16,16]
encoded      [B,256,16,16]
```

## 19.9 模型输出

训练：

```text
6 × [2,1,256,256]
```

测试：

```text
[2,1,256,256]
```

## 19.10 DWT/IDWT

```text
DWT=4
IDWT=4
```

## 19.11 Decoder不变

比较 E0、E1、E2：

```text
align_H/V/D
decoder_fuse0–3
gt_conv5/4/3/2
out_head
outconv
```

模块类型和参数形状必须一致。

## 19.12 梯度

E1检查：

```text
四级LFSS
四级AWGM
四级原ResBlock
DirectionalBandEncoder
decoder
side heads
```

均有非零梯度。

E2检查：

```text
四级LFSS
四级AWGM
四级Transition
DirectionalBandEncoder
decoder
side heads
```

均有非零梯度。

重点检查：

```text
A_logs
Ds
dt_projs_weight
dt_projs_bias
in_proj
out_proj
ffn convs
skip_scale
skip_scale2
```

## 19.13 AMP

RTX 3090 或当前空闲CUDA GPU：

```text
2×1×256×256
FP32 forward/backward
AMP forward/backward
```

不得出现：

```text
NaN
Inf
CUDA illegal memory access
selective_scan kernel error
OOM
```

## 19.14 真数据smoke test

NUAA：

```text
batch=4
patch=256
单步forward
loss
backward
optimizer.step
```

E1/E2均通过后才能启动正式训练。

不得为通过测试而静默降低正式 batch size。

---

# 20. 复杂度测试

新建：

```text
tools/profile_experiment_e_lfss_awgm.py
```

比较：

```text
E0：sd_awgm
E1：LFSS + AWGM + ResBlock
E2：LFSS + AWGM + Transition
```

统一：

```text
输入 [1,1,256,256]
warmup=5
repeat>=20
同一GPU
同一PyTorch环境
eval
AMP关闭
torch.cuda.synchronize()
```

报告：

```text
Parameters
LFSS parameters
Encoder parameters
FLOPs
Latency
FPS
Inference peak memory
Training peak memory
DWT/IDWT
```

说明：

```text
THOP可能无法准确统计selective_scan；
必须同时报告实测latency和显存。
```

---

# 21. 低频特征诊断

新建：

```text
tools/analyze_experiment_e_low_frequency.py
```

仅用于验证集离线诊断，不参与训练和loss。

## 21.1 保存的特征

每个stage记录：

```text
raw_LL
LFSS_LL
guided_LL
E_s
AWGM gate
AWGM direction weights
```

训练时不得长期保存完整tensor。

只在离线分析时开启：

```text
debug_tensors=True
```

## 21.2 GT对齐

使用：

```python
stage_mask = F.adaptive_max_pool2d(
    gt_mask.float(),
    output_size=feature.shape[-2:],
)
```

避免深层小目标消失。

## 21.3 目标/背景响应比

对：

```text
raw_LL
LFSS_LL
guided_LL
```

分别计算：

\[
R(F)
=
\frac{
\operatorname{Mean}(|F|\mid Y=1)
}{
\operatorname{Mean}(|F|\mid Y=0)+\epsilon
}
\]

记录：

```text
raw_target_background_ratio
lfss_target_background_ratio
guided_target_background_ratio
```

重点比较：

```text
LFSS后是否提高目标/背景分离
AWGM后是否进一步提高
```

## 21.4 LFSS变化强度

仅作诊断：

\[
R_{\Delta LF}
=
\frac{
\|LFSS(LL)-LL\|_2
}{
\|LL\|_2+\epsilon
}
\]

这不是外部残差，只用于分析模块改变幅度。

## 21.5 AWGM目标/背景诊断

记录：

```text
target_gate_mean
background_gate_mean
target_background_gate_ratio
mean_G_H
mean_G_V
mean_G_D
```

## 21.6 LFSS内部scale

记录各stage：

```text
skip_scale mean/std/min/max
skip_scale2 mean/std/min/max
```

判断训练后内部残差权重是否退化或异常放大。

---

# 22. 正式训练设置

与现有单解码器 AWGM 基线一致：

```text
seed=42
patch size=256
batch size=4
epochs=1000
optimizer=Adam
initial lr=1e-3
scheduler=现有训练脚本的同一warmup + CosineAnnealingLR
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

全部随机初始化。

不加载：

```text
E0 checkpoint
D0–D7 checkpoint
Wave-Mamba pretrained checkpoint
其他DWTFreqNet checkpoint
```

只复制模块代码，不迁移 Wave-Mamba 低光增强权重。

---

# 23. E0基线使用规则

现有 `sd_awgm` seed42 结果可作为 E0 参考：

```text
NUAA mIoU ≈ 0.7799
NUDT mIoU ≈ 0.9058
IRSTD mIoU ≈ 0.6561
```

但正式记录前必须验证：

```text
模型文件hash
base commit
数据划分
训练参数
loss
评价代码
```

与 Experiment E 基线一致。

若不一致：

```text
不得直接把旧结果作为E0正式对照
```

此时将 E0 重跑加入后续队列，但不要占用当前首批两张GPU。

---

# 24. 初始两张GPU安排

当前只有两张空闲GPU。

首先同时启动：

| 初始空闲GPU | Variant | Dataset |
|---:|---|---|
| 第1张 | E1 | NUAA-SIRST |
| 第2张 | E2 | NUAA-SIRST |

实际 GPU ID 由脚本动态检测，不在规范中假设为0/1。

启动前记录：

```text
GPU ID
GPU name
free memory
utilization
existing compute PID
```

---

# 25. 动态队列顺序

初始两个NUAA任务启动后，每60秒检测空闲GPU。

固定队列：

```text
1. E1 - NUAA-SIRST
2. E2 - NUAA-SIRST
3. E1 - IRSTD-1K
4. E2 - IRSTD-1K
5. E1 - NUDT-SIRST
6. E2 - NUDT-SIRST
```

前两项立即启动。

后四项按顺序等待任何GPU空闲。

这样可以：

```text
优先完成NUAA直接对照
随后完成复杂背景IRSTD
最后验证NUDT收益
```

不得因等待成对GPU而让单张GPU长期空闲。

---

# 26. GPU空闲判定

新建：

```text
scripts/launch_experiment_e_lfss_awgm_queue.sh
```

每60秒检查：

```text
nvidia-smi memory.used
nvidia-smi utilization.gpu
compute process PID
输出目录锁
完成标记
失败标记
```

建议空闲条件：

```text
memory.used < 1000 MiB
utilization.gpu < 10%
无其他用户compute PID
```

实际阈值写入记录。

不得停止或抢占：

```text
Experiment D
其他用户任务
当前E1/E2任务
```

---

# 27. 防重复启动

每个任务目录包含：

```text
RUNNING.lock
FAILED
TRAINING_COMPLETE
launcher.pid
python.pid
```

启动前检查：

```text
若TRAINING_COMPLETE存在：跳过
若RUNNING.lock且PID存活：跳过
若输出目录存在有效checkpoint：不得覆盖
若FAILED存在：记录并等待人工处理，不自动无限重启
```

队列本身使用：

```text
flock
```

或等效单实例锁。

---

# 28. 启动脚本

新建：

```text
scripts/run_experiment_e_lfss_awgm.sh
```

示例：

```bash
bash scripts/run_experiment_e_lfss_awgm.sh \
  e1_lfss_resblock \
  NUAA-SIRST \
  0
```

```bash
bash scripts/run_experiment_e_lfss_awgm.sh \
  e2_lfss_transition \
  NUAA-SIRST \
  1
```

队列：

```bash
bash scripts/launch_experiment_e_lfss_awgm_queue.sh
```

支持：

```text
GPU_ALLOWLIST
MAX_CONCURRENT=2
POLL_SECONDS=60
```

---

# 29. 结果表

| Dataset | ID | Low-frequency path | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | E0 | LL→AWGM→ResBlock | 489* | 0.7799* | 0.7848* | 0.8764* | 0.9466* | 1.935e-5* | | |
| NUAA | E1 | LL→LFSS→AWGM→ResBlock | | | | | | | | |
| NUAA | E2 | LL→LFSS→AWGM→Transition | | | | | | | | |
| IRSTD | E0 | LL→AWGM→ResBlock | 894* | 0.6561* | 0.6477* | 0.7924* | 0.9091* | 1.537e-5* | | |
| IRSTD | E1 | LL→LFSS→AWGM→ResBlock | | | | | | | | |
| IRSTD | E2 | LL→LFSS→AWGM→Transition | | | | | | | | |
| NUDT | E0 | LL→AWGM→ResBlock | 556* | 0.9058* | 0.9019* | 0.9505* | 0.9852* | 4.182e-6* | | |
| NUDT | E1 | LL→LFSS→AWGM→ResBlock | | | | | | | | |
| NUDT | E2 | LL→LFSS→AWGM→Transition | | | | | | | | |

`*`表示旧结果，只有通过E0一致性验证后才转为正式对照。

---

# 30. 结论判定

## 30.1 E1 > E0

说明：

> 在AWGM之前对本级LL进行全局低频建模有价值，原始LL直接接受高频引导不是最优顺序。

## 30.2 E1 ≈ E0

若三数据集平均 mIoU 差异小于约0.002：

> LFSS预处理没有形成稳定收益，AWGM前的低频语义不足可能不是主要瓶颈，或原ResBlock已经能够补偿。

## 30.3 E2 ≈ E1且复杂度更低

说明：

> LFSSBlock可以承担主要低频特征提取职责，完整ResBlock在LFSS之后存在冗余。

此时优先选择E2。

## 30.4 E1 > E2

说明：

> LFSS全局建模不能完全替代卷积ResBlock的局部归纳偏置，二者具有互补性。

## 30.5 E2 > E1

说明：

> LFSS后继续使用完整ResBlock可能过度处理低频或破坏LFSS输出，轻量通道Transition更适合。

## 30.6 NUAA/IRSTD提升、NUDT下降

说明：

> LFSS更有利于复杂背景建模，但可能削弱规则场景中的强小目标响应。

需结合：

```text
target/background response ratio
Pd
Fa
```

判断。

---

# 31. 多数据集平均

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
E1 - E0
E2 - E0
E2 - E1
```

小于约0.002的平均mIoU差异不表述为确定性优势。

首轮只运行seed42，不自动扩展多seed。

---

# 32. 本轮不做

E1/E2完成前不新增：

```text
E3不同LFSS层数
E4仅深层使用LFSS
E5不同d_state
E6不同expand
E7 LFSS后再加DWConv Transition
E8 LFSS与原LL融合
LFSS pretrained权重
Decoder HFE组合
LDRC组合
多seed
```

---

# 33. 实验记录

新建：

```text
EXPERIMENT_E_LFSS_AWGM_RECORD.md
```

记录：

```text
base branch
base commit
new branch
final commit
Wave-Mamba source commit
license
环境版本
模型结构
初始化保护
测试
复杂度
GPU/PID
队列
训练进度
最终结果
低频诊断
结论
```

在：

```text
EXPERIMENT_RECORD.md
README.md
```

只增加 Experiment E 的索引和链接。

---

# 34. Codex最终交付

Codex完成代码、测试和启动后返回：

1. Experiment E 开始时实际 base commit；
2. 新分支名称；
3. 新 Draft PR；
4. 最终 commit SHA；
5. Wave-Mamba 官方 source commit；
6. 提取的原始类和实际适配内容；
7. 许可证与NOTICE文件；
8. 新增和修改文件列表；
9. E1完整结构；
10. E2完整结构；
11. 无外部残差/gamma检查；
12. LFSS特殊初始化保护测试；
13. LFSS→AWGM→PostEncoder顺序hook测试；
14. AWGM低频输入等于LFSS输出测试；
15. E1/E2各stage形状；
16. E2不存在低频主路径ResBlock检查；
17. Decoder结构一致性；
18. DWT/IDWT计数；
19. CPU可构建状态；
20. CUDA FP32/AMP前向反向测试；
21. NUAA batch4真数据smoke test；
22. 参数/FLOPs/延迟/显存；
23. 两张空闲GPU的实际ID；
24. E1-NUAA和E2-NUAA的启动命令；
25. wrapper PID与Python PID；
26. 输出目录；
27. 动态队列PID；
28. 后续IRSTD/NUDT队列状态；
29. 当前epoch；
30. 1000 epoch最终结果；
31. E0/E1/E2完整比较；
32. 低频target/background诊断；
33. `EXPERIMENT_E_LFSS_AWGM_RECORD.md`更新。

建议commit：

```text
Add Experiment E LFSS-preconditioned AWGM encoder
```

---

# 35. 最终结构公式

E0：

\[
L_s^g
=
\operatorname{AWGM}_s
(
L_s,
H_s^e,
V_s^e,
D_s^e
)
\]

\[
E_s
=
\operatorname{ResBlock}_s
(
L_s^g
)
\]

E1：

\[
L_s^e
=
\operatorname{LFSSBlock}_s
(
L_s
)
\]

\[
L_s^g
=
\operatorname{AWGM}_s
(
L_s^e,
H_s^e,
V_s^e,
D_s^e
)
\]

\[
E_s
=
\operatorname{ResBlock}_s
(
L_s^g
)
\]

E2：

\[
L_s^e
=
\operatorname{LFSSBlock}_s
(
L_s
)
\]

\[
L_s^g
=
\operatorname{AWGM}_s
(
L_s^e,
H_s^e,
V_s^e,
D_s^e
)
\]

\[
E_s
=
\operatorname{Transition}_s
(
L_s^g
)
\]

其中：

\[
\operatorname{Transition}
=
\operatorname{GELU}
\circ
\operatorname{BN}
\circ
\operatorname{Conv}_{1\times1}
\]

核心判断：

\[
\boxed{
\text{先用LFSS提取低频语义，再进行高频AWGM引导，是否优于直接引导原始LL}
}
\]
