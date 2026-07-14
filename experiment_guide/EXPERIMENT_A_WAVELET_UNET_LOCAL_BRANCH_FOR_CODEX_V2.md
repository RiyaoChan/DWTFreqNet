# Experiment A v2：以 Wavelet U-Net 简化 Low-Frequency Local Feature Encoder

## 0. 基线代码与版本约束

本方案依据当前 GitHub 主分支重新制定。

- Repository：`RiyaoChan/DWTFreqNet`
- 生成方案时读取的基线 commit：

```text
71dfeb348878517775af3df0767b54747f692c5d
```

Codex 开始修改前必须执行：

```bash
git checkout main
git pull
git rev-parse HEAD
```

将实际 base commit 记录到实验文件中。若实际 `main` 已经晚于上述 commit，必须先重新检查以下文件再实现：

```text
model/DWTFreqNet.py
train_one.py
model/Config.py
decoder_fuse/transformer_dec_fuse_none_posqkv_dropout.py
```

不得基于旧缓存、旧工作区或旧服务器副本修改。

---

# 1. 实验目标

本实验只回答一个问题：

> 原 DWTFreqNet 中类似 UNet++ 的嵌套 Low-Frequency Local Feature Encoder，能否被一条更简单的 Wavelet U-Net 编解码路径替代，同时保持 Global High-Frequency Feature Encoder、第二次 DWT、AWGM/W8M、LDRC 和最终 decoder 的主体设计不变？

实验 A 的唯一核心变量为：

```text
原嵌套 local branch
        ↓
四层编码 + 三层解码的 Wavelet U-Net local branch
```

暂时不执行以下改动：

```text
不删除第二次 DWT
不将 AWGM 前移到 LLE 内部
不修改 AWGM/W8M
不修改 LDRC
不修改最终 decoder
不修改损失函数
不修改数据划分
```

---

# 2. 文件隔离要求

## 2.1 不修改原模型文件

禁止修改：

```text
model/DWTFreqNet.py
```

原模型必须继续支持：

```python
from model.DWTFreqNet import DWTFreqNet
```

并能够加载已有原模型 checkpoint。

## 2.2 新建模型文件

新建：

```text
model/DWTFreqNet_WULLE.py
```

建议类名：

```python
class DWTFreqNet_WULLE(DWTFreqNet):
    ...
```

`WULLE` 表示：

```text
Wavelet U-Net Low-Frequency Local Encoder
```

## 2.3 推荐采用继承而不是复制全部模块

推荐：

```python
from model.DWTFreqNet import DWTFreqNet, Res_block


class DWTFreqNet_WULLE(DWTFreqNet):
    def __init__(...):
        super().__init__(...)
        ...
```

理由：

1. AWGM/W8M 直接复用当前最新实现；
2. 自动保留当前已修正的 LH/HL 扫描对应关系；
3. 自动保留 Haar 方向检查元数据；
4. LDRC、第二次 DWT、最终 decoder 不需要复制出另一份；
5. 后续原模型修复不会因复制代码而产生两个版本。

但必须删除被替代的旧模块，不能只是不调用。

---

# 3. 当前最新代码中必须保留的修正

当前代码已经明确：

```text
代码变量 H = LH
代码变量 V = HL
```

并且当前物理扫描对应关系为：

```text
H/LH → vertical scan
V/HL → horizontal scan
```

新模型不得重新交换、重命名或覆盖该逻辑。

必须直接复用当前文件中的：

```text
HAAR_CODE_BAND_NAMES
W8M_CURRENT_SCAN_AXIS
check_haar_direction_correspondence
build_wave_guidance
DirectionMatchedAWGM
WaveletEightDirectionAWGM
```

新模型中的 AWGM/W8M 输出和 `awgm_backends` 元数据必须与原模型保持一致。

---

# 4. 原 local branch 中需要删除的节点

当前原模型 local branch 包含：

```text
第一列编码节点：
local_encoder1_1
local_encoder2_1
local_encoder3_1
local_encoder4_1

嵌套重构节点：
local_encoder1_2
local_encoder2_2
local_encoder3_2
local_encoder1_3
local_encoder2_3
local_encoder1_4
```

新模型保留第一列四个编码节点：

```text
local_encoder1_1
local_encoder2_1
local_encoder3_1
local_encoder4_1
```

删除六个嵌套节点：

```python
for name in (
    "local_encoder1_2",
    "local_encoder2_2",
    "local_encoder3_2",
    "local_encoder1_3",
    "local_encoder2_3",
    "local_encoder1_4",
):
    delattr(self, name)
```

必须确认删除后这些模块不再出现在：

```python
dict(model.named_modules())
dict(model.named_parameters())
model.state_dict()
```

---

# 5. 删除不再使用的嵌套高频投影

原模型中以下投影只服务于被删除的后续嵌套 local 节点：

```text
global_channel1_3
global_channel2_3
global_channel1_4
```

Experiment A 中不再使用它们，应从新模型实例中删除：

```python
for name in (
    "global_channel1_3",
    "global_channel2_3",
    "global_channel1_4",
):
    delattr(self, name)
```

保留：

```text
global_channel1_2
global_channel2_2
global_channel3_2
```

它们分别作为 Wavelet U-Net 三个解码阶段的高频系数预测头。

注意：

> 删除上述三个投影仅发生在新模型实例中，不修改原 `DWTFreqNet.py`。

---

# 6. 新增 Wavelet U-Net decoder

在 `DWTFreqNet_WULLE.__init__()` 中新增：

```python
C = config.base_channel
block = Res_block

self.wulle_decoder3 = self._make_layer(
    block,
    C * 16,
    C * 8,
    1,
)

self.wulle_decoder2 = self._make_layer(
    block,
    C * 12,
    C * 4,
    1,
)

self.wulle_decoder1 = self._make_layer(
    block,
    C * 6,
    C * 2,
    1,
)
```

通道逻辑：

```text
decoder3：cat(E3=8C, U3=8C) → 8C
decoder2：cat(E2=4C, U2=8C) → 4C
decoder1：cat(E1=2C, U1=4C) → 2C
```

不要修改继承的最终 decoder：

```text
decoder4_channel
decoder3_channel
decoder2_channel
decoder1_channel
decoder3_channel_local
decoder2_channel_local
decoder1_channel_local
```

`wulle_decoder*` 属于 LLE 内部简化路径；原 `decoder*_channel*` 属于网络末端最终 decoder，二者不是同一个模块。

---

# 7. 新模型前向传播：第一列编码器

必须保留输入 stem 和第一级 local stem。

正确实现：

```python
x_input = self.inc(x)

A1, LH1, HL1, HH1 = self.har(x_input)

G1_input = self.conv_wavelet_inchannel_global(
    torch.cat([LH1, HL1, HH1], dim=1)
)
G1 = self.global_encoder1_1(G1_input)

E1_input = self.conv_wavelet_inchannel_local(A1)
E1 = self.local_encoder1_1(E1_input)
```

特别注意：

> 不允许直接写 `E1 = local_encoder1_1(A1)`。

当前原模型在 `A1` 与 `local_encoder1_1` 之间还存在：

```text
conv_wavelet_inchannel_local
```

该模块属于输入 stem，Experiment A 必须保留。

继续向下：

```python
A2, LH2, HL2, HH2 = self.har(E1)
G2 = self.global_encoder2_1(
    torch.cat([LH2, HL2, HH2], dim=1)
)
E2 = self.local_encoder2_1(A2)

A3, LH3, HL3, HH3 = self.har(E2)
G3 = self.global_encoder3_1(
    torch.cat([LH3, HL3, HH3], dim=1)
)
E3 = self.local_encoder3_1(A3)

A4, LH4, HL4, HH4 = self.har(E3)
G4 = self.global_encoder4_1(
    torch.cat([LH4, HL4, HH4], dim=1)
)
E4 = self.local_encoder4_1(A4)
```

以 `C=32`、输入 `256×256` 为例：

```text
E1：[B,  64, 128, 128]
E2：[B, 128,  64,  64]
E3：[B, 256,  32,  32]
E4：[B, 256,  16,  16]

G1：[B,  64, 128, 128]
G2：[B, 128,  64,  64]
G3：[B, 256,  32,  32]
G4：[B, 256,  16,  16]
```

---

# 8. 新 Wavelet U-Net local decoder

## 8.1 E4 → D3

使用第一列深层 global feature `G4` 预测三个 IDWT 高频系数槽位：

```python
pred4 = self.global_channel3_2(G4)
pred4_LH, pred4_HL, pred4_HH = torch.chunk(pred4, 3, dim=1)

U3 = self.inversehar(
    E4,
    pred4_LH,
    pred4_HL,
    pred4_HH,
)

D3 = self.wulle_decoder3(
    torch.cat([E3, U3], dim=1)
)
```

形状：

```text
pred4_LH/HL/HH：[B, 256, 16, 16]
U3：            [B, 256, 32, 32]
D3：            [B, 256, 32, 32]
```

## 8.2 D3 → D2

```python
pred3 = self.global_channel2_2(G3)
pred3_LH, pred3_HL, pred3_HH = torch.chunk(pred3, 3, dim=1)

U2 = self.inversehar(
    D3,
    pred3_LH,
    pred3_HL,
    pred3_HH,
)

D2 = self.wulle_decoder2(
    torch.cat([E2, U2], dim=1)
)
```

形状：

```text
pred3_LH/HL/HH：[B, 256, 32, 32]
U2：            [B, 256, 64, 64]
D2：            [B, 128, 64, 64]
```

## 8.3 D2 → D1

```python
pred2 = self.global_channel1_2(G2)
pred2_LH, pred2_HL, pred2_HH = torch.chunk(pred2, 3, dim=1)

U1 = self.inversehar(
    D2,
    pred2_LH,
    pred2_HL,
    pred2_HH,
)

D1 = self.wulle_decoder1(
    torch.cat([E1, U1], dim=1)
)
```

形状：

```text
pred2_LH/HL/HH：[B, 128, 64, 64]
U1：            [B, 128, 128, 128]
D1：            [B,  64, 128, 128]
```

## 8.4 最终 local feature 对应关系

后续代码使用：

```python
x4_local_output_4_1 = E4
x3_local_output_3_2 = D3
x2_local_output_2_3 = D2
x1_local_output_1_4 = D1
```

不得额外构造：

```text
x1_local_output_1_2
x1_local_output_1_3
x2_local_output_2_2
```

---

# 9. Global High-Frequency Feature Encoder 的兼容性重接线

## 9.1 重要说明

原 global dense branch 会读取嵌套 local 节点的 DWT 高频分量。

删除这些 local 节点后，不可能做到“所有输入张量逐元素完全不变”。

本实验中“global 高频分支保持不变”的严格定义为：

1. `global_encoder1_1` 至 `global_encoder1_4` 等所有 global encoder 模块保留；
2. 每个 global 节点的输出尺度和通道数保持不变；
3. 原 global dense 拓扑保持；
4. 原来来自嵌套 local 节点的输入，替换为同尺度的 U-Net decoder 特征；
5. 不修改 global encoder 内部卷积。

## 9.2 计算 D1、D2 的反馈频率分量

```python
_, D1_LH, D1_HL, D1_HH = self.har(D1)
D1_HVD = torch.cat([D1_LH, D1_HL, D1_HH], dim=1)

_, D2_LH, D2_HL, D2_HH = self.har(D2)
D2_HVD = torch.cat([D2_LH, D2_HL, D2_HH], dim=1)
```

`D1_HVD` 同时替代原来的：

```text
DWT(x1_local_output_1_2) 的三个高频分量
DWT(x1_local_output_1_3) 的三个高频分量
```

`D2_HVD` 替代原来的：

```text
DWT(x2_local_output_2_2) 的三个高频分量
```

这是 Experiment A 为消除嵌套 local 节点所必需的重接线。

## 9.3 保持 global dense 节点

```python
G1_2 = self.global_encoder1_2(
    torch.cat([G1, self.up(G2)], dim=1)
)

G2_2 = self.global_encoder2_2(
    torch.cat([
        G2,
        D1_HVD,
        self.up(G3),
    ], dim=1)
)

G3_2 = self.global_encoder3_2(
    torch.cat([
        G3,
        D2_HVD,
        self.up(G4),
    ], dim=1)
)

G1_3 = self.global_encoder1_3(
    torch.cat([
        G1,
        G1_2,
        self.up(G2_2),
    ], dim=1)
)

G2_3 = self.global_encoder2_3(
    torch.cat([
        G2,
        G2_2,
        D1_HVD,
        self.up(G3_2),
    ], dim=1)
)

G1_4 = self.global_encoder1_4(
    torch.cat([
        G1,
        G1_2,
        G1_3,
        self.up(G2_3),
    ], dim=1)
)
```

最终 HGE 输入保持：

```python
f_input = x_input
f1 = G1_4
f2 = G2_3
f3 = G3_2
```

通道必须为：

```text
f1：[B,  64, 128, 128]
f2：[B, 128,  64,  64]
f3：[B, 256,  32,  32]
```

---

# 10. 第二次 DWT、AWGM/W8M 和 LDRC

以下部分从当前原模型 forward 逻辑迁移，不进行算法修改：

```text
f_input 的两次 DWT
f1/f2/f3 的第二次 DWT
wave_att_input_t
wave_att_f1
wave_att_f2
wave_att_f3
stand_cahnnel_input
stand_cahnnel1–3
TransTo3e
TransTo2e
TransTo1e
TransTo_input
四个尺度的 IDWT 重构
```

保持 LDRC 顺序：

```text
f3 → f2 → f1 → f_input
```

保持：

```python
self.awgm_variant
self.awgm_backends
```

并确保 W8M 新模型仍返回：

```text
haar_band_scan_axis
haar_routing_aligned
share_mode
diagonal_order
diagonal_directions
direction_embedding
mamba_instances_per_awgm
```

Experiment A 不得修改 H/V 物理扫描对应关系。

---

# 11. 最终 decoder 和输出

最终 decoder 完全沿用当前原模型。

仅将 local 输入替换为：

```text
E4
D3
D2
D1
```

不得修改以下继承模块：

```text
decoder4_channel
decoder3_channel
decoder2_channel
decoder1_channel
decoder3_channel_local
decoder2_channel_local
decoder1_channel_local
out4
out3
out2
out1
from_input2out
outc_global
gt_conv5
gt_conv4
gt_conv3
gt_conv2
outconv
```

训练模式必须返回 6 个输出：

```python
(
    torch.sigmoid(gt5),
    torch.sigmoid(gt4),
    torch.sigmoid(gt3),
    torch.sigmoid(gt2),
    torch.sigmoid(d0),
    torch.sigmoid(out),
)
```

测试模式返回：

```python
torch.sigmoid(out)
```

---

# 12. 训练入口修改

允许最小修改：

```text
train_one.py
```

新增导入：

```python
from model.DWTFreqNet_WULLE import DWTFreqNet_WULLE
```

新增参数：

```python
parser.add_argument(
    "--model-variant",
    default="dwtfreqnet_original",
    choices=(
        "dwtfreqnet_original",
        "dwtfreqnet_wulle_a",
    ),
)
```

增加统一构建函数：

```python
def build_model(args, mode):
    kwargs = dict(
        config=get_DWTFreqNet_config(),
        mode=mode,
        deepsuper=True,
        awgm_variant=args.awgm_variant,
        awgm_allow_fallback=args.awgm_allow_fallback,
    )

    if args.model_variant == "dwtfreqnet_original":
        return DWTFreqNet(**kwargs)

    if args.model_variant == "dwtfreqnet_wulle_a":
        return DWTFreqNet_WULLE(**kwargs)

    raise ValueError(args.model_variant)
```

训练和 `eval_only` 都必须调用 `build_model()`。

`run_config.json` 增加：

```json
{
  "model_variant": "dwtfreqnet_wulle_a",
  "model_base_commit": "<actual git commit>",
  "local_branch": "wavelet_unet",
  "local_encoder_nodes": 4,
  "local_decoder_nodes": 3,
  "nested_local_nodes_removed": 6,
  "second_dwt": true,
  "awgm_position": "after_lle",
  "ldrc_unchanged": true,
  "final_decoder_unchanged": true
}
```

---

# 13. Checkpoint 隔离

不同结构禁止共享自动恢复目录。

目录：

```text
runs/experiment_a_v2/<dataset>/dwtfreqnet_original/<awgm_variant>
runs/experiment_a_v2/<dataset>/dwtfreqnet_wulle_a/<awgm_variant>
```

必须将 `model_variant` 写入 checkpoint 的 `args`。

加载时：

- 原模型 checkpoint 不允许静默加载到 WULLE；
- WULLE checkpoint 不允许静默加载到原模型；
- 错误信息必须明确显示 `model_variant` 不匹配。

WULLE 不使用原模型预训练权重进行正式公平对比。

---

# 14. 必须新增的测试

新建：

```text
tools/test_wulle_experiment_a_v2.py
```

## 14.1 原模型不受影响

确认：

```python
DWTFreqNet(...)
```

仍可构建、前向和严格加载原 checkpoint。

## 14.2 删除节点检查

断言新模型中不存在：

```text
local_encoder1_2
local_encoder2_2
local_encoder3_2
local_encoder1_3
local_encoder2_3
local_encoder1_4
global_channel1_3
global_channel2_3
global_channel1_4
```

## 14.3 新节点检查

断言存在：

```text
wulle_decoder3
wulle_decoder2
wulle_decoder1
```

## 14.4 形状检查

输入：

```python
x = torch.randn(2, 1, 256, 256)
```

检查：

```text
E1：[2,  64, 128, 128]
E2：[2, 128,  64,  64]
E3：[2, 256,  32,  32]
E4：[2, 256,  16,  16]

D3：[2, 256,  32,  32]
D2：[2, 128,  64,  64]
D1：[2,  64, 128, 128]

G1_4：[2,  64, 128, 128]
G2_3：[2, 128,  64,  64]
G3_2：[2, 256,  32,  32]
```

训练模式返回 6 个：

```text
[2, 1, 256, 256]
```

测试模式返回一个：

```text
[2, 1, 256, 256]
```

## 14.5 梯度检查

执行一次：

```python
outputs = model(x)
loss = sum(output.mean() for output in outputs)
loss.backward()
```

检查以下模块有非零梯度：

```text
conv_wavelet_inchannel_local
local_encoder1_1–local_encoder4_1
wulle_decoder1–wulle_decoder3
global_encoder1_1–global_encoder1_4
global_encoder2_1–global_encoder2_3
global_encoder3_1–global_encoder3_2
global_encoder4_1
wave_att_input_t
wave_att_f1–wave_att_f3
TransTo_input
TransTo1e–TransTo3e
最终 decoder
```

## 14.6 Haar/W8M 对应检查

运行当前仓库已有：

```bash
python tools/check_haar_direction_mapping.py
python tools/test_w8m_diagonal.py
```

新模型 smoke test 后再次检查：

```python
assert model.awgm_backends["haar_routing_aligned"] is True
```

仅对 W8M variant 执行该断言。

## 14.7 Variant 兼容

至少测试：

```text
awgm_original
dm_awgm_no_dcn
w8m_diag4_subband_shared
w8m_diag4_axial_diag_shared
```

正式实验禁止 fallback。

---

# 15. 复杂度统计

新建：

```text
tools/profile_wulle_experiment_a_v2.py
```

统一输入：

```text
[1, 1, 256, 256]
```

比较：

```text
DWTFreqNet original
DWTFreqNet_WULLE
```

报告：

```text
总参数量
local branch 参数量
FLOPs
单张 latency
FPS
推理峰值显存
训练峰值显存
```

local branch 参数统计：

```python
def is_local_parameter(name):
    return (
        name.startswith("conv_wavelet_inchannel_local")
        or name.startswith("local_encoder")
        or name.startswith("wulle_decoder")
    )
```

必须确认：

```text
WULLE 不包含被删除旧节点的参数
WULLE 总参数量低于原模型
```

若参数量没有下降，停止正式实验并排查无用模块残留。

---

# 16. 正式实验

## 16.1 第一阶段：仅验证 local branch

对比：

```text
A0：dwtfreqnet_original + awgm_original
A1：dwtfreqnet_wulle_a + awgm_original
```

数据集：

```text
NUAA-SIRST
NUDT-SIRST
IRSTD-1K
```

统一训练设置：

```text
seed：42
patch size：256
batch size：4
epochs：1000
optimizer：Adam
initial lr：1e-3
scheduler：CosineAnnealingLR
eta_min：1e-5
eval start：100
eval every：1
save every：20
threshold：0.5
```

当前最新训练入口默认 `eval_every=1`，因此 A0 和 A1 必须都使用 1。

不要将旧的“每 5 个 epoch 评估一次”的历史 best checkpoint 与新实验直接作为唯一公平比较结果。

## 16.2 第二阶段：与选定 W8M 组合

第一阶段完成后，选用同一个 W8M variant：

```text
A2：dwtfreqnet_original + selected W8M
A3：dwtfreqnet_wulle_a + selected W8M
```

原模型和 WULLE 必须使用完全相同的 W8M variant、seed 和训练设置。

## 16.3 多随机种子条件

只有当 A1 满足以下任一条件时才执行多 seed：

1. 至少两个数据集 mIoU 不低于 A0；
2. 三数据集平均 mIoU 下降不超过 0.3 个百分点，且复杂度明显下降；
3. Fa 明显降低且 Pd 基本不下降。

多 seed：

```text
42
3407
2026
```

报告 mean ± standard deviation。

---

# 17. 评价指标与结果表

性能：

```text
mIoU ↑
nIoU ↑
F1 ↑
Pd ↑
Fa ↓
Best epoch
```

效率：

```text
Params ↓
FLOPs ↓
Latency ↓
Peak memory ↓
```

表格：

| Dataset | Model | AWGM | Seed | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | FLOPs | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | Original | original | 42 | | | | | | | | | |
| NUAA | WULLE-A | original | 42 | | | | | | | | | |

---

# 18. 中间特征记录

为了判断简化后哪里发生变化，验证阶段记录：

```text
E1–E4 feature norm
D1–D3 feature norm
G1–G4 feature norm
G1_4/G2_3/G3_2 feature norm
D1_HVD mean absolute response
D2_HVD mean absolute response
AWGM mean_G_H/V/D
AWGM attention mean/std
```

保存同一批测试图的：

```text
原模型 local 最终节点
WULLE D1/D2/D3
原模型预测
WULLE 预测
GT
```

重点观察：

```text
极小目标是否在 D1/D2 中消失
复杂背景边缘是否被过度保留
Fa 变化来自哪些区域
```

---

# 19. 建议新增文件

```text
model/DWTFreqNet_WULLE.py
tools/test_wulle_experiment_a_v2.py
tools/profile_wulle_experiment_a_v2.py
scripts/run_experiment_a_v2.sh
EXPERIMENT_A_WULLE_V2_RECORD.md
```

最小修改：

```text
train_one.py
README.md
```

禁止修改：

```text
model/DWTFreqNet.py
```

---

# 20. 启动脚本

建议：

```bash
bash scripts/run_experiment_a_v2.sh \
  NUAA-SIRST \
  dwtfreqnet_wulle_a \
  awgm_original \
  0
```

内部：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python train_one.py \
  --model-variant "$MODEL_VARIANT" \
  --dataset-name "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 1000 \
  --batch-size 4 \
  --patch-size 256 \
  --workers 0 \
  --eval-start 100 \
  --eval-every 1 \
  --save-every 20 \
  --threshold 0.5 \
  --seed 42 \
  --awgm-variant "$AWGM_VARIANT"
```

---

# 21. Codex 交付要求

完成代码后返回：

1. 实际 base commit；
2. 新分支名；
3. commit SHA；
4. 新增和修改文件；
5. 原模型回归测试；
6. WULLE 删除节点检查；
7. 中间张量形状；
8. 四个 AWGM variant 的 smoke test；
9. Haar 路由检查结果；
10. 新旧参数量、FLOPs、延迟和显存；
11. 三数据集启动命令；
12. PID、GPU 和输出目录；
13. 当前 epoch；
14. `EXPERIMENT_A_WULLE_V2_RECORD.md`。

建议分支：

```text
codex/experiment-a-wulle-v2
```

建议 commit：

```text
Add isolated Wavelet U-Net local branch experiment
```

---

# 22. 实验边界

Experiment A v2 不能同时实施：

```text
删除第二次 DWT
stage-wise AWGM
直接使用第一次 DWT 的 H/V/D
修改 LDRC
替换 Transformer
修改最终 decoder
修改 loss
修改数据增强
加载原模型部分权重
```

本实验的结论必须仅针对：

> 将复杂嵌套 Low-Frequency Local Feature Encoder 简化为 Wavelet U-Net 后，精度、假警和效率如何变化。
