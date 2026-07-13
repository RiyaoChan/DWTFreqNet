# Experiment C：SD-AWGM + Encoder-side LDRC

## 0. 实验定位

本实验以 Experiment B 中的 `sd_awgm` 为唯一基础结构，仅增加 Encoder-side LDRC，用于验证：

> `sd_awgm` 与原始 DWTFreqNet 的性能差距，是否主要来自缺失的同尺度自注意力和跨尺度关系建模。

首轮只实现和运行：

```text
sd_awgm_ldrc
```

本轮不实现 SAM-only、CAM-only、Directional Pyramid、Mamba-LDRC、Linear Attention、新位置编码或新损失。


## 1. 基础分支与代码隔离

基于：

```text
Repository: RiyaoChan/DWTFreqNet
Base branch: codex/experiment-b-single-decoder-directional-pyramid
Base commit: 435ab1827ecee4c6b83b669789bb9833a5fd5320
```

Codex 开始前执行：

```bash
git checkout codex/experiment-b-single-decoder-directional-pyramid
git pull
git rev-parse HEAD
```

建议新分支：

```text
codex/experiment-c-sd-awgm-encoder-ldrc
```

禁止修改：

```text
model/DWTFreqNet.py
model/DWTFreqNet_WULLE.py
model/DWTFreqNet_SingleDecoder.py
train_experiment_b.py
```

新建：

```text
model/DWTFreqNet_SingleDecoder_LDRC.py
```


## 2. 新模型实现方式

建议继承当前 `DWTFreqNet_SingleDecoder`，并固定父类结构为：

```python
sd_variant="sd_awgm"
```

示例：

```python
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder

class DWTFreqNet_SingleDecoder_LDRC(DWTFreqNet_SingleDecoder):
    def __init__(self, config, n_channels=1, n_classes=1,
                 img_size=256, vis=False, mode="train", deepsuper=True):
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
        self.encoder_ldrc = EncoderLDRC()
        self.model_variant = "dwtfreqnet_single_decoder_ldrc"
        self.sd_variant = "sd_awgm_ldrc"
        self.ldrc = True
```

这样可以确保 Stem、四级 DWT、DirectionalBandEncoder、Stage-wise AWGM、高频系数对齐和 Single Wavelet Decoder 与 `sd_awgm` 完全一致。唯一新增变量是 Encoder-side LDRC。


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
  │                 ▼
  └── A ← Stage-wise AWGM
            │
            ▼
      E1 / E2 / E3 / E4
            │
            ▼
      Encoder-side LDRC
      ├── SAM：每个尺度内部自注意力
      ├── CAM：不同尺度之间交叉注意力
      └── FFL：前馈网络
            │
            ▼
      Ê1 / Ê2 / Ê3 / Ê4
            │
            ▼
      Single Wavelet Decoder
            │
            ▼
         Prediction
```

职责划分：

```text
AWGM：同一次 DWT 的 H/V/D 调制同源低频 A
LDRC：对 AWGM 后的多尺度低频语义特征 E1–E4 建模
原始 H/V/D：经 1×1 对齐后继续作为 IDWT 高频系数
```

禁止重新加入第二次 DWT、后置 AWGM、Directional Pyramid、Global Dense Encoder 或双 decoder。


## 4. LDRC 输入与 Token 设计

LDRC 输入固定为：

```text
E1、E2、E3、E4
```

当前形状：

```text
E1: [B,  64, 128, 128]
E2: [B, 128,  64,  64]
E3: [B, 256,  32,  32]
E4: [B, 256,  16,  16]
```

E1 不能直接展平为 16384 tokens。统一投影为：

```text
R1: [B, 128, 64, 64]
R2: [B, 128, 64, 64]
R3: [B, 128, 32, 32]
R4: [B, 128, 16, 16]
```

实现：

```python
self.proj1 = nn.Sequential(
    nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.GELU(),
)
self.proj2 = nn.Sequential(
    nn.Conv2d(128, 128, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.GELU(),
)
self.proj3 = nn.Sequential(
    nn.Conv2d(256, 128, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.GELU(),
)
self.proj4 = nn.Sequential(
    nn.Conv2d(256, 128, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.GELU(),
)
```

展平：

```python
T1 = R1.flatten(2).transpose(1, 2)  # [B, 4096, 128]
T2 = R2.flatten(2).transpose(1, 2)  # [B, 4096, 128]
T3 = R3.flatten(2).transpose(1, 2)  # [B, 1024, 128]
T4 = R4.flatten(2).transpose(1, 2)  # [B,  256, 128]
```


## 5. 复用原始 LDRC

直接复用：

```python
from decoder_fuse.transformer_dec_fuse_none_posqkv_dropout import TransFuseModel
```

首轮保持原始 Block：

```text
SAM
  ↓ residual + LayerNorm
CAM
  ↓ residual + LayerNorm
FFL
  ↓ residual + LayerNorm
```

保持：

```text
num_blocks = 1
embedding dim = 128
head_count = 4
dropout = 0.2
不新增位置编码
不修改 attention scaling
不修改 Q/K/V、FFL 和 LayerNorm
```


## 6. EncoderLDRC 构造

```python
class EncoderLDRC(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj1 = ...
        self.proj2 = ...
        self.proj3 = ...
        self.proj4 = ...

        self.ldrc4 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=256,
            y_channels=128, ny=9216,
        )
        self.ldrc3 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=1024,
            y_channels=128, ny=8448,
        )
        self.ldrc2 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=4096,
            y_channels=128, ny=5376,
        )
        self.ldrc1 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=4096,
            y_channels=128, ny=5376,
        )

        self.back1 = nn.Conv2d(128, 64, 1)
        self.back2 = nn.Conv2d(128, 128, 1)
        self.back3 = nn.Conv2d(128, 256, 1)
        self.back4 = nn.Conv2d(128, 256, 1)

        self.gamma1 = nn.Parameter(torch.full((1, 64, 1, 1), 1e-3))
        self.gamma2 = nn.Parameter(torch.full((1, 128, 1, 1), 1e-3))
        self.gamma3 = nn.Parameter(torch.full((1, 256, 1, 1), 1e-3))
        self.gamma4 = nn.Parameter(torch.full((1, 256, 1, 1), 1e-3))
```

`gamma=1e-3` 使模型初始化时近似 `sd_awgm`，同时保证梯度从第一步进入 LDRC。


## 7. LDRC 级联更新

保持深层到浅层：

```text
T4 → T3 → T2 → T1
```

### 更新 T4

```python
T4e = self.ldrc4(
    T4,
    torch.cat([T1, T2, T3], dim=1),
)
```

```text
Query: 256
Context: 4096 + 4096 + 1024 = 9216
```

### 更新 T3

```python
T3e = self.ldrc3(
    T3,
    torch.cat([T1, T2, T4e], dim=1),
)
```

```text
Query: 1024
Context: 4096 + 4096 + 256 = 8448
```

### 更新 T2

```python
T2e = self.ldrc2(
    T2,
    torch.cat([T1, T3e, T4e], dim=1),
)
```

```text
Query: 4096
Context: 4096 + 1024 + 256 = 5376
```

### 更新 T1

```python
T1e = self.ldrc1(
    T1,
    torch.cat([T2e, T3e, T4e], dim=1),
)
```

```text
Query: 4096
Context: 4096 + 1024 + 256 = 5376
```


## 8. 回投影和残差注入

```python
R1e = T1e.transpose(1, 2).reshape(B, 128, 64, 64)
R2e = T2e.transpose(1, 2).reshape(B, 128, 64, 64)
R3e = T3e.transpose(1, 2).reshape(B, 128, 32, 32)
R4e = T4e.transpose(1, 2).reshape(B, 128, 16, 16)

delta1 = self.back1(F.interpolate(
    R1e, size=E1.shape[-2:], mode="bilinear", align_corners=False
))
delta2 = self.back2(R2e)
delta3 = self.back3(R3e)
delta4 = self.back4(R4e)

E1e = E1 + self.gamma1 * delta1
E2e = E2 + self.gamma2 * delta2
E3e = E3 + self.gamma3 * delta3
E4e = E4 + self.gamma4 * delta4
```

返回：

```python
return E1e, E2e, E3e, E4e
```


## 9. 新模型 forward

重写 forward，但复用父类：

```text
_encode_stage
_refine_coefficients
_idwt
```

关键流程：

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

E1e, E2e, E3e, E4e = self.encoder_ldrc(
    encoded[1], encoded[2], encoded[3], encoded[4]
)

enhanced = {1: E1e, 2: E2e, 3: E3e, 4: E4e}

coefficients, _, statistics = self._refine_coefficients(
    raw_bands,
    pyramid_features=None,
)

def stage_coefficients(stage):
    return tuple(
        coefficients[(stage, d)] for d in ("H", "V", "D")
    )

u3 = self._idwt(enhanced[4], *stage_coefficients(4))
l3 = self.decoder_fuse3(torch.cat([u3, enhanced[3]], dim=1))

u2 = self._idwt(l3, *stage_coefficients(3))
l2 = self.decoder_fuse2(torch.cat([u2, enhanced[2]], dim=1))

u1 = self._idwt(l2, *stage_coefficients(2))
l1 = self.decoder_fuse1(torch.cat([u1, enhanced[1]], dim=1))

u0 = self._idwt(l1, *stage_coefficients(1))
l0 = self.decoder_fuse0(torch.cat([u0, x0], dim=1))

out = self.out_head(l0)
```

必须保持：

```text
DWT = 4
IDWT = 4
第二次 DWT = 0
Pyramid = 0
```


## 10. Deep Supervision

```python
gt5 = self.gt_conv5(E4e)
gt4 = self.gt_conv4(l3)
gt3 = self.gt_conv3(l2)
gt2 = self.gt_conv2(l1)
```

其余上采样、`d0` 融合和返回形式与 `sd_awgm` 完全一致。不要修改 loss。


## 11. 元数据

```python
self.model_variant = "dwtfreqnet_single_decoder_ldrc"
self.sd_variant = "sd_awgm_ldrc"
self.single_decoder = True
self.stage_wise_awgm = True
self.directional_pyramid = False
self.second_dwt = False
self.ldrc = True
self.ldrc_position = "encoder_after_stage_awgm"
self.ldrc_input = "E1_E2_E3_E4"
self.ldrc_order = "E4_E3_E2_E1"
self.ldrc_dim = 128
self.ldrc_blocks = 1
self.mamba = False
self.coefficient_mode = "aligned_raw"
```


## 12. 独立训练入口

新建：

```text
train_experiment_c.py
```

模型固定为：

```python
model = DWTFreqNet_SingleDecoder_LDRC(
    get_DWTFreqNet_config(),
    mode=mode,
    deepsuper=True,
)
```

不需要 `--sd-variant`、`--awgm-variant`、`--pyramid` 或 `--mamba`。

输出目录：

```text
runs/experiment_c/<dataset>/sd_awgm_ldrc/seed42
```

`run_config.json` 至少记录：

```json
{
  "model_variant": "dwtfreqnet_single_decoder_ldrc",
  "sd_variant": "sd_awgm_ldrc",
  "single_decoder": true,
  "stage_wise_awgm": true,
  "directional_pyramid": false,
  "second_dwt": false,
  "ldrc": true,
  "ldrc_input": "E1_E2_E3_E4",
  "ldrc_order": "E4_E3_E2_E1",
  "ldrc_dim": 128,
  "ldrc_blocks": 1,
  "gamma_init": 0.001,
  "coefficient_mode": "aligned_raw"
}
```

正式实验从随机初始化开始，不加载 `sd_awgm` 或其他模型权重。


## 13. 必须新增的测试

新建：

```text
tools/test_sd_awgm_ldrc_experiment_c.py
```

### 输出和形状

输入：

```python
x = torch.randn(2, 1, 256, 256)
```

训练输出：

```text
6 × [2, 1, 256, 256]
```

测试输出：

```text
[2, 1, 256, 256]
```

中间形状：

```text
E1 [2,64,128,128]
E2 [2,128,64,64]
E3 [2,256,32,32]
E4 [2,256,16,16]

R1/R2 [2,128,64,64]
R3 [2,128,32,32]
R4 [2,128,16,16]

T1/T2 [2,4096,128]
T3 [2,1024,128]
T4 [2,256,128]

E1e [2,64,128,128]
E2e [2,128,64,64]
E3e [2,256,32,32]
E4e [2,256,16,16]

L3 [2,256,32,32]
L2 [2,128,64,64]
L1 [2,64,128,128]
L0 [2,32,256,256]
```

调用计数：

```text
DWT = 4
IDWT = 4
```

### 近似 identity 回归

构建：

```text
baseline = DWTFreqNet_SingleDecoder(sd_awgm)
new_model = DWTFreqNet_SingleDecoder_LDRC()
```

复制共同参数，临时将 gamma1–4 置零，在 eval 模式检查：

```python
torch.testing.assert_close(
    baseline_output,
    new_output,
    rtol=1e-5,
    atol=1e-6,
)
```

### 梯度检查

检查非零梯度：

```text
stem
local_encoder1–4
dir_encoder1–4
stage_awgm1–4
align_H/V/D 1–4
encoder_ldrc.proj1–4
encoder_ldrc.ldrc1–4
encoder_ldrc.back1–4
encoder_ldrc.gamma1–4
decoder_fuse0–3
output heads
```

### Haar方向

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


## 14. 复杂度统计

新建：

```text
tools/profile_sd_awgm_ldrc_experiment_c.py
```

比较：

```text
Original DWTFreqNet
WULLE-A
sd_awgm
sd_awgm_ldrc
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
LDRC parameter count
```

额外记录：

```text
SAM attention matrix sizes
CAM attention matrix sizes
sd_awgm → sd_awgm_ldrc 增量参数和增量 FLOPs
```


## 15. 首轮正式实验

本轮只训练：

```text
sd_awgm_ldrc
```

数据集优先级：

```text
1. NUDT-SIRST
2. NUAA-SIRST
3. IRSTD-1K
```

NUDT 优先，因为 `sd_awgm` 与原模型差距最大，最适合验证 LDRC 是否为缺失环节。

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

必须与现有 `sd_awgm` 保持相同数据划分、随机种子、loss、增强、优化器和评价流程。


## 16. GPU 调度与记录

新建：

```text
scripts/run_experiment_c.sh
scripts/launch_experiment_c_queue.sh
EXPERIMENT_C_SD_AWGM_LDRC_RECORD.md
```

要求：

1. 不终止 Experiment A、B 或 W8M 正式任务；
2. 只使用空闲 GPU；
3. 每个数据集独立目录；
4. 记录 PID、GPU、启动时间、epoch；
5. 不覆盖已有结果。

结果表：

| Dataset | Model | Best epoch | mIoU | nIoU | F1 | Pd | Fa | Params | FLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | sd_awgm | existing | | | | | | | |
| NUAA | sd_awgm_ldrc | | | | | | | | |
| NUDT | sd_awgm | existing | | | | | | | |
| NUDT | sd_awgm_ldrc | | | | | | | | |
| IRSTD | sd_awgm | existing | | | | | | | |
| IRSTD | sd_awgm_ldrc | | | | | | | | |

额外记录：

```text
gamma1–4 mean
E1e/E1、E2e/E2、E3e/E3、E4e/E4 norm ratio
SAM output norm
CAM output norm
FFL output norm
```


## 17. 结果判定

### 支持 LDRC 假设

若 NUDT 上 `sd_awgm_ldrc` 明显高于 `sd_awgm`，并且 Pd 提高或 Fa 下降，则说明跨尺度关系建模是缺失环节之一。

### 接近原模型

若 NUDT mIoU 接近约 0.95，说明原模型的大部分优势可能来自 LDRC，而不一定依赖原复杂 Global Dense Branch。

### 提升有限

若提升小于 0.5 个百分点，说明差距更可能来自 Global Dense Encoder、local-global 多阶段反馈或其他耦合结构。

### 性能下降

重点检查：

```text
gamma 是否过大
E1 stride=2 是否损失小目标
CAM 是否被背景 token 主导
SAM/CAM 是否过度平滑
LDRC 输出范数是否远大于输入
```

首轮完成前不要增加 SAM-only、CAM-only、2-block、位置编码、Mamba 或 Pyramid。


## 18. 建议新增文件

```text
model/DWTFreqNet_SingleDecoder_LDRC.py
train_experiment_c.py
tools/test_sd_awgm_ldrc_experiment_c.py
tools/profile_sd_awgm_ldrc_experiment_c.py
scripts/run_experiment_c.sh
scripts/launch_experiment_c_queue.sh
EXPERIMENT_C_SD_AWGM_LDRC_RECORD.md
```

建议 commit：

```text
Add encoder-side LDRC to SD-AWGM
```


## 19. 最终结构主线

```text
第一次 DWT
   │
   ├── H/V/D → DirectionalBandEncoder
   │                 │
   │                 ▼
   └── A ← Stage-wise AWGM
              │
              ▼
        E1/E2/E3/E4
              │
              ▼
        Encoder-side LDRC
        SAM：尺度内部关系
        CAM：尺度之间关系
              │
              ▼
        Ê1/Ê2/Ê3/Ê4
              │
              ▼
     原始 H/V/D + Single Decoder
              │
              ▼
           Prediction
```

\[
oxed{
	ext{AWGM负责频率引导，LDRC负责跨尺度语义关系，原始H/V/D负责IDWT细节重构}
}
\]
