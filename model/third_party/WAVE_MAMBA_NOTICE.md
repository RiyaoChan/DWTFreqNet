# Wave-Mamba LFSS source notice

- Project: Wave-Mamba
- Paper: *Wave-Mamba: Wavelet State Space Model for Ultra-High-Definition Low-Light Image Enhancement*
- Authors: Wenbin Zou, Hongxia Gao, Weipeng Yang, Tongtong Liu
- Official repository: https://github.com/AlexZou14/Wave-Mamba
- Original source file: `basicsr/archs/wavemamba_arch.py`
- Source commit: `7e8c63f37af7640e228345c410c2e2165e216117`
- Extracted components: `SimpleGate`, `ffn`, `SS2D`, `LFSSBlock`
- Local adaptation: removed unused BasicSR/low-light-network imports, added source metadata,
  formatting/type hints, and a parameter-free `NCHW ↔ token` adapter. The SS2D scan,
  special initialization, LFSS internal residuals and gated depthwise FFN are unchanged.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0). See `WAVE_MAMBA_LICENSE`.

Only the minimum LFSS dependencies were extracted. Wave-Mamba's full network, wavelet
operators, high-frequency enhancement modules, BasicSR registration and low-light
training code are not included.
