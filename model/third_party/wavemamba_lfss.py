"""Minimal Wave-Mamba LFSS extraction with an NCHW adapter.

# Extracted and adapted from:
# Wave-Mamba: Wavelet State Space Model for
# Ultra-High-Definition Low-Light Image Enhancement
# Official repository:
# https://github.com/AlexZou14/Wave-Mamba
# Original source:
# basicsr/archs/wavemamba_arch.py
# Source commit: 7e8c63f37af7640e228345c410c2e2165e216117
# License: CC BY-NC-SA 4.0

Only formatting, unused-import removal, type hints, and the parameter-free
NCHW adapter were added. The LFSS and SS2D computations are unchanged.
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.layers import DropPath


WAVE_MAMBA_SOURCE_COMMIT = "7e8c63f37af7640e228345c410c2e2165e216117"
WAVE_MAMBA_SOURCE_FILE = "basicsr/archs/wavemamba_arch.py"
WAVE_MAMBA_SOURCE_URL = "https://github.com/AlexZou14/Wave-Mamba"
WAVE_MAMBA_LICENSE = "CC BY-NC-SA 4.0"


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ffn(nn.Module):
    def __init__(self, num_feat, ffn_expand=2):
        super().__init__()
        dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(
            num_feat, dw_channel, kernel_size=1, padding=0, stride=1
        )
        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
        )
        self.conv3 = nn.Conv2d(
            dw_channel // 2, num_feat, kernel_size=1, padding=0, stride=1
        )
        self.sg = SimpleGate()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv3(x)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([projection.weight for projection in self.x_proj], dim=0)
        )
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([projection.weight for projection in self.dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([projection.bias for projection in self.dt_projs], dim=0)
        )
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        length = height * width
        directions = 4

        x_hwwh = torch.stack(
            [
                x.view(batch, -1, length),
                torch.transpose(x, dim0=2, dim1=3)
                .contiguous()
                .view(batch, -1, length),
            ],
            dim=1,
        ).view(batch, 2, -1, length)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(batch, directions, -1, length),
            self.x_proj_weight,
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(batch, directions, -1, length),
            self.dt_projs_weight,
        )

        xs = xs.float().view(batch, -1, length)
        dts = dts.contiguous().float().view(batch, -1, length)
        Bs = Bs.float().view(batch, directions, -1, length)
        Cs = Cs.float().view(batch, directions, -1, length)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch, directions, -1, length)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(
            batch, 2, -1, length
        )
        wh_y = (
            torch.transpose(
                out_y[:, 1].view(batch, -1, width, height), dim0=2, dim1=3
            )
            .contiguous()
            .view(batch, -1, length)
        )
        invwh_y = (
            torch.transpose(
                inv_y[:, 1].view(batch, -1, width, height), dim0=2, dim1=3
            )
            .contiguous()
            .view(batch, -1, length)
        )
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        batch, height, width, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = (
            torch.transpose(y, dim0=1, dim1=2)
            .contiguous()
            .view(batch, height, width, -1)
        )
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LFSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand,
            dropout=attn_drop_rate,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = ffn(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        batch, _, channels = input.shape
        input = input.view(batch, *x_size, channels).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(
            self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1).contiguous()
        return x.view(batch, -1, channels).contiguous()


class WaveMambaLFSSNCHWAdapter(nn.Module):
    """Parameter-free NCHW/token shape adapter around the original LFSSBlock."""

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
            raise RuntimeError(f"LFSS adapter expects NCHW input, got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise RuntimeError(
                f"LFSS adapter expects {self.channels} channels, got {channels}"
            )
        tokens = (
            x.permute(0, 2, 3, 1)
            .reshape(batch, height * width, channels)
            .contiguous()
        )
        tokens = self.block(tokens, (height, width))
        return (
            tokens.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


__all__ = [
    "LFSSBlock",
    "SS2D",
    "SimpleGate",
    "WaveMambaLFSSNCHWAdapter",
    "WAVE_MAMBA_LICENSE",
    "WAVE_MAMBA_SOURCE_COMMIT",
    "WAVE_MAMBA_SOURCE_FILE",
    "WAVE_MAMBA_SOURCE_URL",
    "ffn",
]
