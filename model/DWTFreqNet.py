# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author  : Qianwen Ma
# @File    : DWTFreqNet.py
# @Software: PyCharm
# coding=utf-8
#Time: 2025.1.24

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from decoder_fuse.transformer_dec_fuse_none_posqkv_dropout import TransFuseModel
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
import numpy as np
from thop import profile

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None


AWGM_VARIANTS = (
    'awgm_original',
    'dm_awgm_full',
    'dm_awgm_no_mamba',
    'dm_awgm_no_dcn',
    'dm_awgm_conv_only',
    'w8m_diag2_subband_shared',
    'w8m_diag4_independent',
    'w8m_diag4_pair_shared',
    'w8m_diag4_subband_shared',
    'w8m_diag4_axial_diag_shared',
    'w8m_diag4_axial_diag_shared_dir_embed',
    'w8m_diag4_all_shared',
)


W8M_VARIANT_CONFIGS = {
    'w8m_diag2_subband_shared': {
        'share_mode': 'subband_shared_3',
        'diag_directions': 2,
        'use_direction_embedding': False,
    },
    'w8m_diag4_independent': {
        'share_mode': 'independent_8',
        'diag_directions': 4,
        'use_direction_embedding': False,
    },
    'w8m_diag4_pair_shared': {
        'share_mode': 'pair_shared_4',
        'diag_directions': 4,
        'use_direction_embedding': False,
    },
    'w8m_diag4_subband_shared': {
        'share_mode': 'subband_shared_3',
        'diag_directions': 4,
        'use_direction_embedding': False,
    },
    'w8m_diag4_axial_diag_shared': {
        'share_mode': 'axial_diag_shared_2',
        'diag_directions': 4,
        'use_direction_embedding': False,
    },
    'w8m_diag4_axial_diag_shared_dir_embed': {
        'share_mode': 'axial_diag_shared_2',
        'diag_directions': 4,
        'use_direction_embedding': True,
    },
    'w8m_diag4_all_shared': {
        'share_mode': 'all_shared_1',
        'diag_directions': 4,
        'use_direction_embedding': False,
    },
}


def get_DWTFreqNet_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    # ********** useless **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super(HaarWaveletTransform, self).__init__()

        # Define Haar wavelet filters
        self.haar_matrix_LL = torch.tensor([
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

        self.haar_matrix_LH = torch.tensor([
            [1 / 2, -1 / 2],
            [1 / 2, -1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

        self.haar_matrix_HL = torch.tensor([
            [1 / 2, 1 / 2],
            [-1 / 2, -1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

        self.haar_matrix_HH = torch.tensor([
            [1 / 2, -1 / 2],
            [-1 / 2, 1 / 2]
        ], dtype=torch.float32).reshape(1, 1, 2, 2)

    def forward(self, x):
        B, C, H, W = x.size()

        # Ensure the input tensor has even dimensions
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("Input dimensions must be even.")

        # Move haar_matrix to the same device as x
        device = x.device
        haar_matrix_LL = self.haar_matrix_LL.to(device)
        haar_matrix_LH = self.haar_matrix_LH.to(device)
        haar_matrix_HL = self.haar_matrix_HL.to(device)
        haar_matrix_HH = self.haar_matrix_HH.to(device)

        # Reshape input tensor for convolution
        x_reshaped = x.view(B*C, 1, H, W)

        # Perform convolutions for all channels at once
        out_LL = F.conv2d(x_reshaped, haar_matrix_LL, stride=2, padding=0).view(B, C, H//2, W//2)
        out_LH = F.conv2d(x_reshaped, haar_matrix_LH, stride=2, padding=0).view(B, C, H//2, W//2)
        out_HL = F.conv2d(x_reshaped, haar_matrix_HL, stride=2, padding=0).view(B, C, H//2, W//2)
        out_HH = F.conv2d(x_reshaped, haar_matrix_HH, stride=2, padding=0).view(B, C, H//2, W//2)

        return out_LL, out_LH, out_HL, out_HH


HAAR_CODE_BAND_NAMES = {
    'H': 'LH',
    'V': 'HL',
    'D': 'HH',
}

# This describes the physical scan assigned to each returned band.  The
# synthetic check below keeps filter response and routing separate so a future
# Haar convention change cannot silently swap the scan directions.
W8M_CURRENT_SCAN_AXIS = {
    'H': 'vertical',
    'V': 'horizontal',
}


def check_haar_direction_correspondence(size=32, device='cpu'):
    """Measure which returned Haar band responds to horizontal/vertical lines.

    The synthetic line is placed at an odd coordinate so a stride-2 Haar
    window crosses it.  An even-aligned line/edge can fall exactly between
    windows and incorrectly produce zero high-frequency response.
    """
    if size < 4 or size % 2:
        raise ValueError('size must be an even integer greater than or equal to 4')
    line_index = size // 2 - 1
    if line_index % 2 == 0:
        line_index -= 1
    device = torch.device(device)
    haar = HaarWaveletTransform().to(device)

    horizontal_line = torch.zeros(1, 1, size, size, device=device)
    horizontal_line[:, :, line_index, :] = 1.0
    vertical_line = torch.zeros(1, 1, size, size, device=device)
    vertical_line[:, :, :, line_index] = 1.0
    horizontal_step = torch.zeros(1, 1, size, size, device=device)
    horizontal_step[:, :, line_index:, :] = 1.0
    vertical_step = torch.zeros(1, 1, size, size, device=device)
    vertical_step[:, :, :, line_index:] = 1.0

    def energies(image):
        _, band_h, band_v, band_d = haar(image)
        return {
            'H': float(band_h.abs().sum().item()),
            'V': float(band_v.abs().sum().item()),
            'D': float(band_d.abs().sum().item()),
        }

    responses = {
        'horizontal_line': energies(horizontal_line),
        'vertical_line': energies(vertical_line),
        'horizontal_step': energies(horizontal_step),
        'vertical_step': energies(vertical_step),
    }
    horizontal_band = max(
        ('H', 'V'), key=lambda band: responses['horizontal_line'][band]
    )
    vertical_band = max(
        ('H', 'V'), key=lambda band: responses['vertical_line'][band]
    )
    if horizontal_band == vertical_band:
        raise AssertionError('Haar direction check did not separate H and V bands')
    band_response_orientation = {
        horizontal_band: 'horizontal',
        vertical_band: 'vertical',
    }
    routing_aligned = all(
        band_response_orientation[band] == W8M_CURRENT_SCAN_AXIS[band]
        for band in ('H', 'V')
    )
    return {
        'size': size,
        'line_index': line_index,
        'code_band_names': dict(HAAR_CODE_BAND_NAMES),
        'responses': responses,
        'band_response_orientation': band_response_orientation,
        'current_scan_axis': dict(W8M_CURRENT_SCAN_AXIS),
        'routing_aligned': routing_aligned,
        'recommended_scan_axis': dict(band_response_orientation),
    }


###反向小波变换######
class InverseHaarWaveletTransform(nn.Module):
    def __init__(self):
        super(InverseHaarWaveletTransform, self).__init__()

        # Define inverse Haar wavelet filters
        self.inv_haar_matrix = torch.tensor([
            [1 / 2, 1 / 2],
            [1 / 2, 1 / 2],
            [1 / 2, -1 / 2],
            [1 / 2, -1 / 2],
            [1 / 2, 1 / 2],
            [-1 / 2, -1 / 2],
            [1 / 2, -1 / 2],
            [-1 / 2, 1 / 2]
        ], dtype=torch.float32).reshape(4, 1, 2, 2)  # Adjusted shape

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.size()

        # Stack the coefficients
        coeffs = torch.stack([LL, LH, HL, HH], dim=2)  # Shape (B, C, 4, H, W)

        # Move inv_haar_matrix to the same device as coefficients
        device = LL.device
        inv_haar_matrix = self.inv_haar_matrix.to(device)

        # Perform the inverse Haar wavelet transform
        output = F.conv_transpose2d(coeffs.view(B * C, 4, H, W), inv_haar_matrix, stride=2, padding=0)

        return output.view(B, C, H * 2, W * 2)

class WaveDownattention(nn.Module):###这个是那个小波注意力机制
    def __init__(self, in_channels):
        super().__init__()

        # self.dwt = DWT_2D(wave='haar')
        self.conv_A_H = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_A_V = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_A_D = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.to_att = nn.Sequential(
                    nn.Conv2d(2, 1, 1, 1, 0),
                    nn.Sigmoid()
        )
        self.att_weights = nn.Parameter(torch.ones(3) / 3)
        # self.pw = nn.Conv2d(in_channels * 4, in_channels * 2, 1, 1, 0)

    def forward(self, A,H,V,D):
        # x = self.dwt(x)
        # x_ll, x_lh, x_hl, x_hh = x.chunk(4, dim=1)
        # get attention
        AH =  self.conv_A_H(A + H)
        AV =  self.conv_A_V(A + V)
        AD = self.conv_A_D(A + D)

##空间注意力机制
        AH_att_maxpool, _ = torch.max(AH, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        AH_att_avgpool = torch.mean(AH, dim=1, keepdim=True)
        AH_att = torch.cat([AH_att_maxpool, AH_att_avgpool], dim=1)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]

        AV_att_maxpool, _ = torch.max(AV, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        AV_att_avgpool = torch.mean(AV, dim=1, keepdim=True)
        AV_att = torch.cat([AV_att_maxpool, AV_att_avgpool], dim=1)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]

        AD_att_maxpool, _ = torch.max(AD, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        AD_att_avgpool = torch.mean(AD, dim=1, keepdim=True)
        AD_att = torch.cat([AD_att_maxpool, AD_att_avgpool], dim=1)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]

        # wave_att = AH_att+AV_att+AD_att
        wave_att = self.att_weights[0] * AH_att + self.att_weights[1] * AV_att + self.att_weights[2] * AD_att

        ##空间注意力机制

        att_map = self.to_att(wave_att)
        # squeeze
        # x_s = self.pw(x)
        o = torch.mul(A, att_map) + A  #这里虽然mul后的两个tensor维度不统一，但是通过广播机制能够将那个1的维度自行复制，以达到维度统一
        # hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        return o #hi_bands #第二个分量好像是高频分量，原网络用于上采样

class FallbackSequenceMixer(nn.Module):
    """Smoke-test fallback. Formal DM-AWGM experiments must use mamba_ssm."""

    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return self.ffn(x)


class ConvBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class HorizontalBiMamba(nn.Module):
    def __init__(self, dim, allow_fallback=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if Mamba is None:
            if not allow_fallback:
                raise RuntimeError(
                    "dm_awgm requires mamba_ssm.Mamba. Install mamba_ssm or "
                    "enable fallback for smoke tests only."
                )
            print("[Warning] mamba_ssm.Mamba is unavailable. "
                  "Using an MLP fallback for smoke test only.")
            self.mamba_lr = FallbackSequenceMixer(dim)
            self.mamba_rl = FallbackSequenceMixer(dim)
            self.backend = "fallback_mlp"
        else:
            self.mamba_lr = Mamba(d_model=dim)
            self.mamba_rl = Mamba(d_model=dim)
            for module in self.mamba_lr.modules():
                module._skip_external_init = True
            for module in self.mamba_rl.modules():
                module._skip_external_init = True
            self.backend = "mamba_ssm.Mamba"
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, channels, height, width = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        sequence = self.norm(sequence)
        left_to_right = self.mamba_lr(sequence)
        right_to_left = torch.flip(
            self.mamba_rl(torch.flip(sequence, dims=[1])), dims=[1]
        )
        output = self.proj(left_to_right + right_to_left)
        output = output.reshape(batch, height, width, channels)
        output = output.permute(0, 3, 1, 2).contiguous()
        return x + output


class VerticalBiMamba(nn.Module):
    def __init__(self, dim, allow_fallback=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if Mamba is None:
            if not allow_fallback:
                raise RuntimeError(
                    "dm_awgm requires mamba_ssm.Mamba. Install mamba_ssm or "
                    "enable fallback for smoke tests only."
                )
            print("[Warning] mamba_ssm.Mamba is unavailable. "
                  "Using an MLP fallback for smoke test only.")
            self.mamba_tb = FallbackSequenceMixer(dim)
            self.mamba_bt = FallbackSequenceMixer(dim)
            self.backend = "fallback_mlp"
        else:
            self.mamba_tb = Mamba(d_model=dim)
            self.mamba_bt = Mamba(d_model=dim)
            for module in self.mamba_tb.modules():
                module._skip_external_init = True
            for module in self.mamba_bt.modules():
                module._skip_external_init = True
            self.backend = "mamba_ssm.Mamba"
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, channels, height, width = x.shape
        sequence = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        sequence = self.norm(sequence)
        top_to_bottom = self.mamba_tb(sequence)
        bottom_to_top = torch.flip(
            self.mamba_bt(torch.flip(sequence, dims=[1])), dims=[1]
        )
        output = self.proj(top_to_bottom + bottom_to_top)
        output = output.reshape(batch, width, height, channels)
        output = output.permute(0, 3, 2, 1).contiguous()
        return x + output


class DeformableDiagonalBranch(nn.Module):
    def __init__(self, dim, kernel_size=3, allow_fallback=False):
        super().__init__()
        padding = kernel_size // 2
        if DeformConv2d is None:
            if not allow_fallback:
                raise RuntimeError(
                    "dm_awgm requires torchvision.ops.DeformConv2d. Install a "
                    "compatible torchvision build or enable smoke-test fallback."
                )
            print("[Warning] torchvision.ops.DeformConv2d is unavailable. "
                  "Falling back to depthwise separable conv for smoke test only.")
            self.fallback = ConvBranch(dim)
            self.backend = "fallback_depthwise_conv"
        else:
            self.offset = nn.Conv2d(
                dim,
                2 * kernel_size * kernel_size,
                kernel_size,
                padding=padding,
            )
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.offset._skip_external_init = True
            self.dcn = DeformConv2d(
                dim, dim, kernel_size=kernel_size, padding=padding
            )
            self.norm = nn.BatchNorm2d(dim)
            self.act = nn.GELU()
            self.pw = nn.Conv2d(dim, dim, 1)
            self.fallback = None
            self.backend = "torchvision.ops.DeformConv2d"

    def forward(self, x):
        if self.fallback is not None:
            return self.fallback(x)
        offset = self.offset(x)
        output = self.dcn(x, offset)
        output = self.pw(self.act(self.norm(output)))
        return x + output


class DirectionFusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 3, 1),
        )
        self.to_att = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, A, FH, FV, FD):
        logits = self.gate(torch.cat([A, FH, FV, FD], dim=1))
        weights = torch.softmax(logits, dim=1)
        fused = (
            weights[:, 0:1] * FH
            + weights[:, 1:2] * FV
            + weights[:, 2:3] * FD
        )
        attention = self.to_att(torch.cat([A, fused], dim=1))
        return attention, weights


class DirectionMatchedAWGM(nn.Module):
    def __init__(
        self,
        in_channels,
        use_mamba=True,
        use_dcn=True,
        fusion='spatial_softmax',
        allow_fallback=False,
    ):
        super().__init__()
        if fusion != 'spatial_softmax':
            raise ValueError("Only fusion='spatial_softmax' is currently supported")
        self.pre_h = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_v = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_d = nn.Conv2d(in_channels, in_channels, 1)

        if use_mamba:
            self.h_branch = HorizontalBiMamba(in_channels, allow_fallback)
            self.v_branch = VerticalBiMamba(in_channels, allow_fallback)
            self.mamba_backend = self.h_branch.backend
        else:
            self.h_branch = ConvBranch(in_channels)
            self.v_branch = ConvBranch(in_channels)
            self.mamba_backend = "not_requested"

        if use_dcn:
            self.d_branch = DeformableDiagonalBranch(
                in_channels, allow_fallback=allow_fallback
            )
            self.dcn_backend = self.d_branch.backend
        else:
            self.d_branch = ConvBranch(in_channels)
            self.dcn_backend = "not_requested"

        self.fusion = DirectionFusionGate(in_channels)
        self.band_scan_axis = dict(W8M_CURRENT_SCAN_AXIS)
        self.last_direction_weights = None
        self.last_attention_map = None
        self.last_branch_norms = None

    def forward(self, A, H, V, D):
        # In this repository H=LH responds to vertical structure, while V=HL
        # responds to horizontal structure.  Preserve H/V fusion ordering but
        # match each raw band to its physical scan axis.
        FH = self.v_branch(self.pre_h(A + H))
        FV = self.h_branch(self.pre_v(A + V))
        FD = self.d_branch(self.pre_d(A + D))
        attention, direction_weights = self.fusion(A, FH, FV, FD)
        if not self.training:
            self.last_direction_weights = direction_weights.detach()
            self.last_attention_map = attention.detach()
            self.last_branch_norms = {
                'axial': 0.5 * (
                    FH.detach().float().square().mean().sqrt().item()
                    + FV.detach().float().square().mean().sqrt().item()
                ),
                'diagonal': FD.detach().float().square().mean().sqrt().item(),
            }
        return A * attention + A


def _build_mamba_mixer(dim, allow_fallback=False):
    if Mamba is None:
        if not allow_fallback:
            raise RuntimeError(
                "W8M requires mamba_ssm.Mamba. Install mamba_ssm or enable "
                "fallback for smoke tests only."
            )
        module = FallbackSequenceMixer(dim)
        backend = "fallback_mlp"
    else:
        module = Mamba(d_model=dim)
        for child in module.modules():
            child._skip_external_init = True
        backend = "mamba_ssm.Mamba"
    return module, backend


def inverse_permutation(index):
    inverse = torch.empty_like(index)
    inverse[index] = torch.arange(index.numel(), device=index.device)
    return inverse


class DiagonalIndexCache:
    """Build and cache diagonal permutations on the target device."""

    _cache = {}

    @staticmethod
    def _walk(height, width, row, column, row_step, column_step):
        diagonal = []
        while 0 <= row < height and 0 <= column < width:
            diagonal.append(row * width + column)
            row += row_step
            column += column_step
        return diagonal

    @classmethod
    def _nwse_groups(cls, height, width):
        starts = [(0, column) for column in range(width)]
        starts.extend((row, 0) for row in range(1, height))
        return [
            cls._walk(height, width, row, column, 1, 1)
            for row, column in starts
        ]

    @classmethod
    def _nesw_groups(cls, height, width):
        starts = [(0, column) for column in range(width - 1, -1, -1)]
        starts.extend((row, width - 1) for row in range(1, height))
        return [
            cls._walk(height, width, row, column, 1, -1)
            for row, column in starts
        ]

    @staticmethod
    def _flatten_groups(groups, order):
        if order == 'concat':
            return [position for diagonal in groups for position in diagonal]
        if order == 'snake':
            return [
                position
                for diagonal_index, diagonal in enumerate(groups)
                for position in (
                    diagonal if diagonal_index % 2 == 0 else reversed(diagonal)
                )
            ]
        raise ValueError("diag_order must be 'concat' or 'snake'")

    @classmethod
    def build(cls, height, width, order='snake', device=None):
        if height <= 0 or width <= 0:
            raise ValueError('height and width must be positive')
        device = torch.device('cpu' if device is None else device)
        key = (height, width, order, device.type, device.index)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached

        nwse = cls._flatten_groups(cls._nwse_groups(height, width), order)
        nesw = cls._flatten_groups(cls._nesw_groups(height, width), order)
        indices = {
            'nwse': torch.tensor(nwse, dtype=torch.long, device=device),
            'senw': torch.tensor(list(reversed(nwse)), dtype=torch.long, device=device),
            'nesw': torch.tensor(nesw, dtype=torch.long, device=device),
            'swne': torch.tensor(list(reversed(nesw)), dtype=torch.long, device=device),
        }
        cached = {}
        for direction, index in indices.items():
            cached['idx_' + direction] = index
            cached['inv_' + direction] = inverse_permutation(index)
        cls._cache[key] = cached
        return cached


class AxialFourDirectionMamba(nn.Module):
    DIRECTIONS = ('lr', 'rl', 'tb', 'bt')

    def __init__(
        self,
        dim,
        share_mode='subband_shared_3',
        use_direction_embedding=False,
        allow_fallback=False,
        shared_mamba=None,
        shared_backend=None,
    ):
        super().__init__()
        self.dim = dim
        self.share_mode = share_mode
        self.use_direction_embedding = use_direction_embedding
        self.band_scan_axis = dict(W8M_CURRENT_SCAN_AXIS)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.proj_h = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.direction_embedding = None
        if use_direction_embedding:
            self.direction_embedding = nn.Parameter(torch.empty(4, 1, dim))
            nn.init.normal_(self.direction_embedding, std=0.02)

        def new_mamba():
            return _build_mamba_mixer(dim, allow_fallback)

        if share_mode == 'independent_8':
            self.mamba_lr, self.backend = new_mamba()
            self.mamba_rl, _ = new_mamba()
            self.mamba_tb, _ = new_mamba()
            self.mamba_bt, _ = new_mamba()
            names = {direction: 'mamba_' + direction for direction in self.DIRECTIONS}
        elif share_mode in ('pair_shared_4', 'subband_shared_3'):
            self.h_mamba, self.backend = new_mamba()
            self.v_mamba, _ = new_mamba()
            names = {'lr': 'h_mamba', 'rl': 'h_mamba',
                     'tb': 'v_mamba', 'bt': 'v_mamba'}
        elif share_mode == 'axial_diag_shared_2':
            self.axial_mamba, self.backend = new_mamba()
            names = {direction: 'axial_mamba' for direction in self.DIRECTIONS}
        elif share_mode == 'all_shared_1':
            if shared_mamba is None:
                raise ValueError('all_shared_1 requires a shared_mamba')
            self.shared_mamba = shared_mamba
            self.backend = shared_backend or 'shared'
            names = {direction: 'shared_mamba' for direction in self.DIRECTIONS}
        else:
            raise ValueError('Unsupported W8M share mode: {}'.format(share_mode))
        self._direction_to_module_name = names

    def get_mamba(self, direction):
        if direction not in self.DIRECTIONS:
            raise KeyError(direction)
        return getattr(self, self._direction_to_module_name[direction])

    def _add_direction_embedding(self, sequence, direction):
        if self.direction_embedding is None:
            return sequence
        index = self.DIRECTIONS.index(direction)
        return sequence + self.direction_embedding[index]

    def _run(self, sequence, direction, norm):
        sequence = self._add_direction_embedding(norm(sequence), direction)
        return self.get_mamba(direction)(sequence)

    def forward(self, horizontal, vertical, return_routes=False):
        batch, channels, height, width = horizontal.shape
        if vertical.shape != horizontal.shape:
            raise ValueError('Horizontal and vertical branch shapes must match')

        h_sequence = horizontal.permute(0, 2, 3, 1).reshape(
            batch * height, width, channels
        )
        route_lr = self._run(h_sequence, 'lr', self.norm_h)
        route_rl = torch.flip(
            self._run(torch.flip(h_sequence, dims=[1]), 'rl', self.norm_h),
            dims=[1],
        )
        h_output = self.proj_h(0.5 * (route_lr + route_rl))
        h_output = h_output.reshape(batch, height, width, channels)
        h_output = h_output.permute(0, 3, 1, 2).contiguous()

        v_sequence = vertical.permute(0, 3, 2, 1).reshape(
            batch * width, height, channels
        )
        route_tb = self._run(v_sequence, 'tb', self.norm_v)
        route_bt = torch.flip(
            self._run(torch.flip(v_sequence, dims=[1]), 'bt', self.norm_v),
            dims=[1],
        )
        v_output = self.proj_v(0.5 * (route_tb + route_bt))
        v_output = v_output.reshape(batch, width, height, channels)
        v_output = v_output.permute(0, 3, 2, 1).contiguous()

        outputs = (horizontal + h_output, vertical + v_output)
        if not return_routes:
            return outputs
        return outputs + ({
            'lr': route_lr,
            'rl': route_rl,
            'tb': route_tb,
            'bt': route_bt,
        },)


class DiagonalFourDirectionMamba(nn.Module):
    ALL_DIRECTIONS = ('nwse', 'senw', 'nesw', 'swne')

    def __init__(
        self,
        dim,
        share_mode='subband_shared_3',
        diag_directions=4,
        diag_order='snake',
        use_direction_embedding=False,
        allow_fallback=False,
        shared_mamba=None,
        shared_backend=None,
    ):
        super().__init__()
        if diag_directions not in (2, 4):
            raise ValueError('diag_directions must be 2 or 4')
        if diag_order not in ('concat', 'snake'):
            raise ValueError("diag_order must be 'concat' or 'snake'")
        self.dim = dim
        self.share_mode = share_mode
        self.diag_directions = diag_directions
        self.diag_order = diag_order
        self.directions = self.ALL_DIRECTIONS[:diag_directions]
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.direction_embedding = None
        if use_direction_embedding:
            self.direction_embedding = nn.Parameter(torch.empty(4, 1, dim))
            nn.init.normal_(self.direction_embedding, std=0.02)

        def new_mamba():
            return _build_mamba_mixer(dim, allow_fallback)

        if share_mode == 'independent_8':
            for direction in self.directions:
                module, backend = new_mamba()
                setattr(self, 'mamba_' + direction, module)
                self.backend = backend
            names = {direction: 'mamba_' + direction for direction in self.directions}
        elif share_mode == 'pair_shared_4':
            self.diag_backslash_mamba, self.backend = new_mamba()
            names = {'nwse': 'diag_backslash_mamba',
                     'senw': 'diag_backslash_mamba'}
            if diag_directions == 4:
                self.diag_slash_mamba, _ = new_mamba()
                names.update({'nesw': 'diag_slash_mamba',
                              'swne': 'diag_slash_mamba'})
        elif share_mode in ('subband_shared_3', 'axial_diag_shared_2'):
            self.diag_mamba, self.backend = new_mamba()
            names = {direction: 'diag_mamba' for direction in self.directions}
        elif share_mode == 'all_shared_1':
            if shared_mamba is None:
                raise ValueError('all_shared_1 requires a shared_mamba')
            self.shared_mamba = shared_mamba
            self.backend = shared_backend or 'shared'
            names = {direction: 'shared_mamba' for direction in self.directions}
        else:
            raise ValueError('Unsupported W8M share mode: {}'.format(share_mode))
        self._direction_to_module_name = names

    def get_mamba(self, direction):
        if direction not in self.directions:
            raise KeyError(direction)
        return getattr(self, self._direction_to_module_name[direction])

    def _add_direction_embedding(self, sequence, direction):
        if self.direction_embedding is None:
            return sequence
        index = self.ALL_DIRECTIONS.index(direction)
        return sequence + self.direction_embedding[index]

    def forward(self, x, return_routes=False):
        batch, channels, height, width = x.shape
        flat = x.flatten(2).transpose(1, 2)
        permutations = DiagonalIndexCache.build(
            height, width, order=self.diag_order, device=x.device
        )
        restored_routes = {}
        for direction in self.directions:
            sequence = flat.index_select(1, permutations['idx_' + direction])
            sequence = self._add_direction_embedding(self.norm(sequence), direction)
            output = self.get_mamba(direction)(sequence)
            restored_routes[direction] = output.index_select(
                1, permutations['inv_' + direction]
            )

        fused = torch.stack(
            [restored_routes[direction] for direction in self.directions], dim=0
        ).mean(dim=0)
        fused = self.proj(fused)
        fused = fused.transpose(1, 2).reshape(batch, channels, height, width)
        output = x + fused
        if return_routes:
            return output, restored_routes
        return output


class WaveletEightDirectionAWGM(nn.Module):
    """Wavelet-aligned axial and diagonal Mamba guidance (W8M-AWGM)."""

    def __init__(
        self,
        in_channels,
        share_mode='subband_shared_3',
        diag_directions=4,
        diag_order='snake',
        use_direction_embedding=False,
        fusion='spatial_softmax',
        allow_fallback=False,
    ):
        super().__init__()
        if fusion != 'spatial_softmax':
            raise ValueError("Only fusion='spatial_softmax' is currently supported")
        self.share_mode = share_mode
        self.diag_directions = diag_directions
        self.diag_order = diag_order
        self.use_direction_embedding = use_direction_embedding
        self.band_scan_axis = dict(W8M_CURRENT_SCAN_AXIS)
        self.pre_h = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_v = nn.Conv2d(in_channels, in_channels, 1)
        self.pre_d = nn.Conv2d(in_channels, in_channels, 1)

        shared_mamba = None
        shared_backend = None
        if share_mode == 'all_shared_1':
            shared_mamba, shared_backend = _build_mamba_mixer(
                in_channels, allow_fallback
            )
            self.shared_mamba = shared_mamba

        self.axial_branch = AxialFourDirectionMamba(
            in_channels,
            share_mode=share_mode,
            use_direction_embedding=use_direction_embedding,
            allow_fallback=allow_fallback,
            shared_mamba=shared_mamba,
            shared_backend=shared_backend,
        )
        self.diagonal_branch = DiagonalFourDirectionMamba(
            in_channels,
            share_mode=share_mode,
            diag_directions=diag_directions,
            diag_order=diag_order,
            use_direction_embedding=use_direction_embedding,
            allow_fallback=allow_fallback,
            shared_mamba=shared_mamba,
            shared_backend=shared_backend,
        )
        self.fusion = DirectionFusionGate(in_channels)
        self.mamba_backend = self.axial_branch.backend
        self.dcn_backend = 'not_requested'
        mixers = [
            self.axial_branch.get_mamba(direction)
            for direction in self.axial_branch.DIRECTIONS
        ]
        mixers.extend(
            self.diagonal_branch.get_mamba(direction)
            for direction in self.diagonal_branch.directions
        )
        self.mamba_instance_count = len({id(module) for module in mixers})
        self.last_direction_weights = None
        self.last_attention_map = None
        self.last_branch_norms = None

    def forward(self, A, H, V, D):
        # AxialFourDirectionMamba expects (horizontal_scan_input,
        # vertical_scan_input).  V=HL is the horizontal-structure band and
        # H=LH is the vertical-structure band for the Haar kernels above.
        FV, FH = self.axial_branch(
            self.pre_v(A + V),
            self.pre_h(A + H),
        )
        FD = self.diagonal_branch(self.pre_d(A + D))
        attention, direction_weights = self.fusion(A, FH, FV, FD)
        if not self.training:
            self.last_direction_weights = direction_weights.detach()
            self.last_attention_map = attention.detach()
            self.last_branch_norms = {
                'axial': 0.5 * (
                    FH.detach().float().square().mean().sqrt().item()
                    + FV.detach().float().square().mean().sqrt().item()
                ),
                'diagonal': FD.detach().float().square().mean().sqrt().item(),
            }
        return A * attention + A


def build_wave_guidance(name, in_channels, allow_fallback=False):
    if name == 'awgm_original':
        return WaveDownattention(in_channels)
    if name == 'dm_awgm_full':
        return DirectionMatchedAWGM(
            in_channels, use_mamba=True, use_dcn=True,
            fusion='spatial_softmax', allow_fallback=allow_fallback,
        )
    if name == 'dm_awgm_no_mamba':
        return DirectionMatchedAWGM(
            in_channels, use_mamba=False, use_dcn=True,
            fusion='spatial_softmax', allow_fallback=allow_fallback,
        )
    if name == 'dm_awgm_no_dcn':
        return DirectionMatchedAWGM(
            in_channels, use_mamba=True, use_dcn=False,
            fusion='spatial_softmax', allow_fallback=allow_fallback,
        )
    if name == 'dm_awgm_conv_only':
        return DirectionMatchedAWGM(
            in_channels, use_mamba=False, use_dcn=False,
            fusion='spatial_softmax', allow_fallback=allow_fallback,
        )
    if name in W8M_VARIANT_CONFIGS:
        settings = W8M_VARIANT_CONFIGS[name]
        return WaveletEightDirectionAWGM(
            in_channels,
            share_mode=settings['share_mode'],
            diag_directions=settings['diag_directions'],
            diag_order='snake',
            use_direction_embedding=settings['use_direction_embedding'],
            fusion='spatial_softmax',
            allow_fallback=allow_fallback,
        )
    raise ValueError(
        "Unknown wave guidance variant: {}. Expected one of {}".format(
            name, AWGM_VARIANTS
        )
    )


class DWTFreqNet(nn.Module):
    def __init__(
        self,
        config,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode='train',
        deepsuper=True,
        awgm_variant='awgm_original',
        awgm_allow_fallback=False,
    ):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.awgm_variant = awgm_variant
        in_channels = config.base_channel  # basic channel 64
        block = Res_block
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_wavelet_inchannel_global = self._make_layer(block, 32*3, in_channels)
        self.conv_wavelet_inchannel_local = self._make_layer(block, 32, in_channels)
        self.inc = self._make_layer(block, n_channels, 32)
        self.global_encoder1_1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.global_encoder2_1 = self._make_layer(block, in_channels * 6, in_channels * 4, 1)  # 64  128
        self.global_encoder3_1 = self._make_layer(block, in_channels * 12, in_channels * 8, 1)  # 64  128
        self.global_encoder4_1 = self._make_layer(block, in_channels * 24, in_channels * 8, 1)  # 64  128
        self.global_encoder1_2 = self._make_layer(block, in_channels * 2 + in_channels * 4, in_channels * 2,1 )
        self.global_encoder2_2 = self._make_layer(block, in_channels * 4 + in_channels * 6 + in_channels * 8, in_channels * 4, 1)
        self.global_encoder3_2 = self._make_layer(block, in_channels * 8 + in_channels * 12 + in_channels * 8, in_channels * 8, 1)
        self.global_encoder1_3 = self._make_layer(block, in_channels * 2 * 2 + in_channels * 4, in_channels * 2, 1)
        self.global_encoder2_3 = self._make_layer(block, in_channels * 4 * 2 + in_channels * 6 + in_channels * 8, in_channels * 4, 1)
        self.global_encoder1_4 = self._make_layer(block, in_channels * 2 * 3 + in_channels * 4, in_channels * 2, 1)


        ##局部低频
        self.local_encoder1_1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.local_encoder2_1 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.local_encoder3_1 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.local_encoder4_1 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128
        self.local_encoder1_2 = self._make_layer(block, in_channels * 2 + in_channels * 4, in_channels * 2)
        self.local_encoder2_2 = self._make_layer(block, in_channels * 4 + in_channels * 2 + in_channels * 8, in_channels * 4, 1)
        self.local_encoder3_2 = self._make_layer(block, in_channels * 8 + in_channels * 4 + in_channels * 8, in_channels * 8, 1)
        self.local_encoder1_3 = self._make_layer(block, in_channels * 2 * 2 + in_channels * 4, in_channels * 2)
        self.local_encoder2_3 = self._make_layer(block, in_channels * 4 * 2 + in_channels * 2 + in_channels * 8, in_channels * 4, 1)
        self.local_encoder1_4 = self._make_layer(block, in_channels * 2 * 3 + in_channels * 4, in_channels * 2)
        ##局部低频

        ##Dense全局结构改变通道
        self.global_channel1_2 = nn.Conv2d(in_channels * 4, in_channels * 4 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel2_2 = nn.Conv2d(in_channels * 8, in_channels * 8 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel3_2 = nn.Conv2d(in_channels * 8, in_channels * 8 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel1_3 = nn.Conv2d(in_channels * 4, in_channels * 4 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel2_3 = nn.Conv2d(in_channels * 8, in_channels * 8 * 3, kernel_size=(1, 1), stride=(1, 1))
        self.global_channel1_4 = nn.Conv2d(in_channels * 4, in_channels * 4 * 3, kernel_size=(1, 1), stride=(1, 1))


        self.from_input2out = self._make_layer(block, in_channels, in_channels // 2, 1)
        self.outc_global = nn.Conv2d(in_channels // 2, 1, kernel_size=(1, 1), stride=(1, 1))

        ##小波解码器通道对齐##

        self.decoder4_channel = self._make_layer(block, in_channels * 8, in_channels * 24, 1)
        self.decoder3_channel = self._make_layer(block, in_channels * 8, in_channels * 12, 1)
        self.decoder2_channel = self._make_layer(block, in_channels * 4, in_channels * 6, 1)
        self.decoder1_channel = self._make_layer(block, in_channels * 2, in_channels * 3, 1)

        self.decoder3_channel_local = self._make_layer(block, in_channels * 8, in_channels * 4, 1)
        self.decoder2_channel_local = self._make_layer(block, in_channels * 4, in_channels * 2, 1)
        self.decoder1_channel_local = self._make_layer(block, in_channels * 2, in_channels * 1, 1)

        ##局部低频
        ##局部低频

        ##小波相关##
        self.har = HaarWaveletTransform()
        self.inversehar = InverseHaarWaveletTransform()

        ##构造我需要的图像尺寸进入Transformer相关##
        self.wavel_channel_down_x1_global_output_1_4 = nn.Conv2d(in_channels * 2, 1, kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_x_inut = nn.Conv2d(32, 1,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_x2_global_output_2_3 = nn.Conv2d(in_channels * 4, 1,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_x3_global_output_3_2 = nn.Conv2d(in_channels * 8, 1,kernel_size=(1, 1), stride=(1, 1))

        self.stand_cahnnel1 = conv1x1_bn_relu(in_channels * 2, 128)
        self.stand_cahnnel2 = conv1x1_bn_relu(in_channels * 4, 128)
        self.stand_cahnnel3 = conv1x1_bn_relu(in_channels * 8, 128)
        self.stand_cahnnel_input = conv1x1_bn_relu(32, 128)

        self.TransTo_input = TransFuseModel(num_blocks=1, x_channels=128, nx=4096, y_channels=128, ny=5376)
        self.TransTo3e = TransFuseModel(num_blocks=1, x_channels=128, nx=256, y_channels=128, ny=9216)
        self.TransTo2e = TransFuseModel(num_blocks=1, x_channels=128, nx=1024, y_channels=128, ny=8448)
        self.TransTo1e = TransFuseModel(num_blocks=1, x_channels=128, nx=4096, y_channels=128, ny=5376)

        self.wavel_channel_down_to_origin_x_inut = nn.Conv2d(128, 32,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_to_origin_x1_global_output_1_4 = nn.Conv2d(128, in_channels * 2,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_to_origin_x2_global_output_2_3 = nn.Conv2d(128, in_channels * 4,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_down_to_origin_x3_global_output_3_2 = nn.Conv2d(128, in_channels * 8,kernel_size=(1, 1), stride=(1, 1))

        self.wavel_channel_up_x_inut = nn.Conv2d(1, 32,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_up_x1_global_output_1_4 = nn.Conv2d(1, in_channels * 2,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_up_x2_global_output_2_3 = nn.Conv2d(1, in_channels * 4,kernel_size=(1, 1), stride=(1, 1))
        self.wavel_channel_up_x3_global_output_3_2 = nn.Conv2d(1, in_channels * 8,kernel_size=(1, 1), stride=(1, 1))

        ##注意力相关##
        self.wave_att_input_t = build_wave_guidance(
            awgm_variant, 32, awgm_allow_fallback
        )
        self.wave_att_f1 = build_wave_guidance(
            awgm_variant, in_channels * 2, awgm_allow_fallback
        )
        self.wave_att_f2 = build_wave_guidance(
            awgm_variant, in_channels * 4, awgm_allow_fallback
        )
        self.wave_att_f3 = build_wave_guidance(
            awgm_variant, in_channels * 8, awgm_allow_fallback
        )
        if isinstance(
            self.wave_att_input_t,
            (DirectionMatchedAWGM, WaveletEightDirectionAWGM),
        ):
            self.awgm_backends = {
                "mamba": self.wave_att_input_t.mamba_backend,
                "dcn": self.wave_att_input_t.dcn_backend,
                "haar_band_scan_axis": self.wave_att_input_t.band_scan_axis,
            }
            if isinstance(self.wave_att_input_t, WaveletEightDirectionAWGM):
                self.awgm_backends.update({
                    "share_mode": self.wave_att_input_t.share_mode,
                    "diagonal_order": self.wave_att_input_t.diag_order,
                    "diagonal_directions": self.wave_att_input_t.diag_directions,
                    "direction_embedding": (
                        self.wave_att_input_t.use_direction_embedding
                    ),
                    "mamba_instances_per_awgm": (
                        self.wave_att_input_t.mamba_instance_count
                    ),
                    "haar_routing_aligned": (
                        check_haar_direction_correspondence()["routing_aligned"]
                    ),
                })
        else:
            self.awgm_backends = {
                "mamba": "not_requested",
                "dcn": "not_requested",
            }
        print(
            "AWGM variant: {} | Mamba: {} | DCN: {} | metadata: {}".format(
                self.awgm_variant,
                self.awgm_backends["mamba"],
                self.awgm_backends["dcn"],
                {
                    key: value for key, value in self.awgm_backends.items()
                    if key not in ("mamba", "dcn")
                },
            )
        )

        ###输出相关的卷积
        self.out4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.out3 = self._make_layer(block, in_channels * 4, in_channels * 4, 1)
        self.out2 = self._make_layer(block, in_channels * 2, in_channels * 2, 1)
        self.out1 = self._make_layer(block, in_channels, in_channels, 1)


        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        ##
        x_inut = self.inc(x)  # 32 256 256
        A1_1,H1_1,V1_1,D1_1 = self.har(x_inut)   ##H/2

    ##第一层重构网络
        concat_hvd_global_1_1 = torch.cat((H1_1, V1_1, D1_1), dim=1)
        x1_global = self.conv_wavelet_inchannel_global(concat_hvd_global_1_1)  ## 32  H/2
        x1_global_output_1_1 = self.global_encoder1_1(x1_global)  # 64 128 128

        x1_local = self.conv_wavelet_inchannel_local(A1_1)
        x1_local_output_1_1 = self.local_encoder1_1(x1_local)  # 64 128 128

    ##第二层重构网络
        A2_1, H2_1, V2_1, D2_1 = self.har(x1_local_output_1_1)
        x2_global_2_1 = torch.cat((H2_1, V2_1, D2_1), dim=1)
        x2_global_output_2_1 = self.global_encoder2_1(x2_global_2_1)  # 128 64  64

        x2_local_output_2_1 = self.local_encoder2_1(A2_1)  # 128 64  64

    ##构建1_2
        x1_global_output_1_2 = self.global_encoder1_2(torch.cat([x1_global_output_1_1,self.up(x2_global_output_2_1)],1))
        x1_global_input_1_2_f_2_1 = self.global_channel1_2(x2_global_output_2_1)
        H_1_2_f_2_1, V_1_2_f_2_1, D_1_2_f_2_1 = torch.chunk(x1_global_input_1_2_f_2_1, 3, 1)
        x2_inversewavel_2_1 = self.inversehar(x2_local_output_2_1, H_1_2_f_2_1, V_1_2_f_2_1, D_1_2_f_2_1)
        x1_local_output_1_2 = self.local_encoder1_2(torch.cat([x1_local_output_1_1, x2_inversewavel_2_1],1))

    ##第三层重构网络
        A3_1, H3_1, V3_1, D3_1 = self.har(x2_local_output_2_1)
        x3_global_3_1 = torch.cat((H3_1, V3_1, D3_1), dim=1)
        x3_global_output_3_1 = self.global_encoder3_1(x3_global_3_1)  # 256 32  32

        x3_local_output_3_1 = self.local_encoder3_1(A3_1)  # 256 32  32

    ##构建2_2 来自2_1 1_2 3_1##调整通道了 不是缩减为1了
        A2_2_onec, H2_2onec, V2_2onec, D2_2onec = self.har(x1_local_output_1_2)
        x2_global_2_2_threec = torch.cat((H2_2onec, V2_2onec, D2_2onec), dim=1)
        x2_global_output_2_2 = self.global_encoder2_2(torch.cat([x2_global_output_2_1, x2_global_2_2_threec, self.up(x3_global_output_3_1)],1))

        x1_global_input_2_2_f_3_1 = self.global_channel2_2(x3_global_output_3_1)
        H_2_2_f_3_1, V_2_2_f_3_1, D_2_2_f_3_1 = torch.chunk(x1_global_input_2_2_f_3_1, 3, 1)
        x3_inversewavel_3_1 = self.inversehar(x3_local_output_3_1, H_2_2_f_3_1, V_2_2_f_3_1, D_2_2_f_3_1)
        x2_local_output_2_2 = self.local_encoder2_2(torch.cat([x2_local_output_2_1, A2_2_onec, x3_inversewavel_3_1], 1))

    ##第四层重构网络
        A4_1, H4_1, V4_1, D4_1 = self.har(x3_local_output_3_1)
        x4_global_4_1 = torch.cat((H4_1, V4_1, D4_1), dim=1)
        x4_global_output_4_1 = self.global_encoder4_1(x4_global_4_1)  # 256 16  16

        x4_local_output_4_1 = self.local_encoder4_1(A4_1)  # 256 16  16

    ##构建3_2 来自 3_1 2_2 4_1##调整了通道
        A3_2_onec, H3_2onec, V3_2onec, D3_2onec = self.har(x2_local_output_2_2)
        x2_global_3_2_threec = torch.cat((H3_2onec, V3_2onec, D3_2onec), dim=1)
        x3_global_output_3_2 = self.global_encoder3_2(torch.cat([x3_global_output_3_1, x2_global_3_2_threec, self.up(x4_global_output_4_1)],1))

        x1_global_input_3_2_f_4_1 = self.global_channel3_2(x4_global_output_4_1)
        H_3_2_f_4_1, V_3_2_f_4_1, D_3_2_f_4_1 = torch.chunk(x1_global_input_3_2_f_4_1, 3, 1)
        x4_inversewavel_4_1 = self.inversehar(x4_local_output_4_1, H_3_2_f_4_1, V_3_2_f_4_1, D_3_2_f_4_1)
        x3_local_output_3_2 = self.local_encoder3_2(torch.cat([x3_local_output_3_1, A3_2_onec, x4_inversewavel_4_1],1))

    ##构建1_3 来自 1_1 1_2 2_2
        x1_global_output_1_3 = self.global_encoder1_3(torch.cat([x1_global_output_1_1, x1_global_output_1_2,self.up(x2_global_output_2_2)],1))
        x1_global_input_1_3_f_2_2 = self.global_channel1_3(x2_global_output_2_2)
        H_1_3_f_2_2, V_1_3_f_2_2, D_1_3_f_2_2 = torch.chunk(x1_global_input_1_3_f_2_2, 3, 1)
        x2_inversewavel_2_2 = self.inversehar(x2_local_output_2_2, H_1_3_f_2_2, V_1_3_f_2_2, D_1_3_f_2_2)
        x1_local_output_1_3 = self.local_encoder1_3(torch.cat([x1_local_output_1_1, x1_local_output_1_2, x2_inversewavel_2_2],1))

    ##构建2_3 来自2_1 2_2 1_3 3_2##调整了通道
        A2_3_onec, H2_3onec, V2_3onec, D2_3onec = self.har(x1_local_output_1_3)
        x2_global_2_3_threec = torch.cat((H2_3onec, V2_3onec, D2_3onec), dim=1)
        x2_global_output_2_3 = self.global_encoder2_3(torch.cat([x2_global_output_2_1, x2_global_output_2_2, x2_global_2_3_threec, self.up(x3_global_output_3_2)],1))

        x1_global_input_2_3_f_3_2 = self.global_channel2_3(x3_global_output_3_2)
        H_2_3_f_3_2, V_2_3_f_3_2, D_2_3_f_3_2 = torch.chunk(x1_global_input_2_3_f_3_2, 3, 1)
        x3_inversewavel_3_2 = self.inversehar(x3_local_output_3_2, H_2_3_f_3_2, V_2_3_f_3_2, D_2_3_f_3_2)
        x2_local_output_2_3 = self.local_encoder2_3(torch.cat([x2_local_output_2_1, x2_local_output_2_2, A2_3_onec, x3_inversewavel_3_2],1))

    ##构建1_4 来自 1_1 1_2 1_3 2_3
        x1_global_output_1_4 = self.global_encoder1_4(torch.cat([x1_global_output_1_1, x1_global_output_1_2, x1_global_output_1_3, self.up(x2_global_output_2_3)],1))
        x1_global_input_1_4_f_2_3 = self.global_channel1_4(x2_global_output_2_3)
        H_1_4_f_2_3, V_1_4_f_2_3, D_1_4_f_2_3 = torch.chunk(x1_global_input_1_4_f_2_3, 3, 1)
        x2_inversewavel_2_3 = self.inversehar(x2_local_output_2_3, H_1_4_f_2_3, V_1_4_f_2_3, D_1_4_f_2_3)
        x1_local_output_1_4 = self.local_encoder1_4(torch.cat([x1_local_output_1_1, x1_local_output_1_2, x1_local_output_1_3,x2_inversewavel_2_3],1))


        f_input = x_inut
        f1 = x1_global_output_1_4
        f2 = x2_global_output_2_3
        f3 = x3_global_output_3_2
        #  CCT

        finput_A,finput_H,finput_V,finput_D = self.har(x_inut)
        finput_AA, finput_HH, finput_VV, finput_DD = self.har(finput_A)
        finput_att = self.wave_att_input_t(finput_AA, finput_HH, finput_VV, finput_DD)
        finput_HHVVDD = self.stand_cahnnel_input(finput_att).flatten(2).permute(0, 2, 1) #64 64=4096

                #f_1
        f1_A,f1_H,f1_V,f1_D = self.har(x1_global_output_1_4)
        f1_att = self.wave_att_f1(f1_A,f1_H,f1_V,f1_D)  ##这个得出的注意力机制，实际上是对A这个低频分量的注意力机制
        f1_HVD = self.stand_cahnnel1(f1_att).flatten(2).permute(0, 2, 1)#64 64=4096
                #f_2
        f2_A,f2_H,f2_V,f2_D = self.har(x2_global_output_2_3)
        f2_att = self.wave_att_f2(f2_A,f2_H,f2_V,f2_D)
        f2_HVD = self.stand_cahnnel2(f2_att).flatten(2).permute(0, 2, 1) #32 32=1024
                #f_3
        f3_A,f3_H,f3_V,f3_D = self.har(x3_global_output_3_2)
        f3_att = self.wave_att_f3(f3_A,f3_H,f3_V,f3_D)
        f3_HVD = self.stand_cahnnel3(f3_att).flatten(2).permute(0, 2, 1)#16 16=256

        ###得到进入trans之前的图像尺寸
        binput, cinput, hinput, winput = finput_att.shape
        b1, c1, h1, w1 = f1_att.shape
        b2, c2, h2, w2 = f2_att.shape
        b3, c3, h3, w3 = f3_att.shape


        ##构造我需要的图像尺寸进入Transformer##

        ##trans的处理##
        f3_HVDe = self.TransTo3e(f3_HVD, torch.cat((finput_HHVVDD, f1_HVD, f2_HVD), dim=1)) #256 9216
        f2_HVDe = self.TransTo2e(f2_HVD, torch.cat((finput_HHVVDD, f1_HVD, f3_HVDe), dim=1))#1024 8448
        f1_HVDe = self.TransTo1e(f1_HVD, torch.cat((finput_HHVVDD, f2_HVDe, f3_HVDe), dim=1))  ##里面加个代码，最后返回的得是图像而不是序列  4096  5376
        finput_HHVVDDe = self.TransTo_input(finput_HHVVDD, torch.cat((f1_HVDe, f2_HVDe, f3_HVDe), dim=1))  ##4096  5376
        ##trans的处理##

        ##从序列变为之前的图像形状##

        f3_HVDe = rearrange(f3_HVDe, 'b (h w) c -> b c h w', h=h3, w=w3)

        f2_HVDe = rearrange(f2_HVDe, 'b (h w) c -> b c h w', h=h2, w=w2)

        f1_HVDe = rearrange(f1_HVDe, 'b (h w) c -> b c h w', h=h1, w=w1)

        finput_HHVVDDe = rearrange(finput_HHVVDDe, 'b (h w) c -> b c h w', h=hinput, w=winput)
        ##从序列变为之前的图像形状##

        ##在给返回回去##
                ##f_input
        finput_HHVVDDe = self.wavel_channel_down_to_origin_x_inut(finput_HHVVDDe)
        finput_A = self.inversehar(finput_HHVVDDe, finput_HH, finput_VV, finput_DD)
        x_inut = self.inversehar(finput_A, finput_H, finput_V, finput_D)


                ##f_1
        f1_HVDe = self.wavel_channel_down_to_origin_x1_global_output_1_4(f1_HVDe)
        x1_global_output_1_4 = self.inversehar(f1_HVDe, f1_H, f1_V, f1_D)

                ##f_2
        f2_HVDe = self.wavel_channel_down_to_origin_x2_global_output_2_3(f2_HVDe)
        x2_global_output_2_3 = self.inversehar(f2_HVDe, f2_H, f2_V, f2_D)
                ##f_3
        f3_HVDe = self.wavel_channel_down_to_origin_x3_global_output_3_2(f3_HVDe)
        x3_global_output_3_2 = self.inversehar(f3_HVDe, f3_H, f3_V, f3_D)

        x_inut = x_inut
        x1_global_output_1_4 = x1_global_output_1_4 + f1
        x2_global_output_2_3 = x2_global_output_2_3 + f2
        x3_global_output_3_2 = x3_global_output_3_2 + f3
        ##第四层的小波上采样
        x4_global_output_de = self.decoder4_channel(x4_global_output_4_1)
        split_tensors_4 = torch.chunk(x4_global_output_de, chunks=3, dim=1)
        H4_de, V4_de, D4_de = split_tensors_4
        x4_out = self.out4(x4_local_output_4_1 + H4_de + V4_de + D4_de)

        x3_local_input_de = self.inversehar(x4_local_output_4_1, H4_de, V4_de, D4_de)

        ##第三层的小波上采样
        x3_global_output_de = self.decoder3_channel(x3_global_output_3_2)
        split_tensors_3 = torch.chunk(x3_global_output_de, chunks=3, dim=1)
        H3_de, V3_de, D3_de = split_tensors_3
        x3_local_output_3_2_de = self.decoder3_channel_local(x3_local_output_3_2+x3_local_input_de)
        x3_out = self.out3(x3_local_output_3_2_de+H3_de+V3_de+D3_de)

        x2_local_input_de = self.inversehar(x3_local_output_3_2_de, H3_de, V3_de, D3_de)

        ##第二层的小波上采样
        x2_global_output_de = self.decoder2_channel(x2_global_output_2_3)
        split_tensors_2 = torch.chunk(x2_global_output_de, chunks=3, dim=1)
        H2_de, V2_de, D2_de = split_tensors_2
        x2_local_output_2_3_de = self.decoder2_channel_local(x2_local_output_2_3+x2_local_input_de)
        x2_out = self.out2(x2_local_output_2_3_de + H2_de + V2_de + D2_de)

        x1_local_input_de = self.inversehar(x2_local_output_2_3_de, H2_de, V2_de, D2_de)

        ##第一层的小波上采样
        x1_global_output_de = self.decoder1_channel(x1_global_output_1_4)
        split_tensors_1 = torch.chunk(x1_global_output_de, chunks=3, dim=1)
        H1_de, V1_de, D1_de = split_tensors_1
        x1_local_output_1_4_de = self.decoder1_channel_local(x1_local_output_1_4+x1_local_input_de)
        x1_out = self.out1(x1_local_output_1_4_de + H1_de + V1_de + D1_de)

        x1_local_final_raw_de = self.inversehar(x1_local_output_1_4_de, H1_de, V1_de, D1_de)
        out = self.outc_global(self.from_input2out(x1_local_final_raw_de + x_inut))

        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(x4_out)
            gt_4 = self.gt_conv4(x3_out)
            gt_3 = self.gt_conv3(x2_out)
            gt_2 = self.gt_conv2(x1_out)
            # 原始深监督
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
            else:
                return torch.sigmoid(out)
        else:
            print("不进入这里")
            return torch.sigmoid(out)


if __name__ == '__main__':
    config_vit = get_DWTFreqNet_config()
    model = DWTFreqNet(config_vit, mode='test', deepsuper=True)
    model = model
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    flops, params = profile(model, (inputs,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
