"""Decoder-side low-frequency guided high-frequency purification for Experiment H.

The mechanism follows the official NS-FPN ``wav_Enhance`` implementation at
commit b857bef068ba48f1258b62de6bf082f73dbafde4, while reusing the Haar bands
already produced by DWTFreqNet instead of introducing a second DWT/IDWT pair.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NS_FPN_SOURCE_COMMIT = "b857bef068ba48f1258b62de6bf082f73dbafde4"
NS_FPN_SOURCE_FILE = "model/NS_FPN.py"
NS_FPN_SOURCE_URL = "https://github.com/mengduann/NS-FPN"


class LFPSpatialAttention(nn.Module):
    """Channel mean/max pooling followed by the fixed 7x7 spatial attention."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size != 7:
            raise ValueError("Experiment H fixes LFP kernel_size=7")
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, low_feature):
        avg_map = torch.mean(low_feature, dim=1, keepdim=True)
        max_map = torch.amax(low_feature, dim=1, keepdim=True)
        attention = self.sigmoid(
            self.conv(torch.cat([avg_map, max_map], dim=1))
        )
        return attention, {
            "avg_map": avg_map,
            "max_map": max_map,
            "attention": attention,
        }


class LearnableDepthwiseGaussian(nn.Module):
    """One positive learnable sigma shared by every H/V/D channel in a stage."""

    def __init__(self, channels, kernel_size=3, sigma_init=1.0):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("Experiment H fixes Gaussian kernel_size=3")
        if sigma_init <= 1e-3:
            raise ValueError("sigma_init must be greater than 1e-3")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        rho_init = math.log(math.expm1(float(sigma_init) - 1e-3))
        self.rho = nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))

    @property
    def sigma(self):
        return F.softplus(self.rho) + 1e-3

    def kernel(self, device=None, dtype=None):
        sigma = self.sigma
        device = sigma.device if device is None else device
        dtype = sigma.dtype if dtype is None else dtype
        coords = torch.arange(
            self.kernel_size,
            device=device,
            dtype=torch.float32,
        ) - self.padding
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma.float().square()))
        kernel = kernel / kernel.sum()
        return kernel.to(dtype=dtype).reshape(1, 1, self.kernel_size, self.kernel_size)

    def reset_control_parameters(self):
        rho_init = math.log(math.expm1(1.0 - 1e-3))
        with torch.no_grad():
            self.rho.fill_(rho_init)

    def forward(self, tensor):
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise RuntimeError(
                f"Gaussian expected N,{self.channels},H,W, got {tuple(tensor.shape)}"
            )
        kernel = self.kernel(device=tensor.device, dtype=tensor.dtype)
        weight = kernel.expand(self.channels, 1, -1, -1).contiguous()
        padded = F.pad(
            tensor,
            (self.padding, self.padding, self.padding, self.padding),
            mode="replicate",
        )
        return F.conv2d(padded, weight, groups=self.channels)


class AdaptiveLFPThreshold(nn.Module):
    """Per-sample/per-channel magnitude threshold with an initial ratio of 0.5."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 8)
        self.channels = int(channels)
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.reset_control_parameters()

    def reset_control_parameters(self):
        final = self.predictor[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, modulated_high):
        magnitude = modulated_high.abs()
        mean_magnitude = F.adaptive_avg_pool2d(magnitude, output_size=1)
        threshold_ratio = torch.sigmoid(self.predictor(mean_magnitude))
        threshold = mean_magnitude * threshold_ratio
        return threshold, {
            "mean_magnitude": mean_magnitude,
            "threshold_ratio": threshold_ratio,
            "threshold": threshold,
        }


class DecoderLFPProcessor(nn.Module):
    """Apply spatial attention and optional Gaussian purification to aligned H/V/D."""

    def __init__(
        self,
        high_channels,
        use_gaussian,
        threshold_mode,
        fixed_tau=0.5,
    ):
        super().__init__()
        if threshold_mode not in ("none", "fixed_hard", "adaptive_soft"):
            raise ValueError(f"Unknown Experiment H threshold mode: {threshold_mode}")
        if bool(use_gaussian) != (threshold_mode != "none"):
            raise ValueError("Gaussian and threshold mode configuration disagree")
        self.high_channels = int(high_channels)
        self.attention = LFPSpatialAttention(kernel_size=7)
        self.use_gaussian = bool(use_gaussian)
        self.threshold_mode = threshold_mode
        self.register_buffer("fixed_tau", torch.tensor(float(fixed_tau)))
        if self.use_gaussian:
            self.gaussian = LearnableDepthwiseGaussian(
                channels=3 * high_channels,
                kernel_size=3,
                sigma_init=1.0,
            )
        if self.threshold_mode == "adaptive_soft":
            self.threshold_predictor = AdaptiveLFPThreshold(
                channels=3 * high_channels
            )

    def reset_control_parameters(self):
        if self.use_gaussian:
            self.gaussian.reset_control_parameters()
        if self.threshold_mode == "adaptive_soft":
            self.threshold_predictor.reset_control_parameters()

    def forward(self, low_source, aligned_h, aligned_v, aligned_d):
        if not (aligned_h.shape == aligned_v.shape == aligned_d.shape):
            raise RuntimeError("LFP requires equal aligned H/V/D shapes")
        if aligned_h.shape[1] != self.high_channels:
            raise RuntimeError(
                f"LFP expected {self.high_channels} high channels, got {aligned_h.shape[1]}"
            )
        if low_source.shape[0] != aligned_h.shape[0] or low_source.shape[-2:] != aligned_h.shape[-2:]:
            raise RuntimeError(
                "LFP low/high shape mismatch: "
                f"low={tuple(low_source.shape)}, high={tuple(aligned_h.shape)}"
            )

        attention, attention_debug = self.attention(low_source)
        high = torch.cat([aligned_h, aligned_v, aligned_d], dim=1)
        modulated = high * attention
        threshold_debug = None

        if not self.use_gaussian:
            purified = modulated
            gaussian = None
            threshold = None
            mask = None
        elif self.threshold_mode == "fixed_hard":
            gaussian = self.gaussian(modulated)
            threshold = self.fixed_tau.to(
                device=modulated.device,
                dtype=modulated.dtype,
            )
            mask = (modulated.abs() < threshold).to(modulated.dtype)
            purified = modulated * (1.0 - mask) + gaussian * mask
        elif self.threshold_mode == "adaptive_soft":
            gaussian = self.gaussian(modulated)
            threshold, threshold_debug = self.threshold_predictor(modulated)
            temperature = 0.1 * threshold + 1e-6
            mask = torch.sigmoid((threshold - modulated.abs()) / temperature)
            purified = modulated * (1.0 - mask) + gaussian * mask
        else:
            raise RuntimeError(f"Invalid LFP threshold mode: {self.threshold_mode}")

        purified_h, purified_v, purified_d = purified.chunk(3, dim=1)
        return purified_h, purified_v, purified_d, {
            "low_source": low_source,
            "attention": attention,
            "attention_debug": attention_debug,
            "aligned_high": high,
            "modulated_high": modulated,
            "gaussian_high": gaussian,
            "threshold": threshold,
            "threshold_debug": threshold_debug,
            "mask": mask,
            "purified_high": purified,
        }


__all__ = [
    "AdaptiveLFPThreshold",
    "DecoderLFPProcessor",
    "LearnableDepthwiseGaussian",
    "LFPSpatialAttention",
    "NS_FPN_SOURCE_COMMIT",
    "NS_FPN_SOURCE_FILE",
    "NS_FPN_SOURCE_URL",
]
