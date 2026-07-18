"""Decoder-side directional sparse high-frequency restoration for Experiment G.

The multi-scale extractor and adaptive sparse support gate are copied from the
Experiment F2 implementation at commit
3034408051d3742d80473650fe9d198fc37e48ab.  Experiment G deliberately keeps
the encoder untouched and applies these blocks only to aligned decoder bands.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_F2_CORE_COMMIT = "3034408051d3742d80473650fe9d198fc37e48ab"


class DirectionalMultiScaleExtractor(nn.Module):
    """Two independent directional depthwise branches followed by 1x1 fusion."""

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
            raise ValueError(f"Unknown DSHF direction: {direction}")

        self.channels = int(channels)
        self.direction = direction
        self.branch1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel1,
            padding=padding1,
            dilation=dilation1,
            groups=channels,
            bias=False,
        )
        self.branch2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel2,
            padding=padding2,
            dilation=dilation2,
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, tensor):
        return self.fuse(
            torch.cat([self.branch1(tensor), self.branch2(tensor)], dim=1)
        )


class AdaptiveSparseSupportGate(nn.Module):
    """Soft magnitude support with a learned per-channel adaptive threshold."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.threshold_predictor = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.reset_control_parameters()

    def reset_control_parameters(self):
        last_conv = self.threshold_predictor[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, feature):
        magnitude = feature.abs()
        mean_magnitude = F.adaptive_avg_pool2d(magnitude, output_size=1)
        threshold_ratio = torch.sigmoid(self.threshold_predictor(mean_magnitude))
        threshold = mean_magnitude * threshold_ratio
        support = torch.sigmoid((magnitude - threshold) / (threshold + 1e-6))
        output = feature * support
        return output, {
            "mean_magnitude": mean_magnitude,
            "threshold_ratio": threshold_ratio,
            "threshold": threshold,
            "support": support,
        }


class DecoderDSHFCore(nn.Module):
    """F2 multi-scale plus sparse core, independently applied to H/V/D."""

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        self.extract_h = DirectionalMultiScaleExtractor(channels, "H")
        self.extract_v = DirectionalMultiScaleExtractor(channels, "V")
        self.extract_d = DirectionalMultiScaleExtractor(channels, "D")
        self.sparse_h = AdaptiveSparseSupportGate(channels)
        self.sparse_v = AdaptiveSparseSupportGate(channels)
        self.sparse_d = AdaptiveSparseSupportGate(channels)

    def reset_control_parameters(self):
        self.sparse_h.reset_control_parameters()
        self.sparse_v.reset_control_parameters()
        self.sparse_d.reset_control_parameters()

    def forward(self, band_h, band_v, band_d):
        multiscale_h = self.extract_h(band_h)
        multiscale_v = self.extract_v(band_v)
        multiscale_d = self.extract_d(band_d)
        residual_h, sparse_h = self.sparse_h(multiscale_h)
        residual_v, sparse_v = self.sparse_v(multiscale_v)
        residual_d, sparse_d = self.sparse_d(multiscale_d)
        return residual_h, residual_v, residual_d, {
            "multiscale_h": multiscale_h,
            "multiscale_v": multiscale_v,
            "multiscale_d": multiscale_d,
            "sparse_h": sparse_h,
            "sparse_v": sparse_v,
            "sparse_d": sparse_d,
            "residual_h": residual_h,
            "residual_v": residual_v,
            "residual_d": residual_d,
        }


class DecoderSemanticDirectionGate(nn.Module):
    """Bounded H/V/D scales conditioned on decoder semantics and band energy."""

    def __init__(self, channels, use_targetness=False):
        super().__init__()
        hidden = min(max(channels // 8, 8), 32)
        self.use_targetness = bool(use_targetness)
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        gate_inputs = hidden + 4 + int(self.use_targetness)
        self.gate = nn.Sequential(
            nn.Conv2d(gate_inputs, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=3, padding=1, bias=True),
        )
        self.reset_control_parameters()

    def reset_control_parameters(self):
        final = self.gate[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def normalize_energy(energy):
        normalized = energy / (energy.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return normalized / (1.0 + normalized)

    def forward(self, low_semantic, residual_h, residual_v, residual_d, targetness=None):
        if self.use_targetness != (targetness is not None):
            expected = "requires" if self.use_targetness else "must not receive"
            raise RuntimeError(f"This semantic direction gate {expected} targetness")
        semantic = self.semantic_proj(low_semantic)
        energy_h = self.normalize_energy(residual_h.abs().mean(dim=1, keepdim=True))
        energy_v = self.normalize_energy(residual_v.abs().mean(dim=1, keepdim=True))
        energy_d = self.normalize_energy(residual_d.abs().mean(dim=1, keepdim=True))
        joint_energy = torch.sqrt(
            energy_h.square() + energy_v.square() + energy_d.square() + 1e-6
        )
        gate_inputs = [semantic, energy_h, energy_v, energy_d, joint_energy]
        if targetness is not None:
            if targetness.shape[-2:] != low_semantic.shape[-2:]:
                targetness = F.interpolate(
                    targetness,
                    size=low_semantic.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate_inputs.append(targetness)
        scales = 2.0 * torch.sigmoid(self.gate(torch.cat(gate_inputs, dim=1)))
        return scales.chunk(3, dim=1), {
            "semantic": semantic,
            "energy_h": energy_h,
            "energy_v": energy_v,
            "energy_d": energy_d,
            "joint_energy": joint_energy,
            "scales": scales,
            "targetness": targetness,
        }


class DecoderHFRefiner(nn.Module):
    """Bounded residual restoration of already aligned decoder coefficients."""

    def __init__(self, channels, use_semantic_gate=False, use_targetness=False):
        super().__init__()
        if use_targetness and not use_semantic_gate:
            raise ValueError("Targetness requires the semantic direction gate")
        self.channels = int(channels)
        self.use_semantic_gate = bool(use_semantic_gate)
        self.use_targetness = bool(use_targetness)
        self.core = DecoderDSHFCore(channels)
        self.semantic_gate = (
            DecoderSemanticDirectionGate(channels, use_targetness)
            if use_semantic_gate
            else None
        )
        self.beta_h = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.beta_v = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.beta_d = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def reset_control_parameters(self):
        self.core.reset_control_parameters()
        if self.semantic_gate is not None:
            self.semantic_gate.reset_control_parameters()
        for beta in (self.beta_h, self.beta_v, self.beta_d):
            nn.init.constant_(beta, 1e-3)

    def forward(self, aligned_h, aligned_v, aligned_d, low_semantic, targetness=None):
        if not (aligned_h.shape == aligned_v.shape == aligned_d.shape):
            raise RuntimeError("Decoder DSHF requires equal aligned H/V/D shapes")
        if aligned_h.shape[1] != self.channels:
            raise RuntimeError(
                f"Decoder DSHF expected {self.channels} channels, got {aligned_h.shape[1]}"
            )
        residual_h, residual_v, residual_d, core_debug = self.core(
            aligned_h, aligned_v, aligned_d
        )
        deltas = tuple(torch.tanh(item) for item in (residual_h, residual_v, residual_d))
        if self.semantic_gate is None:
            if targetness is not None:
                raise RuntimeError("G1 decoder refiner must not receive targetness")
            scales = (
                torch.ones_like(deltas[0][:, :1]),
                torch.ones_like(deltas[1][:, :1]),
                torch.ones_like(deltas[2][:, :1]),
            )
            gate_debug = None
        else:
            scales, gate_debug = self.semantic_gate(
                low_semantic,
                residual_h,
                residual_v,
                residual_d,
                targetness=targetness,
            )
        betas = (self.beta_h, self.beta_v, self.beta_d)
        aligned = (aligned_h, aligned_v, aligned_d)
        restored = tuple(
            base + beta * scale * delta
            for base, beta, scale, delta in zip(aligned, betas, scales, deltas)
        )
        debug = OrderedDict(
            aligned=aligned,
            residuals=(residual_h, residual_v, residual_d),
            deltas=deltas,
            scales=scales,
            restored=restored,
            core=core_debug,
            gate=gate_debug,
        )
        return (*restored, debug)


__all__ = [
    "AdaptiveSparseSupportGate",
    "DecoderDSHFCore",
    "DecoderHFRefiner",
    "DecoderSemanticDirectionGate",
    "DirectionalMultiScaleExtractor",
    "EXPERIMENT_F2_CORE_COMMIT",
]
