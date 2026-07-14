"""Experiment D: decoder-side high-frequency enhancement for SD-AWGM.

The HFE design is independently rewritten from the frequency-interaction idea in
Wave-Mamba: current decoder low-frequency semantics match and correct the H/LH,
V/HL and D/HH coefficients before each IDWT.  No Wave-Mamba source file is
copied here.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder


EXPERIMENT_D_BASE_COMMIT = "435ab1827ecee4c6b83b669789bb9833a5fd5320"


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return (
            self.weight.view(1, -1, 1, 1) * normalized
            + self.bias.view(1, -1, 1, 1)
        )


class SubbandSelectiveFusion(nn.Module):
    """Channel-wise SKFF over aligned H/LH, V/HL and D/HH coefficients."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.PReLU(),
        )
        self.to_h = nn.Conv2d(hidden, channels, 1, bias=False)
        self.to_v = nn.Conv2d(hidden, channels, 1, bias=False)
        self.to_d = nn.Conv2d(hidden, channels, 1, bias=False)
        self.last_weights = None

    def forward(self, band_h, band_v, band_d):
        stacked = torch.stack([band_h, band_v, band_d], dim=1)
        descriptor = self.reduce(band_h + band_v + band_d)
        logits = torch.stack(
            [
                self.to_h(descriptor),
                self.to_v(descriptor),
                self.to_d(descriptor),
            ],
            dim=1,
        )
        weights = torch.softmax(logits, dim=1)
        fused = (stacked * weights).sum(dim=1)
        self.last_weights = weights.detach()
        return fused


class ChannelMatching(nn.Module):
    """Hard top-1 L2 matching between high- and low-frequency channels."""

    def __init__(self):
        super().__init__()
        self.last_indices = None
        self.last_distance_shape = None

    def forward(self, query_feature, candidate_feature):
        if query_feature.shape != candidate_feature.shape:
            raise RuntimeError(
                "ChannelMatching requires equal shapes, got "
                f"{tuple(query_feature.shape)} and {tuple(candidate_feature.shape)}"
            )
        query = query_feature.flatten(2)
        candidate = candidate_feature.flatten(2)
        with torch.no_grad(), torch.autocast(
            device_type=query.device.type, enabled=False
        ):
            distance = torch.cdist(query.float(), candidate.float())
            indices = distance.argmin(dim=-1).detach()
        gather_index = indices.unsqueeze(-1).expand(-1, -1, candidate.shape[-1])
        selected = torch.gather(candidate, dim=1, index=gather_index)
        selected = selected.reshape_as(candidate_feature)
        self.last_indices = indices
        self.last_distance_shape = tuple(distance.shape)
        return selected, indices


class MatchingTransformation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.matching = ChannelMatching()
        combined_channels = channels * 2
        self.gate = nn.Conv2d(combined_channels, combined_channels, 1)
        self.value = nn.Conv2d(
            combined_channels,
            combined_channels,
            3,
            padding=1,
            groups=combined_channels,
            bias=False,
        )
        self.project = nn.Conv2d(combined_channels, channels, 1, bias=False)
        self.record_statistics = True
        self.last_indices = None
        self.last_unique_ratio = None
        self.last_most_selected_channel_frequency = None
        self.last_mean_channel_reuse = None

    def _record_matching_statistics(self, indices):
        channels = indices.shape[1]
        unique_ratios = []
        most_selected = []
        mean_reuse = []
        for sample_indices in indices:
            counts = torch.bincount(sample_indices, minlength=channels).float()
            used = counts > 0
            unique_ratios.append(used.float().mean())
            most_selected.append(counts.max() / float(channels))
            mean_reuse.append(counts[used].mean())
        self.last_unique_ratio = float(torch.stack(unique_ratios).mean().cpu())
        self.last_most_selected_channel_frequency = float(
            torch.stack(most_selected).mean().cpu()
        )
        self.last_mean_channel_reuse = float(torch.stack(mean_reuse).mean().cpu())

    def forward(self, x, perception):
        selected_low, indices = self.matching(x, perception)
        combined = torch.cat([x, selected_low], dim=1)
        output = self.project(torch.sigmoid(self.gate(combined)) * self.value(combined))
        self.last_indices = indices.detach()
        if self.record_statistics:
            self._record_matching_statistics(indices)
        return output


class ChannelMatchedAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.matching = MatchingTransformation(channels)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.last_attention = None

    def forward(self, high_feature, low_feature):
        batch, channels, height, width = high_feature.shape
        q, k, v = self.qkv_dw(self.qkv(high_feature)).chunk(3, dim=1)
        q = self.matching(q, low_feature)
        head_channels = channels // self.num_heads
        q = q.reshape(batch, self.num_heads, head_channels, height * width)
        k = k.reshape(batch, self.num_heads, head_channels, height * width)
        v = v.reshape(batch, self.num_heads, head_channels, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = torch.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        self.last_attention = attention.detach()
        return self.project_out(output)


class ChannelMatchedFFN(nn.Module):
    def __init__(self, channels, expansion=1.0):
        super().__init__()
        hidden = int(channels * expansion)
        if hidden != channels:
            raise ValueError("Experiment D requires expansion=1.0 for channel matching")
        self.project_in = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dwconv1 = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, bias=False
        )
        self.matching = MatchingTransformation(hidden)
        self.dwconv2 = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, bias=False
        )
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, high_feature, low_feature):
        high_feature = self.dwconv1(self.project_in(high_feature))
        high_feature = self.matching(high_feature, low_feature)
        high_feature = F.gelu(self.dwconv2(high_feature))
        return self.project_out(high_feature)


class DecoderHFEBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.high_norm1 = LayerNorm2d(channels)
        self.high_norm2 = LayerNorm2d(channels)
        self.low_norm = LayerNorm2d(channels)
        self.attn = ChannelMatchedAttention(channels, num_heads=num_heads)
        self.ffn = ChannelMatchedFFN(channels, expansion=1.0)

    def forward(self, high_feature, low_feature):
        low = self.low_norm(low_feature)
        high_feature = high_feature + self.attn(
            self.high_norm1(high_feature), low
        )
        high_feature = high_feature + self.ffn(
            self.high_norm2(high_feature), low
        )
        return high_feature


class DirectionalResidualHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, base_direction, refined_shared):
        return self.net(torch.cat([base_direction, refined_shared], dim=1))


def assert_hfe_inputs(low, band_h, band_v, band_d, stage):
    expected = low.shape
    for name, tensor in (("H", band_h), ("V", band_v), ("D", band_d)):
        if tensor.shape != expected:
            raise RuntimeError(
                f"Decoder HFE stage {stage}: {name} shape {tuple(tensor.shape)} "
                f"!= low shape {tuple(expected)}"
            )


class DecoderHFERefiner(nn.Module):
    def __init__(self, channels, num_heads, stage):
        super().__init__()
        self.stage = stage
        self.subband_fusion = SubbandSelectiveFusion(channels, reduction=8)
        self.hfe = DecoderHFEBlock(channels, num_heads=num_heads)
        self.head_h = DirectionalResidualHead(channels)
        self.head_v = DirectionalResidualHead(channels)
        self.head_d = DirectionalResidualHead(channels)
        self.beta_h = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.beta_v = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.beta_d = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.record_statistics = True
        self.last_statistics = None
        self.last_debug = None

    @staticmethod
    def _norm(tensor):
        return tensor.detach().float().square().mean().sqrt()

    @classmethod
    def _ratio(cls, numerator, denominator):
        return float((cls._norm(numerator) / (cls._norm(denominator) + 1e-12)).cpu())

    def _matching_statistics(self, prefix, module):
        return {
            f"{prefix}_matching_unique_ratio": module.last_unique_ratio,
            f"{prefix}_most_selected_channel_frequency": (
                module.last_most_selected_channel_frequency
            ),
            f"{prefix}_mean_channel_reuse": module.last_mean_channel_reuse,
        }

    def forward(self, low_feature, base_h, base_v, base_d):
        assert_hfe_inputs(low_feature, base_h, base_v, base_d, self.stage)
        collect_statistics = not self.training and self.record_statistics
        self.hfe.attn.matching.record_statistics = collect_statistics
        self.hfe.ffn.matching.record_statistics = collect_statistics
        shared_hf = self.subband_fusion(base_h, base_v, base_d)
        refined_hf = self.hfe(shared_hf, low_feature)
        delta_h = self.head_h(base_h, refined_hf)
        delta_v = self.head_v(base_v, refined_hf)
        delta_d = self.head_d(base_d, refined_hf)
        coef_h = base_h + self.beta_h * delta_h
        coef_v = base_v + self.beta_v * delta_v
        coef_d = base_d + self.beta_d * delta_d

        debug = {
            "shared_hf": shared_hf,
            "refined_hf": refined_hf,
            "delta_h": delta_h,
            "delta_v": delta_v,
            "delta_d": delta_d,
            "coef_h": coef_h,
            "coef_v": coef_v,
            "coef_d": coef_d,
        }
        self.last_debug = {key: value.detach() for key, value in debug.items()}

        if collect_statistics:
            weights = self.subband_fusion.last_weights.float()
            statistics = {
                "mean_weight_H": float(weights[:, 0].mean().cpu()),
                "mean_weight_V": float(weights[:, 1].mean().cpu()),
                "mean_weight_D": float(weights[:, 2].mean().cpu()),
                "weight_variance": float(weights.var().cpu()),
                "shared_HF_norm": float(self._norm(shared_hf).cpu()),
                "refined_HF_norm": float(self._norm(refined_hf).cpu()),
                "refined_shared_norm_ratio": self._ratio(refined_hf, shared_hf),
            }
            statistics.update(
                self._matching_statistics("attn", self.hfe.attn.matching)
            )
            statistics.update(
                self._matching_statistics("ffn", self.hfe.ffn.matching)
            )
            for direction, beta, delta, base, final in (
                ("H", self.beta_h, delta_h, base_h, coef_h),
                ("V", self.beta_v, delta_v, base_v, coef_v),
                ("D", self.beta_d, delta_d, base_d, coef_d),
            ):
                statistics[f"beta_{direction}_mean"] = float(
                    beta.detach().float().mean().cpu()
                )
                statistics[f"delta_{direction}_base_norm_ratio"] = self._ratio(
                    delta, base
                )
                statistics[f"beta_delta_{direction}_base_norm_ratio"] = self._ratio(
                    beta * delta, base
                )
                statistics[f"final_{direction}_base_norm_ratio"] = self._ratio(
                    final, base
                )
            self.last_statistics = statistics
        return coef_h, coef_v, coef_d, debug


class DWTFreqNet_SingleDecoder_HFE(DWTFreqNet_SingleDecoder):
    """SD-AWGM with same-scale decoder low-to-high coefficient correction."""

    def __init__(
        self,
        config,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        super().__init__(
            config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
            sd_variant="sd_awgm",
        )
        self.decoder_hfe1 = DecoderHFERefiner(64, 1, stage=1)
        self.decoder_hfe2 = DecoderHFERefiner(128, 2, stage=2)
        self.decoder_hfe3 = DecoderHFERefiner(256, 4, stage=3)
        self.decoder_hfe4 = DecoderHFERefiner(256, 4, stage=4)
        self.model_variant = "dwtfreqnet_single_decoder_hfe"
        self.sd_variant = "sd_awgm_hfe"
        self.model_base_commit = EXPERIMENT_D_BASE_COMMIT
        self.decoder_hfe = True
        self.decoder_hfe_source = "wave_mamba_inspired"
        self.decoder_hfe_matching = "hard_l2_channel_top1"
        self.decoder_hfe_subband_fusion = "channel_skff"
        self.decoder_hfe_residual = True
        self.directional_pyramid = False
        self.second_dwt = False
        self.ldrc = False
        self.mamba = False
        self.coefficient_mode = "aligned_raw_plus_hfe_directional_residual"

    @property
    def experiment_metadata(self):
        metadata = super().experiment_metadata
        metadata.update(
            {
                "model_variant": self.model_variant,
                "sd_variant": self.sd_variant,
                "stage_wise_awgm": True,
                "decoder_hfe": True,
                "decoder_hfe_source": self.decoder_hfe_source,
                "decoder_hfe_matching": self.decoder_hfe_matching,
                "decoder_hfe_subband_fusion": self.decoder_hfe_subband_fusion,
                "decoder_hfe_direction_heads": True,
                "decoder_hfe_beta_init": 0.001,
                "decoder_hfe_residual": True,
                "directional_pyramid": False,
                "second_dwt": False,
                "ldrc": False,
                "mamba": False,
                "coefficient_mode": self.coefficient_mode,
            }
        )
        return metadata

    def _align_stage_coefficients(self, stage, raw_bands):
        _, raw_h, raw_v, raw_d = raw_bands
        return (
            getattr(self, f"align_H{stage}")(raw_h),
            getattr(self, f"align_V{stage}")(raw_v),
            getattr(self, f"align_D{stage}")(raw_d),
        )

    def forward(self, x):
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError(
                f"Experiment D input H/W must be divisible by 16, got {tuple(x.shape[-2:])}"
            )
        self.last_transform_counts = {"dwt": 0, "idwt": 0}
        x0 = self.stem(x)
        encoded = {}
        raw_bands = {}
        directional = {}
        guided = {}
        current = x0
        for stage in range(1, 5):
            encoded[stage], raw_bands[stage], directional[stage], guided[stage] = (
                self._encode_stage(stage, current)
            )
            current = encoded[stage]

        bases = {
            stage: self._align_stage_coefficients(stage, raw_bands[stage])
            for stage in range(1, 5)
        }
        hfe_debug = {}

        coef_h4, coef_v4, coef_d4, hfe_debug[4] = self.decoder_hfe4(
            encoded[4], *bases[4]
        )
        u3 = self._idwt(encoded[4], coef_h4, coef_v4, coef_d4)
        l3 = self.decoder_fuse3(torch.cat([u3, encoded[3]], dim=1))

        coef_h3, coef_v3, coef_d3, hfe_debug[3] = self.decoder_hfe3(
            l3, *bases[3]
        )
        u2 = self._idwt(l3, coef_h3, coef_v3, coef_d3)
        l2 = self.decoder_fuse2(torch.cat([u2, encoded[2]], dim=1))

        coef_h2, coef_v2, coef_d2, hfe_debug[2] = self.decoder_hfe2(
            l2, *bases[2]
        )
        u1 = self._idwt(l2, coef_h2, coef_v2, coef_d2)
        l1 = self.decoder_fuse1(torch.cat([u1, encoded[1]], dim=1))

        coef_h1, coef_v1, coef_d1, hfe_debug[1] = self.decoder_hfe1(
            l1, *bases[1]
        )
        u0 = self._idwt(l1, coef_h1, coef_v1, coef_d1)
        l0 = self.decoder_fuse0(torch.cat([u0, x0], dim=1))
        out = self.out_head(l0)

        self.last_shapes = {
            "X0": tuple(x0.shape),
            "E1": tuple(encoded[1].shape),
            "E2": tuple(encoded[2].shape),
            "E3": tuple(encoded[3].shape),
            "E4": tuple(encoded[4].shape),
            "L3": tuple(l3.shape),
            "L2": tuple(l2.shape),
            "L1": tuple(l1.shape),
            "L0": tuple(l0.shape),
        }
        for stage in range(1, 5):
            self.last_shapes[f"HFE{stage}_shared"] = tuple(
                hfe_debug[stage]["shared_hf"].shape
            )
            self.last_shapes[f"HFE{stage}_refined"] = tuple(
                hfe_debug[stage]["refined_hf"].shape
            )

        if not self.training and self.record_statistics:
            statistics = OrderedDict()
            awgm_stats = [
                getattr(self, f"stage_awgm{stage}").last_statistics
                for stage in range(1, 5)
            ]
            for key in awgm_stats[0]:
                statistics[f"stage_awgm_{key}"] = sum(
                    item[key] for item in awgm_stats
                ) / len(awgm_stats)
            for stage in range(1, 5):
                refiner_stats = getattr(self, f"decoder_hfe{stage}").last_statistics
                for key, value in refiner_stats.items():
                    statistics[f"hfe{stage}_{key}"] = value
            self.last_sd_statistics = dict(statistics)

        if self.debug_tensors:
            self.last_debug = {
                "A": {stage: raw_bands[stage][0].detach() for stage in range(1, 5)},
                "A_guided": {stage: guided[stage].detach() for stage in range(1, 5)},
                "base_coefficients": {
                    stage: tuple(tensor.detach() for tensor in bases[stage])
                    for stage in range(1, 5)
                },
                "hfe": {
                    stage: {
                        key: value.detach() for key, value in hfe_debug[stage].items()
                    }
                    for stage in range(1, 5)
                },
            }

        if not self.deepsuper:
            return torch.sigmoid(out)

        target_size = x.shape[-2:]
        gt5 = F.interpolate(
            self.gt_conv5(encoded[4]), target_size, mode="bilinear", align_corners=False
        )
        gt4 = F.interpolate(
            self.gt_conv4(l3), target_size, mode="bilinear", align_corners=False
        )
        gt3 = F.interpolate(
            self.gt_conv3(l2), target_size, mode="bilinear", align_corners=False
        )
        gt2 = F.interpolate(
            self.gt_conv2(l1), target_size, mode="bilinear", align_corners=False
        )
        d0 = self.outconv(torch.cat([gt2, gt3, gt4, gt5, out], dim=1))
        if self.mode == "train":
            return tuple(
                torch.sigmoid(tensor) for tensor in (gt5, gt4, gt3, gt2, d0, out)
            )
        return torch.sigmoid(out)


__all__ = [
    "ChannelMatchedAttention",
    "ChannelMatchedFFN",
    "ChannelMatching",
    "DecoderHFEBlock",
    "DecoderHFERefiner",
    "DirectionalResidualHead",
    "DWTFreqNet_SingleDecoder_HFE",
    "EXPERIMENT_D_BASE_COMMIT",
    "LayerNorm2d",
    "MatchingTransformation",
    "SubbandSelectiveFusion",
    "assert_hfe_inputs",
]
