"""Experiment D D5-D7 spatial cross-frequency consistency ablations.

D5 modulates the decoder low-frequency feature using same-position cosine
consistency and local contrast. D6 aggregates a fixed 3x3 low-frequency
neighborhood. D7 adds a detached targetness prior from the existing deep
supervision side heads. None of the variants performs global channel matching.
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder_HFE import (
    DirectionalResidualHead,
    LayerNorm2d,
    SubbandSelectiveFusion,
    assert_hfe_inputs,
)
from model.DWTFreqNet_SingleDecoder_HFE_Ablation import (
    AblationChannelAttention,
    AblationFFN,
    DirectFusionTransformation,
    DWTFreqNet_SingleDecoder_HFE_Ablation,
)


EXPERIMENT_D_SPATIAL_BASE_COMMIT = (
    "e5747b7f35fed3ecd3702a5f45332a8c35be8bd3"
)
SPATIAL_HFE_ABLATION_VARIANTS = (
    "d5_same_position",
    "d6_neighborhood",
    "d7_target_neighborhood",
)
SPATIAL_STAGE_CONFIG = {
    1: {"channels": 64, "num_heads": 1, "embed_channels": 16,
        "kernel_size": 3},
    2: {"channels": 128, "num_heads": 2, "embed_channels": 32,
        "kernel_size": 3},
    3: {"channels": 256, "num_heads": 4, "embed_channels": 64,
        "kernel_size": 3},
    4: {"channels": 256, "num_heads": 4, "embed_channels": 64,
        "kernel_size": 3},
}
OFFSETS_3X3 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CENTER_OFFSET_INDEX = 4


def shift_feature(feature, dy, dx):
    """Return output[p] = feature[p + (dy, dx)] with replicate padding."""
    if (dy, dx) not in OFFSETS_3X3:
        raise ValueError(f"Unsupported 3x3 offset: {(dy, dx)}")
    height, width = feature.shape[-2:]
    padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
    y0 = 1 + dy
    x0 = 1 + dx
    return padded[:, :, y0:y0 + height, x0:x0 + width]


def aggregate_shifted_low(low_feature, attention):
    """Aggregate a 3x3 neighborhood without materializing [B,C,9,H,W]."""
    expected = (low_feature.shape[0], 9, *low_feature.shape[-2:])
    if tuple(attention.shape) != expected:
        raise RuntimeError(
            f"Expected attention shape {expected}, got {tuple(attention.shape)}"
        )
    matched_low = torch.zeros_like(low_feature)
    for index, (dy, dx) in enumerate(OFFSETS_3X3):
        matched_low = matched_low + attention[:, index:index + 1].to(
            low_feature.dtype
        ) * shift_feature(low_feature, dy, dx)
    return matched_low


class SamePositionConsistencyFusion(DirectFusionTransformation):
    """D5 same-position consistency and low-frequency local contrast."""

    def __init__(self, channels, embed_channels):
        super().__init__(channels)
        self.embed_channels = int(embed_channels)
        self.q_proj = nn.Conv2d(channels, embed_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, embed_channels, 1, bias=False)
        self.low_response = nn.Conv2d(channels, 1, 1, bias=False)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1, bias=True),
        )
        self.reset_spatial_gate()
        self.last_spatial_scale = None
        self.last_similarity = None

    def reset_spatial_gate(self):
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

    def forward(self, high_feature, low_feature):
        if high_feature.shape != low_feature.shape:
            raise RuntimeError(
                "SamePositionConsistencyFusion requires equal shapes, got "
                f"{tuple(high_feature.shape)} and {tuple(low_feature.shape)}"
            )
        q = self.q_proj(high_feature)
        k = self.k_proj(low_feature)
        with torch.autocast(device_type=q.device.type, enabled=False):
            q_norm = F.normalize(q.float(), dim=1)
            k_norm = F.normalize(k.float(), dim=1)
            similarity = (q_norm * k_norm).sum(dim=1, keepdim=True)

        low_response = self.low_response(low_feature)
        local_mean = F.avg_pool2d(
            low_response, kernel_size=3, stride=1, padding=1
        )
        local_contrast = torch.abs(low_response - local_mean)
        contrast_mean = local_contrast.mean(dim=(2, 3), keepdim=True)
        contrast_normalized = local_contrast / (contrast_mean + 1e-6)
        contrast_normalized = contrast_normalized / (1.0 + contrast_normalized)
        gate_logits = self.spatial_gate(
            torch.cat(
                [similarity.to(low_feature.dtype), contrast_normalized], dim=1
            )
        )
        spatial_scale = 2.0 * torch.sigmoid(gate_logits)
        conditioned_low = spatial_scale * low_feature
        output, base_info = super().forward(high_feature, conditioned_low)
        info = dict(base_info)
        info.update(
            {
                "relation_mode": "same_position_consistency_local_contrast",
                "similarity_shape": tuple(similarity.shape),
                "spatial_scale_shape": tuple(spatial_scale.shape),
            }
        )
        if self.record_statistics:
            similarity_detached = similarity.detach().float()
            contrast_detached = local_contrast.detach().float()
            scale_detached = spatial_scale.detach().float()
            info.update(
                {
                    "similarity_mean": float(similarity_detached.mean().cpu()),
                    "similarity_std": float(similarity_detached.std().cpu()),
                    "similarity_min": float(similarity_detached.min().cpu()),
                    "similarity_max": float(similarity_detached.max().cpu()),
                    "local_contrast_mean": float(contrast_detached.mean().cpu()),
                    "local_contrast_std": float(contrast_detached.std().cpu()),
                    "spatial_scale_mean": float(scale_detached.mean().cpu()),
                    "spatial_scale_std": float(scale_detached.std().cpu()),
                    "spatial_scale_min": float(scale_detached.min().cpu()),
                    "spatial_scale_max": float(scale_detached.max().cpu()),
                    "conditioned_low_norm": self._rms(conditioned_low),
                    "raw_low_norm": self._rms(low_feature),
                    "output_norm": self._rms(output),
                }
            )
            self.last_spatial_scale = spatial_scale.detach()
            self.last_similarity = similarity.detach()
        else:
            self.last_spatial_scale = None
            self.last_similarity = None
        self.last_info = info
        return output, info


class NeighborhoodCrossFrequencyFusion(DirectFusionTransformation):
    """D6 local 3x3 spatial cross-frequency attention."""

    def __init__(self, channels, embed_channels, temperature=0.1):
        super().__init__(channels)
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.embed_channels = int(embed_channels)
        self.q_proj = nn.Conv2d(channels, embed_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, embed_channels, 1, bias=False)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature)).log()
        )
        self.last_attention = None
        self.last_similarity_logits = None

    def _similarity_logits(self, high_feature, low_feature):
        q = self.q_proj(high_feature)
        k = self.k_proj(low_feature)
        with torch.autocast(device_type=q.device.type, enabled=False):
            q_norm = F.normalize(q.float(), dim=1)
            k_norm = F.normalize(k.float(), dim=1)
            logits = torch.cat(
                [
                    (q_norm * shift_feature(k_norm, dy, dx)).sum(
                        dim=1, keepdim=True
                    )
                    for dy, dx in OFFSETS_3X3
                ],
                dim=1,
            )
            temperature = self.log_temperature.exp().clamp(
                min=0.03, max=1.0
            )
        return logits, temperature

    @staticmethod
    def _attention_statistics(attention):
        detached = attention.detach().float()
        entropy = -(
            detached * torch.log(detached.clamp_min(1e-12))
        ).sum(dim=1)
        argmax_index = detached.argmax(dim=1)
        offset_y = detached.new_tensor([item[0] for item in OFFSETS_3X3])
        offset_x = detached.new_tensor([item[1] for item in OFFSETS_3X3])
        selected_y = offset_y[argmax_index]
        selected_x = offset_x[argmax_index]
        center_ratio = (argmax_index == CENTER_OFFSET_INDEX).float().mean()
        return {
            "attention_entropy": float(entropy.mean().cpu()),
            "normalized_attention_entropy": float(
                (entropy / math.log(9.0)).mean().cpu()
            ),
            "center_weight_mean": float(
                detached[:, CENTER_OFFSET_INDEX].mean().cpu()
            ),
            "center_selection_ratio": float(center_ratio.cpu()),
            "neighbor_selection_ratio": float((1.0 - center_ratio).cpu()),
            "mean_abs_dy": float(selected_y.abs().mean().cpu()),
            "mean_abs_dx": float(selected_x.abs().mean().cpu()),
            "mean_offset_distance": float(
                torch.sqrt(selected_y.square() + selected_x.square()).mean().cpu()
            ),
            "offset_selection_ratios": [
                float((argmax_index == index).float().mean().cpu())
                for index in range(9)
            ],
        }

    def _finish_forward(
        self,
        high_feature,
        low_feature,
        attention,
        similarity_logits,
        temperature,
        relation_mode,
        extra_info=None,
    ):
        matched_low = aggregate_shifted_low(low_feature, attention)
        output, base_info = super().forward(high_feature, matched_low)
        info = dict(base_info)
        info.update(
            {
                "relation_mode": relation_mode,
                "attention_shape": tuple(attention.shape),
                "similarity_logits_shape": tuple(similarity_logits.shape),
            }
        )
        if extra_info:
            info.update(extra_info)
        if self.record_statistics:
            info.update(self._attention_statistics(attention))
            info.update(
                {
                    "temperature": float(temperature.detach().cpu()),
                    "matched_low_norm": self._rms(matched_low),
                    "raw_low_norm": self._rms(low_feature),
                }
            )
            self.last_attention = attention.detach()
            self.last_similarity_logits = similarity_logits.detach()
        else:
            self.last_attention = None
            self.last_similarity_logits = None
        self.last_info = info
        return output, info

    def forward(self, high_feature, low_feature):
        if high_feature.shape != low_feature.shape:
            raise RuntimeError(
                "NeighborhoodCrossFrequencyFusion requires equal shapes, got "
                f"{tuple(high_feature.shape)} and {tuple(low_feature.shape)}"
            )
        similarity_logits, temperature = self._similarity_logits(
            high_feature, low_feature
        )
        attention = torch.softmax(similarity_logits / temperature, dim=1)
        return self._finish_forward(
            high_feature,
            low_feature,
            attention,
            similarity_logits,
            temperature,
            "local_3x3_cross_frequency_attention",
        )


class TargetAwareNeighborhoodFusion(NeighborhoodCrossFrequencyFusion):
    """D7 D6 attention augmented by a detached side-head targetness prior."""

    def __init__(
        self,
        channels,
        embed_channels,
        temperature=0.1,
        targetness_scale=1.0,
    ):
        super().__init__(channels, embed_channels, temperature)
        if targetness_scale <= 0:
            raise ValueError(
                f"targetness_scale must be positive, got {targetness_scale}"
            )
        initial_raw = math.log(math.exp(float(targetness_scale)) - 1.0)
        self.raw_targetness_scale = nn.Parameter(torch.tensor(initial_raw))
        self.last_targetness = None
        self.last_target_neighbors = None

    def forward(self, high_feature, low_feature, targetness):
        if targetness.requires_grad:
            raise RuntimeError("D7 targetness prior must be detached")
        expected = (low_feature.shape[0], 1, *low_feature.shape[-2:])
        if tuple(targetness.shape) != expected:
            raise RuntimeError(
                f"Expected targetness shape {expected}, got {tuple(targetness.shape)}"
            )
        similarity_logits, temperature = self._similarity_logits(
            high_feature, low_feature
        )
        target_neighbors = torch.cat(
            [
                shift_feature(targetness, dy, dx)
                for dy, dx in OFFSETS_3X3
            ],
            dim=1,
        )
        targetness_scale = F.softplus(self.raw_targetness_scale)
        target_prior_logits = torch.log(
            target_neighbors.float().clamp_min(1e-6)
        )
        combined_logits = (
            similarity_logits / temperature
            + targetness_scale * target_prior_logits
        )
        attention = torch.softmax(combined_logits, dim=1)
        extra_info = {
            "targetness_shape": tuple(targetness.shape),
            "target_neighbors_shape": tuple(target_neighbors.shape),
            "targetness_detached": not targetness.requires_grad,
        }
        if self.record_statistics:
            extra_info.update(
                {
                    "targetness_scale": float(targetness_scale.detach().cpu()),
                    "targetness_mean": float(
                        targetness.detach().float().mean().cpu()
                    ),
                }
            )
            self.last_targetness = targetness.detach()
            self.last_target_neighbors = target_neighbors.detach()
        else:
            self.last_targetness = None
            self.last_target_neighbors = None
        return self._finish_forward(
            high_feature,
            low_feature,
            attention,
            similarity_logits,
            temperature,
            "targetness_prior_local_3x3_cross_frequency_attention",
            extra_info=extra_info,
        )


class SpatialDecoderHFEBlock(nn.Module):
    def __init__(
        self, channels, num_heads, attn_relation, ffn_relation
    ):
        super().__init__()
        self.high_norm1 = LayerNorm2d(channels)
        self.high_norm2 = LayerNorm2d(channels)
        self.low_norm = LayerNorm2d(channels)
        self.attn = AblationChannelAttention(
            channels=channels,
            num_heads=num_heads,
            relation_module=attn_relation,
        )
        self.ffn = AblationFFN(
            channels=channels, relation_module=ffn_relation
        )

    def forward(self, high_feature, low_feature):
        low = self.low_norm(low_feature)
        attn_out, attn_info = self.attn(self.high_norm1(high_feature), low)
        high_feature = high_feature + attn_out
        ffn_out, ffn_info = self.ffn(self.high_norm2(high_feature), low)
        return high_feature + ffn_out, {
            "attn_relation": attn_info,
            "ffn_relation": ffn_info,
        }


class TargetAwareChannelAttention(AblationChannelAttention):
    def forward(self, high_feature, low_feature, targetness):
        batch, channels, height, width = high_feature.shape
        q, k, v = self.qkv_dw(self.qkv(high_feature)).chunk(3, dim=1)
        q, relation_info = self.relation(q, low_feature, targetness)
        head_channels = channels // self.num_heads
        q = q.reshape(batch, self.num_heads, head_channels, height * width)
        k = k.reshape(batch, self.num_heads, head_channels, height * width)
        v = v.reshape(batch, self.num_heads, head_channels, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = torch.softmax(
            (q @ k.transpose(-2, -1)) * self.temperature, dim=-1
        )
        output = (attention @ v).reshape(batch, channels, height, width)
        self.last_attention = attention.detach()
        self.last_relation_info = relation_info
        return self.project_out(output), relation_info


class TargetAwareFFN(AblationFFN):
    def forward(self, high_feature, low_feature, targetness):
        x = self.dwconv1(self.project_in(high_feature))
        x, relation_info = self.relation(x, low_feature, targetness)
        x = F.gelu(self.dwconv2(x))
        self.last_relation_info = relation_info
        return self.project_out(x), relation_info


class TargetAwareDecoderHFEBlock(nn.Module):
    def __init__(
        self, channels, num_heads, attn_relation, ffn_relation
    ):
        super().__init__()
        self.high_norm1 = LayerNorm2d(channels)
        self.high_norm2 = LayerNorm2d(channels)
        self.low_norm = LayerNorm2d(channels)
        self.attn = TargetAwareChannelAttention(
            channels=channels,
            num_heads=num_heads,
            relation_module=attn_relation,
        )
        self.ffn = TargetAwareFFN(
            channels=channels, relation_module=ffn_relation
        )

    def forward(self, high_feature, low_feature, targetness):
        low = self.low_norm(low_feature)
        attn_out, attn_info = self.attn(
            self.high_norm1(high_feature), low, targetness
        )
        high_feature = high_feature + attn_out
        ffn_out, ffn_info = self.ffn(
            self.high_norm2(high_feature), low, targetness
        )
        return high_feature + ffn_out, {
            "attn_relation": attn_info,
            "ffn_relation": ffn_info,
        }


class SpatialDecoderHFERefiner(nn.Module):
    def __init__(
        self, channels, num_heads, stage, relation_factory
    ):
        super().__init__()
        self.stage = int(stage)
        self.subband_fusion = SubbandSelectiveFusion(channels, reduction=8)
        self.hfe = SpatialDecoderHFEBlock(
            channels=channels,
            num_heads=num_heads,
            attn_relation=relation_factory(),
            ffn_relation=relation_factory(),
        )
        self.head_h = DirectionalResidualHead(channels)
        self.head_v = DirectionalResidualHead(channels)
        self.head_d = DirectionalResidualHead(channels)
        self.beta_h = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.beta_v = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.beta_d = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.record_statistics = True
        self.last_statistics = None
        self.last_relation_info = None
        self.last_debug = None

    @staticmethod
    def _norm(tensor):
        return tensor.detach().float().square().mean().sqrt()

    @classmethod
    def _ratio(cls, numerator, denominator):
        return float(
            (cls._norm(numerator) / (cls._norm(denominator) + 1e-12)).cpu()
        )

    @staticmethod
    def _append_numeric_info(statistics, prefix, info):
        for key, value in info.items():
            if isinstance(value, (int, float)):
                statistics[f"{prefix}_{key}"] = float(value)

    def _run_hfe(self, shared_hf, low_feature, targetness=None):
        if targetness is None:
            return self.hfe(shared_hf, low_feature)
        return self.hfe(shared_hf, low_feature, targetness)

    def forward(
        self, low_feature, base_h, base_v, base_d, targetness=None
    ):
        assert_hfe_inputs(low_feature, base_h, base_v, base_d, self.stage)
        collect_statistics = not self.training and self.record_statistics
        self.hfe.attn.relation.record_statistics = collect_statistics
        self.hfe.ffn.relation.record_statistics = collect_statistics
        shared_hf = self.subband_fusion(base_h, base_v, base_d)
        refined_hf, relation_info = self._run_hfe(
            shared_hf, low_feature, targetness
        )
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
        self.last_relation_info = relation_info
        if collect_statistics:
            weights = self.subband_fusion.last_weights.float()
            statistics = {
                "mean_weight_H": float(weights[:, 0].mean().cpu()),
                "mean_weight_V": float(weights[:, 1].mean().cpu()),
                "mean_weight_D": float(weights[:, 2].mean().cpu()),
                "weight_variance": float(weights.var().cpu()),
                "shared_HF_norm": float(self._norm(shared_hf).cpu()),
                "refined_HF_norm": float(self._norm(refined_hf).cpu()),
                "refined_shared_norm_ratio": self._ratio(
                    refined_hf, shared_hf
                ),
            }
            self._append_numeric_info(
                statistics, "attn_relation", relation_info["attn_relation"]
            )
            self._append_numeric_info(
                statistics, "ffn_relation", relation_info["ffn_relation"]
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
                statistics[
                    f"beta_delta_{direction}_base_norm_ratio"
                ] = self._ratio(beta * delta, base)
                statistics[f"final_{direction}_base_norm_ratio"] = self._ratio(
                    final, base
                )
            self.last_statistics = statistics
        return coef_h, coef_v, coef_d, debug


class TargetAwareDecoderHFERefiner(SpatialDecoderHFERefiner):
    def __init__(
        self,
        channels,
        num_heads,
        stage,
        embed_channels,
        temperature=0.1,
        targetness_scale=1.0,
    ):
        relation_factory = lambda: TargetAwareNeighborhoodFusion(
            channels=channels,
            embed_channels=embed_channels,
            temperature=temperature,
            targetness_scale=targetness_scale,
        )
        super().__init__(channels, num_heads, stage, relation_factory)
        self.hfe = TargetAwareDecoderHFEBlock(
            channels=channels,
            num_heads=num_heads,
            attn_relation=relation_factory(),
            ffn_relation=relation_factory(),
        )

    def forward(
        self, low_feature, base_h, base_v, base_d, targetness
    ):
        return super().forward(
            low_feature,
            base_h,
            base_v,
            base_d,
            targetness=targetness,
        )


class DWTFreqNet_SingleDecoder_HFE_SpatialAblation(
    DWTFreqNet_SingleDecoder_HFE_Ablation
):
    """Isolated D5/D6/D7 spatial relation variants."""

    def __init__(
        self,
        config,
        spatial_hfe_ablation,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if spatial_hfe_ablation not in SPATIAL_HFE_ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown spatial_hfe_ablation={spatial_hfe_ablation!r}; "
                f"expected one of {SPATIAL_HFE_ABLATION_VARIANTS}"
            )
        super().__init__(
            config,
            hfe_ablation="d4_no_matching",
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
        )
        self.spatial_hfe_ablation = spatial_hfe_ablation
        self.spatial_stage_config = SPATIAL_STAGE_CONFIG
        for stage in range(1, 5):
            cfg = SPATIAL_STAGE_CONFIG[stage]
            if spatial_hfe_ablation == "d5_same_position":
                relation_factory = lambda cfg=cfg: SamePositionConsistencyFusion(
                    channels=cfg["channels"],
                    embed_channels=cfg["embed_channels"],
                )
                refiner = SpatialDecoderHFERefiner(
                    channels=cfg["channels"],
                    num_heads=cfg["num_heads"],
                    stage=stage,
                    relation_factory=relation_factory,
                )
            elif spatial_hfe_ablation == "d6_neighborhood":
                relation_factory = lambda cfg=cfg: NeighborhoodCrossFrequencyFusion(
                    channels=cfg["channels"],
                    embed_channels=cfg["embed_channels"],
                    temperature=0.1,
                )
                refiner = SpatialDecoderHFERefiner(
                    channels=cfg["channels"],
                    num_heads=cfg["num_heads"],
                    stage=stage,
                    relation_factory=relation_factory,
                )
            else:
                refiner = TargetAwareDecoderHFERefiner(
                    channels=cfg["channels"],
                    num_heads=cfg["num_heads"],
                    stage=stage,
                    embed_channels=cfg["embed_channels"],
                    temperature=0.1,
                    targetness_scale=1.0,
                )
            setattr(self, f"decoder_hfe{stage}", refiner)

        self.experiment_group = "experiment_d"
        self.experiment_type = "ablation"
        self.ablation_axis = "decoder_hfe_spatial_relation"
        self.model_base_commit = EXPERIMENT_D_SPATIAL_BASE_COMMIT
        self.explicit_channel_matching = False
        self.channel_similarity_matrix = False
        self.channel_candidate_selection = False
        self.directional_pyramid = False
        self.second_dwt = False
        self.ldrc = False
        self.mamba = False
        self.coefficient_mode = (
            "aligned_raw_plus_spatial_hfe_directional_residual"
        )
        if spatial_hfe_ablation == "d5_same_position":
            self.ablation_id = "D5"
            self.model_variant = "dwtfreqnet_single_decoder_hfe_samepos"
            self.sd_variant = "sd_awgm_hfe_samepos"
            self.decoder_hfe_relation = (
                "same_position_consistency_local_contrast"
            )
            self.spatial_offset_search = False
            self.targetness_prior = False
        elif spatial_hfe_ablation == "d6_neighborhood":
            self.ablation_id = "D6"
            self.model_variant = "dwtfreqnet_single_decoder_hfe_neighborhood"
            self.sd_variant = "sd_awgm_hfe_neighborhood"
            self.decoder_hfe_relation = (
                "local_3x3_cross_frequency_attention"
            )
            self.spatial_offset_search = True
            self.targetness_prior = False
        else:
            self.ablation_id = "D7"
            self.model_variant = "dwtfreqnet_single_decoder_hfe_targetlocal"
            self.sd_variant = "sd_awgm_hfe_targetlocal"
            self.decoder_hfe_relation = (
                "targetness_prior_local_3x3_cross_frequency_attention"
            )
            self.spatial_offset_search = True
            self.targetness_prior = True
        self.decoder_hfe_matching = self.decoder_hfe_relation
        self.last_targetness_requires_grad = {}

    def reset_spatial_initialization(self):
        """Restore D5's neutral scale after a repository-wide init pass."""
        if self.spatial_hfe_ablation != "d5_same_position":
            return
        for stage in range(1, 5):
            refiner = getattr(self, f"decoder_hfe{stage}")
            refiner.hfe.attn.relation.reset_spatial_gate()
            refiner.hfe.ffn.relation.reset_spatial_gate()

    @property
    def experiment_metadata(self):
        metadata = super().experiment_metadata
        for inherited_d4_key in (
            "hfe_ablation",
            "hfe_stage_modes",
            "hfe_topk",
            "hfe_initial_temperature",
            "direct_fusion_uses_raw_low",
            "stage1_relation",
            "stage2_relation",
            "stage3_relation",
            "stage4_relation",
        ):
            metadata.pop(inherited_d4_key, None)
        metadata.update(
            {
                "experiment_group": self.experiment_group,
                "experiment_type": self.experiment_type,
                "ablation_axis": self.ablation_axis,
                "ablation_id": self.ablation_id,
                "model_variant": self.model_variant,
                "sd_variant": self.sd_variant,
                "spatial_hfe_ablation": self.spatial_hfe_ablation,
                "relation_mode": self.decoder_hfe_relation,
                "decoder_hfe_relation": self.decoder_hfe_relation,
                "base_fusion_head": "direct_low_fusion",
                "model_base_commit": self.model_base_commit,
                "embed_channels": {
                    str(stage): cfg["embed_channels"]
                    for stage, cfg in SPATIAL_STAGE_CONFIG.items()
                },
                "kernel_size": 3,
                "temperature_init": (
                    None
                    if self.spatial_hfe_ablation == "d5_same_position"
                    else 0.1
                ),
                "explicit_channel_matching": False,
                "channel_similarity_matrix": False,
                "channel_candidate_selection": False,
                "spatial_offset_search": self.spatial_offset_search,
                "neighborhood_size": (
                    3 if self.spatial_offset_search else None
                ),
                "targetness_prior": self.targetness_prior,
                "targetness_prior_detached": self.targetness_prior,
                "targetness_scale_init": (
                    1.0 if self.targetness_prior else None
                ),
                "directional_pyramid": False,
                "second_dwt": False,
                "ldrc": False,
                "mamba": False,
                "dwt_calls": 4,
                "idwt_calls": 4,
            }
        )
        if self.targetness_prior:
            metadata.update(
                {
                    "targetness_prior_source": (
                        "existing_deep_supervision_side_heads"
                    ),
                    "side_head_mapping": {
                        "stage4": "gt_conv5",
                        "stage3": "gt_conv4",
                        "stage2": "gt_conv3",
                        "stage1": "gt_conv2",
                    },
                }
            )
        return metadata

    def forward(self, x):
        if self.spatial_hfe_ablation == "d7_target_neighborhood":
            return self._forward_d7(x)
        return super().forward(x)

    def _forward_d7(self, x):
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError(
                "Experiment D input H/W must be divisible by 16, got "
                f"{tuple(x.shape[-2:])}"
            )
        self.last_transform_counts = {"dwt": 0, "idwt": 0}
        x0 = self.stem(x)
        encoded, raw_bands, directional, guided = {}, {}, {}, {}
        current = x0
        for stage in range(1, 5):
            (
                encoded[stage],
                raw_bands[stage],
                directional[stage],
                guided[stage],
            ) = self._encode_stage(stage, current)
            current = encoded[stage]
        bases = {
            stage: self._align_stage_coefficients(stage, raw_bands[stage])
            for stage in range(1, 5)
        }
        hfe_debug = {}
        side_raw = {}
        targetness = {}

        side_raw[4] = self.gt_conv5(encoded[4])
        targetness[4] = torch.sigmoid(side_raw[4]).detach()
        coef_h4, coef_v4, coef_d4, hfe_debug[4] = self.decoder_hfe4(
            encoded[4], *bases[4], targetness=targetness[4]
        )
        u3 = self._idwt(encoded[4], coef_h4, coef_v4, coef_d4)
        l3 = self.decoder_fuse3(torch.cat([u3, encoded[3]], dim=1))

        side_raw[3] = self.gt_conv4(l3)
        targetness[3] = torch.sigmoid(side_raw[3]).detach()
        coef_h3, coef_v3, coef_d3, hfe_debug[3] = self.decoder_hfe3(
            l3, *bases[3], targetness=targetness[3]
        )
        u2 = self._idwt(l3, coef_h3, coef_v3, coef_d3)
        l2 = self.decoder_fuse2(torch.cat([u2, encoded[2]], dim=1))

        side_raw[2] = self.gt_conv3(l2)
        targetness[2] = torch.sigmoid(side_raw[2]).detach()
        coef_h2, coef_v2, coef_d2, hfe_debug[2] = self.decoder_hfe2(
            l2, *bases[2], targetness=targetness[2]
        )
        u1 = self._idwt(l2, coef_h2, coef_v2, coef_d2)
        l1 = self.decoder_fuse1(torch.cat([u1, encoded[1]], dim=1))

        side_raw[1] = self.gt_conv2(l1)
        targetness[1] = torch.sigmoid(side_raw[1]).detach()
        coef_h1, coef_v1, coef_d1, hfe_debug[1] = self.decoder_hfe1(
            l1, *bases[1], targetness=targetness[1]
        )
        u0 = self._idwt(l1, coef_h1, coef_v1, coef_d1)
        l0 = self.decoder_fuse0(torch.cat([u0, x0], dim=1))
        out = self.out_head(l0)

        self.last_targetness_requires_grad = {
            stage: tensor.requires_grad for stage, tensor in targetness.items()
        }
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
                refiner_stats = getattr(
                    self, f"decoder_hfe{stage}"
                ).last_statistics
                for key, value in refiner_stats.items():
                    statistics[f"hfe{stage}_{key}"] = value
            self.last_sd_statistics = dict(statistics)

        if self.debug_tensors:
            self.last_debug = {
                "A": {
                    stage: raw_bands[stage][0].detach()
                    for stage in range(1, 5)
                },
                "A_guided": {
                    stage: guided[stage].detach() for stage in range(1, 5)
                },
                "base_coefficients": {
                    stage: tuple(tensor.detach() for tensor in bases[stage])
                    for stage in range(1, 5)
                },
                "hfe": {
                    stage: {
                        key: value.detach()
                        for key, value in hfe_debug[stage].items()
                    }
                    for stage in range(1, 5)
                },
                "targetness": {
                    stage: tensor.detach()
                    for stage, tensor in targetness.items()
                },
            }

        if not self.deepsuper:
            return torch.sigmoid(out)
        target_size = x.shape[-2:]
        gt5 = F.interpolate(
            side_raw[4], target_size, mode="bilinear", align_corners=False
        )
        gt4 = F.interpolate(
            side_raw[3], target_size, mode="bilinear", align_corners=False
        )
        gt3 = F.interpolate(
            side_raw[2], target_size, mode="bilinear", align_corners=False
        )
        gt2 = F.interpolate(
            side_raw[1], target_size, mode="bilinear", align_corners=False
        )
        d0 = self.outconv(torch.cat([gt2, gt3, gt4, gt5, out], dim=1))
        if self.mode == "train":
            return tuple(
                torch.sigmoid(tensor)
                for tensor in (gt5, gt4, gt3, gt2, d0, out)
            )
        return torch.sigmoid(out)


__all__ = [
    "CENTER_OFFSET_INDEX",
    "DWTFreqNet_SingleDecoder_HFE_SpatialAblation",
    "EXPERIMENT_D_SPATIAL_BASE_COMMIT",
    "NeighborhoodCrossFrequencyFusion",
    "OFFSETS_3X3",
    "SPATIAL_HFE_ABLATION_VARIANTS",
    "SPATIAL_STAGE_CONFIG",
    "SamePositionConsistencyFusion",
    "SpatialDecoderHFEBlock",
    "SpatialDecoderHFERefiner",
    "TargetAwareChannelAttention",
    "TargetAwareDecoderHFEBlock",
    "TargetAwareDecoderHFERefiner",
    "TargetAwareFFN",
    "TargetAwareNeighborhoodFusion",
    "aggregate_shifted_low",
    "shift_feature",
]
