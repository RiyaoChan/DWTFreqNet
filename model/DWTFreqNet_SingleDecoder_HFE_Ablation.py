"""Experiment D matching ablations for decoder-side HFE.

D2 replaces hard L2 Top-1 matching at all four scales with differentiable
Soft Cosine Top-k matching.  D3 keeps the same deep relation modules at stages
3/4 and replaces only stages 1/2 with a local correlation gate.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder_HFE import (
    DirectionalResidualHead,
    DWTFreqNet_SingleDecoder_HFE,
    LayerNorm2d,
    SubbandSelectiveFusion,
    assert_hfe_inputs,
)


EXPERIMENT_D_HFE_ABLATION_BASE_COMMIT = (
    "6fb19768dd7013aff536447b39652a44c1538912"
)
HFE_ABLATION_VARIANTS = ("d2_softcos_all", "d3_scaleaware")

D2_STAGE_CONFIG = {
    1: {"mode": "soft_cosine_topk", "channels": 64, "num_heads": 1,
        "topk": 8, "temperature": 0.1},
    2: {"mode": "soft_cosine_topk", "channels": 128, "num_heads": 2,
        "topk": 8, "temperature": 0.1},
    3: {"mode": "soft_cosine_topk", "channels": 256, "num_heads": 4,
        "topk": 8, "temperature": 0.1},
    4: {"mode": "soft_cosine_topk", "channels": 256, "num_heads": 4,
        "topk": 8, "temperature": 0.1},
}

D3_STAGE_CONFIG = {
    1: {"mode": "local_correlation_gate", "channels": 64, "num_heads": 1},
    2: {"mode": "local_correlation_gate", "channels": 128, "num_heads": 2},
    3: {"mode": "soft_cosine_topk", "channels": 256, "num_heads": 4,
        "topk": 8, "temperature": 0.1},
    4: {"mode": "soft_cosine_topk", "channels": 256, "num_heads": 4,
        "topk": 8, "temperature": 0.1},
}


class SoftCosineTopKMatching(nn.Module):
    """Differentiable channel matching using cosine similarity and soft Top-k."""

    def __init__(self, channels, topk=8, temperature=0.1):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if topk <= 0 or topk > channels:
            raise ValueError(
                f"topk must be in [1, channels], got topk={topk}, channels={channels}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.channels = int(channels)
        self.topk = int(topk)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(temperature)).log()
        )
        self.record_statistics = True
        self.last_info = None

    @staticmethod
    def dense_aggregate(candidate, topk_indices, topk_weights):
        """Memory-efficient aggregation used by the formal implementation."""
        batch, query_channels, _ = topk_indices.shape
        candidate_channels = candidate.shape[1]
        dense_weights = candidate.new_zeros(
            batch, query_channels, candidate_channels
        )
        dense_weights.scatter_add_(-1, topk_indices, topk_weights)
        return torch.bmm(dense_weights, candidate)

    @staticmethod
    def gather_aggregate(candidate, topk_indices, topk_weights):
        """Reference aggregation used only by the numerical-consistency test."""
        candidate_expanded = candidate.unsqueeze(1).expand(
            -1, topk_indices.shape[1], -1, -1
        )
        gather_index = topk_indices.unsqueeze(-1).expand(
            -1, -1, -1, candidate.shape[-1]
        )
        selected = torch.gather(candidate_expanded, dim=2, index=gather_index)
        return (selected * topk_weights.unsqueeze(-1)).sum(dim=2)

    def _diagnostics(self, similarity, indices, weights, temperature):
        detached_weights = weights.detach().float()
        detached_indices = indices.detach()
        entropy = -(
            detached_weights * torch.log(detached_weights.clamp_min(1e-12))
        ).sum(dim=-1)
        if self.topk > 1:
            normalized_entropy = entropy / math.log(self.topk)
        else:
            normalized_entropy = torch.zeros_like(entropy)

        usage_ratios = []
        most_used = []
        for sample_indices in detached_indices:
            counts = torch.bincount(
                sample_indices.reshape(-1), minlength=self.channels
            ).float()
            usage_ratios.append((counts > 0).float().mean())
            most_used.append(counts.max() / float(sample_indices.numel()))

        return {
            "similarity_shape": tuple(similarity.shape),
            "topk_indices_shape": tuple(indices.shape),
            "topk_weights_shape": tuple(weights.shape),
            "temperature": float(temperature.detach().cpu()),
            "matching_entropy": float(entropy.mean().cpu()),
            "normalized_matching_entropy": float(
                normalized_entropy.mean().cpu()
            ),
            "effective_candidate_count": float(entropy.exp().mean().cpu()),
            "candidate_usage_ratio": float(torch.stack(usage_ratios).mean().cpu()),
            "most_used_candidate_frequency": float(
                torch.stack(most_used).mean().cpu()
            ),
            "topk_weight_max": float(detached_weights.max().cpu()),
            "topk_weight_min": float(detached_weights.min().cpu()),
            "topk_weight_sum_error": float(
                (detached_weights.sum(dim=-1) - 1.0).abs().max().cpu()
            ),
        }

    def forward(self, query_feature, candidate_feature):
        if query_feature.shape != candidate_feature.shape:
            raise RuntimeError(
                "SoftCosineTopKMatching requires equal shapes, got "
                f"{tuple(query_feature.shape)} and {tuple(candidate_feature.shape)}"
            )
        if query_feature.shape[1] != self.channels:
            raise RuntimeError(
                f"Expected {self.channels} channels, got {query_feature.shape[1]}"
            )

        query = query_feature.flatten(2)
        candidate = candidate_feature.flatten(2)
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_float = query.float()
            candidate_float = candidate.float()
            query_norm = F.normalize(query_float, dim=-1)
            candidate_norm = F.normalize(candidate_float, dim=-1)
            similarity = torch.matmul(
                query_norm, candidate_norm.transpose(-2, -1)
            )
            topk_values, topk_indices = torch.topk(
                similarity, k=self.topk, dim=-1
            )
            temperature = self.log_temperature.exp().clamp(min=0.03, max=1.0)
            topk_weights = torch.softmax(topk_values / temperature, dim=-1)
            matched = self.dense_aggregate(
                candidate_float, topk_indices, topk_weights
            )

        matched = matched.reshape_as(candidate_feature).to(candidate_feature.dtype)
        if self.record_statistics:
            info = self._diagnostics(
                similarity, topk_indices, topk_weights, temperature
            )
        else:
            info = {
                "similarity_shape": tuple(similarity.shape),
                "topk_indices_shape": tuple(topk_indices.shape),
                "topk_weights_shape": tuple(topk_weights.shape),
            }
        self.last_info = info
        return matched, info


class SoftMatchingTransformation(nn.Module):
    def __init__(self, channels, topk=8, temperature=0.1):
        super().__init__()
        self.matching = SoftCosineTopKMatching(
            channels=channels, topk=topk, temperature=temperature
        )
        combined = channels * 2
        self.gate = nn.Conv2d(combined, combined, 1)
        self.value = nn.Conv2d(
            combined, combined, 3, padding=1, groups=combined, bias=False
        )
        self.project = nn.Conv2d(combined, channels, 1, bias=False)
        self.record_statistics = True
        self.last_info = None

    def forward(self, x, perception):
        self.matching.record_statistics = self.record_statistics
        matched, info = self.matching(x, perception)
        combined = torch.cat([x, matched], dim=1)
        output = self.project(
            torch.sigmoid(self.gate(combined)) * self.value(combined)
        )
        self.last_info = info
        return output, info


class LocalCorrelationGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.high_value = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
        )
        self.low_value = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
        )
        self.record_statistics = True
        self.last_info = None

    def forward(self, high_feature, low_feature):
        if high_feature.shape != low_feature.shape:
            raise RuntimeError(
                "LocalCorrelationGate requires equal shapes, got "
                f"{tuple(high_feature.shape)} and {tuple(low_feature.shape)}"
            )
        if high_feature.shape[1] != self.channels:
            raise RuntimeError(
                f"Expected {self.channels} channels, got {high_feature.shape[1]}"
            )
        relation = torch.cat(
            [
                high_feature,
                low_feature,
                high_feature * low_feature,
                torch.abs(high_feature - low_feature),
            ],
            dim=1,
        )
        gate = self.gate(relation)
        output = (
            (1.0 - gate) * self.high_value(high_feature)
            + gate * self.low_value(low_feature)
        )
        info = {"gate_shape": tuple(gate.shape)}
        if self.record_statistics:
            detached_gate = gate.detach().float()
            info.update(
                {
                    "gate_mean": float(detached_gate.mean().cpu()),
                    "gate_std": float(detached_gate.std().cpu()),
                    "gate_min": float(detached_gate.min().cpu()),
                    "gate_max": float(detached_gate.max().cpu()),
                    "low_selected_ratio": float(
                        (detached_gate > 0.5).float().mean().cpu()
                    ),
                    "high_selected_ratio": float(
                        (detached_gate <= 0.5).float().mean().cpu()
                    ),
                }
            )
        self.last_info = info
        return output, info


def build_relation_module(mode, channels, topk=8, temperature=0.1):
    if mode == "soft_cosine_topk":
        return SoftMatchingTransformation(
            channels=channels, topk=topk, temperature=temperature
        )
    if mode == "local_correlation_gate":
        return LocalCorrelationGate(channels=channels)
    raise ValueError(f"Unknown HFE relation mode: {mode}")


class AblationChannelAttention(nn.Module):
    def __init__(self, channels, num_heads, relation_module):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.relation = relation_module
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.last_attention = None
        self.last_relation_info = None

    def forward(self, high_feature, low_feature):
        batch, channels, height, width = high_feature.shape
        q, k, v = self.qkv_dw(self.qkv(high_feature)).chunk(3, dim=1)
        q, relation_info = self.relation(q, low_feature)
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


class AblationFFN(nn.Module):
    def __init__(self, channels, relation_module):
        super().__init__()
        self.project_in = nn.Conv2d(channels, channels, 1, bias=False)
        self.dwconv1 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.relation = relation_module
        self.dwconv2 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.last_relation_info = None

    def forward(self, high_feature, low_feature):
        x = self.dwconv1(self.project_in(high_feature))
        x, relation_info = self.relation(x, low_feature)
        x = F.gelu(self.dwconv2(x))
        self.last_relation_info = relation_info
        return self.project_out(x), relation_info


class AblationDecoderHFEBlock(nn.Module):
    def __init__(
        self, channels, num_heads, relation_mode, topk=8, temperature=0.1
    ):
        super().__init__()
        self.high_norm1 = LayerNorm2d(channels)
        self.high_norm2 = LayerNorm2d(channels)
        self.low_norm = LayerNorm2d(channels)
        self.attn = AblationChannelAttention(
            channels=channels,
            num_heads=num_heads,
            relation_module=build_relation_module(
                relation_mode, channels, topk, temperature
            ),
        )
        self.ffn = AblationFFN(
            channels=channels,
            relation_module=build_relation_module(
                relation_mode, channels, topk, temperature
            ),
        )

    def forward(self, high_feature, low_feature):
        low = self.low_norm(low_feature)
        attn_out, attn_info = self.attn(self.high_norm1(high_feature), low)
        high_feature = high_feature + attn_out
        ffn_out, ffn_info = self.ffn(self.high_norm2(high_feature), low)
        high_feature = high_feature + ffn_out
        return high_feature, {
            "attn_relation": attn_info,
            "ffn_relation": ffn_info,
        }


class AblationDecoderHFERefiner(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        stage,
        relation_mode,
        topk=8,
        temperature=0.1,
    ):
        super().__init__()
        self.stage = stage
        self.relation_mode = relation_mode
        self.subband_fusion = SubbandSelectiveFusion(channels, reduction=8)
        self.hfe = AblationDecoderHFEBlock(
            channels=channels,
            num_heads=num_heads,
            relation_mode=relation_mode,
            topk=topk,
            temperature=temperature,
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

    def forward(self, low_feature, base_h, base_v, base_d):
        assert_hfe_inputs(low_feature, base_h, base_v, base_d, self.stage)
        collect_statistics = not self.training and self.record_statistics
        self.hfe.attn.relation.record_statistics = collect_statistics
        self.hfe.ffn.relation.record_statistics = collect_statistics
        shared_hf = self.subband_fusion(base_h, base_v, base_d)
        refined_hf, relation_info = self.hfe(shared_hf, low_feature)
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
                "refined_shared_norm_ratio": self._ratio(refined_hf, shared_hf),
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
                statistics[f"beta_delta_{direction}_base_norm_ratio"] = self._ratio(
                    beta * delta, base
                )
                statistics[f"final_{direction}_base_norm_ratio"] = self._ratio(
                    final, base
                )
            self.last_statistics = statistics
        return coef_h, coef_v, coef_d, debug


class DWTFreqNet_SingleDecoder_HFE_Ablation(DWTFreqNet_SingleDecoder_HFE):
    """Isolated D2/D3 variants; D0 and D1 source modules remain untouched."""

    def __init__(
        self,
        config,
        hfe_ablation,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if hfe_ablation not in HFE_ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown hfe_ablation={hfe_ablation!r}; "
                f"expected one of {HFE_ABLATION_VARIANTS}"
            )
        super().__init__(
            config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
        )
        self.hfe_ablation = hfe_ablation
        self.stage_config = (
            D2_STAGE_CONFIG if hfe_ablation == "d2_softcos_all"
            else D3_STAGE_CONFIG
        )
        for stage in range(1, 5):
            cfg = self.stage_config[stage]
            setattr(
                self,
                f"decoder_hfe{stage}",
                AblationDecoderHFERefiner(
                    channels=cfg["channels"],
                    num_heads=cfg["num_heads"],
                    stage=stage,
                    relation_mode=cfg["mode"],
                    topk=cfg.get("topk", 8),
                    temperature=cfg.get("temperature", 0.1),
                ),
            )

        self.experiment_group = "experiment_d"
        self.experiment_type = "ablation"
        self.ablation_axis = "decoder_hfe_matching"
        self.model_base_commit = EXPERIMENT_D_HFE_ABLATION_BASE_COMMIT
        if hfe_ablation == "d2_softcos_all":
            self.ablation_id = "D2"
            self.model_variant = "dwtfreqnet_single_decoder_hfe_softcos"
            self.sd_variant = "sd_awgm_hfe_softcos"
            self.decoder_hfe_matching = "soft_cosine_topk_all_scales"
        else:
            self.ablation_id = "D3"
            self.model_variant = "dwtfreqnet_single_decoder_hfe_scaleaware"
            self.sd_variant = "sd_awgm_hfe_scaleaware"
            self.decoder_hfe_matching = (
                "local_correlation_gate_shallow_soft_cosine_topk_deep"
            )
        self.directional_pyramid = False
        self.second_dwt = False
        self.ldrc = False
        self.mamba = False

    @property
    def experiment_metadata(self):
        metadata = super().experiment_metadata
        metadata.update(
            {
                "experiment_group": self.experiment_group,
                "experiment_type": self.experiment_type,
                "ablation_axis": self.ablation_axis,
                "ablation_id": self.ablation_id,
                "hfe_ablation": self.hfe_ablation,
                "model_base_commit": self.model_base_commit,
                "hfe_stage_modes": {
                    str(stage): cfg["mode"]
                    for stage, cfg in self.stage_config.items()
                },
                "hfe_topk": 8,
                "hfe_initial_temperature": 0.1,
            }
        )
        return metadata


__all__ = [
    "AblationChannelAttention",
    "AblationDecoderHFEBlock",
    "AblationDecoderHFERefiner",
    "AblationFFN",
    "D2_STAGE_CONFIG",
    "D3_STAGE_CONFIG",
    "DWTFreqNet_SingleDecoder_HFE_Ablation",
    "EXPERIMENT_D_HFE_ABLATION_BASE_COMMIT",
    "HFE_ABLATION_VARIANTS",
    "LocalCorrelationGate",
    "SoftCosineTopKMatching",
    "SoftMatchingTransformation",
    "build_relation_module",
]
