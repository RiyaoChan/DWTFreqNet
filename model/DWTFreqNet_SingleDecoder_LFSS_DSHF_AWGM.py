"""Experiment F: DSHF high-frequency encoder on the Experiment E1 baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    initialize_experiment_e_model,
)


EXPERIMENT_F_BASE_COMMIT = "68ede894be748c8842427e140898f007dbe67953"
EXPERIMENT_F_VARIANTS = (
    "f1_multiscale",
    "f2_sparse",
    "f3_cross_direction",
    "f4_low_guided_full",
)
EXPERIMENT_F_LAUNCH_ORDER = (
    "f1_multiscale",
    "f4_low_guided_full",
    "f2_sparse",
    "f3_cross_direction",
)
DSHF_VARIANT_CONFIGS = {
    "f1_multiscale": {
        "use_sparse_gate": False,
        "use_cross_direction": False,
        "use_low_guidance": False,
    },
    "f2_sparse": {
        "use_sparse_gate": True,
        "use_cross_direction": False,
        "use_low_guidance": False,
    },
    "f3_cross_direction": {
        "use_sparse_gate": True,
        "use_cross_direction": True,
        "use_low_guidance": False,
    },
    "f4_low_guided_full": {
        "use_sparse_gate": True,
        "use_cross_direction": True,
        "use_low_guidance": True,
    },
}
DSHF_STAGE_CONFIG = {
    1: {"channels": 32},
    2: {"channels": 64},
    3: {"channels": 128},
    4: {"channels": 256},
}
VARIANT_METADATA = {
    "f1_multiscale": {
        "ablation_id": "F1",
        "model_variant": "dwtfreqnet_e1_dshf_multiscale",
        "sd_variant": "e1_dshf_multiscale",
    },
    "f2_sparse": {
        "ablation_id": "F2",
        "model_variant": "dwtfreqnet_e1_dshf_sparse",
        "sd_variant": "e1_dshf_sparse",
    },
    "f3_cross_direction": {
        "ablation_id": "F3",
        "model_variant": "dwtfreqnet_e1_dshf_cross_direction",
        "sd_variant": "e1_dshf_cross_direction",
    },
    "f4_low_guided_full": {
        "ablation_id": "F4",
        "model_variant": "dwtfreqnet_e1_dshf_low_guided_full",
        "sd_variant": "e1_dshf_low_guided_full",
    },
}


def assert_hf_inputs(band_h, band_v, band_d, expected_channels):
    tensors = (band_h, band_v, band_d)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise RuntimeError("DSHF expects four-dimensional NCHW tensors")
    if not (band_h.shape == band_v.shape == band_d.shape):
        raise RuntimeError(
            "DSHF H/V/D shapes differ: "
            f"H={tuple(band_h.shape)}, V={tuple(band_v.shape)}, "
            f"D={tuple(band_d.shape)}"
        )
    if band_h.shape[1] != expected_channels:
        raise RuntimeError(
            f"DSHF expected {expected_channels} channels, got {band_h.shape[1]}"
        )


def detach_nested(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_nested(item) for item in value)
    if isinstance(value, list):
        return [detach_nested(item) for item in value]
    return value


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
        return self.fuse(torch.cat([self.branch1(tensor), self.branch2(tensor)], dim=1))


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
        threshold_ratio = torch.sigmoid(
            self.threshold_predictor(mean_magnitude)
        )
        threshold = mean_magnitude * threshold_ratio
        support = torch.sigmoid(
            (magnitude - threshold) / (threshold + 1e-6)
        )
        output = feature * support
        return output, {
            "mean_magnitude": mean_magnitude,
            "threshold_ratio": threshold_ratio,
            "threshold": threshold,
            "support": support,
        }


class CrossDirectionLocalConsistencyGate(nn.Module):
    """Local H/V/D relation gate, optionally conditioned on LFSS contrast."""

    def __init__(self, use_low_guidance):
        super().__init__()
        input_channels = 5 if use_low_guidance else 4
        self.use_low_guidance = bool(use_low_guidance)
        self.gate = nn.Sequential(
            nn.Conv2d(input_channels, 12, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(12, 3, kernel_size=3, padding=1, bias=True),
        )
        self.reset_control_parameters()

    def reset_control_parameters(self):
        last_conv = self.gate[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    @staticmethod
    def normalize_energy(energy):
        normalized = energy / (
            energy.mean(dim=(2, 3), keepdim=True) + 1e-6
        )
        return normalized / (1.0 + normalized)

    def forward(
        self,
        feature_h,
        feature_v,
        feature_d,
        low_contrast=None,
    ):
        if self.use_low_guidance and low_contrast is None:
            raise RuntimeError("Low-guided cross-direction gate requires low contrast")
        if not self.use_low_guidance and low_contrast is not None:
            raise RuntimeError("Non-guided cross-direction gate received low contrast")
        energy_h = self.normalize_energy(feature_h.abs().mean(dim=1, keepdim=True))
        energy_v = self.normalize_energy(feature_v.abs().mean(dim=1, keepdim=True))
        energy_d = self.normalize_energy(feature_d.abs().mean(dim=1, keepdim=True))
        joint_energy = torch.sqrt(
            energy_h.square() + energy_v.square() + energy_d.square() + 1e-6
        )
        relation = [energy_h, energy_v, energy_d, joint_energy]
        if low_contrast is not None:
            relation.append(low_contrast)
        scales = 2.0 * torch.sigmoid(self.gate(torch.cat(relation, dim=1)))
        scale_h, scale_v, scale_d = scales.chunk(3, dim=1)
        return (
            feature_h * scale_h,
            feature_v * scale_v,
            feature_d * scale_d,
            {
                "energy_h": energy_h,
                "energy_v": energy_v,
                "energy_d": energy_d,
                "joint_energy": joint_energy,
                "scales": scales,
            },
        )


class LFSSLowFrequencyLocalContrast(nn.Module):
    """Bounded same-stage local contrast derived only from the LFSS output."""

    def __init__(self, channels):
        super().__init__()
        self.response = nn.Conv2d(channels, 1, kernel_size=1, bias=False)

    def forward(self, low_feature):
        response = self.response(low_feature)
        local_mean = F.avg_pool2d(response, kernel_size=3, stride=1, padding=1)
        contrast = (response - local_mean).abs()
        normalized = contrast / (
            contrast.mean(dim=(2, 3), keepdim=True) + 1e-6
        )
        bounded = normalized / (1.0 + normalized)
        return bounded, {
            "low_response": response,
            "low_contrast_raw": contrast,
            "low_contrast": bounded,
        }


class DSHFBlock(nn.Module):
    """Directional Sparse High-Frequency block used before stage-wise AWGM."""

    def __init__(self, channels, variant):
        super().__init__()
        if variant not in DSHF_VARIANT_CONFIGS:
            raise ValueError(f"Unknown DSHF variant: {variant}")
        config = DSHF_VARIANT_CONFIGS[variant]
        self.channels = int(channels)
        self.variant = variant
        self.use_sparse_gate = config["use_sparse_gate"]
        self.use_cross_direction = config["use_cross_direction"]
        self.use_low_guidance = config["use_low_guidance"]

        self.extract_h = DirectionalMultiScaleExtractor(channels, "H")
        self.extract_v = DirectionalMultiScaleExtractor(channels, "V")
        self.extract_d = DirectionalMultiScaleExtractor(channels, "D")
        if self.use_sparse_gate:
            self.sparse_h = AdaptiveSparseSupportGate(channels)
            self.sparse_v = AdaptiveSparseSupportGate(channels)
            self.sparse_d = AdaptiveSparseSupportGate(channels)
        if self.use_low_guidance:
            self.low_contrast = LFSSLowFrequencyLocalContrast(channels)
        if self.use_cross_direction:
            self.cross_direction = CrossDirectionLocalConsistencyGate(
                use_low_guidance=self.use_low_guidance
            )

    def reset_control_parameters(self):
        if self.use_sparse_gate:
            self.sparse_h.reset_control_parameters()
            self.sparse_v.reset_control_parameters()
            self.sparse_d.reset_control_parameters()
        if self.use_cross_direction:
            self.cross_direction.reset_control_parameters()

    def forward(self, band_h, band_v, band_d, low_feature=None):
        assert_hf_inputs(
            band_h, band_v, band_d, expected_channels=self.channels
        )
        if self.use_low_guidance:
            if low_feature is None:
                raise RuntimeError("F4 requires same-stage LFSS low feature")
            if low_feature.shape != band_h.shape:
                raise RuntimeError(
                    "F4 LFSS/HF shape mismatch: "
                    f"low={tuple(low_feature.shape)}, high={tuple(band_h.shape)}"
                )

        feature_h = self.extract_h(band_h)
        feature_v = self.extract_v(band_v)
        feature_d = self.extract_d(band_d)
        debug = {
            "multiscale_h": feature_h,
            "multiscale_v": feature_v,
            "multiscale_d": feature_d,
        }

        if self.use_sparse_gate:
            feature_h, sparse_h = self.sparse_h(feature_h)
            feature_v, sparse_v = self.sparse_v(feature_v)
            feature_d, sparse_d = self.sparse_d(feature_d)
            debug.update({
                "sparse_h": sparse_h,
                "sparse_v": sparse_v,
                "sparse_d": sparse_d,
                "sparse_feature_h": feature_h,
                "sparse_feature_v": feature_v,
                "sparse_feature_d": feature_d,
            })

        low_contrast = None
        if self.use_low_guidance:
            low_contrast, low_info = self.low_contrast(low_feature)
            debug["low_guidance"] = low_info

        if self.use_cross_direction:
            feature_h, feature_v, feature_d, cross_info = self.cross_direction(
                feature_h,
                feature_v,
                feature_d,
                low_contrast=low_contrast,
            )
            debug.update({
                "cross_direction": cross_info,
                "cross_feature_h": feature_h,
                "cross_feature_v": feature_v,
                "cross_feature_d": feature_d,
            })

        output_h = band_h + feature_h
        output_v = band_v + feature_v
        output_d = band_d + feature_d
        debug.update({
            "residual_h": feature_h,
            "residual_v": feature_v,
            "residual_d": feature_d,
            "output_h": output_h,
            "output_v": output_v,
            "output_d": output_d,
        })
        return output_h, output_v, output_d, debug


def initialize_experiment_f_model(model, baseline_init_fn):
    """Protect LFSS initialization and restore all zero-initialized DSHF controls."""

    initialize_experiment_e_model(model, baseline_init_fn)
    for stage in range(1, 5):
        getattr(model, f"dir_encoder{stage}").reset_control_parameters()


class DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM(
    DWTFreqNet_SingleDecoder_LFSS_AWGM
):
    """Experiment E1 with only its pre-AWGM high-frequency encoder replaced."""

    def __init__(
        self,
        config,
        hf_variant,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if hf_variant not in EXPERIMENT_F_VARIANTS:
            raise ValueError(f"Unknown Experiment F variant: {hf_variant}")
        super().__init__(
            config=config,
            encoder_variant="e1_lfss_resblock",
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
        )
        self.hf_variant = hf_variant
        for stage in range(1, 5):
            setattr(
                self,
                f"dir_encoder{stage}",
                DSHFBlock(
                    channels=DSHF_STAGE_CONFIG[stage]["channels"],
                    variant=hf_variant,
                ),
            )

        config_flags = DSHF_VARIANT_CONFIGS[hf_variant]
        metadata = VARIANT_METADATA[hf_variant]
        self.experiment_group = "experiment_f"
        self.experiment_type = "encoder_high_frequency_ablation"
        self.ablation_axis = "pre_awgm_high_frequency_extractor"
        self.base_low_frequency_variant = "experiment_e_e1_lfss_resblock"
        self.encoder_lfss = True
        self.post_awgm_encoder = "original_res_block"
        self.high_frequency_encoder = "dshf_block"
        self.high_frequency_decoder_source = "raw_haar_aligned"
        self.dshf_multiscale = True
        self.dshf_sparse_gate = config_flags["use_sparse_gate"]
        self.dshf_cross_direction = config_flags["use_cross_direction"]
        self.dshf_low_guidance = config_flags["use_low_guidance"]
        self.decoder_hfe = False
        self.directional_pyramid = False
        self.second_dwt = False
        self.ldrc = False
        self.coefficient_mode = "aligned_raw"
        self.ablation_id = metadata["ablation_id"]
        self.model_variant = metadata["model_variant"]
        self.sd_variant = metadata["sd_variant"]
        self.model_base_commit = EXPERIMENT_F_BASE_COMMIT
        if self.dshf_low_guidance:
            self.dshf_low_guidance_source = "same_stage_lfss_output"
        self._experiment_f_debug = {}

    @property
    def experiment_metadata(self):
        metadata = dict(super().experiment_metadata)
        metadata.update({
            "experiment_group": self.experiment_group,
            "experiment_type": self.experiment_type,
            "ablation_axis": self.ablation_axis,
            "ablation_id": self.ablation_id,
            "hf_variant": self.hf_variant,
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "model_base_commit": self.model_base_commit,
            "base_low_frequency_variant": self.base_low_frequency_variant,
            "encoder_lfss": self.encoder_lfss,
            "post_awgm_encoder": self.post_awgm_encoder,
            "high_frequency_encoder": self.high_frequency_encoder,
            "high_frequency_decoder_source": self.high_frequency_decoder_source,
            "dshf_multiscale": self.dshf_multiscale,
            "dshf_sparse_gate": self.dshf_sparse_gate,
            "dshf_cross_direction": self.dshf_cross_direction,
            "dshf_low_guidance": self.dshf_low_guidance,
            "dshf_low_guidance_source": getattr(
                self, "dshf_low_guidance_source", None
            ),
            "dshf_stage_config": {
                str(stage): dict(config)
                for stage, config in DSHF_STAGE_CONFIG.items()
            },
            "decoder_hfe": self.decoder_hfe,
            "directional_pyramid": self.directional_pyramid,
            "second_dwt": self.second_dwt,
            "ldrc": self.ldrc,
            "coefficient_mode": self.coefficient_mode,
        })
        return metadata

    def _encode_stage(self, stage, tensor):
        band_a, band_h, band_v, band_d = self._dwt(tensor)
        refined_a = self.lfss_blocks[str(stage)](band_a)
        feature_h, feature_v, feature_d, hf_debug = getattr(
            self, f"dir_encoder{stage}"
        )(
            band_h,
            band_v,
            band_d,
            low_feature=(
                refined_a if self.hf_variant == "f4_low_guided_full" else None
            ),
        )
        guided_a = self._apply_stage_awgm(
            stage,
            refined_a,
            feature_h,
            feature_v,
            feature_d,
        )
        encoded = getattr(self, f"local_encoder{stage}")(guided_a)
        if self.debug_tensors:
            self._lfss_debug[stage] = {
                "lfss": refined_a.detach(),
                "encoded": encoded.detach(),
            }
            self._experiment_f_debug[stage] = {
                "raw_ll": band_a.detach(),
                "lfss_ll": refined_a.detach(),
                "raw_h": band_h.detach(),
                "raw_v": band_v.detach(),
                "raw_d": band_d.detach(),
                "dshf": detach_nested(hf_debug),
                "guided_ll": guided_a.detach(),
                "encoded": encoded.detach(),
            }
        return (
            encoded,
            (band_a, band_h, band_v, band_d),
            {"H": feature_h, "V": feature_v, "D": feature_d},
            guided_a,
        )

    def forward(self, tensor):
        self._experiment_f_debug = {}
        output = super().forward(tensor)
        if self.debug_tensors:
            self.last_debug["experiment_f"] = self._experiment_f_debug
        return output


__all__ = [
    "AdaptiveSparseSupportGate",
    "CrossDirectionLocalConsistencyGate",
    "DSHFBlock",
    "DSHF_STAGE_CONFIG",
    "DSHF_VARIANT_CONFIGS",
    "DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM",
    "DirectionalMultiScaleExtractor",
    "EXPERIMENT_F_BASE_COMMIT",
    "EXPERIMENT_F_LAUNCH_ORDER",
    "EXPERIMENT_F_VARIANTS",
    "LFSSLowFrequencyLocalContrast",
    "VARIANT_METADATA",
    "assert_hf_inputs",
    "detach_nested",
    "initialize_experiment_f_model",
]
