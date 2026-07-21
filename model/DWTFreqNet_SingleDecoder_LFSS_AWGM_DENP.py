"""Experiment J: dual-evidence noise-calibrated decoder purification."""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    initialize_experiment_e_model,
)
from model.decoder_denp import DENP_BANDS, DENPPurifier


EXPERIMENT_J_BASE_COMMIT = "68ede894be748c8842427e140898f007dbe67953"
EXPERIMENT_J_DEVELOPMENT_BASE_COMMIT = "ba96c1f119ea1cc4e8f0fdf2dc3818291ff48449"
EXPERIMENT_J_PHASE1_REFERENCE_COMMIT = "356ac5611b4797452d2aeedd954993948e06750d"
EXPERIMENT_J_VARIANTS = (
    "j0_e1_passthrough",
    "j1_bandwise_noise_calibrated",
    "j2_rawll_compactness",
    "j2_decoder_compactness",
    "j3_dual_evidence_fixed",
    "j3_dual_evidence_reliability",
)
DENP_STAGE_CHANNELS = {1: 64, 2: 128, 3: 256, 4: 256}
DENP_VARIANT_CONFIGS = {
    "j0_e1_passthrough": {
        "ablation_id": "J0",
        "noise_calibrated": False,
        "raw_compactness": False,
        "decoder_compactness": False,
        "dual_evidence": False,
        "learnable_reliability": False,
    },
    "j1_bandwise_noise_calibrated": {
        "ablation_id": "J1",
        "noise_calibrated": True,
        "raw_compactness": False,
        "decoder_compactness": False,
        "dual_evidence": False,
        "learnable_reliability": False,
    },
    "j2_rawll_compactness": {
        "ablation_id": "J2-R",
        "noise_calibrated": True,
        "raw_compactness": True,
        "decoder_compactness": False,
        "dual_evidence": False,
        "learnable_reliability": False,
    },
    "j2_decoder_compactness": {
        "ablation_id": "J2-D",
        "noise_calibrated": True,
        "raw_compactness": False,
        "decoder_compactness": True,
        "dual_evidence": False,
        "learnable_reliability": False,
    },
    "j3_dual_evidence_fixed": {
        "ablation_id": "J3-F",
        "noise_calibrated": True,
        "raw_compactness": True,
        "decoder_compactness": True,
        "dual_evidence": True,
        "learnable_reliability": False,
    },
    "j3_dual_evidence_reliability": {
        "ablation_id": "J3-R",
        "noise_calibrated": True,
        "raw_compactness": True,
        "decoder_compactness": True,
        "dual_evidence": True,
        "learnable_reliability": True,
    },
}


def _detach_nested(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_nested(item) for item in value)
    if isinstance(value, list):
        return [_detach_nested(item) for item in value]
    return value


def initialize_experiment_j_model(model, baseline_init_fn):
    """Initialize the fixed E1 body and restore every bounded DENP control."""

    initialize_experiment_e_model(model, baseline_init_fn)
    if getattr(model, "use_denp", False):
        for stage in range(1, 5):
            getattr(model, f"decoder_denp{stage}").reset_control_parameters()


class DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP(
    DWTFreqNet_SingleDecoder_LFSS_AWGM
):
    """Fixed E1 with DENP applied once to aligned raw H/V/D before IDWT."""

    def __init__(
        self,
        config,
        denp_variant,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if denp_variant not in EXPERIMENT_J_VARIANTS:
            raise ValueError(f"Unknown Experiment J variant: {denp_variant}")
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
        variant = DENP_VARIANT_CONFIGS[denp_variant]
        self.denp_variant = denp_variant
        self.ablation_id = variant["ablation_id"]
        self.use_denp = bool(variant["noise_calibrated"])
        self.denp_raw_compactness = bool(variant["raw_compactness"])
        self.denp_decoder_compactness = bool(variant["decoder_compactness"])
        self.denp_dual_evidence = bool(variant["dual_evidence"])
        self.denp_learnable_reliability = bool(variant["learnable_reliability"])
        if self.use_denp:
            for stage, channels in DENP_STAGE_CHANNELS.items():
                setattr(
                    self,
                    f"decoder_denp{stage}",
                    DENPPurifier(high_channels=channels, variant=denp_variant),
                )

        self.experiment_group = "experiment_j"
        self.experiment_type = "dual_evidence_noise_calibrated_purification"
        self.ablation_axis = "noise_scale_and_low_frequency_protection_evidence"
        self.base_encoder = "experiment_e_e1_lfss_resblock_awgm"
        self.model_base_commit = EXPERIMENT_J_BASE_COMMIT
        self.development_base_commit = EXPERIMENT_J_DEVELOPMENT_BASE_COMMIT
        self.phase1_reference_commit = EXPERIMENT_J_PHASE1_REFERENCE_COMMIT
        self.model_variant = f"dwtfreqnet_e1_{denp_variant}"
        self.sd_variant = denp_variant
        self.encoder_lfss = True
        self.encoder_awgm = True
        self.encoder_dshf = False
        self.decoder_dshf = False
        self.decoder_lfp = False
        self.decoder_denp = self.use_denp
        self.denp_high_source = "once_aligned_raw_haar_hvd"
        self.denp_raw_low_source = "same_dwt_raw_ll" if self.denp_raw_compactness else "none"
        self.denp_decoder_low_source = (
            "current_decoder_low" if self.denp_decoder_compactness else "none"
        )
        self.second_dwt = False
        self.directional_pyramid = False
        self.ldrc = False
        self.channel_matching = False
        self.coefficient_mode = (
            "denp_purified_once_aligned_raw_hvd" if self.use_denp else "aligned_raw"
        )
        self.last_denp_statistics = None

    @property
    def experiment_metadata(self):
        metadata = dict(super().experiment_metadata)
        metadata.update({
            "experiment_group": self.experiment_group,
            "experiment_type": self.experiment_type,
            "ablation_axis": self.ablation_axis,
            "ablation_id": self.ablation_id,
            "denp_variant": self.denp_variant,
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "model_base_commit": self.model_base_commit,
            "development_base_commit": self.development_base_commit,
            "phase1_reference_commit": self.phase1_reference_commit,
            "base_encoder": self.base_encoder,
            "encoder_lfss": self.encoder_lfss,
            "encoder_awgm": self.encoder_awgm,
            "encoder_dshf": self.encoder_dshf,
            "decoder_dshf": self.decoder_dshf,
            "decoder_lfp": self.decoder_lfp,
            "decoder_denp": self.decoder_denp,
            "decoder_denp_all_stages": self.use_denp,
            "decoder_denp_stage_channels": dict(DENP_STAGE_CHANNELS),
            "denp_high_source": self.denp_high_source,
            "denp_raw_low_source": self.denp_raw_low_source,
            "denp_decoder_low_source": self.denp_decoder_low_source,
            "denp_noise_estimator": "1.4826_channel_median_mad_detached_fp32",
            "denp_lambda_range": [0.5, 3.0],
            "denp_lambda_init": 1.0,
            "denp_noise_kappa": 0.15,
            "denp_gaussian_kernel": 3,
            "denp_gaussian_sigma_range": [0.5, 2.0],
            "denp_gaussian_sigma_init": 1.0,
            "denp_raw_compactness": self.denp_raw_compactness,
            "denp_decoder_compactness": self.denp_decoder_compactness,
            "denp_dual_evidence": self.denp_dual_evidence,
            "denp_learnable_reliability": self.denp_learnable_reliability,
            "denp_compactness_kernels": [3, 7],
            "denp_compactness_slope_range": [1.0, 20.0],
            "denp_compactness_slope_init": 5.0,
            "denp_compactness_threshold_range": [-0.5, 0.5],
            "denp_compactness_threshold_init": 0.0,
            "denp_gamma_range": [0.5, 2.0] if self.denp_learnable_reliability else None,
            "denp_gamma_init": 1.0 if self.denp_learnable_reliability else None,
            "second_dwt": self.second_dwt,
            "directional_pyramid": self.directional_pyramid,
            "ldrc": self.ldrc,
            "channel_matching": self.channel_matching,
            "coefficient_mode": self.coefficient_mode,
        })
        return metadata

    def _align_stage_high(self, stage, raw_bands):
        _, raw_h, raw_v, raw_d = raw_bands[stage]
        return tuple(
            getattr(self, f"align_{direction}{stage}")(raw)
            for direction, raw in zip(DENP_BANDS, (raw_h, raw_v, raw_d))
        )

    def _purify_stage_high(self, stage, raw_bands, decoder_low):
        aligned = self._align_stage_high(stage, raw_bands)
        if not self.use_denp:
            return (*aligned, {
                "raw_low": raw_bands[stage][0],
                "decoder_low": decoder_low,
                "raw_protection": None,
                "decoder_protection": None,
                "raw_compactness": None,
                "decoder_compactness": None,
                "bands": {
                    name: {
                        "aligned": band,
                        "noise_confidence": None,
                        "noise": None,
                        "gaussian": None,
                        "mask": None,
                        "purified": band,
                        "gamma_raw": None,
                        "gamma_decoder": None,
                    }
                    for name, band in zip(DENP_BANDS, aligned)
                },
            })
        return getattr(self, f"decoder_denp{stage}")(
            raw_bands[stage][0], decoder_low, *aligned
        )

    def _collect_denp_statistics(self, stage_debug):
        statistics = OrderedDict()
        for stage, debug in stage_debug.items():
            if debug["raw_protection"] is not None:
                protection = debug["raw_protection"].detach().float()
                compactness = debug["raw_compactness"]["compactness"].detach().float()
                statistics.update({
                    f"stage{stage}_P_R_mean": float(protection.mean().cpu()),
                    f"stage{stage}_C_R_mean": float(compactness.mean().cpu()),
                    f"stage{stage}_raw_slope": float(
                        debug["raw_compactness"]["slope"].detach().float().cpu()
                    ),
                    f"stage{stage}_raw_threshold": float(
                        debug["raw_compactness"]["threshold"].detach().float().cpu()
                    ),
                })
            if debug["decoder_protection"] is not None:
                protection = debug["decoder_protection"].detach().float()
                compactness = debug["decoder_compactness"]["compactness"].detach().float()
                statistics.update({
                    f"stage{stage}_P_D_mean": float(protection.mean().cpu()),
                    f"stage{stage}_C_D_mean": float(compactness.mean().cpu()),
                    f"stage{stage}_decoder_slope": float(
                        debug["decoder_compactness"]["slope"].detach().float().cpu()
                    ),
                    f"stage{stage}_decoder_threshold": float(
                        debug["decoder_compactness"]["threshold"].detach().float().cpu()
                    ),
                })
            for band in DENP_BANDS:
                item = debug["bands"][band]
                prefix = f"stage{stage}_{band}"
                statistics[f"{prefix}_aligned_norm"] = self._norm(item["aligned"])
                statistics[f"{prefix}_purified_norm"] = self._norm(item["purified"])
                if item["noise_confidence"] is None:
                    continue
                noise = item["noise_confidence"].detach().float()
                mask = item["mask"].detach().float()
                statistics.update({
                    f"{prefix}_sigma_hat": float(item["noise"]["sigma_hat"].mean().cpu()),
                    f"{prefix}_lambda": float(item["noise"]["lambda"].detach().float().cpu()),
                    f"{prefix}_tau": float(item["noise"]["tau"].detach().float().mean().cpu()),
                    f"{prefix}_gaussian_sigma": float(
                        getattr(self, f"decoder_denp{stage}").gaussians[band].sigma.detach().cpu()
                    ),
                    f"{prefix}_N_mean": float(noise.mean().cpu()),
                    f"{prefix}_N_std": float(noise.std(unbiased=False).cpu()),
                    f"{prefix}_N_gt_0_5": float((noise > 0.5).float().mean().cpu()),
                    f"{prefix}_M_mean": float(mask.mean().cpu()),
                    f"{prefix}_change_ratio": float(
                        (item["purified"] - item["aligned"]).detach().float().norm().cpu()
                        / (item["aligned"].detach().float().norm().cpu() + 1e-8)
                    ),
                })
                if item["gamma_raw"] is not None:
                    statistics[f"{prefix}_gamma_R"] = float(
                        item["gamma_raw"].detach().float().cpu()
                    )
                if item["gamma_decoder"] is not None:
                    statistics[f"{prefix}_gamma_D"] = float(
                        item["gamma_decoder"].detach().float().cpu()
                    )
        return dict(statistics)

    def forward(self, x):
        self._lfss_debug = {}
        self.last_transform_counts = {"dwt": 0, "idwt": 0}
        x0 = self.stem(x)
        encoded, raw_bands, directional, guided = {}, {}, {}, {}
        current = x0
        for stage in range(1, 5):
            encoded[stage], raw_bands[stage], directional[stage], guided[stage] = (
                self._encode_stage(stage, current)
            )
            current = encoded[stage]

        stage_debug = {}
        h4, v4, d4, stage_debug[4] = self._purify_stage_high(4, raw_bands, encoded[4])
        u3 = self._idwt(encoded[4], h4, v4, d4)
        l3 = self.decoder_fuse3(torch.cat([u3, encoded[3]], dim=1))
        h3, v3, d3, stage_debug[3] = self._purify_stage_high(3, raw_bands, l3)
        u2 = self._idwt(l3, h3, v3, d3)
        l2 = self.decoder_fuse2(torch.cat([u2, encoded[2]], dim=1))
        h2, v2, d2, stage_debug[2] = self._purify_stage_high(2, raw_bands, l2)
        u1 = self._idwt(l2, h2, v2, d2)
        l1 = self.decoder_fuse1(torch.cat([u1, encoded[1]], dim=1))
        h1, v1, d1, stage_debug[1] = self._purify_stage_high(1, raw_bands, l1)
        u0 = self._idwt(l1, h1, v1, d1)
        l0 = self.decoder_fuse0(torch.cat([u0, x0], dim=1))
        out = self.out_head(l0)

        self.last_shapes = {
            "X0": tuple(x0.shape),
            **{f"E{stage}": tuple(encoded[stage].shape) for stage in range(1, 5)},
            "U3": tuple(u3.shape), "L3": tuple(l3.shape),
            "U2": tuple(u2.shape), "L2": tuple(l2.shape),
            "U1": tuple(u1.shape), "L1": tuple(l1.shape),
            "U0": tuple(u0.shape), "L0": tuple(l0.shape),
        }

        if not self.training and self.record_statistics:
            statistics = self._collect_denp_statistics(stage_debug)
            awgm_stats = [
                getattr(self, f"stage_awgm{stage}").last_statistics
                for stage in range(1, 5)
            ]
            if all(item is not None for item in awgm_stats):
                for key in awgm_stats[0]:
                    statistics[f"stage_awgm_{key}"] = sum(
                        item[key] for item in awgm_stats
                    ) / len(awgm_stats)
            self.last_denp_statistics = statistics
            self.last_sd_statistics = statistics

        if self.debug_tensors:
            coefficient_debug = {
                (stage, band): {
                    "aligned": stage_debug[stage]["bands"][band]["aligned"].detach(),
                    "delta": None,
                    "coefficient": stage_debug[stage]["bands"][band]["purified"].detach(),
                }
                for stage in range(1, 5)
                for band in DENP_BANDS
            }
            self.last_debug = {
                "A": {stage: raw_bands[stage][0].detach() for stage in range(1, 5)},
                "A_lfss": {stage: self._lfss_debug[stage]["lfss"] for stage in range(1, 5)},
                "A_guided": {stage: guided[stage].detach() for stage in range(1, 5)},
                "E": {stage: self._lfss_debug[stage]["encoded"] for stage in range(1, 5)},
                "U": {3: u3.detach(), 2: u2.detach(), 1: u1.detach(), 0: u0.detach()},
                "L": {3: l3.detach(), 2: l2.detach(), 1: l1.detach(), 0: l0.detach()},
                "coefficients": coefficient_debug,
                "decoder_denp": _detach_nested(stage_debug),
                "AWGM_gate": {
                    stage: getattr(self, f"stage_awgm{stage}").last_attention_map
                    for stage in range(1, 5)
                },
                "AWGM_direction_weights": {
                    stage: getattr(self, f"stage_awgm{stage}").last_direction_weights
                    for stage in range(1, 5)
                },
            }

        if not self.deepsuper:
            return torch.sigmoid(out)
        target_size = x.shape[-2:]
        gt5 = F.interpolate(self.gt_conv5(encoded[4]), target_size, mode="bilinear", align_corners=False)
        gt4 = F.interpolate(self.gt_conv4(l3), target_size, mode="bilinear", align_corners=False)
        gt3 = F.interpolate(self.gt_conv3(l2), target_size, mode="bilinear", align_corners=False)
        gt2 = F.interpolate(self.gt_conv2(l1), target_size, mode="bilinear", align_corners=False)
        d0 = self.outconv(torch.cat([gt2, gt3, gt4, gt5, out], dim=1))
        if self.mode == "train":
            return tuple(torch.sigmoid(tensor) for tensor in (gt5, gt4, gt3, gt2, d0, out))
        return torch.sigmoid(out)


__all__ = [
    "DENP_STAGE_CHANNELS",
    "DENP_VARIANT_CONFIGS",
    "DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP",
    "EXPERIMENT_J_BASE_COMMIT",
    "EXPERIMENT_J_DEVELOPMENT_BASE_COMMIT",
    "EXPERIMENT_J_PHASE1_REFERENCE_COMMIT",
    "EXPERIMENT_J_VARIANTS",
    "initialize_experiment_j_model",
]
