"""Experiment K v2: E1-centered dose-calibrated purification."""

import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    initialize_experiment_e_model,
)
from model.decoder_k_dose import DoseCalibratedBandPurifier, K_BANDS


EXPERIMENT_K_J_BASE_COMMIT = "048b01aa0f9c13edfb75a3081dde003e4e9aef4b"
EXPERIMENT_K_PHASE1_REFERENCE_COMMIT = "e7980e064acc4eca06237a23914adc77cabf94fe"
EXPERIMENT_K_VARIANTS = (
    "k0_e1_passthrough",
    "k1_j1_full_dose",
    "k2_dose_calibrated",
    "k3_gr_raw_all",
    "k4_gr_lfss_s123",
    "k5_gr_guided_s123",
    "k6_gr_selected_hybrid",
)
K_STAGE_CHANNELS = {1: 64, 2: 128, 3: 256, 4: 256}
K_VARIANT_CONFIGS = {
    "k0_e1_passthrough": {"id": "K0", "purify": False, "learn_alpha": False},
    "k1_j1_full_dose": {"id": "K1", "purify": True, "learn_alpha": False},
    "k2_dose_calibrated": {"id": "K2", "purify": True, "learn_alpha": True},
    "k3_gr_raw_all": {"id": "K3", "purify": True, "learn_alpha": True},
    "k4_gr_lfss_s123": {"id": "K4", "purify": True, "learn_alpha": True},
    "k5_gr_guided_s123": {"id": "K5", "purify": True, "learn_alpha": True},
    "k6_gr_selected_hybrid": {"id": "K6", "purify": True, "learn_alpha": True},
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


def _source_map(variant, decision_path):
    if variant in ("k0_e1_passthrough", "k1_j1_full_dose", "k2_dose_calibrated"):
        return {stage: None for stage in range(1, 5)}, None
    if variant == "k3_gr_raw_all":
        return {stage: "raw_ll" for stage in range(1, 5)}, None
    if variant == "k4_gr_lfss_s123":
        return {1: "lfss_ll", 2: "lfss_ll", 3: "lfss_ll", 4: None}, None
    if variant == "k5_gr_guided_s123":
        return {1: "guided_ll", 2: "guided_ll", 3: "guided_ll", 4: None}, None
    if not decision_path:
        raise ValueError("K6 requires --decision-json with a locked train-discovery decision")
    path = Path(decision_path).resolve()
    decision = json.loads(path.read_text(encoding="utf-8"))
    if not decision.get("discovery_complete"):
        raise RuntimeError("K6 decision must have discovery_complete=true")
    selected = decision.get("selected_sources")
    if selected:
        mapping = {
            stage: selected.get(str(stage), selected.get(stage)) for stage in range(1, 5)
        }
    else:
        preferred = decision.get("preferred_source")
        aliases = {"raw_ll": "raw_ll", "lfss_ll": "lfss_ll", "guided_ll": "guided_ll"}
        if preferred not in aliases:
            raise RuntimeError("K6 decision has no valid selected_sources/preferred_source")
        active = {int(stage) for stage in decision.get("active_stages", [])}
        mapping = {stage: aliases[preferred] if stage in active else None for stage in range(1, 5)}
    valid = {None, "raw_ll", "lfss_ll", "guided_ll"}
    if any(source not in valid for source in mapping.values()):
        raise RuntimeError(f"K6 decision contains invalid source mapping: {mapping}")
    return mapping, decision


def initialize_experiment_k_model(model, baseline_init_fn):
    """Initialize the fixed E1 body and reset all bounded K controls."""

    initialize_experiment_e_model(model, baseline_init_fn)
    if getattr(model, "use_k_purification", False):
        for stage in range(1, 5):
            getattr(model, f"decoder_k{stage}").reset_control_parameters()


class DWTFreqNet_SingleDecoder_LFSS_AWGM_K(DWTFreqNet_SingleDecoder_LFSS_AWGM):
    """Fixed E1 with one dose-calibrated operation before each decoder IDWT."""

    def __init__(
        self,
        config,
        k_variant,
        decision_path="",
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if k_variant not in EXPERIMENT_K_VARIANTS:
            raise ValueError(f"Unknown Experiment K variant: {k_variant}")
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
        config_k = K_VARIANT_CONFIGS[k_variant]
        source_map, decision = _source_map(k_variant, decision_path)
        self.k_variant = k_variant
        self.ablation_id = config_k["id"]
        self.k_source_map = source_map
        self.k_decision = decision
        self.k_decision_path = str(Path(decision_path).resolve()) if decision_path else None
        self.use_k_purification = bool(config_k["purify"])
        self.k_learnable_alpha = bool(config_k["learn_alpha"])
        if self.use_k_purification:
            for stage, channels in K_STAGE_CHANNELS.items():
                setattr(self, f"decoder_k{stage}", DoseCalibratedBandPurifier(
                    channels=channels,
                    alpha_init=0.05,
                    protection_enabled=source_map[stage] is not None,
                    rho_init=0.05,
                    learnable_alpha=self.k_learnable_alpha,
                    fixed_alpha=1.0 if k_variant == "k1_j1_full_dose" else None,
                ))

        self.experiment_group = "experiment_k"
        self.experiment_type = "e1_centered_dose_calibrated_purification"
        self.ablation_axis = "gaussian_dose_and_read_only_compactness_protection"
        self.model_variant = f"dwtfreqnet_e1_{k_variant}"
        self.sd_variant = k_variant
        self.model_base_commit = EXPERIMENT_K_J_BASE_COMMIT
        self.phase1_reference_commit = EXPERIMENT_K_PHASE1_REFERENCE_COMMIT
        self.encoder_lfss = True
        self.encoder_awgm = True
        self.second_dwt = False
        self.directional_pyramid = False
        self.ldrc = False
        self.encoder_dshf = False
        self.decoder_dshf = False
        self.decoder_lfp = False
        self.decoder_denp = self.use_k_purification
        self.coefficient_mode = (
            "dose_calibrated_once_aligned_raw_hvd"
            if self.use_k_purification else "aligned_raw"
        )
        self.alpha_override = None
        self.rho_override = None
        self.spatial_dose_overrides = {}
        self.last_k_statistics = None
        self._k_low_sources = {}

    @property
    def experiment_metadata(self):
        metadata = dict(super().experiment_metadata)
        metadata.update({
            "experiment_group": self.experiment_group,
            "experiment_type": self.experiment_type,
            "ablation_axis": self.ablation_axis,
            "ablation_id": self.ablation_id,
            "k_variant": self.k_variant,
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "model_base_commit": self.model_base_commit,
            "phase1_reference_commit": self.phase1_reference_commit,
            "decision_path": self.k_decision_path,
            "source_map": {str(stage): source for stage, source in self.k_source_map.items()},
            "alpha_range": [0.0, 1.0] if self.k_learnable_alpha else None,
            "alpha_init": 0.05 if self.k_learnable_alpha else 1.0,
            "alpha_parameters": 12 if self.k_learnable_alpha else 0,
            "rho_range": [0.0, 0.5],
            "rho_init": 0.05,
            "compactness_operator": "gaussian_radial_32_rmax2",
            "compactness_source_detached": True,
            "noise_estimator": "J1_global_post_align_channel_median_MAD",
            "high_source": "once_aligned_signed_raw_haar_hvd",
            "second_dwt": False,
            "extra_idwt": False,
            "dwt_calls": 4,
            "idwt_calls": 4,
        })
        return metadata

    def _encode_stage(self, stage, tensor):
        band_a, band_h, band_v, band_d = self._dwt(tensor)
        feature_h, feature_v, feature_d = getattr(self, f"dir_encoder{stage}")(
            band_h, band_v, band_d
        )
        refined_a = self.lfss_blocks[str(stage)](band_a)
        guided_a = self._apply_stage_awgm(
            stage, refined_a, feature_h, feature_v, feature_d
        )
        encoded = getattr(self, f"local_encoder{stage}")(guided_a)
        self._k_low_sources[stage] = {
            "raw_ll": band_a,
            "lfss_ll": refined_a,
            "guided_ll": guided_a,
            "encoded": encoded,
        }
        if self.debug_tensors:
            self._lfss_debug[stage] = {
                "lfss": refined_a.detach(), "encoded": encoded.detach()
            }
        return (
            encoded,
            (band_a, band_h, band_v, band_d),
            {"H": feature_h, "V": feature_v, "D": feature_d},
            guided_a,
        )

    def _align_stage_high(self, stage, raw_bands):
        _, raw_h, raw_v, raw_d = raw_bands[stage]
        return tuple(
            getattr(self, f"align_{direction}{stage}")(raw)
            for direction, raw in zip(K_BANDS, (raw_h, raw_v, raw_d))
        )

    def _purify_stage_high(self, stage, raw_bands, decoder_low):
        aligned = self._align_stage_high(stage, raw_bands)
        source_name = self.k_source_map[stage]
        if source_name == "decoder_low":
            prior = decoder_low
        else:
            prior = self._k_low_sources[stage].get(source_name) if source_name else None
        if source_name is not None and prior is None:
            raise RuntimeError(
                f"Experiment K source {source_name!r} is unavailable at stage {stage}"
            )
        if not self.use_k_purification:
            return (*aligned, {
                "prior_source": None,
                "prior_low": None,
                "protection": None,
                "compactness": None,
                "rho": None,
                "bands": {
                    name: {
                        "aligned": band,
                        "noise_confidence": None,
                        "noise": None,
                        "gaussian": None,
                        "gaussian_residual": None,
                        "alpha": None,
                        "dose": None,
                        "purified": band,
                    }
                    for name, band in zip(K_BANDS, aligned)
                },
            })
        alpha_override = (
            self.alpha_override.get(stage) if isinstance(self.alpha_override, dict)
            else self.alpha_override
        )
        rho_override = (
            self.rho_override.get(stage) if isinstance(self.rho_override, dict)
            else self.rho_override
        )
        result = getattr(self, f"decoder_k{stage}")(
            *aligned,
            prior_low=prior,
            alpha_override=alpha_override,
            rho_override=rho_override,
            spatial_dose_override=self.spatial_dose_overrides.get(stage),
        )
        result[-1]["prior_source"] = source_name
        return result

    @staticmethod
    def _norm(tensor):
        return float(tensor.detach().float().norm().cpu())

    def _collect_k_statistics(self, stage_debug):
        statistics = OrderedDict()
        for stage, debug in stage_debug.items():
            if debug["protection"] is not None:
                protection = debug["protection"].detach().float()
                ratio = debug["compactness"]["ratio"].detach().float()
                statistics.update({
                    f"stage{stage}_P_mean": float(protection.mean().cpu()),
                    f"stage{stage}_R_GR_mean": float(ratio.mean().cpu()),
                    f"stage{stage}_rho": float(debug["rho"].detach().float().cpu()),
                })
            for band in K_BANDS:
                item = debug["bands"][band]
                prefix = f"stage{stage}_{band}"
                statistics[f"{prefix}_aligned_norm"] = self._norm(item["aligned"])
                statistics[f"{prefix}_purified_norm"] = self._norm(item["purified"])
                if item["noise_confidence"] is None:
                    continue
                statistics.update({
                    f"{prefix}_alpha": float(item["alpha"].detach().float().cpu()),
                    f"{prefix}_lambda": float(item["noise"]["lambda"].detach().float().cpu()),
                    f"{prefix}_sigma_hat": float(item["noise"]["sigma_hat"].mean().cpu()),
                    f"{prefix}_tau": float(item["noise"]["tau"].mean().cpu()),
                    f"{prefix}_N_mean": float(item["noise_confidence"].detach().float().mean().cpu()),
                    f"{prefix}_gaussian_sigma": float(
                        getattr(self, f"decoder_k{stage}").gaussians[band].sigma.detach().cpu()
                    ),
                    f"{prefix}_dose_mean": float(item["dose"].detach().float().mean().cpu()),
                    f"{prefix}_change_ratio": float(
                        (item["purified"] - item["aligned"]).detach().float().norm().cpu()
                        / (item["aligned"].detach().float().norm().cpu() + 1e-8)
                    ),
                })
        return dict(statistics)

    def forward(self, x):
        self._lfss_debug = {}
        self._k_low_sources = {}
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
        h4, v4, d4, stage_debug[4] = self._purify_stage_high(
            4, raw_bands, encoded[4]
        )
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
            statistics = self._collect_k_statistics(stage_debug)
            awgm_stats = [
                getattr(self, f"stage_awgm{stage}").last_statistics for stage in range(1, 5)
            ]
            if all(item is not None for item in awgm_stats):
                for key in awgm_stats[0]:
                    statistics[f"stage_awgm_{key}"] = sum(
                        item[key] for item in awgm_stats
                    ) / len(awgm_stats)
            self.last_k_statistics = statistics
            self.last_sd_statistics = statistics

        if self.debug_tensors:
            self.last_debug = {
                "A": {stage: raw_bands[stage][0].detach() for stage in range(1, 5)},
                "A_lfss": {stage: self._k_low_sources[stage]["lfss_ll"].detach() for stage in range(1, 5)},
                "A_guided": {stage: guided[stage].detach() for stage in range(1, 5)},
                "E": {stage: encoded[stage].detach() for stage in range(1, 5)},
                "U": {3: u3.detach(), 2: u2.detach(), 1: u1.detach(), 0: u0.detach()},
                "L": {3: l3.detach(), 2: l2.detach(), 1: l1.detach(), 0: l0.detach()},
                "decoder_k": _detach_nested(stage_debug),
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
    "DWTFreqNet_SingleDecoder_LFSS_AWGM_K",
    "EXPERIMENT_K_J_BASE_COMMIT",
    "EXPERIMENT_K_PHASE1_REFERENCE_COMMIT",
    "EXPERIMENT_K_VARIANTS",
    "K_STAGE_CHANNELS",
    "K_VARIANT_CONFIGS",
    "initialize_experiment_k_model",
]
