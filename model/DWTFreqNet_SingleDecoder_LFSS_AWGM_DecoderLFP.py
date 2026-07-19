"""Experiment H: decoder LFP purification on the fixed Experiment E1 model."""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    initialize_experiment_e_model,
)
from model.decoder_lfp import (
    DecoderLFPProcessor,
    NS_FPN_SOURCE_COMMIT,
    NS_FPN_SOURCE_FILE,
    NS_FPN_SOURCE_URL,
)


EXPERIMENT_H_BASE_COMMIT = "68ede894be748c8842427e140898f007dbe67953"
EXPERIMENT_H_VARIANTS = (
    "h0_e1_passthrough",
    "h1_rawll_attention",
    "h2_rawll_fixed_gaussian",
    "h3_rawll_adaptive_gaussian",
    "h1_decoder_attention",
    "h2_decoder_fixed_gaussian",
    "h3_decoder_adaptive_gaussian",
)
DECODER_LFP_STAGE_CHANNELS = {1: 64, 2: 128, 3: 256, 4: 256}
DECODER_LFP_VARIANT_CONFIGS = {
    "h0_e1_passthrough": {
        "ablation_id": "H0",
        "low_source": "none",
        "use_attention": False,
        "use_gaussian": False,
        "threshold_mode": "none",
    },
    "h1_rawll_attention": {
        "ablation_id": "H1-R",
        "low_source": "same_dwt_raw_ll",
        "use_attention": True,
        "use_gaussian": False,
        "threshold_mode": "none",
    },
    "h2_rawll_fixed_gaussian": {
        "ablation_id": "H2-R",
        "low_source": "same_dwt_raw_ll",
        "use_attention": True,
        "use_gaussian": True,
        "threshold_mode": "fixed_hard",
    },
    "h3_rawll_adaptive_gaussian": {
        "ablation_id": "H3-R",
        "low_source": "same_dwt_raw_ll",
        "use_attention": True,
        "use_gaussian": True,
        "threshold_mode": "adaptive_soft",
    },
    "h1_decoder_attention": {
        "ablation_id": "H1-D",
        "low_source": "decoder_low_semantic",
        "use_attention": True,
        "use_gaussian": False,
        "threshold_mode": "none",
    },
    "h2_decoder_fixed_gaussian": {
        "ablation_id": "H2-D",
        "low_source": "decoder_low_semantic",
        "use_attention": True,
        "use_gaussian": True,
        "threshold_mode": "fixed_hard",
    },
    "h3_decoder_adaptive_gaussian": {
        "ablation_id": "H3-D",
        "low_source": "decoder_low_semantic",
        "use_attention": True,
        "use_gaussian": True,
        "threshold_mode": "adaptive_soft",
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


def initialize_experiment_h_model(model, baseline_init_fn):
    """Initialize E1 normally while retaining the LFP attention's PyTorch default."""

    attention_defaults = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if name.startswith("decoder_lfp") and name.endswith("attention.conv")
    }
    initialize_experiment_e_model(model, baseline_init_fn)
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, weight in attention_defaults.items():
            modules[name].weight.copy_(weight)
    if getattr(model, "use_decoder_lfp", False):
        for stage in range(1, 5):
            getattr(model, f"decoder_lfp{stage}").reset_control_parameters()


class DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP(
    DWTFreqNet_SingleDecoder_LFSS_AWGM
):
    """E1 plus online H/V/D purification immediately before each decoder IDWT."""

    def __init__(
        self,
        config,
        lfp_variant,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if lfp_variant not in EXPERIMENT_H_VARIANTS:
            raise ValueError(f"Unknown Experiment H variant: {lfp_variant}")
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
        variant = DECODER_LFP_VARIANT_CONFIGS[lfp_variant]
        self.lfp_variant = lfp_variant
        self.low_source_type = variant["low_source"]
        self.use_decoder_lfp = variant["use_attention"]
        self.use_gaussian = variant["use_gaussian"]
        self.threshold_mode = variant["threshold_mode"]
        self.ablation_id = variant["ablation_id"]

        if self.use_decoder_lfp:
            for stage, channels in DECODER_LFP_STAGE_CHANNELS.items():
                setattr(
                    self,
                    f"decoder_lfp{stage}",
                    DecoderLFPProcessor(
                        high_channels=channels,
                        use_gaussian=self.use_gaussian,
                        threshold_mode=self.threshold_mode,
                        fixed_tau=0.5,
                    ),
                )

        self.experiment_group = "experiment_h"
        self.experiment_type = "decoder_lfp_high_frequency_purification"
        self.ablation_axis = "lfp_low_source_and_gaussian_threshold"
        self.base_encoder = "experiment_e_e1_lfss_resblock_awgm"
        self.model_base_commit = EXPERIMENT_H_BASE_COMMIT
        self.model_variant = f"dwtfreqnet_e1_{lfp_variant}"
        self.sd_variant = lfp_variant
        self.encoder_lfss = True
        self.encoder_awgm = True
        self.encoder_dshf = False
        self.decoder_dshf = False
        self.decoder_semantic_direction_gate = False
        self.decoder_targetness = False
        self.decoder_lfp = self.use_decoder_lfp
        self.decoder_lfp_all_stages = self.use_decoder_lfp
        self.decoder_lfp_high_source = "once_aligned_raw_haar_hvd"
        self.decoder_lfp_low_source = self.low_source_type
        self.decoder_lfp_attention = (
            "channel_mean_max_conv7_sigmoid" if self.use_decoder_lfp else "none"
        )
        self.decoder_lfp_gaussian = self.use_gaussian
        self.decoder_lfp_threshold_mode = self.threshold_mode
        self.second_dwt = False
        self.directional_pyramid = False
        self.ldrc = False
        self.channel_matching = False
        self.coefficient_mode = (
            "lfp_purified_once_aligned_raw_hvd"
            if self.use_decoder_lfp
            else "aligned_raw"
        )
        self.ns_fpn_source_commit = NS_FPN_SOURCE_COMMIT
        self.ns_fpn_source_file = NS_FPN_SOURCE_FILE
        self.ns_fpn_source_url = NS_FPN_SOURCE_URL
        self.last_lfp_statistics = None

    @property
    def experiment_metadata(self):
        metadata = dict(super().experiment_metadata)
        metadata.update({
            "experiment_group": self.experiment_group,
            "experiment_type": self.experiment_type,
            "ablation_axis": self.ablation_axis,
            "ablation_id": self.ablation_id,
            "lfp_variant": self.lfp_variant,
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "model_base_commit": self.model_base_commit,
            "base_encoder": self.base_encoder,
            "encoder_lfss": self.encoder_lfss,
            "encoder_awgm": self.encoder_awgm,
            "encoder_dshf": self.encoder_dshf,
            "decoder_dshf": self.decoder_dshf,
            "decoder_semantic_direction_gate": self.decoder_semantic_direction_gate,
            "decoder_targetness": self.decoder_targetness,
            "decoder_lfp": self.decoder_lfp,
            "decoder_lfp_all_stages": self.decoder_lfp_all_stages,
            "decoder_lfp_stage_channels": dict(DECODER_LFP_STAGE_CHANNELS),
            "decoder_lfp_high_source": self.decoder_lfp_high_source,
            "decoder_lfp_low_source": self.decoder_lfp_low_source,
            "decoder_lfp_attention": self.decoder_lfp_attention,
            "decoder_lfp_gaussian": self.decoder_lfp_gaussian,
            "decoder_lfp_threshold_mode": self.decoder_lfp_threshold_mode,
            "decoder_lfp_fixed_tau": 0.5 if self.threshold_mode == "fixed_hard" else None,
            "decoder_lfp_gaussian_kernel": 3 if self.use_gaussian else None,
            "decoder_lfp_sigma_init": 1.0 if self.use_gaussian else None,
            "second_dwt": self.second_dwt,
            "directional_pyramid": self.directional_pyramid,
            "ldrc": self.ldrc,
            "channel_matching": self.channel_matching,
            "coefficient_mode": self.coefficient_mode,
            "ns_fpn_source_commit": self.ns_fpn_source_commit,
            "ns_fpn_source_file": self.ns_fpn_source_file,
            "ns_fpn_source_url": self.ns_fpn_source_url,
        })
        return metadata

    def _align_stage_high(self, stage, raw_bands):
        _, raw_h, raw_v, raw_d = raw_bands[stage]
        return tuple(
            getattr(self, f"align_{direction}{stage}")(raw)
            for direction, raw in zip(("H", "V", "D"), (raw_h, raw_v, raw_d))
        )

    def _select_low_source(self, stage, raw_bands, decoder_low):
        if self.low_source_type == "same_dwt_raw_ll":
            return raw_bands[stage][0]
        if self.low_source_type == "decoder_low_semantic":
            return decoder_low
        raise RuntimeError(f"Invalid Experiment H low source: {self.low_source_type}")

    def _purify_stage_high(self, stage, raw_bands, decoder_low):
        aligned = self._align_stage_high(stage, raw_bands)
        if not self.use_decoder_lfp:
            high = torch.cat(aligned, dim=1)
            return (*aligned, {
                "low_source": None,
                "attention": None,
                "aligned_high": high,
                "modulated_high": high,
                "gaussian_high": None,
                "threshold": None,
                "threshold_debug": None,
                "mask": None,
                "purified_high": high,
            })
        low_source = self._select_low_source(stage, raw_bands, decoder_low)
        return getattr(self, f"decoder_lfp{stage}")(
            low_source, *aligned
        )

    def _collect_lfp_statistics(self, stage_debug):
        statistics = OrderedDict()
        for stage, debug in stage_debug.items():
            statistics[f"stage{stage}_aligned_norm"] = self._norm(debug["aligned_high"])
            statistics[f"stage{stage}_purified_norm"] = self._norm(debug["purified_high"])
            if debug["attention"] is None:
                continue
            attention = debug["attention"].detach().float()
            statistics.update({
                f"stage{stage}_attention_mean": float(attention.mean().cpu()),
                f"stage{stage}_attention_std": float(attention.std(unbiased=False).cpu()),
                f"stage{stage}_attention_min": float(attention.min().cpu()),
                f"stage{stage}_attention_max": float(attention.max().cpu()),
                f"stage{stage}_attention_change_ratio": float(
                    (
                        debug["modulated_high"] - debug["aligned_high"]
                    ).detach().float().norm().cpu()
                    / (debug["aligned_high"].detach().float().norm().cpu() + 1e-8)
                ),
            })
            if debug["gaussian_high"] is not None:
                processor = getattr(self, f"decoder_lfp{stage}")
                statistics[f"stage{stage}_sigma"] = float(
                    processor.gaussian.sigma.detach().float().cpu()
                )
                mask = debug["mask"].detach().float()
                statistics[f"stage{stage}_mask_mean"] = float(mask.mean().cpu())
                statistics[f"stage{stage}_purification_change_ratio"] = float(
                    (
                        debug["purified_high"] - debug["modulated_high"]
                    ).detach().float().norm().cpu()
                    / (debug["modulated_high"].detach().float().norm().cpu() + 1e-8)
                )
                if debug["threshold_debug"] is not None:
                    ratio = debug["threshold_debug"]["threshold_ratio"].detach().float()
                    threshold = debug["threshold"].detach().float()
                    statistics[f"stage{stage}_threshold_ratio_mean"] = float(ratio.mean().cpu())
                    statistics[f"stage{stage}_threshold_mean"] = float(threshold.mean().cpu())
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
        h4, v4, d4, stage_debug[4] = self._purify_stage_high(
            4, raw_bands, encoded[4]
        )
        u3 = self._idwt(encoded[4], h4, v4, d4)
        l3 = self.decoder_fuse3(torch.cat([u3, encoded[3]], dim=1))

        h3, v3, d3, stage_debug[3] = self._purify_stage_high(
            3, raw_bands, l3
        )
        u2 = self._idwt(l3, h3, v3, d3)
        l2 = self.decoder_fuse2(torch.cat([u2, encoded[2]], dim=1))

        h2, v2, d2, stage_debug[2] = self._purify_stage_high(
            2, raw_bands, l2
        )
        u1 = self._idwt(l2, h2, v2, d2)
        l1 = self.decoder_fuse1(torch.cat([u1, encoded[1]], dim=1))

        h1, v1, d1, stage_debug[1] = self._purify_stage_high(
            1, raw_bands, l1
        )
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
            statistics = self._collect_lfp_statistics(stage_debug)
            awgm_stats = [
                getattr(self, f"stage_awgm{stage}").last_statistics
                for stage in range(1, 5)
            ]
            if all(item is not None for item in awgm_stats):
                for key in awgm_stats[0]:
                    statistics[f"stage_awgm_{key}"] = sum(
                        item[key] for item in awgm_stats
                    ) / len(awgm_stats)
            self.last_lfp_statistics = statistics
            self.last_sd_statistics = statistics

        if self.debug_tensors:
            coefficient_debug = {
                (stage, direction): {
                    "aligned": stage_debug[stage]["aligned_high"].chunk(3, dim=1)[index].detach(),
                    "delta": None,
                    "coefficient": stage_debug[stage]["purified_high"].chunk(3, dim=1)[index].detach(),
                }
                for stage in range(1, 5)
                for index, direction in enumerate(("H", "V", "D"))
            }
            self.last_debug = {
                "A": {stage: raw_bands[stage][0].detach() for stage in range(1, 5)},
                "A_lfss": {stage: self._lfss_debug[stage]["lfss"] for stage in range(1, 5)},
                "A_guided": {stage: guided[stage].detach() for stage in range(1, 5)},
                "E": {stage: self._lfss_debug[stage]["encoded"] for stage in range(1, 5)},
                "U": {3: u3.detach(), 2: u2.detach(), 1: u1.detach(), 0: u0.detach()},
                "L": {3: l3.detach(), 2: l2.detach(), 1: l1.detach(), 0: l0.detach()},
                "coefficients": coefficient_debug,
                "decoder_lfp": _detach_nested(stage_debug),
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
    "DECODER_LFP_STAGE_CHANNELS",
    "DECODER_LFP_VARIANT_CONFIGS",
    "DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP",
    "EXPERIMENT_H_BASE_COMMIT",
    "EXPERIMENT_H_VARIANTS",
    "initialize_experiment_h_model",
]
