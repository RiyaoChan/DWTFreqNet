"""Experiment G: decoder-side DSHF restoration on the fixed Experiment E1 encoder."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    initialize_experiment_e_model,
)
from model.decoder_dshf import DecoderHFRefiner, EXPERIMENT_F2_CORE_COMMIT


EXPERIMENT_G_BASE_COMMIT = "68ede894be748c8842427e140898f007dbe67953"
EXPERIMENT_G_VARIANTS = (
    "g0_e1_passthrough",
    "g1_decoder_dshf",
    "g2_decoder_dshf_semantic",
    "g3_decoder_dshf_targetness",
)
DECODER_DSHF_STAGE_CONFIG = {
    1: {"channels": 64},
    2: {"channels": 128},
    3: {"channels": 256},
    4: {"channels": 256},
}
DECODER_STAGE_CHANNELS = {
    stage: config["channels"] for stage, config in DECODER_DSHF_STAGE_CONFIG.items()
}
DECODER_DSHF_VARIANT_CONFIGS = {
    "g0_e1_passthrough": {
        "ablation_id": "G0",
        "use_decoder_dshf": False,
        "use_semantic_gate": False,
        "use_targetness": False,
    },
    "g1_decoder_dshf": {
        "ablation_id": "G1",
        "use_decoder_dshf": True,
        "use_semantic_gate": False,
        "use_targetness": False,
    },
    "g2_decoder_dshf_semantic": {
        "ablation_id": "G2",
        "use_decoder_dshf": True,
        "use_semantic_gate": True,
        "use_targetness": False,
    },
    "g3_decoder_dshf_targetness": {
        "ablation_id": "G3",
        "use_decoder_dshf": True,
        "use_semantic_gate": True,
        "use_targetness": True,
    },
}
VARIANT_CONFIGS = DECODER_DSHF_VARIANT_CONFIGS


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


def initialize_experiment_g_model(model, baseline_init_fn):
    """Preserve LFSS initialization, then restore all G control-point defaults."""

    initialize_experiment_e_model(model, baseline_init_fn)
    if getattr(model, "use_decoder_dshf", False):
        model.reset_decoder_control_parameters()


class DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF(
    DWTFreqNet_SingleDecoder_LFSS_AWGM
):
    """E1 encoder plus optional decoder coefficient restoration before each IDWT."""

    def __init__(
        self,
        config,
        decoder_variant,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if decoder_variant not in EXPERIMENT_G_VARIANTS:
            raise ValueError(f"Unknown Experiment G variant: {decoder_variant}")
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
        variant = VARIANT_CONFIGS[decoder_variant]
        self.decoder_variant = decoder_variant
        self.use_decoder_dshf = variant["use_decoder_dshf"]
        self.use_decoder_semantic_gate = variant["use_semantic_gate"]
        self.use_decoder_targetness = variant["use_targetness"]
        self.ablation_id = variant["ablation_id"]

        if self.use_decoder_dshf:
            for stage, channels in DECODER_STAGE_CHANNELS.items():
                setattr(
                    self,
                    f"decoder_hf_refiner{stage}",
                    DecoderHFRefiner(
                    channels,
                    use_semantic_gate=self.use_decoder_semantic_gate,
                    use_targetness=self.use_decoder_targetness,
                    ),
                )
            self.reset_decoder_control_parameters()

        self.experiment_group = "experiment_g"
        self.experiment_type = "decoder_high_frequency_restoration"
        self.ablation_axis = "decoder_dshf_semantic_targetness"
        self.model_variant = f"dwtfreqnet_e1_{decoder_variant}"
        self.sd_variant = decoder_variant
        self.model_base_commit = EXPERIMENT_G_BASE_COMMIT
        self.base_encoder = "experiment_e_e1_lfss_resblock_awgm"
        self.encoder_lfss = True
        self.encoder_awgm = True
        self.encoder_dshf = False
        self.decoder_hfe = self.use_decoder_dshf
        self.decoder_dshf_source = f"experiment_f2@{EXPERIMENT_F2_CORE_COMMIT}"
        self.decoder_dshf_input = "once_aligned_raw_hvd"
        self.decoder_dshf_residual = "tanh_bounded"
        self.decoder_dshf_beta_init = 1e-3 if self.use_decoder_dshf else None
        self.decoder_concat_position = "post_idwt"
        self.second_dwt = False
        self.directional_pyramid = False
        self.ldrc = False
        self.channel_matching = False
        self.coefficient_mode = (
            "aligned_raw_plus_bounded_decoder_dshf_residual"
            if self.use_decoder_dshf
            else "aligned_raw"
        )
        self.last_decoder_dshf_statistics = None

    @property
    def experiment_metadata(self):
        metadata = dict(super().experiment_metadata)
        metadata.update({
            "experiment_group": self.experiment_group,
            "experiment_type": self.experiment_type,
            "ablation_axis": self.ablation_axis,
            "ablation_id": self.ablation_id,
            "decoder_variant": self.decoder_variant,
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "model_base_commit": self.model_base_commit,
            "base_encoder": self.base_encoder,
            "encoder_lfss": self.encoder_lfss,
            "encoder_awgm": self.encoder_awgm,
            "encoder_dshf": self.encoder_dshf,
            "decoder_hfe": self.decoder_hfe,
            "decoder_dshf_source": self.decoder_dshf_source,
            "decoder_dshf_input": self.decoder_dshf_input,
            "decoder_dshf_residual": self.decoder_dshf_residual,
            "decoder_dshf_beta_init": self.decoder_dshf_beta_init,
            "decoder_stage_channels": dict(DECODER_STAGE_CHANNELS),
            "decoder_semantic_gate": self.use_decoder_semantic_gate,
            "decoder_targetness": self.use_decoder_targetness,
            "targetness_source": (
                "detached_native_existing_side_logits"
                if self.use_decoder_targetness
                else "none"
            ),
            "decoder_concat_position": self.decoder_concat_position,
            "directional_pyramid": self.directional_pyramid,
            "second_dwt": self.second_dwt,
            "ldrc": self.ldrc,
            "channel_matching": self.channel_matching,
            "coefficient_mode": self.coefficient_mode,
        })
        return metadata

    def reset_decoder_control_parameters(self):
        if not self.use_decoder_dshf:
            return
        for stage in range(1, 5):
            getattr(self, f"decoder_hf_refiner{stage}").reset_control_parameters()

    @staticmethod
    def _mean_parameter(parameter):
        return float(parameter.detach().float().mean().cpu())

    def _restore_stage(self, stage, raw_bands, low_semantic, targetness=None):
        _, raw_h, raw_v, raw_d = raw_bands
        aligned = tuple(
            getattr(self, f"align_{direction}{stage}")(raw)
            for direction, raw in zip(("H", "V", "D"), (raw_h, raw_v, raw_d))
        )
        restored = aligned
        refiner_debug = None
        if self.use_decoder_dshf:
            restored_h, restored_v, restored_d, refiner_debug = (
                getattr(self, f"decoder_hf_refiner{stage}")(
                    *aligned,
                    low_semantic=low_semantic,
                    targetness=targetness,
                )
            )
            restored = (restored_h, restored_v, restored_d)
        return restored, {
            "aligned": aligned,
            "restored": restored,
            "refiner": refiner_debug,
            "targetness": targetness,
        }

    def _collect_decoder_statistics(self, stage_debug):
        statistics = OrderedDict()
        for stage, debug in stage_debug.items():
            for index, direction in enumerate(("H", "V", "D")):
                statistics[f"stage{stage}_{direction}_aligned_norm"] = self._norm(
                    debug["aligned"][index]
                )
                statistics[f"stage{stage}_{direction}_restored_norm"] = self._norm(
                    debug["restored"][index]
                )
            if debug["refiner"] is not None:
                refiner = getattr(self, f"decoder_hf_refiner{stage}")
                core = debug["refiner"]
                for index, direction in enumerate(("H", "V", "D")):
                    statistics[f"stage{stage}_{direction}_delta_norm"] = self._norm(
                        core["deltas"][index]
                    )
                    statistics[f"stage{stage}_{direction}_scale_mean"] = float(
                        core["scales"][index].detach().float().mean().cpu()
                    )
                    statistics[f"stage{stage}_{direction}_beta_mean"] = (
                        self._mean_parameter(getattr(refiner, f"beta_{direction.lower()}"))
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
        native_logits = {}

        native_logits[5] = self.gt_conv5(encoded[4])
        targetness4 = (
            torch.sigmoid(native_logits[5]).detach()
            if self.use_decoder_targetness
            else None
        )
        coefficients4, stage_debug[4] = self._restore_stage(
            4, raw_bands[4], encoded[4], targetness4
        )
        u3 = self._idwt(encoded[4], *coefficients4)
        l3 = self.decoder_fuse3(torch.cat([u3, encoded[3]], dim=1))

        native_logits[4] = self.gt_conv4(l3)
        targetness3 = (
            torch.sigmoid(native_logits[4]).detach()
            if self.use_decoder_targetness
            else None
        )
        coefficients3, stage_debug[3] = self._restore_stage(
            3, raw_bands[3], l3, targetness3
        )
        u2 = self._idwt(l3, *coefficients3)
        l2 = self.decoder_fuse2(torch.cat([u2, encoded[2]], dim=1))

        native_logits[3] = self.gt_conv3(l2)
        targetness2 = (
            torch.sigmoid(native_logits[3]).detach()
            if self.use_decoder_targetness
            else None
        )
        coefficients2, stage_debug[2] = self._restore_stage(
            2, raw_bands[2], l2, targetness2
        )
        u1 = self._idwt(l2, *coefficients2)
        l1 = self.decoder_fuse1(torch.cat([u1, encoded[1]], dim=1))

        native_logits[2] = self.gt_conv2(l1)
        targetness1 = (
            torch.sigmoid(native_logits[2]).detach()
            if self.use_decoder_targetness
            else None
        )
        coefficients1, stage_debug[1] = self._restore_stage(
            1, raw_bands[1], l1, targetness1
        )
        u0 = self._idwt(l1, *coefficients1)
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
        if not self.training and self.record_statistics:
            statistics = self._collect_decoder_statistics(stage_debug)
            awgm_stats = [
                getattr(self, f"stage_awgm{stage}").last_statistics
                for stage in range(1, 5)
            ]
            if all(item is not None for item in awgm_stats):
                for key in awgm_stats[0]:
                    statistics[f"stage_awgm_{key}"] = sum(
                        item[key] for item in awgm_stats
                    ) / len(awgm_stats)
            self.last_decoder_dshf_statistics = statistics
            self.last_sd_statistics = statistics

        if self.debug_tensors:
            coefficient_debug = {
                (stage, direction): {
                    "aligned": stage_debug[stage]["aligned"][index].detach(),
                    "delta": (
                        None
                        if stage_debug[stage]["refiner"] is None
                        else stage_debug[stage]["refiner"]["deltas"][index].detach()
                    ),
                    "coefficient": stage_debug[stage]["restored"][index].detach(),
                }
                for stage in range(1, 5)
                for index, direction in enumerate(("H", "V", "D"))
            }
            self.last_debug = {
                "A": {stage: raw_bands[stage][0].detach() for stage in range(1, 5)},
                "A_lfss": {
                    stage: self._lfss_debug[stage]["lfss"] for stage in range(1, 5)
                },
                "A_guided": {stage: guided[stage].detach() for stage in range(1, 5)},
                "E": {
                    stage: self._lfss_debug[stage]["encoded"] for stage in range(1, 5)
                },
                "coefficients": coefficient_debug,
                "decoder_dshf": _detach_nested(stage_debug),
                "native_side_logits": {
                    index: tensor.detach() for index, tensor in native_logits.items()
                },
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
        gt5 = F.interpolate(native_logits[5], target_size, mode="bilinear", align_corners=False)
        gt4 = F.interpolate(native_logits[4], target_size, mode="bilinear", align_corners=False)
        gt3 = F.interpolate(native_logits[3], target_size, mode="bilinear", align_corners=False)
        gt2 = F.interpolate(native_logits[2], target_size, mode="bilinear", align_corners=False)
        d0 = self.outconv(torch.cat([gt2, gt3, gt4, gt5, out], dim=1))
        if self.mode == "train":
            return tuple(
                torch.sigmoid(tensor)
                for tensor in (gt5, gt4, gt3, gt2, d0, out)
            )
        return torch.sigmoid(out)


__all__ = [
    "DECODER_DSHF_STAGE_CONFIG",
    "DECODER_DSHF_VARIANT_CONFIGS",
    "DECODER_STAGE_CHANNELS",
    "DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF",
    "EXPERIMENT_G_BASE_COMMIT",
    "EXPERIMENT_G_VARIANTS",
    "VARIANT_CONFIGS",
    "initialize_experiment_g_model",
]
