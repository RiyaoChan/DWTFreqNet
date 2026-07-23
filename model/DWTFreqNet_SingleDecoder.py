"""Experiment B: single wavelet decoder with directional frequency guidance."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DWTFreqNet import (
    HaarWaveletTransform,
    InverseHaarWaveletTransform,
    Res_block,
    check_haar_direction_correspondence,
)


MODEL_BASE_COMMIT = "b98bb4e25b425d9fdf5f2ccbadca6f76af38b539"
SINGLE_DECODER_VARIANTS = (
    "sd_raw",
    "sd_awgm",
    "sd_pyramid",
    "sd_full",
)


def _conv_norm_act(in_channels, out_channels, kernel_size=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class DirectionalBandEncoder(nn.Module):
    """Encode H/LH, V/HL and D/HH without mixing their directions."""

    def __init__(self, channels):
        super().__init__()
        self.h_branch = self._branch(channels, (5, 1), (2, 0))
        self.v_branch = self._branch(channels, (1, 5), (0, 2))
        self.d_branch = self._branch(channels, (3, 3), (1, 1))

    @staticmethod
    def _branch(channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, band_h, band_v, band_d):
        return (
            band_h + self.h_branch(band_h),
            band_v + self.v_branch(band_v),
            band_d + self.d_branch(band_d),
        )


class StageWiseAWGM(nn.Module):
    """Use same-DWT directional bands to bidirectionally modulate LL."""

    def __init__(self, channels):
        super().__init__()
        self.direction_gate = nn.Conv2d(channels * 4, 3, 1)
        self.modulation_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.last_direction_weights = None
        self.last_attention_map = None
        self.last_statistics = None
        self.record_statistics = True

    def forward(self, band_a, feature_h, feature_v, feature_d):
        weights = torch.softmax(
            self.direction_gate(
                torch.cat([band_a, feature_h, feature_v, feature_d], dim=1)
            ),
            dim=1,
        )
        weight_h, weight_v, weight_d = torch.chunk(weights, 3, dim=1)
        feature_hf = (
            weight_h * feature_h
            + weight_v * feature_v
            + weight_d * feature_d
        )
        gate = torch.tanh(
            self.modulation_gate(torch.cat([band_a, feature_hf], dim=1))
        )
        guided = band_a * (1.0 + self.alpha * gate)

        self.last_direction_weights = weights.detach()
        self.last_attention_map = gate.detach()
        if not self.training and self.record_statistics:
            direction_means = weights.detach().float().mean(dim=(0, 2, 3)).cpu()
            self.last_statistics = {
                "mean_G_H": float(direction_means[0]),
                "mean_G_V": float(direction_means[1]),
                "mean_G_D": float(direction_means[2]),
                "gate_mean": float(gate.detach().float().mean().cpu()),
                "gate_std": float(gate.detach().float().std().cpu()),
                "alpha_mean": float(self.alpha.detach().float().mean().cpu()),
                "A_norm": float(band_a.detach().float().square().mean().sqrt().cpu()),
                "A_guided_norm": float(
                    guided.detach().float().square().mean().sqrt().cpu()
                ),
            }
        return guided


class DirectionalPyramidBranch(nn.Module):
    """One top-down path; H, V and D each receive a separate instance."""

    def __init__(self):
        super().__init__()
        self.lateral4 = _conv_norm_act(256, 256)
        self.lateral3 = _conv_norm_act(128, 256)
        self.fuse3 = _conv_norm_act(512, 256, 3, 1)
        self.reduce3to2 = _conv_norm_act(256, 128)
        self.lateral2 = _conv_norm_act(64, 128)
        self.fuse2 = _conv_norm_act(256, 128, 3, 1)
        self.reduce2to1 = _conv_norm_act(128, 64)
        self.lateral1 = _conv_norm_act(32, 64)
        self.fuse1 = _conv_norm_act(128, 64, 3, 1)

    def forward(self, feature1, feature2, feature3, feature4):
        pyramid4 = self.lateral4(feature4)
        pyramid3 = self.fuse3(torch.cat([
            self.lateral3(feature3),
            F.interpolate(
                pyramid4, size=feature3.shape[-2:], mode="bilinear", align_corners=False
            ),
        ], dim=1))
        pyramid2 = self.fuse2(torch.cat([
            self.lateral2(feature2),
            F.interpolate(
                self.reduce3to2(pyramid3),
                size=feature2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        ], dim=1))
        pyramid1 = self.fuse1(torch.cat([
            self.lateral1(feature1),
            F.interpolate(
                self.reduce2to1(pyramid2),
                size=feature1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        ], dim=1))
        return pyramid1, pyramid2, pyramid3, pyramid4


class DirectionalTopDownPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_path = DirectionalPyramidBranch()
        self.v_path = DirectionalPyramidBranch()
        self.d_path = DirectionalPyramidBranch()

    def forward(self, directional_features):
        outputs = {}
        for direction, path in (
            ("H", self.h_path),
            ("V", self.v_path),
            ("D", self.d_path),
        ):
            stage_outputs = path(*[
                directional_features[stage][direction] for stage in range(1, 5)
            ])
            for stage, tensor in enumerate(stage_outputs, start=1):
                outputs[(stage, direction)] = tensor
        return outputs


class DWTFreqNet_SingleDecoder(nn.Module):
    """Four-level encoder and one coefficient-calibrated wavelet decoder."""

    FORBIDDEN_MODULE_NAMES = (
        "local_encoder1_2", "local_encoder2_2", "local_encoder3_2",
        "local_encoder1_3", "local_encoder2_3", "local_encoder1_4",
        "global_encoder1_2", "global_encoder2_2", "global_encoder3_2",
        "global_encoder1_3", "global_encoder2_3", "global_encoder1_4",
        "wulle_decoder1", "wulle_decoder2", "wulle_decoder3",
        "wave_att_input_t", "wave_att_f1", "wave_att_f2", "wave_att_f3",
        "TransTo_input", "TransTo1e", "TransTo2e", "TransTo3e",
    )

    def __init__(
        self,
        config,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
        sd_variant="sd_full",
    ):
        super().__init__()
        if sd_variant not in SINGLE_DECODER_VARIANTS:
            raise ValueError(f"Unknown single-decoder variant: {sd_variant}")
        channels = int(config.base_channel)
        if channels != 32:
            raise ValueError("Experiment B channel schedule requires base_channel=32")

        self.mode = mode
        self.deepsuper = deepsuper
        self.sd_variant = sd_variant
        self.use_awgm = sd_variant in ("sd_awgm", "sd_full")
        self.use_pyramid = sd_variant in ("sd_pyramid", "sd_full")
        self.model_variant = "dwtfreqnet_single_decoder"
        self.model_base_commit = MODEL_BASE_COMMIT
        self.single_decoder = True
        self.second_dwt = False
        self.ldrc = False
        self.mamba = False
        self.coefficient_mode = (
            "raw_plus_directional_residual" if self.use_pyramid else "aligned_raw"
        )

        self.har = HaarWaveletTransform()
        self.inversehar = InverseHaarWaveletTransform()
        self.stem = Res_block(n_channels, 32)
        self.inc = self.stem
        self.local_encoder1 = Res_block(32, 64)
        self.local_encoder2 = Res_block(64, 128)
        self.local_encoder3 = Res_block(128, 256)
        self.local_encoder4 = Res_block(256, 256)

        stage_channels = (32, 64, 128, 256)
        if self.use_awgm or self.use_pyramid:
            for stage, stage_channel in enumerate(stage_channels, start=1):
                setattr(self, f"dir_encoder{stage}", DirectionalBandEncoder(stage_channel))
        if self.use_awgm:
            for stage, stage_channel in enumerate(stage_channels, start=1):
                setattr(self, f"stage_awgm{stage}", StageWiseAWGM(stage_channel))

        decoder_channels = (64, 128, 256, 256)
        for stage, (input_channel, output_channel) in enumerate(
            zip(stage_channels, decoder_channels), start=1
        ):
            for direction in ("H", "V", "D"):
                setattr(
                    self,
                    f"align_{direction}{stage}",
                    nn.Conv2d(input_channel, output_channel, 1),
                )

        if self.use_pyramid:
            self.directional_pyramid = DirectionalTopDownPyramid()
            for stage, output_channel in enumerate(decoder_channels, start=1):
                for direction in ("H", "V", "D"):
                    setattr(
                        self,
                        f"delta_{direction}{stage}",
                        nn.Conv2d(output_channel, output_channel, 1),
                    )
                    setattr(
                        self,
                        f"beta_{direction}{stage}",
                        nn.Parameter(torch.full((1, output_channel, 1, 1), 0.1)),
                    )

        self.decoder_fuse3 = Res_block(512, 256)
        self.decoder_fuse2 = Res_block(384, 128)
        self.decoder_fuse1 = Res_block(192, 64)
        self.decoder_fuse0 = Res_block(96, 32)
        self.gt_conv5 = nn.Conv2d(256, n_classes, 1)
        self.gt_conv4 = nn.Conv2d(256, n_classes, 1)
        self.gt_conv3 = nn.Conv2d(128, n_classes, 1)
        self.gt_conv2 = nn.Conv2d(64, n_classes, 1)
        self.out_head = nn.Conv2d(32, n_classes, 1)
        self.outconv = nn.Conv2d(5, n_classes, 1)

        self.debug_tensors = False
        self.record_statistics = True
        self.last_debug = None
        self.last_shapes = None
        self.last_transform_counts = {"dwt": 0, "idwt": 0}
        self.last_sd_statistics = None

    @property
    def experiment_metadata(self):
        return {
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "model_base_commit": self.model_base_commit,
            "single_decoder": self.single_decoder,
            "stage_wise_awgm": self.use_awgm,
            "directional_pyramid": self.use_pyramid,
            "second_dwt": self.second_dwt,
            "ldrc": self.ldrc,
            "mamba": self.mamba,
            "coefficient_mode": self.coefficient_mode,
        }

    def _dwt(self, tensor):
        self.last_transform_counts["dwt"] += 1
        return self.har(tensor)

    def _idwt(self, band_a, band_h, band_v, band_d):
        self.last_transform_counts["idwt"] += 1
        return self.inversehar(band_a, band_h, band_v, band_d)

    @staticmethod
    def _norm(tensor):
        return float(tensor.detach().float().square().mean().sqrt().cpu())

    def _encode_stage(self, stage, tensor):
        band_a, band_h, band_v, band_d = self._dwt(tensor)
        if self.use_awgm or self.use_pyramid:
            feature_h, feature_v, feature_d = getattr(
                self, f"dir_encoder{stage}"
            )(band_h, band_v, band_d)
        else:
            feature_h, feature_v, feature_d = band_h, band_v, band_d
        guided_a = (
            self._apply_stage_awgm(stage, band_a, feature_h, feature_v, feature_d)
            if self.use_awgm
            else band_a
        )
        encoded = getattr(self, f"local_encoder{stage}")(guided_a)
        return (
            encoded,
            (band_a, band_h, band_v, band_d),
            {"H": feature_h, "V": feature_v, "D": feature_d},
            guided_a,
        )

    def _apply_stage_awgm(self, stage, band_a, feature_h, feature_v, feature_d):
        module = getattr(self, f"stage_awgm{stage}")
        module.record_statistics = self.record_statistics
        return module(band_a, feature_h, feature_v, feature_d)

    def _refine_coefficients(self, raw_bands, pyramid_features):
        coefficients = {}
        debug = {}
        statistics = OrderedDict()
        collect_statistics = not self.training and self.record_statistics
        for stage in range(1, 5):
            _, raw_h, raw_v, raw_d = raw_bands[stage]
            for direction, raw in (("H", raw_h), ("V", raw_v), ("D", raw_d)):
                aligned = getattr(self, f"align_{direction}{stage}")(raw)
                coefficient = aligned
                delta = None
                if self.use_pyramid:
                    delta = torch.tanh(
                        getattr(self, f"delta_{direction}{stage}")(
                            pyramid_features[(stage, direction)]
                        )
                    )
                    beta = getattr(self, f"beta_{direction}{stage}")
                    coefficient = aligned + beta * delta
                    if collect_statistics:
                        statistics[f"P{stage}{direction}_norm"] = self._norm(
                            pyramid_features[(stage, direction)]
                        )
                        statistics[f"delta_{direction}{stage}_norm"] = self._norm(delta)
                        statistics[f"beta_{direction}{stage}_mean"] = float(
                            beta.detach().float().mean().cpu()
                        )
                if collect_statistics:
                    statistics[f"raw_coef_{direction}{stage}_norm"] = self._norm(aligned)
                    statistics[f"final_coef_{direction}{stage}_norm"] = self._norm(coefficient)
                coefficients[(stage, direction)] = coefficient
                if self.debug_tensors:
                    debug[(stage, direction)] = {
                        "aligned": aligned.detach(),
                        "delta": None if delta is None else delta.detach(),
                        "coefficient": coefficient.detach(),
                    }
        return coefficients, debug, statistics

    def forward(self, x):
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

        pyramid = (
            self.directional_pyramid(directional) if self.use_pyramid else None
        )
        coefficients, coefficient_debug, statistics = self._refine_coefficients(
            raw_bands, pyramid
        )

        def stage_coefficients(stage):
            return tuple(coefficients[(stage, direction)] for direction in ("H", "V", "D"))

        u3 = self._idwt(encoded[4], *stage_coefficients(4))
        l3 = self.decoder_fuse3(torch.cat([u3, encoded[3]], dim=1))
        u2 = self._idwt(l3, *stage_coefficients(3))
        l2 = self.decoder_fuse2(torch.cat([u2, encoded[2]], dim=1))
        u1 = self._idwt(l2, *stage_coefficients(2))
        l1 = self.decoder_fuse1(torch.cat([u1, encoded[1]], dim=1))
        u0 = self._idwt(l1, *stage_coefficients(1))
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
        if self.use_pyramid:
            self.last_shapes.update({
                f"P{stage}{direction}": tuple(pyramid[(stage, direction)].shape)
                for stage in range(1, 5)
                for direction in ("H", "V", "D")
            })

        if not self.training and self.record_statistics:
            if self.use_awgm:
                awgm_stats = [
                    getattr(self, f"stage_awgm{stage}").last_statistics
                    for stage in range(1, 5)
                ]
                for key in awgm_stats[0]:
                    statistics[f"stage_awgm_{key}"] = sum(
                        item[key] for item in awgm_stats
                    ) / len(awgm_stats)
            self.last_sd_statistics = dict(statistics)

        if self.debug_tensors:
            self.last_debug = {
                "A": {stage: raw_bands[stage][0].detach() for stage in range(1, 5)},
                "A_guided": {stage: guided[stage].detach() for stage in range(1, 5)},
                "coefficients": coefficient_debug,
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
            return tuple(torch.sigmoid(tensor) for tensor in (gt5, gt4, gt3, gt2, d0, out))
        return torch.sigmoid(out)


__all__ = [
    "DirectionalBandEncoder",
    "DirectionalTopDownPyramid",
    "DWTFreqNet_SingleDecoder",
    "MODEL_BASE_COMMIT",
    "SINGLE_DECODER_VARIANTS",
    "StageWiseAWGM",
    "check_haar_direction_correspondence",
]
