"""Experiment C: SD-AWGM with encoder-side LDRC."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder_fuse.transformer_dec_fuse_none_posqkv_dropout import TransFuseModel
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder


EXPERIMENT_C_VARIANT = "sd_awgm_ldrc"


def _project(in_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            128,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(128),
        nn.GELU(),
    )


class EncoderLDRC(nn.Module):
    """Model same-scale and cross-scale relations after stage-wise AWGM."""

    def __init__(self):
        super().__init__()
        self.proj1 = _project(64, kernel_size=3, stride=2, padding=1)
        self.proj2 = _project(128)
        self.proj3 = _project(256)
        self.proj4 = _project(256)

        self.ldrc4 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=256,
            y_channels=128, ny=9216,
        )
        self.ldrc3 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=1024,
            y_channels=128, ny=8448,
        )
        self.ldrc2 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=4096,
            y_channels=128, ny=5376,
        )
        self.ldrc1 = TransFuseModel(
            num_blocks=1, x_channels=128, nx=4096,
            y_channels=128, ny=5376,
        )

        self.back1 = nn.Conv2d(128, 64, 1)
        self.back2 = nn.Conv2d(128, 128, 1)
        self.back3 = nn.Conv2d(128, 256, 1)
        self.back4 = nn.Conv2d(128, 256, 1)

        self.gamma1 = nn.Parameter(torch.full((1, 64, 1, 1), 1e-3))
        self.gamma2 = nn.Parameter(torch.full((1, 128, 1, 1), 1e-3))
        self.gamma3 = nn.Parameter(torch.full((1, 256, 1, 1), 1e-3))
        self.gamma4 = nn.Parameter(torch.full((1, 256, 1, 1), 1e-3))

        self.record_statistics = True
        self.last_shapes = None
        self.last_statistics = None
        self._block_statistics = OrderedDict()
        self._register_statistic_hooks()

    @staticmethod
    def _norm(tensor):
        return float(tensor.detach().float().square().mean().sqrt().cpu())

    def _make_norm_hook(self, name):
        def hook(_module, _inputs, output):
            if not self.training and self.record_statistics:
                self._block_statistics[name] = self._norm(output)
        return hook

    def _register_statistic_hooks(self):
        for stage in range(1, 5):
            block = getattr(self, f"ldrc{stage}").blocks[0]
            block.self_attn.register_forward_hook(
                self._make_norm_hook(f"ldrc{stage}_sam_output_norm")
            )
            block.cross_attn.register_forward_hook(
                self._make_norm_hook(f"ldrc{stage}_cam_output_norm")
            )
            block.mlp.register_forward_hook(
                self._make_norm_hook(f"ldrc{stage}_ffl_output_norm")
            )

    def forward(self, e1, e2, e3, e4):
        self._block_statistics = OrderedDict()
        r1 = self.proj1(e1)
        r2 = self.proj2(e2)
        r3 = self.proj3(e3)
        r4 = self.proj4(e4)

        t1 = r1.flatten(2).transpose(1, 2)
        t2 = r2.flatten(2).transpose(1, 2)
        t3 = r3.flatten(2).transpose(1, 2)
        t4 = r4.flatten(2).transpose(1, 2)

        t4e = self.ldrc4(t4, torch.cat([t1, t2, t3], dim=1))
        t3e = self.ldrc3(t3, torch.cat([t1, t2, t4e], dim=1))
        t2e = self.ldrc2(t2, torch.cat([t1, t3e, t4e], dim=1))
        t1e = self.ldrc1(t1, torch.cat([t2e, t3e, t4e], dim=1))

        batch = e1.shape[0]
        r1e = t1e.transpose(1, 2).reshape(batch, 128, *r1.shape[-2:])
        r2e = t2e.transpose(1, 2).reshape(batch, 128, *r2.shape[-2:])
        r3e = t3e.transpose(1, 2).reshape(batch, 128, *r3.shape[-2:])
        r4e = t4e.transpose(1, 2).reshape(batch, 128, *r4.shape[-2:])

        delta1 = self.back1(F.interpolate(
            r1e, size=e1.shape[-2:], mode="bilinear", align_corners=False
        ))
        delta2 = self.back2(r2e)
        delta3 = self.back3(r3e)
        delta4 = self.back4(r4e)

        e1e = e1 + self.gamma1 * delta1
        e2e = e2 + self.gamma2 * delta2
        e3e = e3 + self.gamma3 * delta3
        e4e = e4 + self.gamma4 * delta4

        self.last_shapes = {
            "R1": tuple(r1.shape), "R2": tuple(r2.shape),
            "R3": tuple(r3.shape), "R4": tuple(r4.shape),
            "T1": tuple(t1.shape), "T2": tuple(t2.shape),
            "T3": tuple(t3.shape), "T4": tuple(t4.shape),
            "E1e": tuple(e1e.shape), "E2e": tuple(e2e.shape),
            "E3e": tuple(e3e.shape), "E4e": tuple(e4e.shape),
        }

        if not self.training and self.record_statistics:
            statistics = OrderedDict(self._block_statistics)
            for stage, gamma in enumerate(
                (self.gamma1, self.gamma2, self.gamma3, self.gamma4), start=1
            ):
                statistics[f"ldrc_gamma{stage}_mean"] = float(
                    gamma.detach().float().mean().cpu()
                )
            for stage, source, enhanced in (
                (1, e1, e1e), (2, e2, e2e),
                (3, e3, e3e), (4, e4, e4e),
            ):
                source_norm = self._norm(source)
                statistics[f"ldrc_E{stage}e_E{stage}_norm_ratio"] = (
                    self._norm(enhanced) / max(source_norm, 1e-12)
                )
            for kind in ("sam", "cam", "ffl"):
                values = [
                    statistics[f"ldrc{stage}_{kind}_output_norm"]
                    for stage in range(1, 5)
                ]
                statistics[f"ldrc_{kind}_output_norm"] = sum(values) / len(values)
            self.last_statistics = dict(statistics)

        return e1e, e2e, e3e, e4e


class DWTFreqNet_SingleDecoder_LDRC(DWTFreqNet_SingleDecoder):
    """Experiment B SD-AWGM plus one encoder-side LDRC block per scale."""

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
            config=config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
            sd_variant="sd_awgm",
        )
        self.encoder_ldrc = EncoderLDRC()
        self.model_variant = "dwtfreqnet_single_decoder_ldrc"
        self.sd_variant = EXPERIMENT_C_VARIANT
        self.single_decoder = True
        self.stage_wise_awgm = True
        self.directional_pyramid = False
        self.second_dwt = False
        self.ldrc = True
        self.ldrc_position = "encoder_after_stage_awgm"
        self.ldrc_input = "E1_E2_E3_E4"
        self.ldrc_order = "E4_E3_E2_E1"
        self.ldrc_dim = 128
        self.ldrc_blocks = 1
        self.gamma_init = 1e-3
        self.mamba = False
        self.coefficient_mode = "aligned_raw"

    @property
    def experiment_metadata(self):
        metadata = dict(super().experiment_metadata)
        metadata.update({
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "single_decoder": self.single_decoder,
            "stage_wise_awgm": self.stage_wise_awgm,
            "directional_pyramid": self.directional_pyramid,
            "second_dwt": self.second_dwt,
            "ldrc": self.ldrc,
            "ldrc_position": self.ldrc_position,
            "ldrc_input": self.ldrc_input,
            "ldrc_order": self.ldrc_order,
            "ldrc_dim": self.ldrc_dim,
            "ldrc_blocks": self.ldrc_blocks,
            "gamma_init": self.gamma_init,
            "mamba": self.mamba,
            "coefficient_mode": self.coefficient_mode,
        })
        return metadata

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

        self.encoder_ldrc.record_statistics = self.record_statistics
        enhanced_tuple = self.encoder_ldrc(
            encoded[1], encoded[2], encoded[3], encoded[4]
        )
        enhanced = {stage: tensor for stage, tensor in enumerate(enhanced_tuple, start=1)}

        coefficients, coefficient_debug, statistics = self._refine_coefficients(
            raw_bands, pyramid_features=None
        )

        def stage_coefficients(stage):
            return tuple(
                coefficients[(stage, direction)] for direction in ("H", "V", "D")
            )

        u3 = self._idwt(enhanced[4], *stage_coefficients(4))
        l3 = self.decoder_fuse3(torch.cat([u3, enhanced[3]], dim=1))
        u2 = self._idwt(l3, *stage_coefficients(3))
        l2 = self.decoder_fuse2(torch.cat([u2, enhanced[2]], dim=1))
        u1 = self._idwt(l2, *stage_coefficients(2))
        l1 = self.decoder_fuse1(torch.cat([u1, enhanced[1]], dim=1))
        u0 = self._idwt(l1, *stage_coefficients(1))
        l0 = self.decoder_fuse0(torch.cat([u0, x0], dim=1))
        out = self.out_head(l0)

        self.last_shapes = {
            "X0": tuple(x0.shape),
            "E1": tuple(encoded[1].shape), "E2": tuple(encoded[2].shape),
            "E3": tuple(encoded[3].shape), "E4": tuple(encoded[4].shape),
            **self.encoder_ldrc.last_shapes,
            "L3": tuple(l3.shape), "L2": tuple(l2.shape),
            "L1": tuple(l1.shape), "L0": tuple(l0.shape),
        }

        if not self.training and self.record_statistics:
            awgm_stats = [
                getattr(self, f"stage_awgm{stage}").last_statistics
                for stage in range(1, 5)
            ]
            for key in awgm_stats[0]:
                statistics[f"stage_awgm_{key}"] = sum(
                    item[key] for item in awgm_stats
                ) / len(awgm_stats)
            statistics.update(self.encoder_ldrc.last_statistics or {})
            self.last_sd_statistics = dict(statistics)

        if self.debug_tensors:
            self.last_debug = {
                "A": {stage: raw_bands[stage][0].detach() for stage in range(1, 5)},
                "A_guided": {stage: guided[stage].detach() for stage in range(1, 5)},
                "encoded": {stage: encoded[stage].detach() for stage in range(1, 5)},
                "enhanced": {stage: enhanced[stage].detach() for stage in range(1, 5)},
                "coefficients": coefficient_debug,
            }

        if not self.deepsuper:
            return torch.sigmoid(out)

        target_size = x.shape[-2:]
        gt5 = F.interpolate(
            self.gt_conv5(enhanced[4]), target_size,
            mode="bilinear", align_corners=False,
        )
        gt4 = F.interpolate(
            self.gt_conv4(l3), target_size,
            mode="bilinear", align_corners=False,
        )
        gt3 = F.interpolate(
            self.gt_conv3(l2), target_size,
            mode="bilinear", align_corners=False,
        )
        gt2 = F.interpolate(
            self.gt_conv2(l1), target_size,
            mode="bilinear", align_corners=False,
        )
        d0 = self.outconv(torch.cat([gt2, gt3, gt4, gt5, out], dim=1))
        if self.mode == "train":
            return tuple(
                torch.sigmoid(tensor) for tensor in (gt5, gt4, gt3, gt2, d0, out)
            )
        return torch.sigmoid(out)


__all__ = [
    "DWTFreqNet_SingleDecoder_LDRC",
    "EncoderLDRC",
    "EXPERIMENT_C_VARIANT",
]
