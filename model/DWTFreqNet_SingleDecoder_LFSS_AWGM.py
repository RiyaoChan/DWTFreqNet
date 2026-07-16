"""Experiment E: Wave-Mamba LFSS preconditioning before encoder AWGM."""

from importlib import metadata as importlib_metadata

import torch
import torch.nn as nn

from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.third_party.wavemamba_lfss import (
    WAVE_MAMBA_LICENSE,
    WAVE_MAMBA_SOURCE_COMMIT,
    WAVE_MAMBA_SOURCE_FILE,
    WAVE_MAMBA_SOURCE_URL,
    WaveMambaLFSSNCHWAdapter,
    selective_scan_fn,
)


EXPERIMENT_E_BASE_COMMIT = "435ab1827ecee4c6b83b669789bb9833a5fd5320"
EXPERIMENT_E_VARIANTS = (
    "e1_lfss_resblock",
    "e2_lfss_transition",
)
LFSS_STAGE_CONFIG = {
    1: {
        "channels": 32,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
    2: {
        "channels": 64,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
    3: {
        "channels": 128,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
    4: {
        "channels": 256,
        "d_state": 16,
        "expand": 2.0,
        "drop_path": 0.0,
        "attn_drop_rate": 0.0,
    },
}


def _package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


class LowFrequencyTransition(nn.Module):
    """Fixed E2 post-AWGM channel transition."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


def snapshot_lfss_special_parameters(model):
    """Clone every LFSS parameter whose original initialization is protected."""

    suffixes = (
        "dt_projs_weight",
        "dt_projs_bias",
        "A_logs",
        "Ds",
        "skip_scale",
        "skip_scale2",
    )
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("lfss_blocks.") and name.endswith(suffixes)
    }


def initialize_experiment_e_model(model, baseline_init_fn):
    """Apply baseline initialization while preserving every LFSS submodule."""

    for name, module in model.named_modules():
        if name.startswith("lfss_blocks."):
            continue
        baseline_init_fn(module)


def lfss_initialization_max_difference(before, after):
    if before.keys() != after.keys():
        raise RuntimeError("LFSS initialization snapshots contain different parameters")
    return max(
        (before[name] - after[name]).abs().max().item() for name in before
    ) if before else 0.0


class DWTFreqNet_SingleDecoder_LFSS_AWGM(DWTFreqNet_SingleDecoder):
    """Single decoder with raw Wave-Mamba LFSS applied before each AWGM."""

    def __init__(
        self,
        config,
        encoder_variant,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
    ):
        if encoder_variant not in EXPERIMENT_E_VARIANTS:
            raise ValueError(f"Unknown Experiment E variant: {encoder_variant}")
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

        self.encoder_variant = encoder_variant
        self.lfss_blocks = nn.ModuleDict(
            {
                str(stage): WaveMambaLFSSNCHWAdapter(**LFSS_STAGE_CONFIG[stage])
                for stage in range(1, 5)
            }
        )
        # This guard complements the explicit named-module initialization routine.
        # It prevents the repository baseline initializer from touching an LFSS
        # descendant if a caller accidentally uses model.apply(init_weights).
        for module in self.lfss_blocks.modules():
            module._skip_external_init = True

        if encoder_variant == "e2_lfss_transition":
            transitions = (
                (32, 64),
                (64, 128),
                (128, 256),
                (256, 256),
            )
            for stage, (in_channels, out_channels) in enumerate(
                transitions, start=1
            ):
                setattr(
                    self,
                    f"local_encoder{stage}",
                    LowFrequencyTransition(in_channels, out_channels),
                )

        self.experiment_group = "experiment_e"
        self.experiment_type = "encoder_ablation"
        self.ablation_axis = "pre_awgm_low_frequency_extractor"
        self.encoder_lfss = True
        self.lfss_source = "wave_mamba"
        self.lfss_outer_residual = False
        self.lfss_blocks_per_stage = 1
        self.awgm_input_low = "lfss_refined_ll"
        self.decoder_hfe = False
        self.second_dwt = False
        self.ldrc = False
        self.mamba = True
        self.coefficient_mode = "aligned_raw"
        self.model_base_commit = EXPERIMENT_E_BASE_COMMIT

        if encoder_variant == "e1_lfss_resblock":
            self.ablation_id = "E1"
            self.model_variant = (
                "dwtfreqnet_single_decoder_lfss_awgm_resblock"
            )
            self.sd_variant = "sd_awgm_lfss_resblock"
            self.post_awgm_encoder = "original_res_block"
        else:
            self.ablation_id = "E2"
            self.model_variant = (
                "dwtfreqnet_single_decoder_lfss_awgm_transition"
            )
            self.sd_variant = "sd_awgm_lfss_transition"
            self.post_awgm_encoder = "conv1x1_bn_gelu_transition"

        self.wave_mamba_source_commit = WAVE_MAMBA_SOURCE_COMMIT
        self.wave_mamba_source_file = WAVE_MAMBA_SOURCE_FILE
        self.wave_mamba_source_url = WAVE_MAMBA_SOURCE_URL
        self.wave_mamba_license = WAVE_MAMBA_LICENSE
        self.mamba_ssm_version = _package_version("mamba-ssm")
        self.torch_version = torch.__version__
        self.cuda_version = torch.version.cuda
        self.selective_scan_backend = (
            f"{selective_scan_fn.__module__}.{selective_scan_fn.__name__}"
        )
        self._lfss_debug = {}

    @property
    def experiment_metadata(self):
        metadata = dict(super().experiment_metadata)
        metadata.update({
            "experiment_group": self.experiment_group,
            "experiment_type": self.experiment_type,
            "ablation_axis": self.ablation_axis,
            "ablation_id": self.ablation_id,
            "encoder_variant": self.encoder_variant,
            "model_variant": self.model_variant,
            "sd_variant": self.sd_variant,
            "model_base_commit": self.model_base_commit,
            "encoder_lfss": self.encoder_lfss,
            "lfss_source": self.lfss_source,
            "lfss_outer_residual": self.lfss_outer_residual,
            "lfss_blocks_per_stage": self.lfss_blocks_per_stage,
            "lfss_stage_config": {
                str(stage): dict(config)
                for stage, config in LFSS_STAGE_CONFIG.items()
            },
            "awgm_input_low": self.awgm_input_low,
            "post_awgm_encoder": self.post_awgm_encoder,
            "decoder_hfe": self.decoder_hfe,
            "directional_pyramid": False,
            "second_dwt": self.second_dwt,
            "ldrc": self.ldrc,
            "mamba": self.mamba,
            "coefficient_mode": self.coefficient_mode,
            "wave_mamba_source_commit": self.wave_mamba_source_commit,
            "wave_mamba_source_file": self.wave_mamba_source_file,
            "wave_mamba_source_url": self.wave_mamba_source_url,
            "wave_mamba_license": self.wave_mamba_license,
            "mamba_ssm_version": self.mamba_ssm_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "selective_scan_backend": self.selective_scan_backend,
        })
        return metadata

    def _encode_stage(self, stage, tensor):
        band_a, band_h, band_v, band_d = self._dwt(tensor)
        feature_h, feature_v, feature_d = getattr(
            self, f"dir_encoder{stage}"
        )(band_h, band_v, band_d)
        refined_a = self.lfss_blocks[str(stage)](band_a)
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
        return (
            encoded,
            (band_a, band_h, band_v, band_d),
            {"H": feature_h, "V": feature_v, "D": feature_d},
            guided_a,
        )

    def forward(self, x):
        self._lfss_debug = {}
        output = super().forward(x)
        if self.debug_tensors:
            self.last_debug["A_lfss"] = {
                stage: self._lfss_debug[stage]["lfss"] for stage in range(1, 5)
            }
            self.last_debug["E"] = {
                stage: self._lfss_debug[stage]["encoded"] for stage in range(1, 5)
            }
            self.last_debug["AWGM_gate"] = {
                stage: getattr(self, f"stage_awgm{stage}").last_attention_map
                for stage in range(1, 5)
            }
            self.last_debug["AWGM_direction_weights"] = {
                stage: getattr(
                    self, f"stage_awgm{stage}"
                ).last_direction_weights
                for stage in range(1, 5)
            }
        return output

    def lfss_scale_statistics(self):
        statistics = {}
        for stage in range(1, 5):
            block = self.lfss_blocks[str(stage)].block
            for name, parameter in (
                ("skip_scale", block.skip_scale),
                ("skip_scale2", block.skip_scale2),
            ):
                values = parameter.detach().float()
                prefix = f"lfss{stage}_{name}"
                statistics.update({
                    f"{prefix}_mean": float(values.mean().cpu()),
                    f"{prefix}_std": float(values.std(unbiased=False).cpu()),
                    f"{prefix}_min": float(values.min().cpu()),
                    f"{prefix}_max": float(values.max().cpu()),
                })
        return statistics


__all__ = [
    "DWTFreqNet_SingleDecoder_LFSS_AWGM",
    "EXPERIMENT_E_BASE_COMMIT",
    "EXPERIMENT_E_VARIANTS",
    "LFSS_STAGE_CONFIG",
    "LowFrequencyTransition",
    "initialize_experiment_e_model",
    "lfss_initialization_max_difference",
    "snapshot_lfss_special_parameters",
]
