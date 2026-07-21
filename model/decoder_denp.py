"""Dual-Evidence Noise-Calibrated Purification for Experiment J.

The module keeps robust high-frequency noise estimation, low-frequency
compactness evidence, and the final purification decision as separate
operations.  It consumes once-aligned signed raw Haar H/V/D coefficients and
never introduces another DWT/IDWT pair.
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


DENP_BANDS = ("H", "V", "D")
DENP_TRAIN_VARIANTS = (
    "j1_bandwise_noise_calibrated",
    "j2_rawll_compactness",
    "j2_decoder_compactness",
    "j3_dual_evidence_fixed",
    "j3_dual_evidence_reliability",
)


def _bounded_logit(value, lower, upper):
    probability = (float(value) - float(lower)) / (float(upper) - float(lower))
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Initial value {value} must lie inside ({lower}, {upper})")
    return math.log(probability / (1.0 - probability))


def _autocast_disabled(tensor):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=tensor.device.type, enabled=False)
    if tensor.device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


class RobustBandNoiseEstimator(nn.Module):
    """Per-sample robust band scale with a learned bounded threshold multiple."""

    def __init__(self, lambda_init=1.0, kappa=0.15, epsilon=1e-6):
        super().__init__()
        self.lambda_logit = nn.Parameter(torch.tensor(
            _bounded_logit(lambda_init, 0.5, 3.0), dtype=torch.float32
        ))
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)

    @property
    def lambda_value(self):
        return 0.5 + 2.5 * torch.sigmoid(self.lambda_logit)

    def reset_control_parameters(self):
        with torch.no_grad():
            self.lambda_logit.fill_(_bounded_logit(1.0, 0.5, 3.0))

    def robust_scale(self, band):
        if band.ndim != 4:
            raise RuntimeError(f"DENP band must be NCHW, got {tuple(band.shape)}")
        with _autocast_disabled(band):
            values = band.float().flatten(start_dim=2)
            spatial_median = values.median(dim=-1, keepdim=True).values
            channel_mad = (values - spatial_median).abs().median(dim=-1).values
            sigma_hat = 1.4826 * channel_mad.median(dim=1, keepdim=True).values
            sigma_hat = sigma_hat.view(band.shape[0], 1, 1, 1).detach()
        return sigma_hat

    def forward(self, band):
        sigma_hat = self.robust_scale(band)
        lambda_value = self.lambda_value
        tau_float = lambda_value.float() * sigma_hat
        tau = tau_float.to(dtype=band.dtype)
        temperature = self.kappa * tau + self.epsilon
        noise_confidence = torch.sigmoid((tau - band.abs()) / temperature)
        return noise_confidence, {
            "sigma_hat": sigma_hat,
            "lambda": lambda_value,
            "tau": tau,
            "temperature": temperature,
        }


class LearnableBandGaussian(nn.Module):
    """A bounded 3x3 Gaussian shared across channels within one Haar band."""

    def __init__(self, channels, sigma_init=1.0):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = 3
        self.padding = 1
        self.sigma_logit = nn.Parameter(torch.tensor(
            _bounded_logit(sigma_init, 0.5, 2.0), dtype=torch.float32
        ))

    @property
    def sigma(self):
        return 0.5 + 1.5 * torch.sigmoid(self.sigma_logit)

    def reset_control_parameters(self):
        with torch.no_grad():
            self.sigma_logit.fill_(_bounded_logit(1.0, 0.5, 2.0))

    def kernel(self, device=None, dtype=None):
        sigma = self.sigma.float()
        device = sigma.device if device is None else device
        dtype = sigma.dtype if dtype is None else dtype
        coords = torch.arange(3, device=device, dtype=torch.float32) - 1.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma.square()))
        kernel = kernel / kernel.sum()
        return kernel.to(dtype=dtype).reshape(1, 1, 3, 3)

    def forward(self, band):
        if band.ndim != 4 or band.shape[1] != self.channels:
            raise RuntimeError(
                f"Band Gaussian expected N,{self.channels},H,W, got {tuple(band.shape)}"
            )
        kernel = self.kernel(device=band.device, dtype=band.dtype)
        weight = kernel.expand(self.channels, 1, 3, 3).contiguous()
        padded = F.pad(band, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, weight, groups=self.channels)


class LowFrequencyCompactness(nn.Module):
    """Monotone local compactness evidence from a low-frequency feature map."""

    def __init__(self, slope_init=5.0, threshold_init=0.0, epsilon=1e-6):
        super().__init__()
        self.slope_logit = nn.Parameter(torch.tensor(
            _bounded_logit(slope_init, 1.0, 20.0), dtype=torch.float32
        ))
        if not -0.5 < float(threshold_init) < 0.5:
            raise ValueError("Compactness threshold must lie inside (-0.5, 0.5)")
        self.threshold_raw = nn.Parameter(torch.tensor(
            math.atanh(2.0 * float(threshold_init)), dtype=torch.float32
        ))
        self.epsilon = float(epsilon)

    @property
    def slope(self):
        return 1.0 + 19.0 * torch.sigmoid(self.slope_logit)

    @property
    def threshold(self):
        return 0.5 * torch.tanh(self.threshold_raw)

    def reset_control_parameters(self):
        with torch.no_grad():
            self.slope_logit.fill_(_bounded_logit(5.0, 1.0, 20.0))
            self.threshold_raw.zero_()

    def forward(self, low):
        if low.ndim != 4:
            raise RuntimeError(f"DENP low-frequency input must be NCHW, got {tuple(low.shape)}")
        with _autocast_disabled(low):
            energy = torch.sqrt(low.float().square().mean(dim=1, keepdim=True) + self.epsilon)
            center = F.avg_pool2d(
                F.pad(energy, (1, 1, 1, 1), mode="replicate"), 3, stride=1
            )
            large = F.avg_pool2d(
                F.pad(energy, (3, 3, 3, 3), mode="replicate"), 7, stride=1
            )
            ring = (49.0 * large - 9.0 * center) / 40.0
            compactness = (center - ring) / (center + ring + self.epsilon)
            protection = torch.sigmoid(
                self.slope.float() * (compactness - self.threshold.float())
            )
        return protection.to(dtype=low.dtype), {
            "energy": energy,
            "center": center,
            "ring": ring,
            "compactness": compactness,
            "slope": self.slope,
            "threshold": self.threshold,
        }


class DENPPurifier(nn.Module):
    """One decoder-stage DENP processor for three independent Haar bands."""

    def __init__(self, high_channels, variant):
        super().__init__()
        if variant not in DENP_TRAIN_VARIANTS:
            raise ValueError(f"Unknown trainable DENP variant: {variant}")
        self.high_channels = int(high_channels)
        self.variant = variant
        self.noise_estimators = nn.ModuleDict({
            band: RobustBandNoiseEstimator() for band in DENP_BANDS
        })
        self.gaussians = nn.ModuleDict({
            band: LearnableBandGaussian(self.high_channels) for band in DENP_BANDS
        })
        self.use_raw_compactness = variant in (
            "j2_rawll_compactness",
            "j3_dual_evidence_fixed",
            "j3_dual_evidence_reliability",
        )
        self.use_decoder_compactness = variant in (
            "j2_decoder_compactness",
            "j3_dual_evidence_fixed",
            "j3_dual_evidence_reliability",
        )
        self.use_reliability = variant == "j3_dual_evidence_reliability"
        if self.use_raw_compactness:
            self.raw_compactness = LowFrequencyCompactness()
        if self.use_decoder_compactness:
            self.decoder_compactness = LowFrequencyCompactness()
        if self.use_reliability:
            gamma_init = _bounded_logit(1.0, 0.5, 2.0)
            self.gamma_raw_logits = nn.Parameter(torch.full((3,), gamma_init))
            self.gamma_decoder_logits = nn.Parameter(torch.full((3,), gamma_init))

    @staticmethod
    def _gamma(logit):
        return 0.5 + 1.5 * torch.sigmoid(logit)

    @property
    def gamma_raw(self):
        return self._gamma(self.gamma_raw_logits) if self.use_reliability else None

    @property
    def gamma_decoder(self):
        return self._gamma(self.gamma_decoder_logits) if self.use_reliability else None

    def reset_control_parameters(self):
        for module in self.noise_estimators.values():
            module.reset_control_parameters()
        for module in self.gaussians.values():
            module.reset_control_parameters()
        if self.use_raw_compactness:
            self.raw_compactness.reset_control_parameters()
        if self.use_decoder_compactness:
            self.decoder_compactness.reset_control_parameters()
        if self.use_reliability:
            initial = _bounded_logit(1.0, 0.5, 2.0)
            with torch.no_grad():
                self.gamma_raw_logits.fill_(initial)
                self.gamma_decoder_logits.fill_(initial)

    def _validate(self, raw_low, decoder_low, bands):
        if len(bands) != 3 or not (bands[0].shape == bands[1].shape == bands[2].shape):
            raise RuntimeError("DENP requires equal aligned H/V/D shapes")
        if bands[0].shape[1] != self.high_channels:
            raise RuntimeError(
                f"DENP expected {self.high_channels} high channels, got {bands[0].shape[1]}"
            )
        for name, low, enabled in (
            ("raw LL", raw_low, self.use_raw_compactness),
            ("decoder low", decoder_low, self.use_decoder_compactness),
        ):
            if not enabled:
                continue
            if low is None or low.shape[0] != bands[0].shape[0] or low.shape[-2:] != bands[0].shape[-2:]:
                raise RuntimeError(
                    f"DENP {name}/high shape mismatch: low={None if low is None else tuple(low.shape)}, "
                    f"high={tuple(bands[0].shape)}"
                )

    def forward(self, raw_low, decoder_low, aligned_h, aligned_v, aligned_d):
        bands = (aligned_h, aligned_v, aligned_d)
        self._validate(raw_low, decoder_low, bands)
        raw_protection = raw_debug = None
        decoder_protection = decoder_debug = None
        if self.use_raw_compactness:
            raw_protection, raw_debug = self.raw_compactness(raw_low)
        if self.use_decoder_compactness:
            decoder_protection, decoder_debug = self.decoder_compactness(decoder_low)

        purified, band_debug = [], {}
        for index, (name, band) in enumerate(zip(DENP_BANDS, bands)):
            noise, noise_debug = self.noise_estimators[name](band)
            gaussian = self.gaussians[name](band)
            mask = noise
            gamma_raw = gamma_decoder = None
            if raw_protection is not None:
                if self.use_reliability:
                    gamma_raw = self.gamma_raw[index].view(1, 1, 1, 1)
                    mask = mask * (1.0 - raw_protection).pow(gamma_raw)
                else:
                    mask = mask * (1.0 - raw_protection)
            if decoder_protection is not None:
                if self.use_reliability:
                    gamma_decoder = self.gamma_decoder[index].view(1, 1, 1, 1)
                    mask = mask * (1.0 - decoder_protection).pow(gamma_decoder)
                else:
                    mask = mask * (1.0 - decoder_protection)
            output = (1.0 - mask) * band + mask * gaussian
            purified.append(output)
            band_debug[name] = {
                "aligned": band,
                "noise_confidence": noise,
                "noise": noise_debug,
                "gaussian": gaussian,
                "mask": mask,
                "purified": output,
                "gamma_raw": gamma_raw,
                "gamma_decoder": gamma_decoder,
            }
        return (*purified, {
            "raw_low": raw_low,
            "decoder_low": decoder_low,
            "raw_protection": raw_protection,
            "decoder_protection": decoder_protection,
            "raw_compactness": raw_debug,
            "decoder_compactness": decoder_debug,
            "bands": band_debug,
        })


__all__ = [
    "DENP_BANDS",
    "DENP_TRAIN_VARIANTS",
    "DENPPurifier",
    "LearnableBandGaussian",
    "LowFrequencyCompactness",
    "RobustBandNoiseEstimator",
]
