"""E1-centered dose-calibrated high-frequency purification for Experiment K.

The module reuses Experiment J's robust MAD estimator and learnable bandwise
Gaussian without changing either implementation.  It adds a learnable dose
that can fall back to E1 and an optional, read-only Gaussian-radial prior.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder_denp import LearnableBandGaussian, RobustBandNoiseEstimator


K_BANDS = ("H", "V", "D")


def _logit(probability):
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Probability must lie inside (0, 1), got {probability}")
    return math.log(probability / (1.0 - probability))


def build_gaussian_radial_offsets(num_points=32, radius=2.0):
    """Build the exact float64 Gaussian-radial geometry used by Phase 1."""

    if int(num_points) != 32:
        raise ValueError("Experiment K requires exactly 32 Gaussian-radial points")
    quantiles = (torch.arange(num_points, dtype=torch.float64) + 0.5) / num_points
    radii = torch.sqrt(-2.0 * torch.log(1.0 - 0.85 * quantiles))
    angles = torch.arange(num_points, dtype=torch.float64) * (
        math.pi * (3.0 - math.sqrt(5.0))
    )
    points = torch.stack(
        (radii * torch.cos(angles), radii * torch.sin(angles)), dim=-1
    )
    points = points / points.norm(dim=-1).max() * float(radius)
    return points


def sample_offsets(feature, offsets, chunk_size=8):
    """Bilinearly sample NCHW ``feature`` at dense pixel-space offsets.

    Returned shape is N,C,H,W,K.  Concatenating K grids along output width
    keeps each grid_sample call small while preserving exact offset ordering.
    """

    if feature.ndim != 4:
        raise RuntimeError(f"Expected NCHW feature, got {tuple(feature.shape)}")
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise RuntimeError(f"Expected Kx2 offsets, got {tuple(offsets.shape)}")
    n, _, height, width = feature.shape
    work_dtype = torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(height, device=feature.device, dtype=work_dtype),
        torch.arange(width, device=feature.device, dtype=work_dtype),
        indexing="ij",
    )
    sampled = []
    offsets = offsets.to(device=feature.device, dtype=work_dtype)
    for start in range(0, offsets.shape[0], int(chunk_size)):
        chunk = offsets[start:start + int(chunk_size)]
        grids = []
        for offset in chunk:
            sample_x = xx + offset[0]
            sample_y = yy + offset[1]
            grid_x = (2.0 * sample_x + 1.0) / max(width, 1) - 1.0
            grid_y = (2.0 * sample_y + 1.0) / max(height, 1) - 1.0
            grids.append(torch.stack((grid_x, grid_y), dim=-1))
        grid = torch.cat(grids, dim=1).unsqueeze(0).expand(n, -1, -1, -1)
        values = F.grid_sample(
            feature.float(), grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        )
        values = values.reshape(n, feature.shape[1], height, len(chunk), width)
        sampled.append(values.permute(0, 1, 2, 4, 3))
    return torch.cat(sampled, dim=-1).to(dtype=feature.dtype)


class GaussianRadialCompactness(nn.Module):
    """Fixed Gaussian-radial protection evidence with a detached source."""

    def __init__(self, radius=2.0, epsilon=1e-6):
        super().__init__()
        offsets = build_gaussian_radial_offsets(radius=radius)
        radii = offsets.norm(dim=-1)
        self.register_buffer("offsets", offsets.float(), persistent=True)
        self.register_buffer("inner_mask", radii <= 1.0, persistent=True)
        self.register_buffer("outer_mask", radii > 1.0, persistent=True)
        self.epsilon = float(epsilon)
        if int(self.inner_mask.sum()) != 14 or int(self.outer_mask.sum()) != 18:
            raise RuntimeError("Gaussian-radial split must be inner=14 and outer=18")

    def forward(self, low):
        if low is None or low.ndim != 4:
            raise RuntimeError("Gaussian-radial compactness requires an NCHW low source")
        source = low.detach()
        energy = source.float().abs().mean(dim=1, keepdim=True)
        samples = sample_offsets(energy, self.offsets, chunk_size=8).float()
        inner = samples[..., self.inner_mask].mean(dim=-1)
        outer = samples[..., self.outer_mask].mean(dim=-1)
        support = 0.5 * (energy + inner)
        ratio = (support + self.epsilon) / (outer + self.epsilon)
        protection = (
            F.relu(ratio - 1.0) / (ratio + self.epsilon)
        ).clamp(0.0, 1.0)
        return protection.to(dtype=low.dtype), {
            "energy": energy,
            "inner": inner,
            "outer": outer,
            "support": support,
            "ratio": ratio,
            "protection": protection,
        }


class StagePriorProtection(nn.Module):
    """A bounded stage-wise protection strength in (0, 0.5)."""

    def __init__(self, rho_init=0.05, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)
        self.rho_init = float(rho_init)
        if self.enabled:
            self.rho_logit = nn.Parameter(torch.tensor(
                _logit(self.rho_init / 0.5), dtype=torch.float32
            ))

    @property
    def rho(self):
        if not self.enabled:
            return None
        return 0.5 * torch.sigmoid(self.rho_logit)

    def reset_control_parameters(self):
        if self.enabled:
            with torch.no_grad():
                self.rho_logit.fill_(_logit(self.rho_init / 0.5))


class DoseCalibratedBandPurifier(nn.Module):
    """One stage of learnable E1-to-J1 Gaussian purification dose."""

    def __init__(
        self,
        channels,
        alpha_init=0.05,
        protection_enabled=False,
        rho_init=0.05,
        learnable_alpha=True,
        fixed_alpha=None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.alpha_init = float(alpha_init)
        self.learnable_alpha = bool(learnable_alpha)
        self.fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)
        if self.learnable_alpha:
            self.alpha_logits = nn.Parameter(torch.full(
                (3,), _logit(self.alpha_init), dtype=torch.float32
            ))
        elif self.fixed_alpha is None:
            raise ValueError("A non-learnable purifier requires fixed_alpha")
        self.noise_estimators = nn.ModuleDict({
            band: RobustBandNoiseEstimator() for band in K_BANDS
        })
        self.gaussians = nn.ModuleDict({
            band: LearnableBandGaussian(self.channels) for band in K_BANDS
        })
        self.protection_enabled = bool(protection_enabled)
        if self.protection_enabled:
            self.compactness = GaussianRadialCompactness()
            self.prior_protection = StagePriorProtection(rho_init=rho_init, enabled=True)

    @property
    def alpha(self):
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logits)
        return torch.full(
            (3,), self.fixed_alpha, device=next(self.parameters()).device,
            dtype=torch.float32,
        )

    @property
    def rho(self):
        return self.prior_protection.rho if self.protection_enabled else None

    def reset_control_parameters(self):
        for estimator in self.noise_estimators.values():
            estimator.reset_control_parameters()
        for gaussian in self.gaussians.values():
            gaussian.reset_control_parameters()
        if self.learnable_alpha:
            with torch.no_grad():
                self.alpha_logits.fill_(_logit(self.alpha_init))
        if self.protection_enabled:
            self.prior_protection.reset_control_parameters()

    def forward(
        self,
        aligned_h,
        aligned_v,
        aligned_d,
        prior_low=None,
        alpha_override=None,
        rho_override=None,
        spatial_dose_override=None,
    ):
        bands = (aligned_h, aligned_v, aligned_d)
        if not (bands[0].shape == bands[1].shape == bands[2].shape):
            raise RuntimeError("Experiment K requires equal aligned H/V/D shapes")
        if bands[0].shape[1] != self.channels:
            raise RuntimeError(
                f"Expected {self.channels} aligned channels, got {bands[0].shape[1]}"
            )
        protection = compactness_debug = None
        rho = None
        if self.protection_enabled:
            protection, compactness_debug = self.compactness(prior_low)
            rho = self.rho if rho_override is None else torch.as_tensor(
                rho_override, device=bands[0].device, dtype=torch.float32
            )

        if alpha_override is None:
            alpha = self.alpha
        else:
            alpha = torch.full(
                (3,), float(alpha_override), device=bands[0].device, dtype=torch.float32
            )

        outputs, band_debug = [], {}
        bypass_protection = rho_override is not None and float(rho_override) == 0.0
        for index, (name, band) in enumerate(zip(K_BANDS, bands)):
            noise, noise_debug = self.noise_estimators[name](band)
            gaussian = self.gaussians[name](band)
            alpha_band = alpha[index].to(device=band.device, dtype=band.dtype).view(1, 1, 1, 1)
            dose = alpha_band * noise
            if protection is not None and not bypass_protection:
                dose = dose * (1.0 - rho.to(dtype=band.dtype) * protection)
            if spatial_dose_override is not None:
                override = spatial_dose_override
                if isinstance(spatial_dose_override, dict):
                    override = spatial_dose_override.get(name)
                if override is not None:
                    override = override.to(device=band.device, dtype=band.dtype)
                    dose = torch.where(override >= 0.0, override, dose)

            if alpha_override is not None and float(alpha_override) == 0.0:
                output = band
            elif ((not self.learnable_alpha and self.fixed_alpha == 1.0)
                  or (alpha_override is not None and float(alpha_override) == 1.0)) \
                    and (protection is None or bypass_protection) \
                    and spatial_dose_override is None:
                # Preserve the exact J1 arithmetic order for the strict regression.
                # A spatial override is an explicit counterfactual intervention and
                # must use the dose computed above instead of bypassing it here.
                output = (1.0 - noise) * band + noise * gaussian
                dose = noise
            else:
                output = band + dose * (gaussian - band)
            outputs.append(output)
            band_debug[name] = {
                "aligned": band,
                "noise_confidence": noise,
                "noise": noise_debug,
                "gaussian": gaussian,
                "gaussian_residual": gaussian - band,
                "alpha": alpha_band,
                "dose": dose,
                "purified": output,
            }
        return (*outputs, {
            "prior_low": prior_low,
            "protection": protection,
            "compactness": compactness_debug,
            "rho": rho,
            "bands": band_debug,
        })


__all__ = [
    "DoseCalibratedBandPurifier",
    "GaussianRadialCompactness",
    "K_BANDS",
    "StagePriorProtection",
    "build_gaussian_radial_offsets",
    "sample_offsets",
]
