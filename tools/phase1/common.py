"""Shared data, geometry, statistics, and feature-extraction utilities.

The Phase 1 tools are intentionally read-only with respect to models and
checkpoints.  They operate on raw datasets and, when requested, expose E1
intermediate tensors through forward hooks without modifying model code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize, stats


EPS = 1e-8
IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
NORM_CONFIGS = {
    "NUAA-SIRST": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "NUDT-SIRST": {"mean": 107.80905151367188, "std": 33.02274703979492},
    "IRSTD-1K": {"mean": 87.4661865234375, "std": 39.71953201293945},
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(path, rows, fieldnames=None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path, block_size=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_head(root=None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def runtime_metadata(dataset_dir=None, checkpoint=None, command=None):
    metadata = {
        "timestamp": now_iso(),
        "git_commit": git_head(Path(__file__).resolve().parents[2]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "command": command,
    }
    try:
        import scipy

        metadata["scipy"] = scipy.__version__
    except ImportError:
        metadata["scipy"] = None
    try:
        import torch

        metadata.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available() else None
            ),
        })
    except ImportError:
        metadata.update({"torch": None, "cuda": None, "gpu": None})
    if checkpoint:
        metadata["checkpoint"] = str(checkpoint)
        metadata["checkpoint_sha256"] = sha256_file(checkpoint)
    return metadata


def split_file(dataset_dir, dataset_name, split) -> Path:
    return (
        Path(dataset_dir) / dataset_name / "img_idx"
        / f"{split}_{dataset_name}.txt"
    )


def read_split(dataset_dir, dataset_name, split):
    path = split_file(dataset_dir, dataset_name, split)
    with path.open("r", encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip()]


def locate_image_mask(dataset_dir, dataset_name, image_id):
    root = Path(dataset_dir) / dataset_name
    image_path = None
    mask_path = None
    for extension in IMAGE_EXTENSIONS:
        candidate = root / "images" / f"{image_id}{extension}"
        if candidate.is_file():
            image_path = candidate
            break
    for suffix in ("", "_pixels0"):
        for extension in IMAGE_EXTENSIONS:
            candidate = root / "masks" / f"{image_id}{suffix}{extension}"
            if candidate.is_file():
                mask_path = candidate
                break
        if mask_path is not None:
            break
    if image_path is None or mask_path is None:
        raise FileNotFoundError(
            f"Missing image/mask pair for {dataset_name}/{image_id} under {root}"
        )
    return image_path, mask_path


def load_sample(dataset_dir, dataset_name, image_id):
    image_path, mask_path = locate_image_mask(dataset_dir, dataset_name, image_id)
    raw = np.asarray(Image.open(image_path).convert("I"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > (0.5 if mask.max() <= 1.0 else 127.5)
    norm = NORM_CONFIGS.get(dataset_name)
    if norm is None:
        norm = {"mean": float(raw.mean()), "std": float(raw.std() + EPS)}
    normalized = (raw - norm["mean"]) / max(norm["std"], EPS)
    return {
        "raw": raw,
        "linear": (raw - raw.min()) / max(float(raw.max() - raw.min()), EPS),
        "normalized": normalized.astype(np.float32),
        "mask": mask,
        "image_path": image_path,
        "mask_path": mask_path,
        "height": int(raw.shape[0]),
        "width": int(raw.shape[1]),
    }


def pad_to_multiple(array, multiple=32, value=0.0):
    height, width = array.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    padding = [(0, 0)] * array.ndim
    padding[-2] = (0, pad_h)
    padding[-1] = (0, pad_w)
    return np.pad(array, padding, mode="constant", constant_values=value)


def connected_components(mask):
    labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3)))
    records = []
    for component_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == component_id)
        if not len(xs):
            continue
        area = int(len(xs))
        cx, cy = float(xs.mean()), float(ys.mean())
        if area > 1:
            covariance = np.cov(np.stack([xs, ys]), bias=True)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            vector = eigenvectors[:, order[0]]
            major = float(4.0 * math.sqrt(eigenvalues[0] + EPS))
            minor = float(4.0 * math.sqrt(eigenvalues[1] + EPS))
            orientation = float(math.atan2(vector[1], vector[0]))
        else:
            major = minor = 1.0
            orientation = 0.0
        records.append({
            "component_id": component_id,
            "area": area,
            "bbox_x0": int(xs.min()), "bbox_y0": int(ys.min()),
            "bbox_x1": int(xs.max() + 1), "bbox_y1": int(ys.max() + 1),
            "binary_centroid_x": cx, "binary_centroid_y": cy,
            "equivalent_radius": float(math.sqrt(area / math.pi)),
            "equivalent_diameter": float(2.0 * math.sqrt(area / math.pi)),
            "major_axis": major, "minor_axis": minor,
            "mask_orientation": orientation,
        })
    return labels, records


def _intensity_center(image, background, region_mask, fallback_x, fallback_y):
    weights = np.maximum(image - background, 0.0) * region_mask
    total = float(weights.sum())
    if total <= EPS:
        return float(fallback_x), float(fallback_y)
    ys, xs = np.indices(image.shape)
    return (
        float((xs * weights).sum() / total),
        float((ys * weights).sum() / total),
    )


def build_candidate_catalog(
    image,
    mask,
    dataset,
    split,
    image_id,
    hard_per_target=2,
    easy_per_target=1,
    seed=42,
):
    """Build target, intensity-matched hard-negative, and easy catalogs."""
    labels, components = connected_components(mask)
    background = ndimage.gaussian_filter(image.astype(np.float64), sigma=3.0)
    residual = image.astype(np.float64) - background
    exclusion = np.zeros_like(mask, dtype=bool)
    targets = []
    for component in components:
        component_mask = labels == component["component_id"]
        radius = int(math.ceil(max(5.0, 3.0 * component["equivalent_radius"])))
        exclusion |= ndimage.binary_dilation(component_mask, iterations=radius)
        ix, iy = _intensity_center(
            image, background, component_mask,
            component["binary_centroid_x"], component["binary_centroid_y"],
        )
        values = residual[component_mask]
        peak_index = np.argmax(values)
        peak_y, peak_x = np.argwhere(component_mask)[peak_index]
        record = dict(component)
        record.update({
            "dataset": dataset, "split": split, "image_id": image_id,
            "instance_id": f"{image_id}:target:{component['component_id']}",
            "candidate_id": f"{image_id}:target:{component['component_id']}",
            "sample_type": "target",
            "center_x": ix, "center_y": iy,
            "intensity_centroid_x": ix, "intensity_centroid_y": iy,
            "peak_x": float(peak_x), "peak_y": float(peak_y),
            "peak_contrast": float(residual[peak_y, peak_x]),
            "boundary_truncated": bool(
                component["bbox_x0"] == 0 or component["bbox_y0"] == 0
                or component["bbox_x1"] == image.shape[1]
                or component["bbox_y1"] == image.shape[0]
            ),
        })
        targets.append(record)

    allowed = ~exclusion
    maxima = residual == ndimage.maximum_filter(residual, size=3, mode="nearest")
    allowed_values = residual[allowed]
    threshold = (
        float(np.percentile(allowed_values, 85.0)) if allowed_values.size else np.inf
    )
    peak_mask = allowed & maxima & (residual > max(threshold, 0.0))
    peak_positions = np.argwhere(peak_mask)
    if len(peak_positions):
        peak_positions = peak_positions[
            np.argsort(residual[peak_positions[:, 0], peak_positions[:, 1]])[::-1]
        ]

    used = []
    hard = []
    for target in targets:
        if not len(peak_positions):
            break
        target_contrast = target["peak_contrast"]
        scores = []
        for py, px in peak_positions:
            if any((px - ux) ** 2 + (py - uy) ** 2 < 25 for uy, ux in used):
                continue
            contrast = float(residual[py, px])
            closeness = abs(math.log((max(contrast, EPS)) / max(target_contrast, EPS)))
            local = residual[max(0, py - 2): py + 3, max(0, px - 2): px + 3]
            high_energy = float(np.mean(np.abs(local))) if local.size else 0.0
            scores.append((closeness - 0.05 * high_energy, int(py), int(px), contrast))
        for _, py, px, contrast in sorted(scores)[:hard_per_target]:
            used.append((py, px))
            index = len(hard) + 1
            hard.append({
                "dataset": dataset, "split": split, "image_id": image_id,
                "component_id": -1,
                "instance_id": f"{image_id}:hard:{index}",
                "candidate_id": f"{image_id}:hard:{index}",
                "sample_type": "hard_negative",
                "matched_component_id": target["component_id"],
                "area": target["area"],
                "equivalent_radius": target["equivalent_radius"],
                "equivalent_diameter": target["equivalent_diameter"],
                "major_axis": target["major_axis"],
                "minor_axis": target["minor_axis"],
                "mask_orientation": target["mask_orientation"],
                "binary_centroid_x": float(px), "binary_centroid_y": float(py),
                "intensity_centroid_x": float(px), "intensity_centroid_y": float(py),
                "center_x": float(px), "center_y": float(py),
                "peak_x": float(px), "peak_y": float(py),
                "peak_contrast": contrast,
                "boundary_truncated": False,
            })

    rng = np.random.default_rng(seed + sum(map(ord, str(image_id))))
    easy = []
    easy_forbidden = exclusion.copy()
    for py, px in used:
        marker = np.zeros_like(mask, dtype=bool)
        marker[py, px] = True
        easy_forbidden |= ndimage.binary_dilation(marker, iterations=7)
    easy_positions = np.argwhere(~easy_forbidden)
    requested = easy_per_target * max(1, len(targets))
    if len(easy_positions):
        selected = rng.choice(len(easy_positions), size=min(requested, len(easy_positions)), replace=False)
        for index, position_index in enumerate(np.atleast_1d(selected), start=1):
            py, px = easy_positions[int(position_index)]
            template = targets[(index - 1) % len(targets)] if targets else {
                "area": 1, "equivalent_radius": 1 / math.sqrt(math.pi),
                "equivalent_diameter": 2 / math.sqrt(math.pi),
                "major_axis": 1.0, "minor_axis": 1.0, "mask_orientation": 0.0,
            }
            easy.append({
                "dataset": dataset, "split": split, "image_id": image_id,
                "component_id": -1,
                "instance_id": f"{image_id}:easy:{index}",
                "candidate_id": f"{image_id}:easy:{index}",
                "sample_type": "easy_background",
                "matched_component_id": template.get("component_id", -1),
                "area": template["area"],
                "equivalent_radius": template["equivalent_radius"],
                "equivalent_diameter": template["equivalent_diameter"],
                "major_axis": template["major_axis"],
                "minor_axis": template["minor_axis"],
                "mask_orientation": template["mask_orientation"],
                "binary_centroid_x": float(px), "binary_centroid_y": float(py),
                "intensity_centroid_x": float(px), "intensity_centroid_y": float(py),
                "center_x": float(px), "center_y": float(py),
                "peak_x": float(px), "peak_y": float(py),
                "peak_contrast": float(residual[py, px]),
                "boundary_truncated": False,
            })
    return targets + hard + easy, exclusion


def extract_patch(array, center_x, center_y, radius, order=1):
    radius = int(radius)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(offsets + center_y, offsets + center_x, indexing="ij")
    patch = ndimage.map_coordinates(
        np.asarray(array, dtype=np.float64), [yy, xx], order=order, mode="nearest"
    )
    truncated = bool(
        center_x - radius < 0 or center_y - radius < 0
        or center_x + radius >= array.shape[1]
        or center_y + radius >= array.shape[0]
    )
    return patch, truncated


def estimate_background(patch, method="local_plane", inner_radius=3.0):
    patch = np.asarray(patch, dtype=np.float64)
    height, width = patch.shape
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outer = rr >= max(inner_radius, 0.35 * min(height, width))
    if outer.sum() < 3:
        outer = np.ones_like(patch, dtype=bool)
    if method == "annulus_median":
        return np.full_like(patch, float(np.median(patch[outer])))
    if method != "local_plane":
        raise ValueError(f"Unknown background method: {method}")
    design = np.stack([np.ones(outer.sum()), xx[outer], yy[outer]], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, patch[outer], rcond=None)
    return coefficients[0] + coefficients[1] * xx + coefficients[2] * yy


def elliptical_gaussian(parameters, xx, yy):
    amplitude, x0, y0, sigma_x, sigma_y, theta, residual = parameters
    cosine, sine = math.cos(theta), math.sin(theta)
    xp = cosine * (xx - x0) + sine * (yy - y0)
    yp = -sine * (xx - x0) + cosine * (yy - y0)
    exponent = -0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2)
    return amplitude * np.exp(exponent) + residual


def _weighted_moments(patch):
    values = np.maximum(patch - np.percentile(patch, 20), 0.0)
    total = float(values.sum())
    height, width = patch.shape
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    if total <= EPS:
        return (width - 1) / 2, (height - 1) / 2, 1.0, 1.0, 0.0
    x0 = float((xx * values).sum() / total)
    y0 = float((yy * values).sum() / total)
    dx, dy = xx - x0, yy - y0
    covariance = np.array([
        [(values * dx * dx).sum(), (values * dx * dy).sum()],
        [(values * dx * dy).sum(), (values * dy * dy).sum()],
    ]) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.25)
    vector = eigenvectors[:, order[0]]
    return (
        x0, y0, float(math.sqrt(eigenvalues[0])),
        float(math.sqrt(eigenvalues[1])), float(math.atan2(vector[1], vector[0])),
    )


def _safe_correlation(a, b, method="pearson"):
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    function = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(function(a, b).statistic)


def fit_gaussian_patch(patch, expected_center=None):
    patch = np.asarray(patch, dtype=np.float64)
    height, width = patch.shape
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    x0, y0, sigma_x, sigma_y, theta = _weighted_moments(patch)
    amplitude = max(float(patch.max() - np.median(patch)), EPS)
    residual = float(np.median(patch))
    initial = np.array([amplitude, x0, y0, sigma_x, sigma_y, theta, residual])
    scale = max(float(np.ptp(patch)), float(np.std(patch)), 1.0)
    lower = [EPS, 0.0, 0.0, 0.3, 0.3, -math.pi / 2, -2.0 * scale]
    upper = [max(amplitude * 10, scale * 10), width - 1, height - 1,
             width, height, math.pi / 2, 2.0 * scale]

    def residuals(parameters):
        return (elliptical_gaussian(parameters, xx, yy) - patch).ravel()

    result = optimize.least_squares(
        residuals, np.clip(initial, lower, upper), bounds=(lower, upper),
        loss="soft_l1", f_scale=max(float(np.std(patch)), 1e-3), max_nfev=500,
    )
    parameters = result.x
    fitted = elliptical_gaussian(parameters, xx, yy)
    errors = patch - fitted
    ss_res = float(np.square(errors).sum())
    ss_total = float(np.square(patch - patch.mean()).sum())
    r2 = 1.0 - ss_res / max(ss_total, EPS)
    n, p = patch.size, len(parameters)
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    amplitude, fitted_x, fitted_y, sx, sy, fitted_theta, fitted_c = parameters
    if sx >= sy:
        sigma_major, sigma_minor, orientation = sx, sy, fitted_theta
    else:
        sigma_major, sigma_minor = sy, sx
        orientation = fitted_theta + math.pi / 2
    orientation = ((orientation + math.pi / 2) % math.pi) - math.pi / 2
    cosine, sine = math.cos(fitted_theta), math.sin(fitted_theta)
    xp = cosine * (xx - fitted_x) + sine * (yy - fitted_y)
    yp = -sine * (xx - fitted_x) + cosine * (yy - fitted_y)
    rho = np.sqrt((xp / max(sx, EPS)) ** 2 + (yp / max(sy, EPS)) ** 2)
    bins = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3)]
    positive = np.maximum(patch - fitted_c, 0.0)
    radial = []
    for lo, hi in bins:
        selection = (rho >= lo) & (rho < hi)
        radial.append(float(positive[selection].mean()) if selection.any() else 0.0)
    monotonicity = sum(
        radial[index] >= radial[index + 1] for index in range(len(radial) - 1)
    ) / max(len(radial) - 1, 1)
    inner = positive[rho < 1]
    outer = positive[(rho >= 2) & (rho < 3)]
    compactness = (
        float(inner.mean() / (outer.mean() + EPS))
        if inner.size and outer.size else 0.0
    )
    center_energy = float(positive[rho < 0.5].mean()) if (rho < 0.5).any() else 0.0
    inner_ring = float(positive[(rho >= 0.5) & (rho < 1.5)].mean()) if ((rho >= 0.5) & (rho < 1.5)).any() else 0.0
    outer_ring = float(positive[(rho >= 2) & (rho < 3)].mean()) if ((rho >= 2) & (rho < 3)).any() else 0.0
    weighted_x, weighted_y, *_ = _weighted_moments(patch)
    peak_y, peak_x = np.unravel_index(np.argmax(patch), patch.shape)
    expected_x, expected_y = expected_center or ((width - 1) / 2, (height - 1) / 2)
    slopes = np.diff(radial)
    return {
        "fit_success": bool(result.success),
        "fit_status": int(result.status),
        "fit_message": str(result.message),
        "R2": float(r2), "adjusted_R2": float(adjusted_r2),
        "NRMSE": float(math.sqrt(np.mean(errors ** 2)) / max(np.ptp(patch), EPS)),
        "MAE": float(np.mean(np.abs(errors))),
        "pearson": _safe_correlation(patch.ravel(), fitted.ravel(), "pearson"),
        "spearman": _safe_correlation(patch.ravel(), fitted.ravel(), "spearman"),
        "amplitude": float(amplitude),
        "fitted_x": float(fitted_x), "fitted_y": float(fitted_y),
        "sigma_major": float(sigma_major), "sigma_minor": float(sigma_minor),
        "sigma_ratio": float(sigma_major / max(sigma_minor, EPS)),
        "FWHM_major": float(2.354820045 * sigma_major),
        "FWHM_minor": float(2.354820045 * sigma_minor),
        "theta": float(orientation), "residual_background": float(fitted_c),
        "fitted_center_offset_to_reference": float(
            math.hypot(fitted_x - expected_x, fitted_y - expected_y)
        ),
        "fitted_center_offset_to_intensity": float(
            math.hypot(fitted_x - weighted_x, fitted_y - weighted_y)
        ),
        "fitted_center_offset_to_peak": float(
            math.hypot(fitted_x - peak_x, fitted_y - peak_y)
        ),
        "peak_background_contrast": float(patch.max() - np.median(patch)),
        "center_inner_energy_ratio": float(center_energy / (inner_ring + EPS)),
        "center_outer_energy_ratio": float(center_energy / (outer_ring + EPS)),
        "inner_outer_energy_ratio": float(inner_ring / (outer_ring + EPS)),
        "radial_monotonicity": float(monotonicity),
        "radial_decay_slope": float(np.mean(slopes)) if len(slopes) else 0.0,
        "radial_profile_residual": float(np.mean(np.abs(errors))),
        "compactness": compactness,
        "orientation_anisotropy": float(sigma_major / max(sigma_minor, EPS)),
        "radial_bins": radial,
    }


def analyze_gaussian_candidate(image, candidate, background_method, radius_scale, intensity_mode):
    radius = int(min(20, max(7, math.ceil(radius_scale * candidate["equivalent_radius"]))))
    patch, truncated = extract_patch(
        image, candidate["center_x"], candidate["center_y"], radius, order=1
    )
    if intensity_mode == "linear_normalized":
        patch = (patch - patch.min()) / max(float(np.ptp(patch)), EPS)
    background = estimate_background(
        patch, method=background_method,
        inner_radius=max(3.0, 2.0 * candidate["equivalent_radius"]),
    )
    residual = patch - background
    record = dict(candidate)
    record.update({
        "background_method": background_method,
        "radius_scale": radius_scale,
        "intensity_mode": intensity_mode,
        "patch_radius": radius,
        "patch_truncated": truncated,
    })
    try:
        record.update(fit_gaussian_patch(residual, expected_center=(radius, radius)))
    except Exception as error:
        record.update({
            "fit_success": False, "fit_status": -1,
            "fit_message": f"{type(error).__name__}: {error}",
        })
    return record


def finite_values(rows, key, sample_type=None):
    values = []
    for row in rows:
        if sample_type is not None and row.get("sample_type") != sample_type:
            continue
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y):
        return float("nan")
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def roc_auc_positive_high(positive, negative):
    if not len(positive) or not len(negative):
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = stats.rankdata(combined)
    rank_sum = ranks[: len(positive)].sum()
    u = rank_sum - len(positive) * (len(positive) + 1) / 2
    return float(u / (len(positive) * len(negative)))


def pr_auc_positive_high(positive, negative):
    if not len(positive) or not len(negative):
        return float("nan")
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    recall = tp / max(len(positive), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def best_balanced_accuracy(positive, negative):
    if not len(positive) or not len(negative):
        return float("nan")
    scores = np.unique(np.concatenate([positive, negative]))
    if len(scores) > 512:
        scores = np.quantile(scores, np.linspace(0, 1, 512))
    best = 0.0
    for threshold in scores:
        sensitivity = float(np.mean(positive >= threshold))
        specificity = float(np.mean(negative < threshold))
        best = max(best, 0.5 * (sensitivity + specificity))
    return best


def bootstrap_effect_ci(x, y, repeats=1000, seed=42):
    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        values.append(cliffs_delta(xb, yb))
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def compare_two_groups(rows, metric, group_a="target", group_b="hard_negative", bootstrap=1000, seed=42):
    a = finite_values(rows, metric, group_a)
    b = finite_values(rows, metric, group_b)
    if not len(a) or not len(b):
        return {
            "metric": metric, "group_a": group_a, "group_b": group_b,
            "n_a": len(a), "n_b": len(b), "insufficient": True,
        }
    test = stats.mannwhitneyu(a, b, alternative="two-sided")
    effect = cliffs_delta(a, b)
    ci_low, ci_high = bootstrap_effect_ci(a, b, bootstrap, seed)
    return {
        "metric": metric, "group_a": group_a, "group_b": group_b,
        "n_a": int(len(a)), "n_b": int(len(b)),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "q25_a": float(np.percentile(a, 25)), "q75_a": float(np.percentile(a, 75)),
        "q25_b": float(np.percentile(b, 25)), "q75_b": float(np.percentile(b, 75)),
        "mannwhitney_u": float(test.statistic), "p_value": float(test.pvalue),
        "cliffs_delta": effect, "cliffs_delta_ci_low": ci_low,
        "cliffs_delta_ci_high": ci_high,
        "roc_auc": roc_auc_positive_high(a, b),
        "pr_auc": pr_auc_positive_high(a, b),
        "best_balanced_accuracy": best_balanced_accuracy(a, b),
        "insufficient": bool(len(a) < 30 or len(b) < 30),
    }


def benjamini_hochberg(records, p_key="p_value", output_key="p_fdr"):
    valid = [(index, float(record[p_key])) for index, record in enumerate(records)
             if record.get(p_key) is not None and np.isfinite(float(record[p_key]))]
    if not valid:
        return records
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        _, pvalue = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, pvalue * len(ordered) / rank)
        adjusted[reverse_index] = min(running, 1.0)
    for (record_index, _), value in zip(ordered, adjusted):
        records[record_index][output_key] = value
    return records


def paired_consistency(a, b, expected_sign):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pair = 1.0 - np.abs(a - expected_sign * b) / (np.abs(a) + np.abs(b) + EPS)
    balance = 2.0 * np.minimum(np.abs(a), np.abs(b)) / (np.abs(a) + np.abs(b) + EPS)
    weights = np.abs(a) + np.abs(b)
    return float(np.sum(pair * balance * weights) / (weights.sum() + EPS))


def quadrant_consistency(values, template):
    values = np.asarray(values, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    cosine = np.sum(values * template, axis=1) / (
        np.linalg.norm(values, axis=1) * np.linalg.norm(template) + EPS
    )
    sign_agreement = np.mean(np.sign(values) == np.sign(template), axis=1)
    magnitudes = np.abs(values)
    balance = magnitudes.min(axis=1) / (magnitudes.max(axis=1) + EPS)
    weights = magnitudes.sum(axis=1)
    denominator = max(float(weights.sum()), EPS)
    return {
        "cosine": float(np.sum(cosine * weights) / denominator),
        "sign_agreement": float(np.sum(sign_agreement * weights) / denominator),
        "amplitude_balance": float(np.sum(balance * weights) / denominator),
    }


def bilinear_sample(feature, points):
    feature = np.asarray(feature, dtype=np.float64)
    if feature.ndim == 2:
        feature = feature[None, ...]
    points = np.asarray(points, dtype=np.float64)
    coordinates = np.stack([points[:, 1], points[:, 0]], axis=0)
    return np.stack([
        ndimage.map_coordinates(channel, coordinates, order=1, mode="nearest")
        for channel in feature
    ], axis=0)


def sample_pair_metric(feature, center, axis, radius, expected_sign):
    x, y = center
    if axis == "x":
        points = np.array([[x - radius, y], [x + radius, y]])
    else:
        points = np.array([[x, y - radius], [x, y + radius]])
    values = bilinear_sample(feature, points)
    return paired_consistency(values[:, 0], values[:, 1], expected_sign)


def sample_quadrants(feature, center, radius, template):
    x, y = center
    points = np.array([
        [x + radius, y + radius], [x - radius, y + radius],
        [x - radius, y - radius], [x + radius, y - radius],
    ])
    return quadrant_consistency(bilinear_sample(feature, points), template)


def radial_compactness(feature, center, scale=1.0, max_radius=4.0):
    feature = np.asarray(feature, dtype=np.float64)
    if feature.ndim == 3:
        magnitude = np.mean(np.abs(feature), axis=0)
    else:
        magnitude = np.abs(feature)
    radius = max(4, int(math.ceil(max_radius * max(scale, 0.5))))
    patch, _ = extract_patch(magnitude, center[0], center[1], radius, order=1)
    yy, xx = np.indices(patch.shape)
    c = radius
    rho = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / max(scale, 0.5)
    center_energy = patch[rho < 1].mean() if (rho < 1).any() else 0.0
    inner_energy = patch[(rho >= 1) & (rho < 2)].mean() if ((rho >= 1) & (rho < 2)).any() else 0.0
    outer_energy = patch[(rho >= 2) & (rho < 4)].mean() if ((rho >= 2) & (rho < 4)).any() else 0.0
    return float((center_energy + inner_energy) / (outer_energy + EPS))


def geometry_points(name, radius, num_points=32, seed=42):
    if num_points != 32:
        raise ValueError("Phase 1 fairness protocol currently requires num_points=32")
    radius = float(radius)
    if name == "grid":
        axis = np.linspace(-1.0, 1.0, 6)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        order = np.argsort(np.linalg.norm(points, axis=1))[:num_points]
        points = points[order]
    elif name == "ring":
        radii = np.linspace(0.25, 1.0, 4)
        angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        points = np.array([[r * math.cos(a), r * math.sin(a)] for r in radii for a in angles])
    elif name == "spiral":
        points = []
        for head in range(8):
            for index in range(4):
                angle = 2 * math.pi * index / 4 + 2 * math.pi * head / 8
                r = (index + 1) / 4
                points.append([r * math.cos(angle), r * math.sin(angle)])
        points = np.asarray(points)
    elif name == "random":
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * math.pi, num_points)
        radii = np.sqrt(rng.uniform(0, 1, num_points))
        points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    elif name == "gaussian_radial":
        quantiles = (np.arange(num_points) + 0.5) / num_points
        radii = np.sqrt(-2.0 * np.log(1.0 - 0.85 * quantiles))
        angles = np.arange(num_points) * (math.pi * (3 - math.sqrt(5)))
        points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    else:
        raise ValueError(f"Unknown geometry: {name}")
    max_norm = float(np.linalg.norm(points, axis=1).max())
    points = points / max(max_norm, EPS) * radius
    return points.astype(np.float64)


def center_perturbations(seed=42):
    perturbations = [{"name": "zero", "dx": 0.0, "dy": 0.0, "magnitude": 0.0}]
    rng = np.random.default_rng(seed)
    for magnitude in (0.5, 1.0, 2.0):
        for name, angle in (
            ("right", 0.0), ("left", math.pi), ("down", math.pi / 2),
            ("up", -math.pi / 2), ("diag_pos", math.pi / 4),
            ("diag_neg", -math.pi / 4), ("random", float(rng.uniform(-math.pi, math.pi))),
        ):
            perturbations.append({
                "name": f"{name}_{magnitude:g}",
                "dx": float(magnitude * math.cos(angle)),
                "dy": float(magnitude * math.sin(angle)),
                "magnitude": magnitude,
            })
    return perturbations


def paired_geometry_statistics(rows, metric, left="spiral", right="grid"):
    pairs = {}
    for row in rows:
        if row.get("geometry") not in (left, right):
            continue
        key = tuple(row.get(field) for field in (
            "dataset", "split", "image_id", "candidate_id", "stage",
            "feature_source", "radius", "perturbation", "random_repeat",
        ))
        try:
            pairs.setdefault(key, {})[row["geometry"]] = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
    differences = np.asarray([
        value[left] - value[right] for value in pairs.values()
        if left in value and right in value
    ], dtype=np.float64)
    if not len(differences):
        return {"metric": metric, "left": left, "right": right, "n": 0}
    nonzero = differences[np.abs(differences) > EPS]
    if len(nonzero):
        test = stats.wilcoxon(nonzero, alternative="two-sided")
        ranks = stats.rankdata(np.abs(nonzero))
        rank_effect = float(
            (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum())
            / max(float(ranks.sum()), EPS)
        )
    else:
        test = type("Result", (), {"statistic": 0.0, "pvalue": 1.0})()
        rank_effect = 0.0
    rng = np.random.default_rng(42)
    boot = [float(np.mean(differences[rng.integers(0, len(differences), len(differences))])) for _ in range(1000)]
    return {
        "metric": metric, "left": left, "right": right, "n": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "wilcoxon_statistic": float(test.statistic), "p_value": float(test.pvalue),
        "rank_biserial": rank_effect,
    }


def calibrate_haar(size=64, device="cpu"):
    """Calibrate code-band response axes, pair signs, and D template."""
    import torch
    from model.DWTFreqNet import HaarWaveletTransform, check_haar_direction_correspondence

    if size % 2:
        raise ValueError("Calibration size must be even")
    transform = HaarWaveletTransform().to(device)
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij"
    )
    center = (size - 1) / 2
    sigma = 4.0
    gaussian = torch.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))[None, None]
    impulse = torch.zeros_like(gaussian)
    impulse[..., size // 2, size // 2] = 1
    patterns = {"center_gaussian": gaussian, "impulse": impulse}
    line_index = size // 2 - 1
    horizontal = torch.zeros_like(gaussian); horizontal[..., line_index, :] = 1
    vertical = torch.zeros_like(gaussian); vertical[..., :, line_index] = 1
    diag_pos = torch.eye(size, device=device)[None, None]
    diag_neg = torch.fliplr(torch.eye(size, device=device))[None, None]
    horizontal_step = torch.zeros_like(gaussian); horizontal_step[..., line_index:, :] = 1
    vertical_step = torch.zeros_like(gaussian); vertical_step[..., :, line_index:] = 1
    patterns.update({
        "horizontal_line": horizontal, "vertical_line": vertical,
        "diag_pos_line": diag_pos, "diag_neg_line": diag_neg,
        "horizontal_step": horizontal_step, "vertical_step": vertical_step,
    })
    response_maps, energies = {}, {}
    with torch.no_grad():
        for name, image in patterns.items():
            bands = transform(image)
            response_maps[name] = [band[0, 0].cpu().numpy() for band in bands]
            energies[name] = {
                band_name: float(band.abs().sum().cpu())
                for band_name, band in zip(("LL", "H", "V", "D"), bands)
            }
    standard = check_haar_direction_correspondence(size=size, device=device)
    feature_center = ((size // 2) / 2.0, (size // 2) / 2.0)
    pair_rules = {}
    for band_name, band_index in (("H", 1), ("V", 2)):
        feature = response_maps["center_gaussian"][band_index]
        candidates = []
        for axis in ("x", "y"):
            for sign in (-1, 1):
                score = np.mean([
                    sample_pair_metric(feature, feature_center, axis, radius, sign)
                    for radius in (1, 2, 3, 4)
                ])
                candidates.append((score, axis, sign))
        score, axis, sign = max(candidates)
        pair_rules[band_name] = {
            "sensitive_axis": axis, "expected_sign": int(sign),
            "synthetic_score": float(score),
        }
    d_feature = response_maps["center_gaussian"][3]
    d_values = bilinear_sample(d_feature, np.array([
        [feature_center[0] + 2, feature_center[1] + 2],
        [feature_center[0] - 2, feature_center[1] + 2],
        [feature_center[0] - 2, feature_center[1] - 2],
        [feature_center[0] + 2, feature_center[1] - 2],
    ]))[0]
    d_template = np.where(d_values >= 0, 1, -1).astype(int).tolist()
    return {
        "size": size,
        "code_band_names": standard["code_band_names"],
        "band_response_orientation": standard["band_response_orientation"],
        "recommended_scan_axis": standard["recommended_scan_axis"],
        "routing_aligned": standard["routing_aligned"],
        "energies": energies,
        "pair_rules": pair_rules,
        "d_quadrant_template": d_template,
        "d_quadrant_values": d_values.tolist(),
    }


def checkpoint_state_dict(checkpoint, model):
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_keys = set(model.state_dict())
    if set(state_dict) == model_keys:
        return state_dict
    for prefix in ("model.", "module.", "module.model."):
        normalized = {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }
        if set(normalized) == model_keys:
            return normalized
    raise RuntimeError("Checkpoint keys do not strictly match the requested model")


class E1FeatureExtractor:
    """Expose E1 tensors through hooks without changing the model source."""

    def __init__(self, checkpoint, device="cuda:0"):
        import torch
        from model.Config import get_DWTFreqNet_config
        from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import DWTFreqNet_SingleDecoder_LFSS_AWGM

        self.torch = torch
        self.device = torch.device(device)
        self.model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
            get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
            mode="test", deepsuper=True,
        ).to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint_state_dict(payload, self.model), strict=True)
        self.model.eval(); self.model.debug_tensors = True; self.model.record_statistics = False
        self.raw_bands = []
        self.decoder_outputs = {}
        self.handles = [self.model.har.register_forward_hook(self._dwt_hook)]
        for name, stage in (("decoder_fuse3", 3), ("decoder_fuse2", 2), ("decoder_fuse1", 1)):
            self.handles.append(getattr(self.model, name).register_forward_hook(self._decoder_hook(stage)))

    def _dwt_hook(self, module, inputs, output):
        self.raw_bands.append(tuple(tensor.detach() for tensor in output))

    def _decoder_hook(self, stage):
        def hook(module, inputs, output):
            self.decoder_outputs[stage] = output.detach()
        return hook

    def extract(self, normalized_image):
        torch = self.torch
        self.raw_bands = []; self.decoder_outputs = {}
        padded = pad_to_multiple(np.asarray(normalized_image, dtype=np.float32), 32)
        tensor = torch.from_numpy(padded[None, None]).to(self.device)
        with torch.no_grad():
            prediction = self.model(tensor)
        debug = self.model.last_debug
        features = {}
        for stage, bands in enumerate(self.raw_bands, start=1):
            for name, tensor_value in zip(("raw_LL", "raw_H", "raw_V", "raw_D"), bands):
                features[(stage, name)] = tensor_value[0].float().cpu().numpy()
            features[(stage, "lfss_LL")] = debug["A_lfss"][stage][0].float().cpu().numpy()
            features[(stage, "guided_LL")] = debug["A_guided"][stage][0].float().cpu().numpy()
            for direction in ("H", "V", "D"):
                features[(stage, f"aligned_{direction}")] = (
                    debug["coefficients"][(stage, direction)]["aligned"][0].float().cpu().numpy()
                )
        features[(4, "decoder_low")] = debug["E"][4][0].float().cpu().numpy()
        for stage in (3, 2, 1):
            features[(stage, "decoder_low")] = self.decoder_outputs[stage][0].float().cpu().numpy()
        return prediction[0, 0].float().cpu().numpy(), features

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def map_center_to_feature(candidate, original_shape, feature_shape):
    original_h, original_w = original_shape
    feature_h, feature_w = feature_shape
    return (
        float(candidate["center_x"] * feature_w / original_w),
        float(candidate["center_y"] * feature_h / original_h),
    )


def map_scale_to_feature(candidate, original_shape, feature_shape):
    original_h, original_w = original_shape
    feature_h, feature_w = feature_shape
    scale = 0.5 * (feature_h / original_h + feature_w / original_w)
    return max(0.5, float(candidate["equivalent_radius"] * scale))
