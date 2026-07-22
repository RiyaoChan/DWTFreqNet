"""P1: validate Gaussian/elliptical compactness of IR small targets."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.phase1.common import (
    analyze_gaussian_candidate,
    benjamini_hochberg,
    build_candidate_catalog,
    compare_two_groups,
    elliptical_gaussian,
    ensure_dir,
    estimate_background,
    extract_patch,
    load_sample,
    now_iso,
    read_json,
    read_split,
    runtime_metadata,
    seed_everything,
    write_csv,
    write_json,
)


P1_METRICS = (
    "R2", "adjusted_R2", "NRMSE", "MAE", "pearson", "spearman",
    "sigma_major", "sigma_minor", "sigma_ratio", "FWHM_major", "FWHM_minor",
    "fitted_center_offset_to_reference", "fitted_center_offset_to_intensity",
    "fitted_center_offset_to_peak", "peak_background_contrast",
    "center_inner_energy_ratio", "center_outer_energy_ratio",
    "inner_outer_energy_ratio", "radial_monotonicity", "radial_decay_slope",
    "radial_profile_residual", "compactness", "orientation_anisotropy",
)
PRIMARY_METRICS = ("R2", "compactness", "radial_monotonicity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hard-per-target", type=int, default=2)
    parser.add_argument("--easy-per-target", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-visuals", type=int, default=12)
    return parser.parse_args()


def _assign_strata(candidates, thresholds):
    q33, q66 = thresholds["area_q33"], thresholds["area_q66"]
    contrast = thresholds["contrast_median"]
    for candidate in candidates:
        area = float(candidate.get("area", 0))
        candidate["size_group"] = (
            "tiny" if area <= q33 else "medium" if area <= q66 else "large"
        )
        candidate["contrast_group"] = (
            "low" if float(candidate.get("peak_contrast", 0)) <= contrast else "high"
        )


def _make_visuals(rows, samples, output_dir, limit):
    if not rows or limit <= 0:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    visual_dir = ensure_dir(Path(output_dir) / "visuals")
    generated = []
    primary = [row for row in rows if row.get("background_method") == "local_plane"
               and float(row.get("radius_scale", 0)) == 4.0
               and row.get("intensity_mode") == "raw"]
    for metric in PRIMARY_METRICS + ("sigma_ratio",):
        figure, axis = plt.subplots(figsize=(6, 4))
        for sample_type, color in (("target", "tab:red"), ("hard_negative", "tab:blue"),
                                   ("easy_background", "tab:gray")):
            values = [float(row[metric]) for row in primary
                      if row.get("sample_type") == sample_type and row.get(metric) is not None]
            values = [value for value in values if np.isfinite(value)]
            if values:
                axis.hist(values, bins=30, alpha=0.45, density=True,
                          label=sample_type, color=color)
        axis.set_title(metric); axis.set_xlabel(metric); axis.set_ylabel("density")
        axis.legend(); figure.tight_layout()
        path = visual_dir / f"distribution_{metric}.png"
        figure.savefig(path, dpi=150); plt.close(figure); generated.append(str(path))
    summary_rows = sorted(
        [row for row in primary if row.get("fit_success")],
        key=lambda row: float(row.get("R2", -np.inf)),
    )
    examples = summary_rows[: limit // 2] + summary_rows[-(limit - limit // 2):]
    write_json(visual_dir / "selected_examples.json", examples)
    generated.append(str(visual_dir / "selected_examples.json"))
    for index, row in enumerate(examples):
        sample = samples[row["image_id"]]
        radius = int(row["patch_radius"])
        patch, _ = extract_patch(
            sample["raw"], float(row["center_x"]), float(row["center_y"]),
            radius, order=1,
        )
        background = estimate_background(
            patch, row["background_method"],
            inner_radius=max(3.0, 2.0 * float(row["equivalent_radius"])),
        )
        residual = patch - background
        yy, xx = np.indices(patch.shape, dtype=float)
        parameters = [
            row["amplitude"], row["fitted_x"], row["fitted_y"],
            row["sigma_major"], row["sigma_minor"], row["theta"],
            row["residual_background"],
        ]
        fitted = elliptical_gaussian(parameters, xx, yy)
        figure, axes = plt.subplots(1, 5, figsize=(15, 3))
        for axis, array, title in zip(
            axes[:4], (patch, residual, fitted, residual - fitted),
            ("raw patch", "background removed", "Gaussian fit", "residual"),
        ):
            axis.imshow(array, cmap="gray"); axis.set_title(title); axis.axis("off")
        radial = row.get("radial_bins", [])
        axes[4].plot(np.arange(len(radial)), radial, marker="o")
        axes[4].set_title("elliptical radial profile")
        axes[4].set_xlabel("rho bin"); axes[4].grid(alpha=0.3)
        figure.suptitle(
            f"{row['sample_type']} {row['image_id']} R2={float(row['R2']):.3f}"
        )
        figure.tight_layout()
        path = visual_dir / f"fit_example_{index:03d}.png"
        figure.savefig(path, dpi=150); plt.close(figure); generated.append(str(path))
    return generated


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    image_ids = read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[: args.max_samples]

    all_candidates = []
    samples = {}
    image_quality = []
    for image_id in image_ids:
        sample = load_sample(args.dataset_dir, args.dataset_name, image_id)
        candidates, exclusion = build_candidate_catalog(
            sample["raw"], sample["mask"], args.dataset_name, args.split, image_id,
            hard_per_target=args.hard_per_target,
            easy_per_target=args.easy_per_target, seed=args.seed,
        )
        samples[image_id] = sample
        all_candidates.extend(candidates)
        image_quality.append({
            "dataset": args.dataset_name, "split": args.split, "image_id": image_id,
            "height": sample["height"], "width": sample["width"],
            "target_count": sum(c["sample_type"] == "target" for c in candidates),
            "hard_negative_count": sum(c["sample_type"] == "hard_negative" for c in candidates),
            "easy_background_count": sum(c["sample_type"] == "easy_background" for c in candidates),
            "gt_exclusion_fraction": float(exclusion.mean()),
        })

    target_candidates = [c for c in all_candidates if c["sample_type"] == "target"]
    if args.split == "train" or not args.thresholds:
        areas = np.asarray([c["area"] for c in target_candidates], dtype=float)
        contrasts = np.asarray([c["peak_contrast"] for c in target_candidates], dtype=float)
        thresholds = {
            "source_split": args.split,
            "area_q33": float(np.percentile(areas, 33)) if len(areas) else 0.0,
            "area_q66": float(np.percentile(areas, 66)) if len(areas) else 0.0,
            "contrast_median": float(np.median(contrasts)) if len(contrasts) else 0.0,
            "test_leakage_warning": args.split == "test" and not bool(args.thresholds),
        }
    else:
        thresholds = read_json(args.thresholds)
        if not thresholds:
            raise FileNotFoundError(f"Missing training thresholds: {args.thresholds}")
    _assign_strata(all_candidates, thresholds)
    write_json(output_dir / "thresholds.json", thresholds)
    write_csv(output_dir / "instance_catalog.csv", target_candidates)
    write_csv(output_dir / "hard_negative_catalog.csv",
              [c for c in all_candidates if c["sample_type"] == "hard_negative"])
    write_csv(output_dir / "easy_background_catalog.csv",
              [c for c in all_candidates if c["sample_type"] == "easy_background"])
    write_csv(output_dir / "image_quality.csv", image_quality)

    configurations = [
        (background, radius_scale, intensity)
        for background in ("local_plane", "annulus_median")
        for radius_scale in (3.0, 4.0, 5.0)
        for intensity in ("raw", "linear_normalized")
    ]
    tasks = []
    for candidate in all_candidates:
        sample = samples[candidate["image_id"]]
        for background, radius_scale, intensity in configurations:
            source = sample["raw"] if intensity == "raw" else sample["linear"]
            tasks.append((source, candidate, background, radius_scale, intensity))

    def execute(task):
        return analyze_gaussian_candidate(*task)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(execute, tasks))
    else:
        rows = [execute(task) for task in tasks]
    write_csv(output_dir / "gaussian_instance_metrics.csv", rows)

    comparisons = []
    grouping_fields = ("background_method", "radius_scale", "intensity_mode")
    for configuration in configurations:
        subset = [row for row in rows if all(
            str(row[field]) == str(value) for field, value in zip(grouping_fields, configuration)
        )]
        for metric in P1_METRICS:
            for group_a, group_b in (
                ("target", "hard_negative"), ("target", "easy_background"),
                ("hard_negative", "easy_background"),
            ):
                result = compare_two_groups(
                    subset, metric, group_a, group_b,
                    bootstrap=args.bootstrap, seed=args.seed,
                )
                result.update(dict(zip(grouping_fields, configuration)))
                comparisons.append(result)
    benjamini_hochberg(comparisons)
    write_csv(output_dir / "statistical_comparisons.csv", comparisons)

    primary_rows = [row for row in rows
                    if row.get("background_method") == "local_plane"
                    and float(row.get("radius_scale", 0)) == 4.0
                    and row.get("intensity_mode") == "raw"]
    stratified = []
    for stratum_field, stratum_values in (
        ("size_group", ("tiny", "medium", "large")),
        ("contrast_group", ("low", "high")),
    ):
        for stratum_value in stratum_values:
            subset = [row for row in primary_rows if row.get(stratum_field) == stratum_value]
            for metric in P1_METRICS:
                result = compare_two_groups(
                    subset, metric, "target", "hard_negative",
                    bootstrap=args.bootstrap, seed=args.seed,
                )
                result.update({"stratum_field": stratum_field,
                               "stratum_value": stratum_value})
                stratified.append(result)
    benjamini_hochberg(stratified)
    write_csv(output_dir / "stratified_comparisons.csv", stratified)

    primary = [row for row in comparisons
               if row.get("background_method") == "local_plane"
               and float(row.get("radius_scale", 0)) == 4.0
               and row.get("intensity_mode") == "raw"
               and row.get("group_a") == "target"
               and row.get("group_b") == "hard_negative"
               and row.get("metric") in PRIMARY_METRICS]
    for row in primary:
        robustness_rows = [item for item in comparisons
                           if item.get("metric") == row.get("metric")
                           and item.get("group_a") == "target"
                           and item.get("group_b") == "hard_negative"
                           and not item.get("insufficient")]
        row["robust_direction_fraction"] = (
            float(np.mean([float(item.get("cliffs_delta", 0)) > 0
                           for item in robustness_rows]))
            if robustness_rows else 0.0
        )
    supported = [row for row in primary if not row.get("insufficient")
                 and float(row.get("cliffs_delta", 0)) >= 0.20
                 and float(row.get("roc_auc", 0)) >= 0.65
                 and float(row.get("robust_direction_fraction", 0)) >= 0.75]
    class_counts = {
        sample_type: sum(c["sample_type"] == sample_type for c in all_candidates)
        for sample_type in ("target", "hard_negative", "easy_background")
    }
    descriptive_only = any(value < 30 for value in class_counts.values())
    if descriptive_only:
        decision = "Descriptive only"
    elif len(supported) >= 2:
        decision = "Go"
    elif supported:
        decision = "Partial Go"
    else:
        decision = "No-Go"
    fit_rows = [row for row in rows if row.get("background_method") == "local_plane"
                and float(row.get("radius_scale", 0)) == 4.0
                and row.get("intensity_mode") == "raw"]
    fit_success = {
        sample_type: float(np.mean([
            bool(row.get("fit_success")) for row in fit_rows
            if row.get("sample_type") == sample_type
        ])) if class_counts[sample_type] else None
        for sample_type in class_counts
    }
    summary = {
        "task": "P1_gaussian_geometry", "dataset": args.dataset_name,
        "split": args.split, "started_or_updated": now_iso(),
        "images": len(image_ids), "class_counts": class_counts,
        "fit_success_rate_primary": fit_success,
        "thresholds": thresholds, "primary_comparisons": primary,
        "supported_primary_metrics": [row["metric"] for row in supported],
        "decision": decision,
        "data_quality": {
            "boundary_patch_truncation_rate": float(np.mean([
                bool(row.get("patch_truncated")) for row in fit_rows
            ])) if fit_rows else None,
            "fewer_than_30": {key: value < 30 for key, value in class_counts.items()},
        },
        "runtime": runtime_metadata(args.dataset_dir, command=" ".join(sys.argv)),
    }
    summary["visuals"] = _make_visuals(rows, samples, output_dir, args.max_visuals)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
