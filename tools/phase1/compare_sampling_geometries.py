"""P3: fair, training-free comparison of structured sampling geometries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.phase1.common import (
    E1FeatureExtractor,
    benjamini_hochberg,
    bilinear_sample,
    build_candidate_catalog,
    center_perturbations,
    ensure_dir,
    geometry_points,
    load_sample,
    map_center_to_feature,
    pad_to_multiple,
    paired_geometry_statistics,
    read_json,
    read_split,
    runtime_metadata,
    seed_everything,
    write_csv,
    write_json,
)


GEOMETRIES = ("grid", "ring", "spiral", "random", "gaussian_radial")
RADII = (2, 3, 4, 5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--num-random-repeats", type=int, default=20)
    parser.add_argument("--selected-radii", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hard-per-target", type=int, default=1)
    parser.add_argument("--max-visuals", type=int, default=30)
    return parser.parse_args()


def haar_numpy(image):
    image = np.asarray(image, dtype=np.float64)
    if image.shape[0] % 2:
        image = np.pad(image, ((0, 1), (0, 0)))
    if image.shape[1] % 2:
        image = np.pad(image, ((0, 0), (0, 1)))
    a = image[0::2, 0::2]
    b = image[0::2, 1::2]
    c = image[1::2, 0::2]
    d = image[1::2, 1::2]
    return (
        0.5 * (a + b + c + d),
        0.5 * (a - b + c - d),
        0.5 * (a + b - c - d),
        0.5 * (a - b - c + d),
    )


def resize_mask(mask, shape):
    zoom = (shape[0] / mask.shape[0], shape[1] / mask.shape[1])
    resized = ndimage.zoom(mask.astype(float), zoom, order=0) > 0.5
    output = np.zeros(shape, dtype=bool)
    h, w = min(shape[0], resized.shape[0]), min(shape[1], resized.shape[1])
    output[:h, :w] = resized[:h, :w]
    return output


def region_maps(mask, shape):
    interior = resize_mask(mask, shape)
    dilated1 = ndimage.binary_dilation(interior, iterations=1)
    dilated2 = ndimage.binary_dilation(interior, iterations=2)
    boundary = dilated1 & ~interior
    near = dilated2 & ~dilated1
    far = ~dilated2
    return {"interior": interior, "boundary": boundary, "near": near, "far": far}


def feature_sources(normalized, features):
    source_map = {(0, "input_intensity"): normalized[None]}
    ll, h, v, d = haar_numpy(normalized)
    source_map[(0, "input_dwt_LL")] = ll[None]
    source_map[(0, "input_dwt_HVD_abs")] = np.stack([np.abs(h), np.abs(v), np.abs(d)])
    for stage in range(1, 5):
        source_map[(stage, "same_dwt_raw_LL")] = features[(stage, "raw_LL")]
        source_map[(stage, "same_dwt_raw_HVD")] = np.concatenate([
            np.abs(features[(stage, "raw_H")]),
            np.abs(features[(stage, "raw_V")]),
            np.abs(features[(stage, "raw_D")]),
        ], axis=0)
        source_map[(stage, "decoder_low")] = features[(stage, "decoder_low")]
        source_map[(stage, "aligned_HVD")] = np.concatenate([
            np.abs(features[(stage, "aligned_H")]),
            np.abs(features[(stage, "aligned_V")]),
            np.abs(features[(stage, "aligned_D")]),
        ], axis=0)
    return source_map


def sampling_metrics(feature, regions, center, offsets):
    points = offsets + np.asarray(center)[None]
    feature = np.asarray(feature)
    magnitude_map = (
        np.mean(np.abs(feature), axis=0) if feature.ndim == 3 else np.abs(feature)
    )
    magnitudes = bilinear_sample(magnitude_map, points)[0]
    region_values = {
        name: bilinear_sample(region.astype(float), points)[0]
        for name, region in regions.items()
    }
    hits = {name: float(np.mean(values >= 0.5)) for name, values in region_values.items()}
    target_count = (hits["interior"] + hits["boundary"]) * len(points)
    far_count = hits["far"] * len(points)
    # A one-point pseudocount prevents an unbounded ratio when no far-background
    # point is sampled, while retaining the count-based protocol definition.
    usr = target_count / (far_count + 1.0)
    radii = np.linalg.norm(offsets, axis=1)
    midpoint = max(float(radii.max()) / 2, 1e-6)
    inner = magnitudes[radii <= midpoint]
    outer = magnitudes[radii > midpoint]
    inner_outer = float(inner.mean() / (outer.mean() + 1e-8)) if len(inner) and len(outer) else 0.0
    pairwise = np.linalg.norm(offsets[:, None, :] - offsets[None, :, :], axis=-1)
    pairwise[pairwise == 0] = np.inf
    rounded_unique = len(np.unique(np.round(points, 3), axis=0))
    value_distance = np.abs(magnitudes[:, None] - magnitudes[None, :])
    upper = value_distance[np.triu_indices_from(value_distance, 1)]
    redundancy_correlation = float(np.mean(np.exp(
        -upper / (float(np.std(magnitudes)) + 1e-8)
    ))) if len(upper) else 0.0
    return {
        "target_interior_hit": hits["interior"],
        "target_boundary_hit": hits["boundary"],
        "near_background_hit": hits["near"],
        "far_background_hit": hits["far"],
        "useful_support_ratio": usr,
        "feature_mean_abs": float(magnitudes.mean()),
        "inner_outer_contrast": inner_outer,
        "minimum_point_distance": float(pairwise.min()),
        "effective_independent_samples": int(rounded_unique),
        "redundancy_correlation": redundancy_correlation,
    }


def aggregate_random(feature, regions, center, radius, repeats, seed, num_points):
    records = []
    for repeat in range(repeats):
        offsets = geometry_points("random", radius, num_points, seed + repeat)
        records.append(sampling_metrics(feature, regions, center, offsets))
    result = {}
    for key in records[0]:
        values = np.asarray([record[key] for record in records], dtype=float)
        result[key] = float(values.mean())
        result[f"{key}_std"] = float(values.std())
        result[f"{key}_cv"] = float(values.std() / (abs(values.mean()) + 1e-8))
        result[f"{key}_worst"] = float(values.min())
    return result


def attach_response_ratios(rows):
    target_by_key = {}
    hard_by_key = {}
    grouping = (
        "image_id", "stage", "feature_source", "radius", "geometry", "perturbation"
    )
    for index, row in enumerate(rows):
        match_id = row.get("match_id")
        key = (match_id,) + tuple(row[field] for field in grouping)
        if row["sample_type"] == "target":
            target_by_key.setdefault(key, []).append(index)
        elif row["sample_type"] == "hard_negative":
            hard_by_key.setdefault(key, []).append(index)
    for key in set(target_by_key) & set(hard_by_key):
        target_value = np.mean([rows[i]["feature_mean_abs"] for i in target_by_key[key]])
        hard_value = np.mean([rows[i]["feature_mean_abs"] for i in hard_by_key[key]])
        ratio = float(target_value / (hard_value + 1e-8))
        for index in target_by_key[key] + hard_by_key[key]:
            rows[index]["feature_response_ratio"] = ratio
    for row in rows:
        row.setdefault("feature_response_ratio", float("nan"))


def make_visuals(example_payloads, output_dir, max_visuals):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    visual_dir = ensure_dir(Path(output_dir) / "visuals")
    generated = []
    for index, payload in enumerate(example_payloads[:max_visuals]):
        image, mask, center, radius = payload
        figure, axes = plt.subplots(1, 4, figsize=(12, 3))
        for axis, geometry in zip(axes, ("grid", "ring", "spiral", "random")):
            axis.imshow(image, cmap="gray")
            axis.contour(mask, levels=[0.5], colors="lime", linewidths=0.7)
            offsets = geometry_points(geometry, radius, 32, 42)
            axis.scatter(offsets[:, 0] + center[0], offsets[:, 1] + center[1], s=8)
            axis.set_title(geometry); axis.axis("off")
        figure.tight_layout()
        path = visual_dir / f"sampling_example_{index:03d}.png"
        figure.savefig(path, dpi=140); plt.close(figure); generated.append(str(path))
    return generated


def main():
    args = parse_args()
    if args.num_points != 32:
        raise ValueError("The predefined fair comparison requires exactly 32 points")
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    image_ids = read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[: args.max_samples]
    fixed_radii = read_json(args.selected_radii, {}) if args.selected_radii else {}
    if args.split == "test" and not fixed_radii:
        raise ValueError("Confirmation split requires --selected-radii from the train split")
    perturbations = center_perturbations(args.seed)
    zero_perturbation = [row for row in perturbations if row["magnitude"] == 0.0]
    extractor = E1FeatureExtractor(args.checkpoint, args.device)
    rows, fairness, candidate_catalog, example_payloads = [], [], [], []
    try:
        for image_index, image_id in enumerate(image_ids, start=1):
            sample = load_sample(args.dataset_dir, args.dataset_name, image_id)
            candidates, _ = build_candidate_catalog(
                sample["raw"], sample["mask"], args.dataset_name, args.split, image_id,
                hard_per_target=args.hard_per_target, easy_per_target=0, seed=args.seed,
            )
            candidates = [c for c in candidates if c["sample_type"] in ("target", "hard_negative")]
            candidate_catalog.extend(candidates)
            _, features = extractor.extract(sample["normalized"])
            sources = feature_sources(sample["normalized"], features)
            padded_mask = pad_to_multiple(sample["mask"], 32, value=0.0).astype(bool)
            if len(example_payloads) < args.max_visuals:
                for candidate in candidates:
                    if candidate["sample_type"] == "target":
                        example_payloads.append((sample["raw"], sample["mask"],
                                                 (candidate["center_x"], candidate["center_y"]),
                                                 max(5, 4 * candidate["equivalent_radius"])))
                        break
            for (stage, source_name), feature in sources.items():
                shape = feature.shape[-2:]
                source_mask = sample["mask"] if stage == 0 else padded_mask
                source_shape = sample["raw"].shape if stage == 0 else padded_mask.shape
                regions = region_maps(source_mask, shape)
                for candidate in candidates:
                    center = map_center_to_feature(candidate, source_shape, shape)
                    match_id = int(candidate.get("matched_component_id", candidate.get("component_id", -1)))
                    for radius in RADII:
                        geometry_cache = {}
                        for geometry in GEOMETRIES:
                            if geometry != "random":
                                offsets = geometry_points(geometry, radius, args.num_points, args.seed)
                                geometry_cache[geometry] = offsets
                                fairness.append({
                                    "geometry": geometry, "radius": radius,
                                    "num_points": len(offsets),
                                    "actual_max_radius": float(np.linalg.norm(offsets, axis=1).max()),
                                    "duplicate_points": int(len(offsets) - len(np.unique(np.round(offsets, 8), axis=0))),
                                })
                        active_perturbations = zero_perturbation
                        if (args.split == "test"
                                and radius == int(fixed_radii.get(str(stage), -1))):
                            active_perturbations = perturbations
                        for perturbation in active_perturbations:
                            shifted = (center[0] + perturbation["dx"], center[1] + perturbation["dy"])
                            for geometry in GEOMETRIES:
                                base = {
                                    "dataset": args.dataset_name, "split": args.split,
                                    "image_id": image_id, "instance_id": candidate["instance_id"],
                                    "candidate_id": candidate["candidate_id"],
                                    "sample_type": candidate["sample_type"], "match_id": match_id,
                                    "stage": stage, "feature_source": source_name,
                                    "radius": radius, "geometry": geometry,
                                    "num_points": args.num_points,
                                    "center_x": center[0], "center_y": center[1],
                                    "perturbation": perturbation["name"],
                                    "perturbation_magnitude": perturbation["magnitude"],
                                    "random_repeat": 0,
                                }
                                if geometry == "random":
                                    metrics = aggregate_random(
                                        feature, regions, shifted, radius,
                                        args.num_random_repeats,
                                        args.seed + image_index * 1009 + int(radius * 31),
                                        args.num_points,
                                    )
                                else:
                                    metrics = sampling_metrics(feature, regions, shifted, geometry_cache[geometry])
                                base.update(metrics); rows.append(base)
            if image_index % 10 == 0:
                print(f"[{args.dataset_name}/{args.split}] {image_index}/{len(image_ids)}", flush=True)
    finally:
        extractor.close()

    attach_response_ratios(rows)
    unique_fairness = {}
    for row in fairness:
        unique_fairness[(row["geometry"], row["radius"])] = row
    fairness = list(unique_fairness.values())
    for radius in RADII:
        random_sets = [geometry_points("random", radius, args.num_points, args.seed + index)
                       for index in range(args.num_random_repeats)]
        fairness.append({
            "geometry": "random", "radius": radius, "num_points": args.num_points,
            "actual_max_radius": float(np.mean([
                np.linalg.norm(points, axis=1).max() for points in random_sets
            ])),
            "duplicate_points": int(np.mean([
                len(points) - len(np.unique(np.round(points, 8), axis=0)) for points in random_sets
            ])),
        })
    write_csv(output_dir / "candidate_catalog.csv", candidate_catalog)
    write_csv(output_dir / "sampling_instance_metrics.csv", rows)
    write_csv(output_dir / "geometry_fairness.csv", fairness)

    if args.split == "train":
        selected = {}
        for stage in range(0, 5):
            subset = [row for row in rows if row["stage"] == stage
                      and row["sample_type"] == "target"
                      and row["perturbation"] == "zero"]
            scores = {}
            for radius in RADII:
                current = [row for row in subset if row["radius"] == radius]
                scores[radius] = float(np.mean([
                    row["useful_support_ratio"] - row["far_background_hit"]
                    for row in current
                ])) if current else -np.inf
            selected[str(stage)] = int(max(scores, key=scores.get))
        write_json(output_dir / "selected_radii.json", selected)
    else:
        selected = {str(key): int(value) for key, value in fixed_radii.items()}
        write_json(output_dir / "selected_radii_used.json", selected)

    core = [row for row in rows if row["sample_type"] == "target"
            and row["perturbation"] == "zero"
            and row["radius"] == selected.get(str(row["stage"]), row["radius"])
            and (row["feature_source"] in ("same_dwt_raw_LL", "same_dwt_raw_HVD")
                 or (row["stage"] == 0 and row["feature_source"] == "input_dwt_LL"))]
    comparisons = []
    for stage in range(0, 5):
        stage_rows = [row for row in core if row["stage"] == stage]
        for metric in ("useful_support_ratio", "feature_response_ratio",
                       "inner_outer_contrast", "far_background_hit"):
            for right in ("grid", "ring", "random", "gaussian_radial"):
                result = paired_geometry_statistics(stage_rows, metric, "spiral", right)
                result["stage"] = stage; comparisons.append(result)
    benjamini_hochberg(comparisons)
    write_csv(output_dir / "paired_statistical_comparisons.csv", comparisons)

    friedman_rows = []
    for stage in range(0, 5):
        stage_rows = [row for row in core if row["stage"] == stage]
        for metric in ("useful_support_ratio", "feature_response_ratio",
                       "inner_outer_contrast", "far_background_hit"):
            grouped = {}
            for row in stage_rows:
                key = (row["image_id"], row["candidate_id"], row["feature_source"],
                       row["radius"], row["perturbation"])
                try:
                    grouped.setdefault(key, {})[row["geometry"]] = float(row[metric])
                except (TypeError, ValueError):
                    continue
            complete = [value for value in grouped.values()
                        if all(name in value and np.isfinite(value[name]) for name in GEOMETRIES)]
            if len(complete) >= 3:
                try:
                    result = stats.friedmanchisquare(*[
                        [value[name] for value in complete] for name in GEOMETRIES
                    ])
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                friedman_rows.append({
                    "stage": stage, "metric": metric, "n": len(complete),
                    "statistic": statistic, "p_value": p_value,
                })
            else:
                friedman_rows.append({
                    "stage": stage, "metric": metric, "n": len(complete),
                    "statistic": None, "p_value": None,
                })
    benjamini_hochberg(friedman_rows)
    write_csv(output_dir / "friedman_tests.csv", friedman_rows)

    spiral_grid = [row for row in comparisons if row["right"] == "grid"
                   and row["metric"] in ("useful_support_ratio", "feature_response_ratio")
                   and row.get("n", 0) >= 30]
    supported = []
    for result in spiral_grid:
        stage_rows = [row for row in core if row["stage"] == result["stage"]]
        grid_values = [row[result["metric"]] for row in stage_rows if row["geometry"] == "grid"
                       and np.isfinite(row[result["metric"]])]
        relative = result.get("mean_difference", 0.0) / (abs(np.mean(grid_values)) + 1e-8) if grid_values else 0.0
        result["relative_improvement"] = float(relative)
        if result.get("rank_biserial", 0) >= 0.15 and relative >= 0.05:
            supported.append(result)
    decision = "Go" if len({row["stage"] for row in supported}) >= 2 else (
        "Partial Go" if supported else "No-Go"
    )
    candidate_counts = {
        sample_type: sum(row["sample_type"] == sample_type for row in candidate_catalog)
        for sample_type in ("target", "hard_negative")
    }
    if any(count < 30 for count in candidate_counts.values()):
        decision = "Descriptive only"
    perturbation_summary = []
    for magnitude in (0.0, 0.5, 1.0, 2.0):
        subset = [row for row in rows if row["sample_type"] == "target"
                  and row["geometry"] in ("spiral", "grid")
                  and row["perturbation_magnitude"] == magnitude
                  and row["radius"] == selected.get(str(row["stage"]), row["radius"])]
        for geometry in ("spiral", "grid"):
            values = [row["useful_support_ratio"] for row in subset if row["geometry"] == geometry]
            perturbation_summary.append({
                "magnitude": magnitude, "geometry": geometry, "n": len(values),
                "mean_usr": float(np.mean(values)) if values else None,
                "std_usr": float(np.std(values)) if values else None,
            })
    write_csv(output_dir / "center_perturbation_summary.csv", perturbation_summary)
    summary = {
        "task": "P3_sampling_geometry", "dataset": args.dataset_name,
        "split": args.split, "images": len(image_ids),
        "candidate_counts": candidate_counts,
        "selected_radii": selected, "fairness_passed": all(
            row["num_points"] == args.num_points
            and abs(row["actual_max_radius"] - row["radius"]) < 1e-6
            for row in fairness
        ),
        "supported_spiral_comparisons": supported,
        "friedman_tests": friedman_rows,
        "decision": decision, "perturbation_summary": perturbation_summary,
        "visuals": make_visuals(example_payloads, output_dir, args.max_visuals),
        "runtime": runtime_metadata(args.dataset_dir, args.checkpoint, " ".join(sys.argv)),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
