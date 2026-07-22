"""P2: validate LL compactness and calibrated H/V/D consistency."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.phase1.common import (
    E1FeatureExtractor,
    benjamini_hochberg,
    build_candidate_catalog,
    calibrate_haar,
    compare_two_groups,
    ensure_dir,
    load_sample,
    map_center_to_feature,
    map_scale_to_feature,
    pad_to_multiple,
    radial_compactness,
    read_split,
    runtime_metadata,
    sample_pair_metric,
    sample_quadrants,
    seed_everything,
    write_csv,
    write_json,
)


SOURCE_FAMILIES = {
    "same_dwt_raw": ("raw_LL", "raw_H", "raw_V", "raw_D"),
    "raw_ll_aligned_hvd": ("raw_LL", "aligned_H", "aligned_V", "aligned_D"),
    "lfss_ll_aligned_hvd": ("lfss_LL", "aligned_H", "aligned_V", "aligned_D"),
    "guided_ll_aligned_hvd": ("guided_LL", "aligned_H", "aligned_V", "aligned_D"),
    "decoder_low_aligned_hvd": ("decoder_low", "aligned_H", "aligned_V", "aligned_D"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-variant", default="e1_lfss_resblock")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hard-per-target", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def calibration_markdown(calibration):
    orientation = calibration["band_response_orientation"]
    h_rule = calibration["pair_rules"]["H"]
    v_rule = calibration["pair_rules"]["V"]
    template = calibration["d_quadrant_template"]
    return f"""# H/V/D 方向校准

- 代码变量 `H` 对应 `{calibration['code_band_names']['H']}`，对 **{orientation['H']}** 结构响应最大；成对统计沿 `{h_rule['sensitive_axis']}` 轴采样，预期符号 `s_d={h_rule['expected_sign']}`。
- 代码变量 `V` 对应 `{calibration['code_band_names']['V']}`，对 **{orientation['V']}** 结构响应最大；成对统计沿 `{v_rule['sensitive_axis']}` 轴采样，预期符号 `s_d={v_rule['expected_sign']}`。
- 代码变量 `D` 对应 `{calibration['code_band_names']['D']}`，中心 Gaussian 校准的四象限模板为 `{template}`。
- 当前扫描路由与实测方向：`{calibration['routing_aligned']}`。

后续 P2 公式读取本校准结果，不根据 H/V 变量名硬编码采样轴或符号关系。
"""


def family_metrics(features, stage, family, candidate, original_shape, calibration):
    ll_name, h_name, v_name, d_name = SOURCE_FAMILIES[family]
    ll = features[(stage, ll_name)]
    h = features[(stage, h_name)]
    v = features[(stage, v_name)]
    d = features[(stage, d_name)]
    center = map_center_to_feature(candidate, original_shape, ll.shape[-2:])
    scale = map_scale_to_feature(candidate, original_shape, ll.shape[-2:])
    c_ll = radial_compactness(ll, center, scale=scale)
    directional = {}
    for name, feature in (("H", h), ("V", v)):
        rule = calibration["pair_rules"][name]
        directional[name] = float(np.mean([
            sample_pair_metric(feature, center, rule["sensitive_axis"], radius,
                               rule["expected_sign"])
            for radius in (1, 2, 3, 4)
        ]))
    d_components = [
        sample_quadrants(d, center, radius, calibration["d_quadrant_template"])
        for radius in (1, 2, 3, 4)
    ]
    c_d_cosine = float(np.mean([item["cosine"] for item in d_components]))
    c_d_sign = float(np.mean([item["sign_agreement"] for item in d_components]))
    c_d_balance = float(np.mean([item["amplitude_balance"] for item in d_components]))
    c_d = c_d_cosine * c_d_balance
    normalized = [c_ll / (1.0 + max(c_ll, 0.0))]
    normalized += [np.clip((directional[key] + 1) / 2, 1e-6, 1.0) for key in ("H", "V")]
    normalized += [np.clip((c_d + 1) / 2, 1e-6, 1.0)]
    c_joint = float(np.exp(np.mean(np.log(np.clip(normalized, 1e-6, None)))))
    return {
        "center_x_stage": center[0], "center_y_stage": center[1],
        "equivalent_radius_stage": scale,
        "C_LL": c_ll, "C_H": directional["H"], "C_V": directional["V"],
        "C_D": c_d, "C_D_cosine": c_d_cosine,
        "C_D_sign_agreement": c_d_sign,
        "C_D_amplitude_balance": c_d_balance, "C_joint": c_joint,
    }


def _make_plots(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    visual_dir = ensure_dir(Path(output_dir) / "visuals")
    generated = []
    for metric in ("C_LL", "C_H", "C_V", "C_D", "C_joint"):
        figure, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
        for stage, axis in enumerate(axes, start=1):
            subset = [row for row in rows if row["stage"] == stage
                      and row["feature_source"] == "same_dwt_raw"]
            for sample_type, color in (("target", "tab:red"), ("hard_negative", "tab:blue")):
                values = [row[metric] for row in subset if row["sample_type"] == sample_type]
                if values:
                    axis.hist(values, bins=25, alpha=0.5, density=True,
                              label=sample_type, color=color)
            axis.set_title(f"stage {stage}")
        axes[0].set_ylabel("density"); axes[-1].legend()
        figure.suptitle(f"same-DWT {metric}"); figure.tight_layout()
        path = visual_dir / f"stagewise_{metric}.png"
        figure.savefig(path, dpi=150); plt.close(figure); generated.append(str(path))
    return generated


def main():
    args = parse_args()
    if args.model_variant != "e1_lfss_resblock":
        raise ValueError("Phase 1 P2 is fixed to the E1 lfss_resblock checkpoint")
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    calibration = calibrate_haar(size=64, device=args.device)
    write_json(output_dir / "HVD_DIRECTION_CALIBRATION.json", calibration)
    (output_dir / "HVD_DIRECTION_CALIBRATION.md").write_text(
        calibration_markdown(calibration), encoding="utf-8"
    )
    image_ids = read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[: args.max_samples]
    extractor = E1FeatureExtractor(args.checkpoint, args.device)
    rows, candidate_catalog, collision_records = [], [], []
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
            analysis_shape = pad_to_multiple(sample["mask"], 32).shape
            for stage in range(1, 5):
                shape = features[(stage, "raw_LL")].shape[-2:]
                mapped = [map_center_to_feature(c, analysis_shape, shape) for c in candidates]
                rounded = [(round(x), round(y)) for x, y in mapped]
                collision_count = len(rounded) - len(set(rounded))
                collision_records.append({
                    "dataset": args.dataset_name, "split": args.split,
                    "image_id": image_id, "stage": stage,
                    "candidate_count": len(candidates), "collision_count": collision_count,
                    "collision_rate": collision_count / max(len(candidates), 1),
                })
            for candidate in candidates:
                for stage in range(1, 5):
                    for family in SOURCE_FAMILIES:
                        row = {
                            "dataset": args.dataset_name, "split": args.split,
                            "image_id": image_id,
                            "instance_id": candidate["instance_id"],
                            "candidate_id": candidate["candidate_id"],
                            "sample_type": candidate["sample_type"],
                            "stage": stage, "feature_source": family,
                            "area": candidate["area"],
                            "peak_contrast": candidate["peak_contrast"],
                        }
                        row.update(family_metrics(
                            features, stage, family, candidate, analysis_shape, calibration
                        ))
                        rows.append(row)
            if image_index % 25 == 0:
                print(f"[{args.dataset_name}/{args.split}] {image_index}/{len(image_ids)}", flush=True)
    finally:
        extractor.close()

    write_csv(output_dir / "candidate_catalog.csv", candidate_catalog)
    write_csv(output_dir / "instance_consistency_metrics.csv", rows)
    write_csv(output_dir / "stage_collision.csv", collision_records)
    comparisons = []
    for stage in range(1, 5):
        for family in SOURCE_FAMILIES:
            subset = [row for row in rows if row["stage"] == stage and row["feature_source"] == family]
            for metric in ("C_LL", "C_H", "C_V", "C_D", "C_joint"):
                result = compare_two_groups(
                    subset, metric, "target", "hard_negative",
                    bootstrap=args.bootstrap, seed=args.seed,
                )
                result.update({"stage": stage, "feature_source": family})
                comparisons.append(result)
    benjamini_hochberg(comparisons)
    write_csv(output_dir / "statistical_comparisons.csv", comparisons)

    raw = [row for row in comparisons if row["feature_source"] == "same_dwt_raw"]
    directional_stage_support = {}
    for stage in range(1, 5):
        supported = [row["metric"] for row in raw if row["stage"] == stage
                     and row["metric"] in ("C_H", "C_V", "C_D")
                     and not row.get("insufficient")
                     and float(row.get("cliffs_delta", 0)) >= 0.20
                     and float(row.get("roc_auc", 0)) >= 0.62]
        directional_stage_support[str(stage)] = supported
    directional_go_stages = [stage for stage, metrics in directional_stage_support.items()
                             if len(metrics) >= 2]
    low_frequency_support = [row for row in comparisons if row["metric"] == "C_LL"
                             and not row.get("insufficient")
                             and float(row.get("cliffs_delta", 0)) >= 0.20
                             and float(row.get("roc_auc", 0)) >= 0.62]
    directional_decision = (
        "Go" if len(directional_go_stages) >= 2
        else "Partial Go" if directional_go_stages else "No-Go"
    )
    low_frequency_decision = "Go" if len(low_frequency_support) >= 2 else (
        "Partial Go" if low_frequency_support else "No-Go"
    )
    candidate_counts = {
        sample_type: sum(c["sample_type"] == sample_type for c in candidate_catalog)
        for sample_type in ("target", "hard_negative")
    }
    if any(count < 30 for count in candidate_counts.values()):
        directional_decision = "Descriptive only"
        low_frequency_decision = "Descriptive only"
    summary = {
        "task": "P2_wavelet_directional_consistency",
        "dataset": args.dataset_name, "split": args.split,
        "images": len(image_ids),
        "candidate_counts": candidate_counts,
        "calibration": calibration,
        "directional_stage_support": directional_stage_support,
        "directional_decision": directional_decision,
        "low_frequency_supported_comparisons": low_frequency_support,
        "low_frequency_decision": low_frequency_decision,
        "collision_rate": float(np.mean([r["collision_rate"] for r in collision_records]))
        if collision_records else 0.0,
        "visuals": _make_plots(rows, output_dir),
        "runtime": runtime_metadata(args.dataset_dir, args.checkpoint, " ".join(sys.argv)),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
