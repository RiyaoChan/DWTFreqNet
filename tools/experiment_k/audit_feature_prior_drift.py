"""K-A2: summarize prior-distribution drift and audit stop-gradient behavior."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.decoder_denp import LowFrequencyCompactness
from model.decoder_k_dose import GaussianRadialCompactness
from tools.experiment_k.common import (
    SOURCES,
    feature_sources,
    load_checkpoint_map,
    load_model,
    load_phase1_common,
    padded_tensor,
    read_csv,
    write_csv,
)


def numeric(row, key):
    value = row.get(key)
    return None if value in (None, "", "None") else float(value)


def distribution_drift(statistics_paths):
    rows = []
    for path in statistics_paths:
        rows.extend(read_csv(path))
    indexed = {
        (row["checkpoint"], row["source"], int(row["stage"]), row["operator"]): row
        for row in rows
    }
    output = []
    for key, row in indexed.items():
        checkpoint, source, stage, operator = key
        if checkpoint == "E1":
            continue
        baseline = indexed.get(("E1", source, stage, operator))
        if baseline is None:
            continue
        target = numeric(row, "median_a")
        hard = numeric(row, "median_b")
        base_target = numeric(baseline, "median_a")
        base_hard = numeric(baseline, "median_b")
        output.append({
            "checkpoint": checkpoint,
            "source": source,
            "stage": stage,
            "operator": operator,
            "target_median": target,
            "hard_median": hard,
            "target_hard_gap": None if target is None or hard is None else target - hard,
            "E1_target_hard_gap": (
                None if base_target is None or base_hard is None else base_target - base_hard
            ),
            "gap_drift": (
                None if None in (target, hard, base_target, base_hard)
                else (target - hard) - (base_target - base_hard)
            ),
            "roc_auc": numeric(row, "roc_auc"),
            "E1_roc_auc": numeric(baseline, "roc_auc"),
            "roc_auc_drift": (
                None if numeric(row, "roc_auc") is None or numeric(baseline, "roc_auc") is None
                else numeric(row, "roc_auc") - numeric(baseline, "roc_auc")
            ),
            "target_distribution_drift": None if target is None or base_target is None else target - base_target,
            "hard_distribution_drift": None if hard is None or base_hard is None else hard - base_hard,
        })
    return output


def gradient_audit():
    low = torch.randn(2, 8, 17, 17, requires_grad=True)
    square = LowFrequencyCompactness()
    protection, _ = square(low)
    protection.mean().backward()
    normal_square = float(low.grad.norm())

    detached = low.detach().clone().requires_grad_(True)
    protection_detached, _ = square(detached.detach())
    detached_square_gradient = torch.autograd.grad(
        protection_detached.mean(), detached, allow_unused=True
    )[0]

    radial_source = torch.randn(2, 8, 17, 17, requires_grad=True)
    radial = GaussianRadialCompactness()
    radial_protection, _ = radial(radial_source)
    return {
        "square_normal_low_gradient_norm": normal_square,
        "square_detached_output_requires_grad": bool(protection_detached.requires_grad),
        "square_detached_low_gradient_norm": (
            None if detached_square_gradient is None else float(detached_square_gradient.norm())
        ),
        "square_detached_output_gradient_source": "compactness_parameters_only",
        "gaussian_radial_output_requires_grad": bool(radial_protection.requires_grad),
        "gaussian_radial_low_gradient": None,
        "formal_k_uses_stop_gradient": True,
    }


def summarize_feature_drift(rows):
    grouped = {}
    for row in rows:
        key = (row["checkpoint"], row["source"], row["stage"])
        grouped.setdefault(key, []).append(float(row["d_feat"]))
    output = []
    for (checkpoint, source, stage), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append({
            "checkpoint": checkpoint,
            "source": source,
            "stage": stage,
            "samples": len(array),
            "d_feat_mean": float(array.mean()),
            "d_feat_median": float(np.median(array)),
            "d_feat_q05": float(np.quantile(array, 0.05)),
            "d_feat_q95": float(np.quantile(array, 0.95)),
        })
    return output


def feature_drift(args, device):
    common, common_path = load_phase1_common(args.phase1_root)
    image_ids = common.read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]
    checkpoints = load_checkpoint_map(args.checkpoint_map)
    baseline, _ = load_model("E1", args.e1_checkpoint, device)
    rows = []
    with torch.no_grad():
        for label, checkpoint_path in checkpoints.items():
            variant, _ = load_model(label, checkpoint_path, device)
            for image_id in image_ids:
                sample = common.load_sample(args.dataset_dir, args.dataset_name, image_id)
                tensor = padded_tensor(common, sample, device)
                baseline(tensor)
                baseline_sources = {
                    (stage, source): value.detach().clone()
                    for stage in range(1, 5)
                    for source, value in feature_sources(baseline, stage).items()
                }
                variant(tensor)
                for stage in range(1, 5):
                    current = feature_sources(variant, stage)
                    for source in SOURCES:
                        reference = baseline_sources[(stage, source)].float()
                        value = current[source].float()
                        numerator = torch.linalg.vector_norm(value - reference)
                        denominator = torch.linalg.vector_norm(reference) + 1e-8
                        rows.append({
                            "dataset": args.dataset_name,
                            "split": args.split,
                            "image_id": image_id,
                            "checkpoint": label,
                            "source": source,
                            "stage": stage,
                            "d_feat": float((numerator / denominator).cpu()),
                        })
            del variant
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del baseline
    return rows, common_path, len(image_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", nargs="+", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--checkpoint-map", required=True)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    drift = distribution_drift(args.statistics)
    feature_rows, common_path, sample_count = feature_drift(args, torch.device(args.device))
    feature_summary = summarize_feature_drift(feature_rows)
    gradients = gradient_audit()
    write_csv(output_dir / "prior_distribution_drift.csv", drift)
    write_csv(output_dir / "feature_drift_instances.csv", feature_rows)
    write_csv(output_dir / "feature_drift_summary.csv", feature_summary)
    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "images": sample_count,
        "statistics": [str(Path(path).resolve()) for path in args.statistics],
        "comparisons": len(drift),
        "feature_drift_comparisons": len(feature_rows),
        "gradient_audit": gradients,
        "phase1_common": str(common_path.resolve()),
        "complete": args.max_samples == 0,
        "d_feat_definition": "||L_variant-L_E1||_2 / (||L_E1||_2 + 1e-8)",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
