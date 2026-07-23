"""K-A4: test whether compactness predicts local Gaussian treatment harm."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiment_k.common import (
    OPERATORS,
    SOURCES,
    candidate_catalog,
    candidate_operator_values,
    dense_compactness_maps,
    feature_sources,
    load_phase1_common,
    padded_tensor,
    write_csv,
)
from tools.experiment_k.run_j1_counterfactual_protection import load_j1_as_k


def local_mask(shape, center_x, center_y, radius, device):
    height, width = shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    return ((xx.float() - float(center_x)).square() + (yy.float() - float(center_y)).square()
            <= float(radius) ** 2)


def local_effect(probability_on, probability_off, target, mask):
    on = probability_on[mask].clamp(1e-6, 1.0 - 1e-6)
    off = probability_off[mask].clamp(1e-6, 1.0 - 1e-6)
    truth = target[mask]
    loss_on = F.binary_cross_entropy(on, truth)
    loss_off = F.binary_cross_entropy(off, truth)
    prediction_on = on >= 0.5
    prediction_off = off >= 0.5
    truth_bool = truth >= 0.5
    union_on = (prediction_on | truth_bool).sum()
    union_off = (prediction_off | truth_bool).sum()
    iou_on = ((prediction_on & truth_bool).sum().float() / union_on.clamp_min(1)).item()
    iou_off = ((prediction_off & truth_bool).sum().float() / union_off.clamp_min(1)).item()
    return {
        "loss_on": float(loss_on), "loss_off": float(loss_off),
        "delta_loss": float(loss_on - loss_off),
        "probability_on": float(on.mean()), "probability_off": float(off.mean()),
        "delta_probability": float(on.mean() - off.mean()),
        "iou_on": iou_on, "iou_off": iou_off, "delta_iou": iou_on - iou_off,
        "false_positive_on": float(prediction_on.float().mean()),
        "false_positive_off": float(prediction_off.float().mean()),
        "delta_false_positive": float(prediction_on.float().mean() - prediction_off.float().mean()),
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], int(row["stage"]), row["operator"])].append(row)
    output = []
    for (source, stage, operator), subset in sorted(grouped.items()):
        compactness = np.asarray([float(row["compactness"]) for row in subset])
        effect = np.asarray([float(row["delta_loss"]) for row in subset])
        if len(subset) >= 3 and np.std(compactness) > 0 and np.std(effect) > 0:
            result = stats.spearmanr(compactness, effect)
            correlation, p_value = float(result.statistic), float(result.pvalue)
        else:
            correlation, p_value = None, None
        targets = [float(row["delta_loss"]) for row in subset if row["sample_type"] == "target"]
        hard = [float(row["delta_loss"]) for row in subset if row["sample_type"] == "hard_negative"]
        output.append({
            "source": source, "stage": stage, "operator": operator,
            "count": len(subset), "spearman_compactness_delta_loss": correlation,
            "p_value": p_value,
            "target_delta_loss_median": float(np.median(targets)) if targets else None,
            "hard_delta_loss_median": float(np.median(hard)) if hard else None,
            "treatment_group_gap": (
                float(np.median(targets) - np.median(hard)) if targets and hard else None
            ),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--j1-checkpoint", required=True)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hard-per-target", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common, common_path = load_phase1_common(args.phase1_root)
    image_ids = common.read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]
    model, checkpoint = load_j1_as_k(args.j1_checkpoint, device)
    model.debug_tensors = True
    model.k_source_map = {stage: "raw_ll" for stage in range(1, 5)}
    model.rho_override = 0.0
    rows = []
    for image_id in image_ids:
        sample = common.load_sample(args.dataset_dir, args.dataset_name, image_id)
        candidates = candidate_catalog(
            common, sample, args.dataset_name, args.split, image_id, args.hard_per_target
        )
        tensor = padded_tensor(common, sample, device)
        model.spatial_dose_overrides = {}
        with torch.no_grad():
            probability_on = model(tensor)[0, 0].float().detach()
        on_sources = {
            stage: {name: value.detach().clone() for name, value in feature_sources(model, stage).items()}
            for stage in range(1, 5)
        }
        on_u = {stage: model.last_debug["U"][stage - 1].detach().clone() for stage in range(1, 5)}
        dense = {
            (stage, source): dense_compactness_maps(on_sources[stage][source])
            for stage in range(1, 5) for source in SOURCES
        }
        padded_target = common.pad_to_multiple(sample["mask"], 32)
        target_tensor = torch.from_numpy(padded_target.astype(np.float32)).to(device)
        original_shape = sample["mask"].shape
        for stage in range(1, 5):
            stage_shape = on_sources[stage]["raw_ll"].shape[-2:]
            for candidate in candidates:
                center = common.map_center_to_feature(candidate, original_shape, stage_shape)
                override = torch.full(
                    (1, 1, *stage_shape), -1.0, device=device, dtype=tensor.dtype
                )
                region = local_mask(stage_shape, center[0], center[1], 2.0, device)
                override[0, 0, region] = 0.0
                model.spatial_dose_overrides = {
                    stage: {band: override for band in "HVD"}
                }
                with torch.no_grad():
                    probability_off = model(tensor)[0, 0].float().detach()
                full_radius = max(2.0, 2.0 * float(candidate["equivalent_radius"]))
                full_region = local_mask(
                    probability_on.shape,
                    candidate["center_x"], candidate["center_y"], full_radius, device,
                )
                effect = local_effect(probability_on, probability_off, target_tensor, full_region)
                idwt_difference = float(
                    (on_u[stage] - model.last_debug["U"][stage - 1]).abs().mean().cpu()
                )
                for source in SOURCES:
                    values = candidate_operator_values(
                        common, on_sources[stage][source], candidate, original_shape,
                        dense=dense[(stage, source)],
                    )
                    for operator in OPERATORS:
                        rows.append({
                            "dataset": args.dataset_name,
                            "split": args.split,
                            "image_id": image_id,
                            "candidate_id": candidate["candidate_id"],
                            "sample_type": candidate["sample_type"],
                            "stage": stage,
                            "source": source,
                            "operator": operator,
                            "compactness": values[operator],
                            **effect,
                            "idwt_feature_change": idwt_difference,
                        })
        model.spatial_dose_overrides = {}
    summary_rows = [
        {"dataset": args.dataset_name, "split": args.split, **row}
        for row in summarize(rows)
    ]
    write_csv(output_dir / "treatment_instances.csv", rows)
    write_csv(output_dir / "treatment_correlations.csv", summary_rows)
    gr_supported = [
        row for row in summary_rows
        if row["operator"] == "C_GR"
        and row["spearman_compactness_delta_loss"] is not None
        and row["spearman_compactness_delta_loss"] >= 0.10
        and row["treatment_group_gap"] is not None
    ]
    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "images": len(image_ids),
        "j1_checkpoint": str(Path(args.j1_checkpoint).resolve()),
        "j1_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "phase1_common": str(common_path.resolve()),
        "complete": args.max_samples == 0,
        "dataset_support": len(gr_supported) >= 2,
        "supported_gr_comparisons": len(gr_supported),
        "effect_sign": "delta_loss = Gaussian-on BCE - Gaussian-off BCE; positive means harmful",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
