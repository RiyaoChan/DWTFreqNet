"""K-A5: audit global/local MAD and spatial Gaussian diffusion on J1."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiment_k.common import (
    candidate_catalog,
    load_model,
    load_phase1_common,
    padded_tensor,
    read_csv,
    write_csv,
)


def channel_mad(values):
    flattened = values.reshape(values.shape[0], -1)
    medians = np.median(flattened, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(flattened - medians), axis=1)


def local_channel_mad(values, center, kernel):
    radius = kernel // 2
    x = int(round(center[0])); y = int(round(center[1]))
    padded = np.pad(values, ((0, 0), (radius, radius), (radius, radius)), mode="edge")
    x += radius; y += radius
    patch = padded[:, y - radius:y + radius + 1, x - radius:x + radius + 1]
    return channel_mad(patch)


def ring_value(values, center, inner, outer):
    channels, height, width = values.shape
    yy, xx = np.indices((height, width))
    rho = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = (rho >= inner) & (rho < outer)
    return float(np.mean(np.abs(values[:, mask]))) if bool(mask.any()) else None


def noise_confidence(magnitude, sigma):
    magnitude = np.asarray(magnitude, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    logits = (sigma - magnitude) / (0.15 * sigma + 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def treatment_correlations(rows, treatment_path):
    if not treatment_path:
        return []
    treatment = read_csv(treatment_path)
    effects = defaultdict(list)
    for row in treatment:
        key = (row["candidate_id"], int(row["stage"]))
        effects[key].append(float(row["delta_loss"]))
    effect_index = {key: float(np.mean(values)) for key, values in effects.items()}
    output = []
    for metric in ("N_raw_global", "N_aligned_global", "N_aligned_per_channel",
                   "N_local7", "N_local9"):
        paired = [(float(row[metric]), effect_index[(row["candidate_id"], int(row["stage"]))])
                  for row in rows if (row["candidate_id"], int(row["stage"])) in effect_index]
        if len(paired) >= 3:
            result = stats.spearmanr([item[0] for item in paired], [item[1] for item in paired])
            output.append({"metric": metric, "spearman_delta_loss": float(result.statistic),
                           "p_value": float(result.pvalue), "count": len(paired)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--j1-checkpoint", required=True)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--treatment-instances", default="")
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
    model, checkpoint = load_model("j1_bandwise_noise_calibrated", args.j1_checkpoint, device)
    captured_raw = {}
    handles = []
    for stage in range(1, 5):
        for band in "HVD":
            module = getattr(model, f"align_{band}{stage}")
            handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, stage=stage, band=band:
                captured_raw.__setitem__((stage, band), inputs[0].detach())
            ))
    rows = []
    for image_id in image_ids:
        sample = common.load_sample(args.dataset_dir, args.dataset_name, image_id)
        candidates = candidate_catalog(
            common, sample, args.dataset_name, args.split, image_id, args.hard_per_target
        )
        captured_raw.clear()
        with torch.no_grad():
            model(padded_tensor(common, sample, device))
        original_shape = sample["mask"].shape
        for stage in range(1, 5):
            for band in "HVD":
                raw = captured_raw[(stage, band)].float().cpu().numpy()[0]
                item = model.last_debug["decoder_denp"][stage]["bands"][band]
                aligned = item["aligned"].float().cpu().numpy()[0]
                gaussian = item["gaussian"].float().cpu().numpy()[0]
                noise = item["noise_confidence"].float().cpu().numpy()[0]
                raw_global = float(np.median(channel_mad(raw)))
                aligned_channel = channel_mad(aligned)
                aligned_global = float(np.median(aligned_channel))
                for candidate in candidates:
                    center = common.map_center_to_feature(candidate, original_shape, aligned.shape[-2:])
                    local7 = local_channel_mad(aligned, center, 7)
                    local9 = local_channel_mad(aligned, center, 9)
                    sampled_noise = common.bilinear_sample(noise, [center])
                    raw_magnitude = np.abs(common.bilinear_sample(raw, [center])[:, 0])
                    aligned_magnitude = np.abs(common.bilinear_sample(aligned, [center])[:, 0])
                    sign_flip = (aligned * gaussian) < 0
                    rows.append({
                        "dataset": args.dataset_name,
                        "split": args.split,
                        "image_id": image_id,
                        "candidate_id": candidate["candidate_id"],
                        "sample_type": candidate["sample_type"],
                        "stage": stage,
                        "band": band,
                        "raw_global_sigma": raw_global,
                        "aligned_global_sigma": aligned_global,
                        "aligned_per_channel_sigma_mean": float(aligned_channel.mean()),
                        "local7_sigma": float(local7.mean()),
                        "local9_sigma": float(local9.mean()),
                        "N_raw_global": float(noise_confidence(raw_magnitude, raw_global).mean()),
                        "N_aligned_global": float(np.mean(sampled_noise)),
                        "N_aligned_per_channel": float(
                            noise_confidence(aligned_magnitude, aligned_channel).mean()
                        ),
                        "N_local7": float(noise_confidence(aligned_magnitude, local7).mean()),
                        "N_local9": float(noise_confidence(aligned_magnitude, local9).mean()),
                        "gaussian_center_energy": ring_value(gaussian - aligned, center, 0.0, 0.75),
                        "gaussian_ring1_energy": ring_value(gaussian - aligned, center, 0.75, 1.5),
                        "gaussian_ring2_energy": ring_value(gaussian - aligned, center, 1.5, 2.5),
                        "gaussian_ring3_energy": ring_value(gaussian - aligned, center, 2.5, 3.5),
                        "sign_flip_rate": ring_value(sign_flip.astype(np.float32), center, 0.0, 3.5),
                        "local_l1_change": ring_value(gaussian - aligned, center, 0.0, 2.5),
                        "local_l2_change": ring_value((gaussian - aligned) ** 2, center, 0.0, 2.5),
                    })
    for handle in handles:
        handle.remove()
    correlations = treatment_correlations(rows, args.treatment_instances)
    write_csv(output_dir / "mad_gaussian_instances.csv", rows)
    write_csv(output_dir / "mad_treatment_correlations.csv", correlations)
    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "images": len(image_ids),
        "j1_checkpoint": str(Path(args.j1_checkpoint).resolve()),
        "j1_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "phase1_common": str(common_path.resolve()),
        "complete": args.max_samples == 0,
        "mad_dimensions": {
            "raw_global": "sample×stage×band after channel-median aggregation",
            "aligned_global": "sample×stage×band; current J1 definition",
            "aligned_per_channel": "sample×stage×band×channel",
            "local7_local9": "candidate×stage×band×channel",
        },
        "local_mad_changes_model": False,
        "experiment_l_candidate_only": True,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
