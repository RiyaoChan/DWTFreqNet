"""Cross-analyze Experiment H attention against P1/P2 task priors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.phase1.common import (
    E1FeatureExtractor,
    bilinear_sample,
    build_candidate_catalog,
    checkpoint_state_dict,
    ensure_dir,
    load_sample,
    map_center_to_feature,
    pad_to_multiple,
    read_csv,
    read_json,
    read_split,
    runtime_metadata,
    seed_everything,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--h-checkpoint-map", required=True)
    parser.add_argument("--p1-metrics", default="")
    parser.add_argument("--p2-metrics", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hard-per-target", type=int, default=1)
    return parser.parse_args()


def segmentation_metrics(prediction, mask):
    prediction = prediction[: mask.shape[0], : mask.shape[1]] >= 0.5
    mask = mask.astype(bool)
    intersection = np.logical_and(prediction, mask).sum()
    union = np.logical_or(prediction, mask).sum()
    false_alarm = np.logical_and(prediction, ~mask).sum() / max(mask.size, 1)
    return {
        "iou": float(intersection / max(union, 1)),
        "fa": float(false_alarm),
    }


def lookup_prior_metrics(p1_path, p2_path):
    p1_index, p2_index = {}, {}
    for row in read_csv(p1_path) if p1_path else []:
        if (row.get("background_method") == "local_plane"
                and float(row.get("radius_scale", 0)) == 4.0
                and row.get("intensity_mode") == "raw"):
            p1_index[row["candidate_id"]] = {
                "P1_R2": row.get("R2"), "P1_compactness": row.get("compactness"),
                "P1_radial_monotonicity": row.get("radial_monotonicity"),
            }
    for row in read_csv(p2_path) if p2_path else []:
        if row.get("feature_source") != "same_dwt_raw":
            continue
        p2_index[(row["candidate_id"], int(row["stage"]))] = {
            key: row.get(key) for key in ("C_LL", "C_H", "C_V", "C_D", "C_joint")
        }
    return p1_index, p2_index


def load_h_model(variant, checkpoint, device):
    import torch
    from model.Config import get_DWTFreqNet_config
    from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP import (
        DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP,
    )

    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP(
        get_DWTFreqNet_config(), lfp_variant=variant, mode="test", deepsuper=True,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(payload, model), strict=True)
    model.eval(); model.debug_tensors = True; model.record_statistics = False
    return model


def region_statistics(attention, mask):
    target = ndimage.zoom(mask.astype(float),
                          (attention.shape[-2] / mask.shape[0], attention.shape[-1] / mask.shape[1]),
                          order=0) > 0.5
    resized = np.zeros(attention.shape[-2:], dtype=bool)
    h, w = min(resized.shape[0], target.shape[0]), min(resized.shape[1], target.shape[1])
    resized[:h, :w] = target[:h, :w]
    boundary = ndimage.binary_dilation(resized, iterations=1) & ~resized
    near = ndimage.binary_dilation(resized, iterations=2) & ~(resized | boundary)
    far = ~(resized | boundary | near)
    magnitude = np.mean(attention, axis=0) if attention.ndim == 3 else attention
    def mean(selection):
        return float(magnitude[selection].mean()) if selection.any() else float("nan")
    return {
        "attention_target_mean": mean(resized),
        "attention_boundary_mean": mean(boundary),
        "attention_near_background_mean": mean(near),
        "attention_far_background_mean": mean(far),
    }


def safe_spearman(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None
             and np.isfinite(float(a)) and np.isfinite(float(b))]
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None, "p_value": None}
    a, b = zip(*pairs)
    result = stats.spearmanr(a, b)
    return {"n": len(pairs), "rho": float(result.statistic), "p_value": float(result.pvalue)}


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_map = read_json(args.h_checkpoint_map)
    if not checkpoint_map:
        raise FileNotFoundError(f"Invalid H checkpoint map: {args.h_checkpoint_map}")
    image_ids = read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[: args.max_samples]
    p1_index, p2_index = lookup_prior_metrics(args.p1_metrics, args.p2_metrics)

    samples, catalogs, e1_metrics = {}, {}, {}
    e1 = E1FeatureExtractor(args.e1_checkpoint, args.device)
    try:
        for index, image_id in enumerate(image_ids, start=1):
            sample = load_sample(args.dataset_dir, args.dataset_name, image_id)
            candidates, _ = build_candidate_catalog(
                sample["raw"], sample["mask"], args.dataset_name, args.split, image_id,
                hard_per_target=args.hard_per_target, easy_per_target=0, seed=args.seed,
            )
            catalogs[image_id] = [c for c in candidates
                                  if c["sample_type"] in ("target", "hard_negative")]
            prediction, _ = e1.extract(sample["normalized"])
            e1_metrics[image_id] = segmentation_metrics(prediction, sample["mask"])
            samples[image_id] = sample
            if index % 50 == 0:
                print(f"[E1 {args.dataset_name}] {index}/{len(image_ids)}", flush=True)
    finally:
        e1.close()

    import torch
    rows = []
    for variant, checkpoint in checkpoint_map.items():
        if not Path(checkpoint).is_file():
            print(f"Skipping missing checkpoint {variant}: {checkpoint}", flush=True)
            continue
        model = load_h_model(variant, checkpoint, args.device)
        try:
            for image_index, image_id in enumerate(image_ids, start=1):
                sample = samples[image_id]
                normalized = pad_to_multiple(sample["normalized"], 32)
                padded_mask = pad_to_multiple(sample["mask"], 32).astype(bool)
                tensor = torch.from_numpy(normalized[None, None].astype(np.float32)).to(args.device)
                with torch.no_grad():
                    prediction = model(tensor)[0, 0].float().cpu().numpy()
                current = segmentation_metrics(prediction, sample["mask"])
                delta_iou = current["iou"] - e1_metrics[image_id]["iou"]
                delta_fa = current["fa"] - e1_metrics[image_id]["fa"]
                for stage in range(1, 5):
                    attention_tensor = model.last_debug["decoder_lfp"][stage]["attention"]
                    if attention_tensor is None:
                        continue
                    attention = attention_tensor[0].float().cpu().numpy()
                    region_stats = region_statistics(attention, padded_mask)
                    for candidate in catalogs[image_id]:
                        center = map_center_to_feature(
                            candidate, padded_mask.shape, attention.shape[-2:]
                        )
                        candidate_attention = float(np.mean(bilinear_sample(attention, [center])))
                        row = {
                            "dataset": args.dataset_name, "split": args.split,
                            "image_id": image_id, "instance_id": candidate["instance_id"],
                            "candidate_id": candidate["candidate_id"],
                            "sample_type": candidate["sample_type"],
                            "variant": variant, "stage": stage,
                            "low_source": "raw_LL" if "rawll" in variant else "decoder_low",
                            "candidate_attention": candidate_attention,
                            "sample_iou": current["iou"], "e1_sample_iou": e1_metrics[image_id]["iou"],
                            "delta_iou": delta_iou, "sample_fa": current["fa"],
                            "e1_sample_fa": e1_metrics[image_id]["fa"], "delta_fa": delta_fa,
                        }
                        row.update(region_stats)
                        row.update(p1_index.get(candidate["candidate_id"], {}))
                        row.update(p2_index.get((candidate["candidate_id"], stage), {}))
                        rows.append(row)
                if image_index % 50 == 0:
                    print(f"[{variant} {args.dataset_name}] {image_index}/{len(image_ids)}", flush=True)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(output_dir / "h_prior_alignment_instances.csv", rows)
    correlations = []
    for variant in sorted(set(row["variant"] for row in rows)):
        for stage in range(1, 5):
            subset = [row for row in rows if row["variant"] == variant and row["stage"] == stage]
            for prior in ("P1_R2", "P1_compactness", "P1_radial_monotonicity",
                          "C_LL", "C_H", "C_V", "C_D", "C_joint",
                          "delta_iou", "delta_fa"):
                result = safe_spearman(
                    [row.get("candidate_attention") for row in subset],
                    [row.get(prior) for row in subset],
                )
                result.update({"variant": variant, "stage": stage, "prior": prior})
                correlations.append(result)
    write_csv(output_dir / "attention_prior_correlations.csv", correlations)

    separation = []
    for variant in sorted(set(row["variant"] for row in rows)):
        for stage in range(1, 5):
            subset = [row for row in rows if row["variant"] == variant and row["stage"] == stage]
            target = [row["candidate_attention"] for row in subset if row["sample_type"] == "target"]
            hard = [row["candidate_attention"] for row in subset if row["sample_type"] == "hard_negative"]
            separation.append({
                "variant": variant, "stage": stage,
                "target_attention_mean": float(np.mean(target)) if target else None,
                "hard_attention_mean": float(np.mean(hard)) if hard else None,
                "target_hard_gap": float(np.mean(target) - np.mean(hard)) if target and hard else None,
                "n_target": len(target), "n_hard": len(hard),
            })
    write_csv(output_dir / "attention_target_hard_separation.csv", separation)
    summary = {
        "task": "H_cross_analysis", "dataset": args.dataset_name,
        "split": args.split, "images": len(image_ids),
        "checkpoint_sha256": {
            variant: sha256_file(path) for variant, path in checkpoint_map.items()
            if Path(path).is_file()
        },
        "attention_separation": separation,
        "strong_correlations": [row for row in correlations
                                if row.get("rho") is not None and abs(row["rho"]) >= 0.30],
        "runtime": runtime_metadata(args.dataset_dir, args.e1_checkpoint, " ".join(sys.argv)),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
