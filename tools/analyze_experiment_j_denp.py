"""Offline region, candidate, paired-sample and threshold diagnostics for DENP."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import DWTFreqNet_SingleDecoder_LFSS_AWGM
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP,
    EXPERIMENT_J_VARIANTS,
)
from train_one import Metrics, checkpoint_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--checkpoint-map", required=True,
                        help="JSON object mapping DENP variant to checkpoint")
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--phase1-root", default="",
                        help="Phase 1 checkout containing tools/phase1/common.py")
    parser.add_argument("--p1-metrics", default="")
    parser.add_argument("--p2-metrics", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hard-per-target", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_phase1_common(explicit_root):
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    root = Path(__file__).resolve().parents[1]
    candidates.extend([
        root.parent / "DWTFreqNet_PHASE1_TASK_PRIOR_VALIDATION",
        root.parent / "DWTFreqNet-phase1",
    ])
    for candidate in candidates:
        path = candidate / "tools" / "phase1" / "common.py"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("experiment_j_phase1_common", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
    raise FileNotFoundError(
        "Phase 1 common.py is required to reuse the formal hard-negative rule; "
        "pass --phase1-root"
    )


def read_csv(path):
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lookup_phase1_metrics(p1_path, p2_path):
    p1, p2 = {}, {}
    for row in read_csv(p1_path):
        if (row.get("background_method") == "local_plane"
                and float(row.get("radius_scale", 0)) == 4.0
                and row.get("intensity_mode") == "raw"):
            p1[row["candidate_id"]] = {
                "P1_Gaussian_R2": row.get("R2"),
                "P1_compactness": row.get("compactness"),
            }
    for row in read_csv(p2_path):
        if row.get("feature_source") == "same_dwt_raw":
            p2[(row["candidate_id"], int(row["stage"]))] = {
                "P2_C_LL": row.get("C_LL"), "P2_C_joint": row.get("C_joint")
            }
    return p1, p2


def load_e1(checkpoint_path, device):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
        mode="test", deepsuper=True,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(checkpoint, model), strict=True)
    model.eval()
    return model


def load_denp(variant, checkpoint_path, device):
    if variant not in EXPERIMENT_J_VARIANTS[1:]:
        raise ValueError(f"Formal diagnostic variant required, got {variant}")
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP(
        get_DWTFreqNet_config(), denp_variant=variant, mode="test", deepsuper=True
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_variant = (checkpoint.get("args", {}).get("denp_variant")
                          if isinstance(checkpoint, dict) else None)
    if checkpoint_variant is not None and checkpoint_variant != variant:
        raise RuntimeError(f"Checkpoint is {checkpoint_variant}, requested {variant}")
    model.load_state_dict(checkpoint_state_dict(checkpoint, model), strict=True)
    model.eval()
    model.debug_tensors = True
    model.record_statistics = False
    return model, checkpoint


def segmentation_metrics(probability, mask, threshold):
    height, width = mask.shape
    target = torch.from_numpy(mask.astype(np.float32))
    prediction = torch.from_numpy(probability[:height, :width].astype(np.float32))
    metrics = Metrics()
    metrics.update(prediction, target, height, width, threshold)
    return metrics.get()


def region_masks(mask, shape):
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    target = F.interpolate(tensor, size=shape, mode="nearest") > 0.5
    boundary_outer = F.max_pool2d(target.float(), 3, stride=1, padding=1) > 0.5
    near_outer = F.max_pool2d(target.float(), 5, stride=1, padding=2) > 0.5
    return {
        "target_interior": target,
        "target_boundary": boundary_outer & ~target,
        "near_background": near_outer & ~boundary_outer,
        "far_background": ~near_outer,
    }


def masked_mean(tensor, mask):
    mask = mask.to(tensor.device).expand(tensor.shape[0], tensor.shape[1], -1, -1)
    if not bool(mask.any()):
        return None
    return float(tensor.detach().float()[mask].mean().cpu())


def sample_feature(common, tensor, candidate, original_shape):
    feature = tensor.detach().float().cpu().numpy()[0]
    center = common.map_center_to_feature(candidate, original_shape, feature.shape[-2:])
    return float(np.mean(common.bilinear_sample(feature, [center])))


def append_region(rows, base, feature_name, tensor, masks):
    for region, mask in masks.items():
        value = masked_mean(tensor, mask)
        if value is not None:
            rows.append({**base, "feature": feature_name, "region": region, "value": value})


def summarize_regions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["stage"], row["band"],
                 row["feature"], row["region"])].append(float(row["value"]))
    return [{
        "variant": key[0], "stage": key[1], "band": key[2],
        "feature": key[3], "region": key[4], "mean": float(np.mean(values)),
        "std": float(np.std(values)), "count": len(values),
    } for key, values in grouped.items()]


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common, phase1_common_path = load_phase1_common(args.phase1_root)
    checkpoint_map = json.loads(Path(args.checkpoint_map).read_text(encoding="utf-8"))
    p1_index, p2_index = lookup_phase1_metrics(args.p1_metrics, args.p2_metrics)
    image_ids = common.read_split(args.dataset_dir, args.dataset_name, "test")
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]

    samples, catalogs, baseline_predictions = {}, {}, {}
    e1 = load_e1(args.e1_checkpoint, device)
    with torch.no_grad():
        for image_id in image_ids:
            sample = common.load_sample(args.dataset_dir, args.dataset_name, image_id)
            candidates, _ = common.build_candidate_catalog(
                sample["raw"], sample["mask"], args.dataset_name, "test", image_id,
                hard_per_target=args.hard_per_target, easy_per_target=0, seed=42,
            )
            catalogs[image_id] = [candidate for candidate in candidates
                                  if candidate["sample_type"] in ("target", "hard_negative")]
            normalized = common.pad_to_multiple(sample["normalized"], 32)
            tensor = torch.from_numpy(normalized[None, None].astype(np.float32)).to(device)
            baseline_predictions[image_id] = e1(tensor)[0, 0].float().cpu().numpy()
            samples[image_id] = sample
    del e1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    region_rows, candidate_rows, per_sample_rows, threshold_rows = [], [], [], []
    checkpoint_metadata = {}
    for variant, checkpoint_path in checkpoint_map.items():
        model, checkpoint = load_denp(variant, checkpoint_path, device)
        checkpoint_metadata[variant] = {
            "path": str(Path(checkpoint_path).resolve()),
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        }
        predictions = {}
        with torch.no_grad():
            for image_id in image_ids:
                sample = samples[image_id]
                normalized = common.pad_to_multiple(sample["normalized"], 32)
                padded_mask = common.pad_to_multiple(sample["mask"], 32)
                tensor = torch.from_numpy(normalized[None, None].astype(np.float32)).to(device)
                prediction = model(tensor)[0, 0].float().cpu().numpy()
                predictions[image_id] = prediction
                current = segmentation_metrics(prediction, sample["mask"], 0.5)
                baseline = segmentation_metrics(
                    baseline_predictions[image_id], sample["mask"], 0.5
                )
                per_sample_rows.append({
                    "dataset": args.dataset_name, "image_id": image_id,
                    "variant": variant, "target_area": int(sample["mask"].sum()),
                    "E1_IoU": baseline["mIoU"], "J_IoU": current["mIoU"],
                    "delta_IoU": current["mIoU"] - baseline["mIoU"],
                    "E1_Pd": baseline["Pd"], "J_Pd": current["Pd"],
                    "delta_Pd": current["Pd"] - baseline["Pd"],
                    "E1_Fa": baseline["Fa"], "J_Fa": current["Fa"],
                    "delta_Fa": current["Fa"] - baseline["Fa"],
                    "local_contrast": float(
                        sample["raw"][sample["mask"].astype(bool)].mean()
                        - sample["raw"][~sample["mask"].astype(bool)].mean()
                    ) if bool(sample["mask"].any()) else None,
                })
                debug = model.last_debug
                for stage in range(1, 5):
                    stage_debug = debug["decoder_denp"][stage]
                    masks = region_masks(padded_mask, stage_debug["bands"]["H"]["aligned"].shape[-2:])
                    decoder_low = debug["E"][4] if stage == 4 else debug["L"][stage]
                    baseline_idwt = model._idwt(
                        decoder_low, *(stage_debug["bands"][band]["aligned"] for band in "HVD")
                    )
                    idwt_difference = (debug["U"][stage - 1] - baseline_idwt).abs()
                    append_region(region_rows, {
                        "dataset": args.dataset_name, "image_id": image_id,
                        "variant": variant, "stage": stage, "band": "all",
                    }, "idwt_abs_difference", idwt_difference,
                                  region_masks(padded_mask, idwt_difference.shape[-2:]))
                    for band in "HVD":
                        item = stage_debug["bands"][band]
                        base = {"dataset": args.dataset_name, "image_id": image_id,
                                "variant": variant, "stage": stage, "band": band}
                        append_region(region_rows, base, "N", item["noise_confidence"], masks)
                        append_region(region_rows, base, "M", item["mask"], masks)
                        if stage_debug["raw_compactness"] is not None:
                            append_region(region_rows, base, "C_R",
                                          stage_debug["raw_compactness"]["compactness"], masks)
                            append_region(region_rows, base, "P_R",
                                          stage_debug["raw_protection"], masks)
                        if stage_debug["decoder_compactness"] is not None:
                            append_region(region_rows, base, "C_D",
                                          stage_debug["decoder_compactness"]["compactness"], masks)
                            append_region(region_rows, base, "P_D",
                                          stage_debug["decoder_protection"], masks)
                        for candidate in catalogs[image_id]:
                            record = {
                                **base, "candidate_id": candidate["candidate_id"],
                                "sample_type": candidate["sample_type"],
                                "target_area": candidate["area"],
                                "background_high_frequency_energy": sample_feature(
                                    common, item["aligned"].abs(), candidate, padded_mask.shape
                                ),
                                "sigma_hat": float(item["noise"]["sigma_hat"].mean()),
                                "lambda": float(item["noise"]["lambda"]),
                                "tau": float(item["noise"]["tau"].mean()),
                                "gaussian_sigma": float(
                                    getattr(model, f"decoder_denp{stage}").gaussians[band].sigma
                                ),
                                "N": sample_feature(common, item["noise_confidence"],
                                                    candidate, padded_mask.shape),
                                "M": sample_feature(common, item["mask"],
                                                    candidate, padded_mask.shape),
                                "P_R": (sample_feature(common, stage_debug["raw_protection"],
                                                       candidate, padded_mask.shape)
                                        if stage_debug["raw_protection"] is not None else None),
                                "P_D": (sample_feature(common, stage_debug["decoder_protection"],
                                                       candidate, padded_mask.shape)
                                        if stage_debug["decoder_protection"] is not None else None),
                                "C_R": (sample_feature(common,
                                                       stage_debug["raw_compactness"]["compactness"],
                                                       candidate, padded_mask.shape)
                                        if stage_debug["raw_compactness"] is not None else None),
                                "C_D": (sample_feature(common,
                                                       stage_debug["decoder_compactness"]["compactness"],
                                                       candidate, padded_mask.shape)
                                        if stage_debug["decoder_compactness"] is not None else None),
                            }
                            record["P_both_high"] = (
                                int(record["P_R"] > 0.5 and record["P_D"] > 0.5)
                                if record["P_R"] is not None and record["P_D"] is not None else None
                            )
                            record["P_both_low"] = (
                                int(record["P_R"] <= 0.5 and record["P_D"] <= 0.5)
                                if record["P_R"] is not None and record["P_D"] is not None else None
                            )
                            record["P_conflict"] = (
                                int((record["P_R"] > 0.5) != (record["P_D"] > 0.5))
                                if record["P_R"] is not None and record["P_D"] is not None else None
                            )
                            processor = getattr(model, f"decoder_denp{stage}")
                            record["gamma_R"] = (float(processor.gamma_raw["HVD".index(band)])
                                                 if processor.use_reliability else None)
                            record["gamma_D"] = (float(processor.gamma_decoder["HVD".index(band)])
                                                 if processor.use_reliability else None)
                            record.update(p1_index.get(candidate["candidate_id"], {}))
                            record.update(p2_index.get((candidate["candidate_id"], stage), {}))
                            candidate_rows.append(record)
        for threshold in np.arange(0.10, 0.901, 0.05):
            metrics = Metrics()
            for image_id, probability in predictions.items():
                mask = samples[image_id]["mask"]
                metrics.update(torch.from_numpy(probability[:mask.shape[0], :mask.shape[1]]),
                               torch.from_numpy(mask.astype(np.float32)),
                               mask.shape[0], mask.shape[1], float(threshold))
            threshold_rows.append({"dataset": args.dataset_name, "variant": variant,
                                   "threshold": float(round(threshold, 2)), **metrics.get()})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "region_metrics.csv", region_rows)
    write_csv(output_dir / "region_summary.csv", summarize_regions(region_rows))
    write_csv(output_dir / "candidate_metrics.csv", candidate_rows)
    write_csv(output_dir / "per_sample_pairing.csv", per_sample_rows)
    write_csv(output_dir / "threshold_scan.csv", threshold_rows)
    payload = {
        "dataset": args.dataset_name,
        "samples": len(image_ids),
        "e1_checkpoint": str(Path(args.e1_checkpoint).resolve()),
        "checkpoints": checkpoint_metadata,
        "phase1_common": str(phase1_common_path.resolve()),
        "hard_negative_rule": "Phase 1 build_candidate_catalog; target exclusion + local maxima + intensity matching",
        "regions": ["target_interior", "target_boundary", "hard_negative",
                    "near_background", "far_background"],
        "formal_threshold": 0.5,
        "threshold_scan": {"start": 0.1, "stop": 0.9, "step": 0.05},
        "outputs": ["region_metrics.csv", "region_summary.csv", "candidate_metrics.csv",
                    "per_sample_pairing.csv", "threshold_scan.csv"],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "diagnostic_manifest.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
