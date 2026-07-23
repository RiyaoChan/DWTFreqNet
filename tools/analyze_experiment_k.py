"""Post-training region, candidate and E1-paired diagnostics for Experiment K."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import DWTFreqNet_SingleDecoder_LFSS_AWGM
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_K import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_K,
    EXPERIMENT_K_VARIANTS,
)
from tools.experiment_k.common import (
    candidate_catalog,
    load_phase1_common,
    padded_tensor,
    write_csv,
)
from train_one import Metrics, checkpoint_state_dict


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
    return float(tensor.detach().float()[mask].mean().cpu()) if bool(mask.any()) else None


def sample_feature(common, tensor, candidate, original_shape):
    feature = tensor.detach().float().cpu().numpy()[0]
    center = common.map_center_to_feature(candidate, original_shape, feature.shape[-2:])
    return float(np.mean(common.bilinear_sample(feature, [center])))


def metrics_for(probability, mask, threshold=0.5):
    metrics = Metrics()
    height, width = mask.shape
    metrics.update(
        torch.from_numpy(probability[:height, :width]),
        torch.from_numpy(mask.astype(np.float32)), height, width, threshold,
    )
    return metrics.get()


def load_e1(path, device):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
        mode="test", deepsuper=True,
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(checkpoint, model), strict=True)
    return model.eval()


def load_k(variant, path, decision, device):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_K(
        get_DWTFreqNet_config(), k_variant=variant, decision_path=decision,
        mode="test", deepsuper=True,
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(checkpoint, model), strict=True)
    model.eval(); model.debug_tensors = True; model.record_statistics = False
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--variant", required=True, choices=EXPERIMENT_K_VARIANTS[2:])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--decision-json", default="")
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common, common_path = load_phase1_common(args.phase1_root)
    image_ids = common.read_split(args.dataset_dir, args.dataset_name, "test")
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]
    e1 = load_e1(args.e1_checkpoint, device)
    model, checkpoint = load_k(args.variant, args.checkpoint, args.decision_json, device)
    region_rows, candidate_rows, pair_rows = [], [], []
    predictions = {}
    with torch.no_grad():
        for image_id in image_ids:
            sample = common.load_sample(args.dataset_dir, args.dataset_name, image_id)
            tensor = padded_tensor(common, sample, device)
            e1_probability = e1(tensor)[0, 0].float().cpu().numpy()
            probability = model(tensor)[0, 0].float().cpu().numpy()
            predictions[image_id] = (probability, sample["mask"])
            e1_metrics = metrics_for(e1_probability, sample["mask"])
            k_metrics = metrics_for(probability, sample["mask"])
            pair_rows.append({
                "dataset": args.dataset_name, "image_id": image_id,
                "variant": args.variant,
                **{f"E1_{key}": value for key, value in e1_metrics.items()},
                **{f"K_{key}": value for key, value in k_metrics.items()},
                "delta_mIoU": k_metrics["mIoU"] - e1_metrics["mIoU"],
                "delta_Fa": k_metrics["Fa"] - e1_metrics["Fa"],
            })
            candidates = candidate_catalog(
                common, sample, args.dataset_name, "test", image_id, hard_per_target=2
            )
            padded_mask = common.pad_to_multiple(sample["mask"], 32)
            for stage in range(1, 5):
                stage_debug = model.last_debug["decoder_k"][stage]
                masks = region_masks(padded_mask, stage_debug["bands"]["H"]["aligned"].shape[-2:])
                for band in "HVD":
                    item = stage_debug["bands"][band]
                    for feature_name, feature in (
                        ("N", item["noise_confidence"]), ("dose", item["dose"]),
                    ):
                        for region, mask in masks.items():
                            region_rows.append({
                                "dataset": args.dataset_name, "image_id": image_id,
                                "variant": args.variant, "stage": stage, "band": band,
                                "feature": feature_name, "region": region,
                                "value": masked_mean(feature, mask),
                            })
                    if stage_debug["protection"] is not None:
                        for region, mask in masks.items():
                            region_rows.append({
                                "dataset": args.dataset_name, "image_id": image_id,
                                "variant": args.variant, "stage": stage, "band": band,
                                "feature": "P", "region": region,
                                "value": masked_mean(stage_debug["protection"], mask),
                            })
                    for candidate in candidates:
                        candidate_rows.append({
                            "dataset": args.dataset_name, "image_id": image_id,
                            "candidate_id": candidate["candidate_id"],
                            "sample_type": candidate["sample_type"],
                            "variant": args.variant, "stage": stage, "band": band,
                            "alpha": float(item["alpha"]),
                            "lambda": float(item["noise"]["lambda"]),
                            "tau": float(item["noise"]["tau"].mean()),
                            "gaussian_sigma": float(getattr(model, f"decoder_k{stage}").gaussians[band].sigma),
                            "rho": float(stage_debug["rho"]) if stage_debug["rho"] is not None else None,
                            "N": sample_feature(common, item["noise_confidence"], candidate, padded_mask.shape),
                            "P": (sample_feature(common, stage_debug["protection"], candidate, padded_mask.shape)
                                  if stage_debug["protection"] is not None else None),
                            "dose": sample_feature(common, item["dose"], candidate, padded_mask.shape),
                        })
    threshold_rows = []
    for threshold in np.arange(0.10, 0.901, 0.05):
        metrics = Metrics()
        for probability, mask in predictions.values():
            height, width = mask.shape
            metrics.update(torch.from_numpy(probability[:height, :width]),
                           torch.from_numpy(mask.astype(np.float32)), height, width, float(threshold))
        threshold_rows.append({"threshold": float(round(threshold, 2)), **metrics.get()})
    write_csv(output_dir / "region_metrics.csv", region_rows)
    write_csv(output_dir / "candidate_metrics.csv", candidate_rows)
    write_csv(output_dir / "e1_paired_metrics.csv", pair_rows)
    write_csv(output_dir / "threshold_scan.csv", threshold_rows)
    payload = {
        "dataset": args.dataset_name, "variant": args.variant,
        "images": len(image_ids),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "phase1_common": str(common_path.resolve()),
        "ideal_dose_relation": "dose_hard_negative > dose_target_and_boundary",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
