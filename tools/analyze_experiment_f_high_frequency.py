"""Offline target/background diagnostics for Experiment F best checkpoints."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import TestSetLoader
from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM,
    EXPERIMENT_F_VARIANTS,
)
from train_one import checkpoint_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-variant", required=True, choices=EXPERIMENT_F_VARIANTS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def masked_values(tensor, mask):
    expanded = mask.expand(-1, tensor.shape[1], -1, -1).bool()
    return tensor[expanded]


def masked_mean(tensor, mask):
    values = masked_values(tensor.float(), mask)
    return None if values.numel() == 0 else float(values.mean().cpu())


def response_statistics(feature, mask, epsilon=1e-8):
    magnitude = feature.detach().abs()
    target = masked_mean(magnitude, mask)
    background = masked_mean(magnitude, mask <= 0.5)
    return {
        "target_mean": target,
        "background_mean": background,
        "target_background_ratio": (
            None
            if target is None or background is None
            else target / (background + epsilon)
        ),
    }


def masked_correlation(left, right, mask):
    selected_left = masked_values(left.float(), mask)
    selected_right = masked_values(right.float(), mask)
    if selected_left.numel() < 2:
        return None
    selected_left = selected_left - selected_left.mean()
    selected_right = selected_right - selected_right.mean()
    denominator = selected_left.norm() * selected_right.norm()
    if float(denominator.cpu()) <= 1e-12:
        return None
    return float((selected_left * selected_right).sum().cpu() / denominator.cpu())


def append_metric(destination, key, value):
    if value is not None and np.isfinite(value):
        destination.setdefault(key, []).append(float(value))


def append_response(destination, prefix, feature, mask):
    for suffix, value in response_statistics(feature, mask).items():
        append_metric(destination, f"{prefix}_{suffix}", value)


def direction_entropy(features, mask):
    energies = torch.cat(
        [feature.detach().abs().mean(dim=1, keepdim=True) for feature in features],
        dim=1,
    )
    probabilities = energies / (energies.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(
        probabilities * torch.log(probabilities + 1e-8)
    ).sum(dim=1, keepdim=True) / math.log(3.0)
    return masked_mean(entropy, mask)


def collect_sparse(destination, direction, info, mask):
    support = info["support"]
    target_support = masked_mean(support, mask)
    background_support = masked_mean(support, mask <= 0.5)
    append_metric(destination, f"{direction}_target_support_mean", target_support)
    append_metric(destination, f"{direction}_background_support_mean", background_support)
    if target_support is not None and background_support is not None:
        append_metric(
            destination,
            f"{direction}_target_background_support_ratio",
            target_support / (background_support + 1e-8),
        )
    append_metric(
        destination,
        f"{direction}_target_active_fraction",
        masked_mean((support > 0.5).float(), mask),
    )
    append_metric(
        destination,
        f"{direction}_background_active_fraction",
        masked_mean((support > 0.5).float(), mask <= 0.5),
    )
    threshold = info["threshold"].detach().float()
    ratio = info["threshold_ratio"].detach().float()
    for suffix, value in (
        ("mean", threshold.mean()),
        ("std", threshold.std(unbiased=False)),
        ("min", threshold.min()),
        ("max", threshold.max()),
    ):
        append_metric(destination, f"{direction}_threshold_{suffix}", float(value.cpu()))
    append_metric(
        destination,
        f"{direction}_threshold_ratio_mean",
        float(ratio.mean().cpu()),
    )


def collect_cross(destination, info, mask):
    scales = info["scales"]
    for index, direction in enumerate(("H", "V", "D")):
        scale = scales[:, index:index + 1]
        target = masked_mean(scale, mask)
        background = masked_mean(scale, mask <= 0.5)
        append_metric(destination, f"target_scale_{direction}", target)
        append_metric(destination, f"background_scale_{direction}", background)
        if target is not None and background is not None:
            append_metric(
                destination,
                f"target_background_scale_ratio_{direction}",
                target / (background + 1e-8),
            )
    joint = info["joint_energy"]
    target_joint = masked_mean(joint, mask)
    background_joint = masked_mean(joint, mask <= 0.5)
    append_metric(destination, "target_joint_energy", target_joint)
    append_metric(destination, "background_joint_energy", background_joint)
    if target_joint is not None and background_joint is not None:
        append_metric(
            destination,
            "target_background_joint_energy_ratio",
            target_joint / (background_joint + 1e-8),
        )


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM(
        get_DWTFreqNet_config(),
        hf_variant=args.hf_variant,
        mode="test",
        deepsuper=True,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(checkpoint, model))
    model.eval()
    model.debug_tensors = True
    model.record_statistics = False

    dataset = TestSetLoader(
        args.dataset_dir, args.dataset_name, args.dataset_name, img_norm_cfg=None
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    collected = {str(stage): {} for stage in range(1, 5)}

    with torch.no_grad():
        for sample_index, (image, target) in enumerate(loader):
            if args.max_samples and sample_index >= args.max_samples:
                break
            image = image.to(device)
            target = target.to(device)
            model(image)
            for stage in range(1, 5):
                stage_data = collected[str(stage)]
                debug = model.last_debug["experiment_f"][stage]
                dshf = debug["dshf"]
                mask = F.adaptive_max_pool2d(
                    target.float(), output_size=debug["raw_h"].shape[-2:]
                )
                raw_features = []
                final_features = []
                for direction, lower in (("H", "h"), ("V", "v"), ("D", "d")):
                    raw = debug[f"raw_{lower}"]
                    multiscale = dshf[f"multiscale_{lower}"]
                    final = dshf[f"output_{lower}"]
                    raw_features.append(raw)
                    final_features.append(final)
                    append_response(stage_data, f"raw_{direction}", raw, mask)
                    append_response(
                        stage_data, f"multiscale_{direction}", multiscale, mask
                    )
                    if f"sparse_feature_{lower}" in dshf:
                        append_response(
                            stage_data,
                            f"sparse_{direction}",
                            dshf[f"sparse_feature_{lower}"],
                            mask,
                        )
                        collect_sparse(
                            stage_data,
                            direction,
                            dshf[f"sparse_{lower}"],
                            mask,
                        )
                    if f"cross_feature_{lower}" in dshf:
                        append_response(
                            stage_data,
                            f"cross_{direction}",
                            dshf[f"cross_feature_{lower}"],
                            mask,
                        )
                    append_response(stage_data, f"final_{direction}", final, mask)
                    append_metric(
                        stage_data,
                        f"residual_strength_{direction}",
                        float(
                            (final - raw).float().norm().cpu()
                            / (raw.float().norm().cpu() + 1e-8)
                        ),
                    )

                append_metric(
                    stage_data,
                    "target_direction_entropy",
                    direction_entropy(final_features, mask),
                )
                append_metric(
                    stage_data,
                    "background_direction_entropy",
                    direction_entropy(final_features, mask <= 0.5),
                )

                if "cross_direction" in dshf:
                    collect_cross(stage_data, dshf["cross_direction"], mask)
                if "low_guidance" in dshf:
                    low_contrast = dshf["low_guidance"]["low_contrast"]
                    target_low = masked_mean(low_contrast, mask)
                    background_low = masked_mean(low_contrast, mask <= 0.5)
                    append_metric(stage_data, "target_low_contrast_mean", target_low)
                    append_metric(
                        stage_data, "background_low_contrast_mean", background_low
                    )
                    if target_low is not None and background_low is not None:
                        append_metric(
                            stage_data,
                            "target_background_low_contrast_ratio",
                            target_low / (background_low + 1e-8),
                        )
                    joint = dshf["cross_direction"]["joint_energy"]
                    append_metric(
                        stage_data,
                        "target_low_joint_correlation",
                        masked_correlation(low_contrast, joint, mask),
                    )
                    append_metric(
                        stage_data,
                        "background_low_joint_correlation",
                        masked_correlation(low_contrast, joint, mask <= 0.5),
                    )

                append_response(stage_data, "lfss_LL", debug["lfss_ll"], mask)
                append_response(stage_data, "guided_LL", debug["guided_ll"], mask)
                gate = model.last_debug["AWGM_gate"][stage]
                append_metric(stage_data, "awgm_target_gate", masked_mean(gate, mask))
                append_metric(
                    stage_data,
                    "awgm_background_gate",
                    masked_mean(gate, mask <= 0.5),
                )
                weights = model.last_debug["AWGM_direction_weights"][stage]
                means = weights.float().mean(dim=(0, 2, 3)).cpu()
                for direction, value in zip(("H", "V", "D"), means):
                    append_metric(stage_data, f"awgm_mean_G_{direction}", float(value))

    stage_results = {
        stage: {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "samples": len(values),
            }
            for key, values in metrics.items()
        }
        for stage, metrics in collected.items()
    }
    payload = {
        "hf_variant": args.hf_variant,
        "dataset": args.dataset_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "samples": min(len(dataset), args.max_samples) if args.max_samples else len(dataset),
        "stages": stage_results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
