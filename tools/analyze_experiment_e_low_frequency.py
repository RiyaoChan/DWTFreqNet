"""Offline target/background diagnostics for trained Experiment E checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import TestSetLoader
from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    EXPERIMENT_E_VARIANTS,
)
from train_one import checkpoint_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-variant", required=True, choices=EXPERIMENT_E_VARIANTS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def masked_mean(tensor, mask):
    expanded = mask.expand(-1, tensor.shape[1], -1, -1).bool()
    if not expanded.any():
        return None
    return float(tensor[expanded].float().mean().cpu())


def response_ratio(feature, mask, epsilon=1e-8):
    values = feature.detach().abs()
    target_mean = masked_mean(values, mask)
    background_mean = masked_mean(values, mask <= 0.5)
    if target_mean is None or background_mean is None:
        return None
    return target_mean / (background_mean + epsilon)


def append_if_finite(destination, key, value):
    if value is not None and np.isfinite(value):
        destination.setdefault(key, []).append(float(value))


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(),
        encoder_variant=args.encoder_variant,
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
            debug = model.last_debug
            for stage in range(1, 5):
                stage_data = collected[str(stage)]
                raw = debug["A"][stage]
                lfss = debug["A_lfss"][stage]
                guided = debug["A_guided"][stage]
                mask = F.adaptive_max_pool2d(
                    target.float(), output_size=raw.shape[-2:]
                )
                append_if_finite(
                    stage_data,
                    "raw_target_background_ratio",
                    response_ratio(raw, mask),
                )
                append_if_finite(
                    stage_data,
                    "lfss_target_background_ratio",
                    response_ratio(lfss, mask),
                )
                append_if_finite(
                    stage_data,
                    "guided_target_background_ratio",
                    response_ratio(guided, mask),
                )
                delta_ratio = float(
                    (lfss - raw).float().norm().cpu()
                    / (raw.float().norm().cpu() + 1e-8)
                )
                append_if_finite(stage_data, "lfss_change_norm_ratio", delta_ratio)

                gate = debug["AWGM_gate"][stage]
                target_gate = masked_mean(gate, mask)
                background_gate = masked_mean(gate, mask <= 0.5)
                append_if_finite(stage_data, "target_gate_mean", target_gate)
                append_if_finite(stage_data, "background_gate_mean", background_gate)
                if target_gate is not None and background_gate is not None:
                    append_if_finite(
                        stage_data,
                        "target_background_gate_ratio",
                        target_gate / (abs(background_gate) + 1e-8),
                    )

                weights = debug["AWGM_direction_weights"][stage]
                means = weights.float().mean(dim=(0, 2, 3)).cpu()
                for direction, value in zip(("H", "V", "D"), means):
                    append_if_finite(stage_data, f"mean_G_{direction}", float(value))

    stage_results = {}
    for stage, metrics in collected.items():
        stage_results[stage] = {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "samples": len(values),
            }
            for key, values in metrics.items()
        }
        block = model.lfss_blocks[stage].block
        for name, parameter in (
            ("skip_scale", block.skip_scale),
            ("skip_scale2", block.skip_scale2),
        ):
            values = parameter.detach().float().cpu()
            stage_results[stage][name] = {
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    payload = {
        "encoder_variant": args.encoder_variant,
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
