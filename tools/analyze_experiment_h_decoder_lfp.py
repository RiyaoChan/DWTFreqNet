"""Region-wise decoder-LFP diagnostics for a trained Experiment H checkpoint."""

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
)
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP,
    EXPERIMENT_H_VARIANTS,
)
from train_one import Metrics, checkpoint_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfp-variant", required=True, choices=EXPERIMENT_H_VARIANTS[1:])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--e1-checkpoint", default="")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--visual-dir", default="")
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def append_value(destination, key, value):
    value = float(value)
    if np.isfinite(value):
        destination.setdefault(key, []).append(value)


def masked_mean(tensor, mask):
    mask = mask.bool().expand(-1, tensor.shape[1], -1, -1)
    if not bool(mask.any()):
        return None
    return tensor.detach().float()[mask].mean()


def region_masks(target, size):
    target = F.interpolate(target.float(), size=size, mode="nearest") > 0.5
    target_float = target.float()
    dilated3 = F.max_pool2d(target_float, kernel_size=3, stride=1, padding=1)
    dilated5 = F.max_pool2d(target_float, kernel_size=5, stride=1, padding=2)
    interior = target
    boundary = (dilated3 > 0.5) & ~target
    near = (dilated5 > 0.5) & ~(dilated3 > 0.5)
    far = ~(dilated5 > 0.5)
    return {
        "target_interior": interior,
        "target_boundary": boundary,
        "near_background": near,
        "far_background": far,
    }


def collect_region_metric(destination, prefix, tensor, masks):
    for region, mask in masks.items():
        value = masked_mean(tensor, mask)
        if value is not None:
            append_value(destination, f"{prefix}_{region}", value)


def summarize(metrics):
    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "samples": len(values),
        }
        for name, values in metrics.items()
    }


def single_sample_metrics(probability, target, height, width):
    metrics = Metrics()
    metrics.update(
        probability[0, 0, :height, :width],
        target[0, 0, :height, :width],
        height, width, 0.5,
    )
    return metrics.get()


def save_visual(path, image, target, prediction, attention, change):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    arrays = [image, target, prediction, attention, change]
    titles = ["image", "target", "prediction", "stage1 attention", "stage1 |purified-modulated|"]
    figure, axes = plt.subplots(1, 5, figsize=(15, 3))
    for axis, array, title in zip(axes, arrays, titles):
        axis.imshow(array, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP(
        get_DWTFreqNet_config(), lfp_variant=args.lfp_variant,
        mode="test", deepsuper=True,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_variant = checkpoint_args.get("lfp_variant")
    if checkpoint_variant is not None and checkpoint_variant != args.lfp_variant:
        raise RuntimeError(
            f"Checkpoint variant {checkpoint_variant} does not match {args.lfp_variant}"
        )
    model.load_state_dict(checkpoint_state_dict(checkpoint, model), strict=True)
    model.eval()
    model.debug_tensors = True
    model.record_statistics = False
    e1_model = None
    if args.e1_checkpoint:
        e1_model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
            get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
            mode="test", deepsuper=True,
        ).to(device)
        e1_checkpoint = torch.load(
            args.e1_checkpoint, map_location=device, weights_only=False
        )
        e1_model.load_state_dict(checkpoint_state_dict(e1_checkpoint, e1_model), strict=True)
        e1_model.eval()

    dataset = TestSetLoader(
        args.dataset_dir, args.dataset_name, args.dataset_name, img_norm_cfg=None
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    collected = {str(stage): {} for stage in range(1, 5)}
    visual_dir = Path(args.visual_dir) if args.visual_dir else None
    if visual_dir:
        visual_dir.mkdir(parents=True, exist_ok=True)
    visuals = []
    per_sample = []
    processed = 0

    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            if args.max_samples and sample_index >= args.max_samples:
                break
            image, target, original_size, sample_name = batch
            image, target = image.to(device), target.to(device)
            prediction = model(image)
            e1_prediction = e1_model(image) if e1_model is not None else None
            debug = model.last_debug
            processed += 1
            height = int(original_size[0][0])
            width = int(original_size[1][0])
            sample_metrics = single_sample_metrics(prediction, target, height, width)
            e1_metrics = (
                single_sample_metrics(e1_prediction, target, height, width)
                if e1_prediction is not None else None
            )
            target_crop = target[:, :, :height, :width]
            image_crop = image[:, :, :height, :width]
            binary_target = target_crop > 0.5
            target_mean = masked_mean(image_crop.abs(), binary_target)
            background_mean = masked_mean(image_crop.abs(), ~binary_target)
            sample_record = {
                "sample": sample_name[0],
                "target_area": int(binary_target.sum()),
                "target_background_contrast": (
                    float(target_mean / (background_mean + 1e-8))
                    if target_mean is not None and background_mean is not None else None
                ),
                "h_metrics": sample_metrics,
                "e1_metrics": e1_metrics,
                "delta_mIoU": (
                    sample_metrics["mIoU"] - e1_metrics["mIoU"]
                    if e1_metrics is not None else None
                ),
                "delta_Pd": (
                    sample_metrics["Pd"] - e1_metrics["Pd"]
                    if e1_metrics is not None else None
                ),
                "delta_Fa": (
                    sample_metrics["Fa"] - e1_metrics["Fa"]
                    if e1_metrics is not None else None
                ),
                "stages": {},
            }
            for stage in range(1, 5):
                stage_debug = debug["decoder_lfp"][stage]
                metrics = collected[str(stage)]
                masks = region_masks(target, stage_debug["attention"].shape[-2:])
                collect_region_metric(metrics, "attention", stage_debug["attention"], masks)
                attention = stage_debug["attention"].float()
                append_value(metrics, "attention_mean", attention.mean())
                append_value(metrics, "attention_std", attention.std(unbiased=False))
                append_value(metrics, "attention_min", attention.min())
                append_value(metrics, "attention_max", attention.max())
                entropy = -attention * torch.log(attention + 1e-8) - (
                    1.0 - attention
                ) * torch.log(1.0 - attention + 1e-8)
                append_value(metrics, "attention_entropy", entropy.mean())
                attention_target = masked_mean(attention, masks["target_interior"])
                attention_boundary = masked_mean(attention, masks["target_boundary"])
                attention_far = masked_mean(attention, masks["far_background"])
                if attention_target is not None and attention_far is not None:
                    append_value(metrics, "attention_target_far_ratio", attention_target / (attention_far + 1e-8))
                if attention_boundary is not None and attention_far is not None:
                    append_value(metrics, "attention_boundary_far_ratio", attention_boundary / (attention_far + 1e-8))

                for direction, index in zip(("H", "V", "D"), range(3)):
                    slices = {
                        "aligned": stage_debug["aligned_high"].chunk(3, dim=1)[index],
                        "modulated": stage_debug["modulated_high"].chunk(3, dim=1)[index],
                        "purified": stage_debug["purified_high"].chunk(3, dim=1)[index],
                    }
                    if stage_debug["gaussian_high"] is not None:
                        slices["gaussian"] = stage_debug["gaussian_high"].chunk(3, dim=1)[index]
                    for coefficient_name, coefficient in slices.items():
                        collect_region_metric(
                            metrics, f"{direction}_{coefficient_name}_abs",
                            coefficient.abs(), masks,
                        )
                change = (stage_debug["purified_high"] - stage_debug["modulated_high"]).abs()
                collect_region_metric(metrics, "purification_abs_change", change, masks)
                relative_change = change / (stage_debug["modulated_high"].abs() + 1e-6)
                collect_region_metric(metrics, "purification_relative_change", relative_change, masks)
                attention_change_ratio = float(
                    (stage_debug["modulated_high"] - stage_debug["aligned_high"]).float().norm()
                    / (stage_debug["aligned_high"].float().norm() + 1e-8)
                )
                purification_change_ratio = float(
                    (stage_debug["purified_high"] - stage_debug["modulated_high"]).float().norm()
                    / (stage_debug["modulated_high"].float().norm() + 1e-8)
                )
                append_value(metrics, "R_attention", attention_change_ratio)
                append_value(metrics, "R_purification", purification_change_ratio)
                if stage_debug["mask"] is not None:
                    collect_region_metric(metrics, "low_magnitude_mask", stage_debug["mask"], masks)
                    append_value(metrics, "mask_mean", stage_debug["mask"].mean())
                    append_value(metrics, "mask_active_fraction", (stage_debug["mask"] > 0.5).float().mean())
                    if args.lfp_variant.startswith("h2"):
                        append_value(
                            metrics, "fixed_tau_percentile",
                            (stage_debug["modulated_high"].abs() < 0.5).float().mean(),
                        )
                if stage_debug["threshold_debug"] is not None:
                    append_value(
                        metrics, "threshold_ratio_mean",
                        stage_debug["threshold_debug"]["threshold_ratio"].mean(),
                    )
                    append_value(metrics, "threshold_mean", stage_debug["threshold"].mean())
                    append_value(metrics, "threshold_std", stage_debug["threshold"].std(unbiased=False))
                    append_value(metrics, "threshold_min", stage_debug["threshold"].min())
                    append_value(metrics, "threshold_max", stage_debug["threshold"].max())
                    append_value(
                        metrics, "threshold_ratio_std",
                        stage_debug["threshold_debug"]["threshold_ratio"].std(unbiased=False),
                    )
                    append_value(metrics, "soft_mask_std", stage_debug["mask"].std(unbiased=False))
                processor = getattr(model, f"decoder_lfp{stage}")
                if processor.use_gaussian:
                    append_value(metrics, "sigma", processor.gaussian.sigma)
                    append_value(metrics, "sigma_change_from_init", processor.gaussian.sigma - 1.0)

                decoder_low = debug["E"][4] if stage == 4 else debug["L"][stage]
                aligned = stage_debug["aligned_high"].chunk(3, dim=1)
                baseline_idwt = model._idwt(decoder_low, *aligned)
                purified_idwt = debug["U"][stage - 1]
                idwt_difference = (purified_idwt - baseline_idwt).abs()
                idwt_masks = region_masks(target, idwt_difference.shape[-2:])
                collect_region_metric(metrics, "idwt_abs_change", idwt_difference, idwt_masks)
                idwt_target = masked_mean(idwt_difference, idwt_masks["target_interior"])
                idwt_boundary = masked_mean(idwt_difference, idwt_masks["target_boundary"])
                idwt_far = masked_mean(idwt_difference, idwt_masks["far_background"])
                if idwt_target is not None and idwt_far is not None:
                    append_value(metrics, "idwt_target_far_ratio", idwt_target / (idwt_far + 1e-8))
                if idwt_boundary is not None and idwt_far is not None:
                    append_value(metrics, "idwt_boundary_far_ratio", idwt_boundary / (idwt_far + 1e-8))
                sample_record["stages"][str(stage)] = {
                    "attention_target_far_ratio": (
                        float(attention_target / (attention_far + 1e-8))
                        if attention_target is not None and attention_far is not None else None
                    ),
                    "mask_far_fraction": (
                        float(masked_mean(stage_debug["mask"], masks["far_background"]))
                        if stage_debug["mask"] is not None
                        and masked_mean(stage_debug["mask"], masks["far_background"]) is not None
                        else None
                    ),
                    "background_high_frequency_energy": (
                        float(masked_mean(stage_debug["aligned_high"].abs(), masks["far_background"]))
                        if masked_mean(stage_debug["aligned_high"].abs(), masks["far_background"]) is not None
                        else None
                    ),
                }

            per_sample.append(sample_record)

            if visual_dir and len(visuals) < args.max_visuals:
                name = sample_name[0]
                lfp1 = debug["decoder_lfp"][1]
                attention = F.interpolate(
                    lfp1["attention"], size=image.shape[-2:], mode="bilinear", align_corners=False
                )
                change = (lfp1["purified_high"] - lfp1["modulated_high"]).abs().mean(1, keepdim=True)
                change = F.interpolate(change, size=image.shape[-2:], mode="bilinear", align_corners=False)
                path = visual_dir / f"{sample_index:04d}_{name}.png"
                saved = save_visual(
                    path,
                    image[0, 0, :height, :width].detach().float().cpu().numpy(),
                    target[0, 0, :height, :width].detach().float().cpu().numpy(),
                    prediction[0, 0, :height, :width].detach().float().cpu().numpy(),
                    attention[0, 0, :height, :width].detach().float().cpu().numpy(),
                    change[0, 0, :height, :width].detach().float().cpu().numpy(),
                )
                if saved:
                    visuals.append(str(path))

    payload = {
        "lfp_variant": args.lfp_variant,
        "dataset": args.dataset_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "e1_checkpoint": str(Path(args.e1_checkpoint).resolve()) if args.e1_checkpoint else None,
        "device": str(device),
        "samples": processed,
        "region_definition": {
            "target_interior": "stage-pooled target",
            "target_boundary": "3x3 dilation minus target",
            "near_background": "5x5 dilation minus 3x3 dilation",
            "far_background": "outside 5x5 dilation",
        },
        "stages": {stage: summarize(metrics) for stage, metrics in collected.items()},
        "per_sample": per_sample,
        "visualizations": visuals,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
