"""Offline target/background diagnostics for trained D5-D7 checkpoints."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import TestSetLoader
from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_HFE_SpatialAblation import (
    DWTFreqNet_SingleDecoder_HFE_SpatialAblation,
    OFFSETS_3X3,
    SPATIAL_HFE_ABLATION_VARIANTS,
)
from train_one import checkpoint_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spatial-hfe-ablation",
        required=True,
        choices=SPATIAL_HFE_ABLATION_VARIANTS,
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def masked_mean(tensor, mask):
    expanded = mask.expand_as(tensor)
    count = expanded.sum()
    if not bool(count):
        return None
    return float(tensor.float()[expanded].mean().cpu())


def add_metric(sums, counts, name, value):
    if value is not None:
        sums[name] += float(value)
        counts[name] += 1


def relation_region_metrics(variant, relation, target_mask, prefix):
    background_mask = ~target_mask
    metrics = {}
    if variant == "d5_same_position":
        scale = relation.last_spatial_scale
        similarity = relation.last_similarity
        target_scale = masked_mean(scale, target_mask)
        background_scale = masked_mean(scale, background_mask)
        target_similarity = masked_mean(similarity, target_mask)
        background_similarity = masked_mean(similarity, background_mask)
        metrics.update(
            {
                f"{prefix}_target_spatial_scale_mean": target_scale,
                f"{prefix}_background_spatial_scale_mean": background_scale,
                f"{prefix}_target_background_scale_ratio": (
                    None
                    if target_scale is None or background_scale is None
                    else target_scale / (background_scale + 1e-12)
                ),
                f"{prefix}_target_similarity_mean": target_similarity,
                f"{prefix}_background_similarity_mean": background_similarity,
                f"{prefix}_target_background_similarity_ratio": (
                    None
                    if target_similarity is None or background_similarity is None
                    else target_similarity / (background_similarity + 1e-12)
                ),
            }
        )
        return metrics

    attention = relation.last_attention.float()
    entropy = -(attention * torch.log(attention.clamp_min(1e-12))).sum(
        dim=1, keepdim=True
    )
    center = attention[:, 4:5]
    argmax_index = attention.argmax(dim=1, keepdim=True)
    neighbor = (argmax_index != 4).float()
    offset_y = attention.new_tensor([item[0] for item in OFFSETS_3X3])
    offset_x = attention.new_tensor([item[1] for item in OFFSETS_3X3])
    distance = torch.sqrt(
        offset_y[argmax_index].square() + offset_x[argmax_index].square()
    )
    for region_name, mask in (
        ("target", target_mask),
        ("background", background_mask),
    ):
        metrics[f"{prefix}_{region_name}_attention_entropy"] = masked_mean(
            entropy, mask
        )
        metrics[f"{prefix}_{region_name}_center_weight"] = masked_mean(
            center, mask
        )
        metrics[
            f"{prefix}_{region_name}_neighbor_selection_ratio"
        ] = masked_mean(neighbor, mask)
        metrics[
            f"{prefix}_{region_name}_mean_offset_distance"
        ] = masked_mean(distance, mask)
    if variant == "d7_target_neighborhood":
        targetness = relation.last_targetness.float()
        target_value = masked_mean(targetness, target_mask)
        background_value = masked_mean(targetness, background_mask)
        metrics.update(
            {
                f"{prefix}_target_targetness_mean": target_value,
                f"{prefix}_background_targetness_mean": background_value,
                f"{prefix}_targetness_separation_ratio": (
                    None
                    if target_value is None or background_value is None
                    else target_value / (background_value + 1e-12)
                ),
                f"{prefix}_targetness_scale": float(
                    F.softplus(relation.raw_targetness_scale).detach().cpu()
                ),
            }
        )
    return metrics


def residual_region_metrics(refiner, target_mask, prefix):
    background_mask = ~target_mask
    metrics = {}
    for direction, beta_name, delta_name in (
        ("H", "beta_h", "delta_h"),
        ("V", "beta_v", "delta_v"),
        ("D", "beta_d", "delta_d"),
    ):
        residual = torch.abs(
            getattr(refiner, beta_name) * refiner.last_debug[delta_name]
        )
        target_value = masked_mean(residual, target_mask)
        background_value = masked_mean(residual, background_mask)
        metrics[f"{prefix}_{direction}_target_background_residual_ratio"] = (
            None
            if target_value is None or background_value is None
            else target_value / (background_value + 1e-12)
        )
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = DWTFreqNet_SingleDecoder_HFE_SpatialAblation(
        get_DWTFreqNet_config(),
        spatial_hfe_ablation=args.spatial_hfe_ablation,
        mode="test",
        deepsuper=True,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(checkpoint, model))
    model.eval()
    model.record_statistics = True
    model.debug_tensors = True

    dataset = TestSetLoader(
        args.dataset_dir,
        args.dataset_name,
        args.dataset_name,
        img_norm_cfg=None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    sums, counts = defaultdict(float), defaultdict(int)
    samples = 0
    with torch.no_grad():
        for batch in loader:
            image, mask = batch[:2]
            image = image.to(device)
            mask = mask.to(device)
            model(image)
            for stage in range(1, 5):
                refiner = getattr(model, f"decoder_hfe{stage}")
                relation = refiner.hfe.attn.relation
                if args.spatial_hfe_ablation == "d5_same_position":
                    relation_map = relation.last_spatial_scale
                else:
                    relation_map = relation.last_attention[:, :1]
                mask_stage = F.adaptive_max_pool2d(
                    mask.float(), output_size=relation_map.shape[-2:]
                )
                target_mask = mask_stage > 0.5
                metrics = relation_region_metrics(
                    args.spatial_hfe_ablation,
                    relation,
                    target_mask,
                    f"stage{stage}",
                )
                metrics.update(
                    residual_region_metrics(refiner, target_mask, f"stage{stage}")
                )
                for name, value in metrics.items():
                    add_metric(sums, counts, name, value)
            samples += 1
            if args.max_samples and samples >= args.max_samples:
                break
    averaged = {
        name: sums[name] / counts[name]
        for name in sorted(sums)
        if counts[name]
    }
    payload = {
        "spatial_hfe_ablation": args.spatial_hfe_ablation,
        "dataset": args.dataset_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples": samples,
        "gt_alignment": "adaptive_max_pool2d",
        "metrics": averaged,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
