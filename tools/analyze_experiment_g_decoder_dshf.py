"""Offline target/background diagnostics for trained Experiment G checkpoints."""

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
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF import DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF, EXPERIMENT_G_VARIANTS
from train_one import checkpoint_state_dict


def masked_mean(tensor, mask):
    expanded = mask.expand(-1, tensor.shape[1], -1, -1).bool()
    return float(tensor[expanded].float().mean().cpu()) if expanded.any() else None


def append(store, name, value):
    if value is not None and np.isfinite(value): store.setdefault(name, []).append(float(value))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--decoder-variant", required=True, choices=EXPERIMENT_G_VARIANTS[1:]); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True); parser.add_argument("--dataset-name", required=True); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--output", default="")
    args = parser.parse_args(); device = torch.device(args.device)
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF(get_DWTFreqNet_config(), decoder_variant=args.decoder_variant, mode="test", deepsuper=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(checkpoint_state_dict(checkpoint, model)); model.eval(); model.debug_tensors = True; model.record_statistics = False
    dataset = TestSetLoader(args.dataset_dir, args.dataset_name, args.dataset_name, img_norm_cfg=None); loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    collected = {str(stage): {} for stage in range(1, 5)}
    with torch.no_grad():
        for index, (image, target) in enumerate(loader):
            if args.max_samples and index >= args.max_samples: break
            model(image.to(device)); target = target.to(device); debug = model.last_debug["decoder_dshf"]
            for stage in range(1, 5):
                data = debug[stage]; mask = F.adaptive_max_pool2d(target.float(), data["aligned"][0].shape[-2:]); store = collected[str(stage)]
                for band_index, direction in enumerate("HVD"):
                    aligned = data["aligned"][band_index].abs(); restored = data["restored"][band_index].abs(); delta = data["refiner"]["deltas"][band_index].abs(); scale = data["refiner"]["scales"][band_index]
                    for label, tensor in (("aligned", aligned), ("restored", restored), ("delta", delta), ("scale", scale)):
                        target_mean = masked_mean(tensor, mask); background_mean = masked_mean(tensor, mask <= 0.5)
                        append(store, f"{direction}_{label}_target_mean", target_mean); append(store, f"{direction}_{label}_background_mean", background_mean)
                        if target_mean is not None and background_mean is not None: append(store, f"{direction}_{label}_target_background_ratio", target_mean / (abs(background_mean) + 1e-8))
                targetness = data["targetness"]
                if targetness is not None:
                    append(store, "targetness_target_mean", masked_mean(targetness, mask)); append(store, "targetness_background_mean", masked_mean(targetness, mask <= 0.5))
    stages = {stage: {key: {"mean": float(np.mean(values)), "std": float(np.std(values)), "samples": len(values)} for key, values in metrics.items()} for stage, metrics in collected.items()}
    payload = {"decoder_variant": args.decoder_variant, "dataset": args.dataset_name, "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_epoch": checkpoint.get("epoch"), "device": str(device), "stages": stages}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2); print(rendered)
    if args.output: Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__": main()
