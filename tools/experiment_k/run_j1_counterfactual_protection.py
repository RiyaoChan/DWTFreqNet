"""K-A3: J1 inference-only source/operator/stage/rho counterfactual sweep."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_K import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_K,
)
from model.decoder_denp import LowFrequencyCompactness
from tools.experiment_k.common import load_phase1_common, padded_tensor, write_csv
from train_one import Metrics


SOURCE_VARIANTS = ("raw_ll", "lfss_ll", "guided_ll", "decoder_low")
STAGE_SETS = ((1,), (2,), (3,), (4,), (1, 2), (1, 2, 3), (1, 2, 3, 4))
RHOS = (0.10, 0.25, 0.50, 1.00)


class SquareProtectionAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.square = LowFrequencyCompactness()

    def forward(self, low):
        protection, debug = self.square(low.detach())
        return protection, {**debug, "ratio": debug["compactness"]}


def load_j1_as_k(checkpoint_path, device):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_K(
        get_DWTFreqNet_config(), k_variant="k3_gr_raw_all", mode="test", deepsuper=True
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    for prefix in ("model.", "module.", "module.model."):
        if any(name.startswith(prefix) for name in state):
            state = {
                name[len(prefix):] if name.startswith(prefix) else name: value
                for name, value in state.items()
            }
    mapped = {
        name.replace("decoder_denp", "decoder_k"): value
        for name, value in state.items()
        if name.replace("decoder_denp", "decoder_k") in model.state_dict()
    }
    model.load_state_dict(mapped, strict=False)
    model.to(device).eval()
    model.record_statistics = False
    model.alpha_override = 1.0
    return model, checkpoint


def evaluate_predictions(predictions, samples, threshold):
    metrics = Metrics()
    for image_id, probability in predictions.items():
        mask = samples[image_id]["mask"]
        metrics.update(
            torch.from_numpy(probability[:mask.shape[0], :mask.shape[1]]),
            torch.from_numpy(mask.astype(np.float32)),
            mask.shape[0], mask.shape[1], float(threshold),
        )
    return metrics.get()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--j1-checkpoint", required=True)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common, common_path = load_phase1_common(args.phase1_root)
    image_ids = common.read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]
    samples = {
        image_id: common.load_sample(args.dataset_dir, args.dataset_name, image_id)
        for image_id in image_ids
    }
    tensors = {
        image_id: padded_tensor(common, sample, device) for image_id, sample in samples.items()
    }
    model, checkpoint = load_j1_as_k(args.j1_checkpoint, device)
    rows, threshold_rows = [], []
    for operator in ("square", "gaussian_radial"):
        for stage in range(1, 5):
            processor = getattr(model, f"decoder_k{stage}")
            if operator == "square":
                processor.compactness = SquareProtectionAdapter().to(device).eval()
        for source in SOURCE_VARIANTS:
            model.k_source_map = {stage: source for stage in range(1, 5)}
            for active_stages in STAGE_SETS:
                for rho in RHOS:
                    model.rho_override = {
                        stage: rho if stage in active_stages else 0.0 for stage in range(1, 5)
                    }
                    predictions = {}
                    with torch.no_grad():
                        for image_id, tensor in tensors.items():
                            predictions[image_id] = model(tensor)[0, 0].float().cpu().numpy()
                    metrics = evaluate_predictions(predictions, samples, 0.5)
                    row = {
                        "dataset": args.dataset_name,
                        "split": args.split,
                        "operator": operator,
                        "source": source,
                        "active_stages": "-".join(map(str, active_stages)),
                        "rho": rho,
                        **metrics,
                    }
                    rows.append(row)
                    for threshold in np.arange(0.10, 0.901, 0.05):
                        threshold_rows.append({
                            **{key: row[key] for key in (
                                "dataset", "split", "operator", "source", "active_stages", "rho"
                            )},
                            "threshold": float(round(threshold, 2)),
                            **evaluate_predictions(predictions, samples, threshold),
                        })
    write_csv(output_dir / "counterfactual_metrics.csv", rows)
    write_csv(output_dir / "threshold_scan.csv", threshold_rows)
    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "images": len(image_ids),
        "j1_checkpoint": str(Path(args.j1_checkpoint).resolve()),
        "j1_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "phase1_common": str(common_path.resolve()),
        "configurations": len(rows),
        "complete": args.max_samples == 0,
        "selection_rule": "Only train split may choose source/operator/stages/rho; test is confirmation only.",
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
