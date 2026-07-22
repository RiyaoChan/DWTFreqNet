"""K-A1/A2: audit operator fidelity, source mismatch and stage behavior."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.experiment_k.common import (
    OPERATORS,
    SOURCES,
    candidate_catalog,
    candidate_operator_values,
    compactness_statistics,
    dense_compactness_maps,
    feature_sources,
    fidelity_correlations,
    load_checkpoint_map,
    load_model,
    load_phase1_common,
    padded_tensor,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--checkpoint-map", required=True)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hard-per-target", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    common, common_path = load_phase1_common(args.phase1_root)
    checkpoints = {"E1": args.e1_checkpoint, **load_checkpoint_map(args.checkpoint_map)}
    image_ids = common.read_split(args.dataset_dir, args.dataset_name, args.split)
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]
    samples = {
        image_id: common.load_sample(args.dataset_dir, args.dataset_name, image_id)
        for image_id in image_ids
    }
    catalogs = {
        image_id: candidate_catalog(
            common, sample, args.dataset_name, args.split, image_id, args.hard_per_target
        )
        for image_id, sample in samples.items()
    }

    rows, checkpoint_metadata = [], {}
    for label, checkpoint_path in checkpoints.items():
        model, checkpoint = load_model(label, checkpoint_path, device)
        checkpoint_metadata[label] = {
            "path": str(Path(checkpoint_path).resolve()),
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        }
        with torch.no_grad():
            for image_id, sample in samples.items():
                model(padded_tensor(common, sample, device))
                original_shape = sample["mask"].shape
                for stage in range(1, 5):
                    sources = feature_sources(model, stage)
                    for source_name in SOURCES:
                        tensor = sources[source_name]
                        dense = dense_compactness_maps(tensor)
                        for candidate in catalogs[image_id]:
                            values = candidate_operator_values(
                                common, tensor, candidate, original_shape, dense=dense
                            )
                            base = {
                                "dataset": args.dataset_name,
                                "split": args.split,
                                "checkpoint": label,
                                "image_id": image_id,
                                "candidate_id": candidate["candidate_id"],
                                "sample_type": candidate["sample_type"],
                                "source": source_name,
                                "stage": stage,
                                "equivalent_radius": candidate["equivalent_radius"],
                            }
                            rows.extend({**base, "operator": operator, "value": values[operator]}
                                        for operator in OPERATORS)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparisons = compactness_statistics(common, rows)
    correlations = fidelity_correlations(rows)
    write_csv(output_dir / "compactness_instances.csv", rows)
    write_csv(output_dir / "operator_statistics.csv", comparisons)
    write_csv(output_dir / "operator_fidelity.csv", correlations)
    payload = {
        "dataset": args.dataset_name,
        "split": args.split,
        "images": len(image_ids),
        "candidate_records": len(rows),
        "checkpoints": checkpoint_metadata,
        "sources": list(SOURCES),
        "operators": list(OPERATORS),
        "phase1_common": str(common_path.resolve()),
        "discovery": args.split == "train",
        "complete": args.max_samples == 0,
        "outputs": ["compactness_instances.csv", "operator_statistics.csv", "operator_fidelity.csv"],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
