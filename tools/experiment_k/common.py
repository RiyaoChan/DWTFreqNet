"""Shared loading, candidate and compactness utilities for Experiment K-A."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import DWTFreqNet_SingleDecoder_LFSS_AWGM
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP,
)
from model.decoder_denp import LowFrequencyCompactness
from model.decoder_k_dose import GaussianRadialCompactness
from train_one import checkpoint_state_dict


J_VARIANTS = (
    "j1_bandwise_noise_calibrated",
    "j2_rawll_compactness",
    "j2_decoder_compactness",
    "j3_dual_evidence_fixed",
    "j3_dual_evidence_reliability",
)
SOURCES = ("raw_ll", "lfss_ll", "guided_ll", "decoder_low")
OPERATORS = ("C_P2", "C_square", "C_GR")


def load_phase1_common(explicit_root):
    candidates = [Path(explicit_root)] if explicit_root else []
    candidates.extend([
        ROOT.parent / "DWTFreqNet_PHASE1_TASK_PRIOR_VALIDATION",
        ROOT.parent / "DWTFreqNet-phase1",
    ])
    for candidate in candidates:
        path = candidate / "tools" / "phase1" / "common.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("experiment_k_phase1_common", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, path
    raise FileNotFoundError("Phase 1 common.py is required; pass --phase1-root")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_checkpoint_map(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint map must be a JSON object")
    return payload


def load_model(label, checkpoint_path, device):
    if label == "E1":
        model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
            get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
            mode="test", deepsuper=True,
        )
    elif label in J_VARIANTS:
        model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP(
            get_DWTFreqNet_config(), denp_variant=label, mode="test", deepsuper=True
        )
    else:
        raise ValueError(f"Unsupported K-A checkpoint label: {label}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_state_dict(checkpoint, model), strict=True)
    model.to(device).eval()
    model.debug_tensors = True
    model.record_statistics = False
    # E1 keeps encoder tensors in last_debug but omits L1-L3.  Capture those
    # decoder-low tensors externally so the frozen E1/J source files remain
    # untouched while K-A evaluates the same source definition for both.
    model._experiment_k_decoder_low = {}

    def capture_decoder_low(stage):
        def hook(_module, _inputs, output):
            model._experiment_k_decoder_low[stage] = output.detach()
        return hook

    for stage, module_name in (
        (3, "decoder_fuse3"), (2, "decoder_fuse2"), (1, "decoder_fuse1")
    ):
        getattr(model, module_name).register_forward_hook(capture_decoder_low(stage))
    return model, checkpoint


def padded_tensor(common, sample, device):
    normalized = common.pad_to_multiple(sample["normalized"], 32)
    return torch.from_numpy(normalized[None, None].astype(np.float32)).to(device)


def feature_sources(model, stage):
    debug = model.last_debug
    if stage == 4:
        decoder_low = debug["E"][4]
    elif "L" in debug:
        decoder_low = debug["L"][stage]
    else:
        decoder_low = model._experiment_k_decoder_low[stage]
    return {
        "raw_ll": debug["A"][stage],
        "lfss_ll": debug["A_lfss"][stage],
        "guided_ll": debug["A_guided"][stage],
        "decoder_low": decoder_low,
    }


def dense_compactness_maps(tensor):
    square = LowFrequencyCompactness().to(tensor.device).eval()
    radial = GaussianRadialCompactness().to(tensor.device).eval()
    with torch.no_grad():
        _, square_debug = square(tensor)
        _, radial_debug = radial(tensor)
    return {
        "C_square": square_debug["compactness"].detach().float().cpu().numpy()[0],
        "C_GR": radial_debug["ratio"].detach().float().cpu().numpy()[0],
    }


def candidate_operator_values(common, tensor, candidate, original_shape, dense=None):
    feature = tensor.detach().float().cpu().numpy()[0]
    center = common.map_center_to_feature(candidate, original_shape, feature.shape[-2:])
    scale = common.map_scale_to_feature(candidate, original_shape, feature.shape[-2:])
    dense = dense_compactness_maps(tensor) if dense is None else dense
    values = {
        "C_P2": common.radial_compactness(feature, center, scale=scale),
    }
    for name, compactness_map in dense.items():
        values[name] = float(np.mean(common.bilinear_sample(compactness_map, [center])))
    return values


def candidate_catalog(common, sample, dataset, split, image_id, hard_per_target=2):
    rows, _ = common.build_candidate_catalog(
        sample["raw"], sample["mask"], dataset, split, image_id,
        hard_per_target=hard_per_target, easy_per_target=0, seed=42,
    )
    return [row for row in rows if row["sample_type"] in ("target", "hard_negative")]


def compactness_statistics(common, rows):
    comparisons = []
    keys = sorted({
        (row["checkpoint"], row["source"], int(row["stage"]), row["operator"])
        for row in rows
    })
    for checkpoint, source, stage, operator in keys:
        subset = [
            row for row in rows
            if row["checkpoint"] == checkpoint and row["source"] == source
            and int(row["stage"]) == stage and row["operator"] == operator
        ]
        comparison = common.compare_two_groups(
            subset, "value", "target", "hard_negative", bootstrap=1000, seed=42
        )
        comparison.update({
            "checkpoint": checkpoint, "source": source,
            "stage": stage, "operator": operator,
        })
        comparisons.append(comparison)
    common.benjamini_hochberg(comparisons)
    return comparisons


def fidelity_correlations(rows):
    from scipy import stats

    output = []
    keys = sorted({
        (row["checkpoint"], row["source"], int(row["stage"])) for row in rows
    })
    for checkpoint, source, stage in keys:
        subset = [
            row for row in rows
            if row["checkpoint"] == checkpoint and row["source"] == source
            and int(row["stage"]) == stage
        ]
        by_candidate = {}
        for row in subset:
            by_candidate.setdefault(row["candidate_id"], {})[row["operator"]] = float(row["value"])
        complete = [values for values in by_candidate.values() if set(OPERATORS) <= set(values)]
        for operator in ("C_square", "C_GR"):
            if len(complete) < 3:
                correlation, p_value = None, None
            else:
                result = stats.spearmanr(
                    [values["C_P2"] for values in complete],
                    [values[operator] for values in complete],
                )
                correlation, p_value = float(result.statistic), float(result.pvalue)
            output.append({
                "checkpoint": checkpoint, "source": source, "stage": stage,
                "reference": "C_P2", "operator": operator,
                "spearman": correlation, "p_value": p_value,
                "count": len(complete),
            })
    return output
