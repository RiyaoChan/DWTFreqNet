"""Complexity profile for Experiment D D0-D7 relation variants."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import HaarWaveletTransform, InverseHaarWaveletTransform
from model.DWTFreqNet_SingleDecoder_HFE_SpatialAblation import (
    DWTFreqNet_SingleDecoder_HFE_SpatialAblation,
)
from tools.profile_experiment_d_hfe_matching_ablation import (
    make_model as make_d0_d4_model,
)


MODEL_NAMES = (
    "sd_awgm",
    "sd_awgm_hfe",
    "sd_awgm_hfe_softcos",
    "sd_awgm_hfe_scaleaware",
    "sd_awgm_hfe_nomatch",
    "sd_awgm_hfe_samepos",
    "sd_awgm_hfe_neighborhood",
    "sd_awgm_hfe_targetlocal",
)
SPATIAL_VARIANTS = {
    "sd_awgm_hfe_samepos": "d5_same_position",
    "sd_awgm_hfe_neighborhood": "d6_neighborhood",
    "sd_awgm_hfe_targetlocal": "d7_target_neighborhood",
}


def make_model(name, mode):
    if name in SPATIAL_VARIANTS:
        return DWTFreqNet_SingleDecoder_HFE_SpatialAblation(
            get_DWTFreqNet_config(),
            spatial_hfe_ablation=SPATIAL_VARIANTS[name],
            mode=mode,
            deepsuper=True,
        )
    return make_d0_d4_model(name, mode)


def count_flops(model, sample):
    try:
        from thop import profile
    except ImportError:
        return None
    flops, _ = profile(model, inputs=(sample,), verbose=False)
    return int(flops)


def transform_counts(model, sample):
    counts = {"dwt": 0, "idwt": 0}
    handles = []
    for module in model.modules():
        if isinstance(module, HaarWaveletTransform):
            handles.append(
                module.register_forward_hook(
                    lambda _m, _i, _o: counts.__setitem__(
                        "dwt", counts["dwt"] + 1
                    )
                )
            )
        elif isinstance(module, InverseHaarWaveletTransform):
            handles.append(
                module.register_forward_hook(
                    lambda _m, _i, _o: counts.__setitem__(
                        "idwt", counts["idwt"] + 1
                    )
                )
            )
    with torch.no_grad():
        model(sample)
    for handle in handles:
        handle.remove()
    return counts


def disable_statistics(model):
    if hasattr(model, "record_statistics"):
        model.record_statistics = False
    for stage in range(1, 5):
        refiner = getattr(model, f"decoder_hfe{stage}", None)
        if refiner is not None and hasattr(refiner, "record_statistics"):
            refiner.record_statistics = False


def profile_one(name, device, warmup, repeats):
    model = make_model(name, "test").to(device).eval()
    disable_statistics(model)
    sample = torch.randn(1, 1, 256, 256, device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    hfe_parameters = sum(
        parameter.numel()
        for parameter_name, parameter in model.named_parameters()
        if parameter_name.startswith("decoder_hfe")
    )
    relation_parameters = sum(
        parameter.numel()
        for parameter_name, parameter in model.named_parameters()
        if parameter_name.startswith("decoder_hfe")
        and (".relation." in parameter_name or ".matching." in parameter_name)
    )
    flops = count_flops(model, sample)
    counts = transform_counts(model, sample)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(1000.0 * (time.perf_counter() - start))
    inference_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    del model, sample
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_model = make_model(name, "train").to(device).train()
    disable_statistics(train_model)
    train_input = torch.randn(1, 1, 256, 256, device=device)
    outputs = train_model(train_input)
    loss = sum(output.mean() for output in outputs)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    del train_model, train_input, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_latency = statistics.mean(timings)
    return {
        "model": name,
        "parameters": parameters,
        "hfe_parameters": hfe_parameters,
        "relation_parameters": relation_parameters,
        "thop_flops": flops,
        "latency_ms_mean": mean_latency,
        "latency_ms_std": statistics.pstdev(timings),
        "fps": 1000.0 / mean_latency,
        "inference_peak_bytes": inference_peak,
        "training_peak_bytes": training_peak,
        "dwt_calls": counts["dwt"],
        "idwt_calls": counts["idwt"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [
        profile_one(name, device, args.warmup, args.repeats)
        for name in MODEL_NAMES
    ]
    assert all(item["dwt_calls"] == 4 for item in results)
    assert all(item["idwt_calls"] == 4 for item in results)
    payload = {
        "input_shape": [1, 1, 256, 256],
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "flops_note": (
            "THOP may omit shifts, local similarity and attention weighting; "
            "latency and peak memory are measured separately."
        ),
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
