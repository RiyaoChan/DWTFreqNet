"""Complexity profile for Experiment D D0-D4 relation variants."""

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
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.DWTFreqNet_SingleDecoder_HFE import DWTFreqNet_SingleDecoder_HFE
from model.DWTFreqNet_SingleDecoder_HFE_Ablation import (
    DWTFreqNet_SingleDecoder_HFE_Ablation,
)


MODEL_NAMES = (
    "sd_awgm",
    "sd_awgm_hfe",
    "sd_awgm_hfe_softcos",
    "sd_awgm_hfe_scaleaware",
    "sd_awgm_hfe_nomatch",
)


def make_model(name, mode):
    config = get_DWTFreqNet_config()
    if name == "sd_awgm":
        return DWTFreqNet_SingleDecoder(
            config, mode=mode, deepsuper=True, sd_variant="sd_awgm"
        )
    if name == "sd_awgm_hfe":
        return DWTFreqNet_SingleDecoder_HFE(
            config, mode=mode, deepsuper=True
        )
    if name == "sd_awgm_hfe_softcos":
        return DWTFreqNet_SingleDecoder_HFE_Ablation(
            config,
            hfe_ablation="d2_softcos_all",
            mode=mode,
            deepsuper=True,
        )
    if name == "sd_awgm_hfe_scaleaware":
        return DWTFreqNet_SingleDecoder_HFE_Ablation(
            config,
            hfe_ablation="d3_scaleaware",
            mode=mode,
            deepsuper=True,
        )
    if name == "sd_awgm_hfe_nomatch":
        return DWTFreqNet_SingleDecoder_HFE_Ablation(
            config,
            hfe_ablation="d4_no_matching",
            mode=mode,
            deepsuper=True,
        )
    raise ValueError(name)


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


def profile_one(name, device, warmup, repeats):
    model = make_model(name, "test").to(device).eval()
    if hasattr(model, "record_statistics"):
        model.record_statistics = False
    for stage in range(1, 5):
        refiner = getattr(model, f"decoder_hfe{stage}", None)
        if refiner is not None and hasattr(refiner, "record_statistics"):
            refiner.record_statistics = False
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
        if ".relation" in parameter_name
        or (
            parameter_name.startswith("decoder_hfe")
            and ".matching" in parameter_name
        )
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [
        profile_one(name, device, args.warmup, args.repeats)
        for name in MODEL_NAMES
    ]
    by_name = {item["model"]: item for item in results}
    for name in MODEL_NAMES:
        assert by_name[name]["dwt_calls"] == 4
        assert by_name[name]["idwt_calls"] == 4
    assert by_name["sd_awgm_hfe_softcos"]["hfe_parameters"] > 0
    assert by_name["sd_awgm_hfe_scaleaware"]["hfe_parameters"] > 0
    assert by_name["sd_awgm_hfe_nomatch"]["hfe_parameters"] > 0
    assert (
        by_name["sd_awgm_hfe"]["relation_parameters"]
        == by_name["sd_awgm_hfe_nomatch"]["relation_parameters"]
    )
    assert (
        by_name["sd_awgm_hfe_softcos"]["relation_parameters"]
        == by_name["sd_awgm_hfe_nomatch"]["relation_parameters"] + 8
    )
    payload = {
        "input_shape": [1, 1, 256, 256],
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "flops_note": (
            "THOP omits direct cosine/matching and attention matrix products; "
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
