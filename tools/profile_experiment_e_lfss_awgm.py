"""Complexity, latency and memory profile for Experiment E0/E1/E2."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    initialize_experiment_e_model,
)
from train_one import init_weights


NAMES = (
    "e0_sd_awgm",
    "e1_lfss_resblock",
    "e2_lfss_transition",
)


def make_model(name, mode):
    config = get_DWTFreqNet_config()
    if name == "e0_sd_awgm":
        model = DWTFreqNet_SingleDecoder(
            config, mode=mode, deepsuper=True, sd_variant="sd_awgm"
        )
        model.apply(init_weights)
        return model
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        config, encoder_variant=name, mode=mode, deepsuper=True
    )
    initialize_experiment_e_model(model, init_weights)
    return model


def count_prefix(model, prefixes):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def count_flops(model, sample):
    try:
        from thop import profile
    except ImportError as error:
        return None, f"thop unavailable: {error}"
    try:
        flops, _ = profile(model, inputs=(sample,), verbose=False)
        return int(flops), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def profile_one(name, device, warmup, repeats):
    model = make_model(name, "test").to(device).eval()
    model.record_statistics = False
    sample = torch.randn(1, 1, 256, 256, device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    lfss_parameters = count_prefix(model, ("lfss_blocks.",))
    encoder_parameters = count_prefix(
        model,
        ("stem", "local_encoder", "dir_encoder", "stage_awgm", "lfss_blocks."),
    )
    flops, flops_error = count_flops(model, sample)

    with torch.no_grad():
        model(sample)
    counts = dict(model.last_transform_counts)
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
    train_counts = dict(train_model.last_transform_counts)
    del train_model, train_input, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_latency = statistics.mean(timings)
    return {
        "model": name,
        "parameters": parameters,
        "lfss_parameters": lfss_parameters,
        "encoder_parameters": encoder_parameters,
        "flops": flops,
        "flops_error": flops_error,
        "latency_ms_mean": mean_latency,
        "latency_ms_std": statistics.pstdev(timings),
        "fps": 1000.0 / mean_latency,
        "inference_peak_bytes": inference_peak,
        "training_peak_bytes": training_peak,
        "dwt_calls": counts["dwt"],
        "idwt_calls": counts["idwt"],
        "training_dwt_calls": train_counts["dwt"],
        "training_idwt_calls": train_counts["idwt"],
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
        profile_one(name, device, args.warmup, args.repeats) for name in NAMES
    ]
    for result in results:
        assert result["dwt_calls"] == result["idwt_calls"] == 4
        assert result["training_dwt_calls"] == result["training_idwt_calls"] == 4
    payload = {
        "input_shape": [1, 1, 256, 256],
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "thop_note": "selective_scan may be omitted from THOP FLOPs",
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
