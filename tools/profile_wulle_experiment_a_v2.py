"""Profile original DWTFreqNet and Experiment-A WULLE under one protocol."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import DWTFreqNet
from model.DWTFreqNet_WULLE import DWTFreqNet_WULLE


def is_local_parameter(name):
    return (
        name.startswith("conv_wavelet_inchannel_local")
        or name.startswith("local_encoder")
        or name.startswith("wulle_decoder")
    )


def count_parameters(model, predicate=lambda _name: True):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if predicate(name)
    )


def flops(model, sample):
    try:
        from thop import profile
    except ImportError:
        return None
    value, _ = profile(model, inputs=(sample,), verbose=False)
    return int(value)


def profile_model(name, model_class, device, warmup, repeats):
    model = model_class(
        get_DWTFreqNet_config(), mode="test", deepsuper=True,
        awgm_variant="awgm_original", awgm_allow_fallback=False,
    ).to(device).eval()
    sample = torch.randn(1, 1, 256, 256, device=device)
    total = count_parameters(model)
    local = count_parameters(model, is_local_parameter)
    flop_count = flops(model, sample)

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
            timings.append(1000 * (time.perf_counter() - start))
    inference_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_model = model_class(
        get_DWTFreqNet_config(), mode="train", deepsuper=True,
        awgm_variant="awgm_original", awgm_allow_fallback=False,
    ).to(device).train()
    train_input = torch.randn(1, 1, 256, 256, device=device)
    outputs = train_model(train_input)
    sum(output.mean() for output in outputs).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    mean_latency = statistics.mean(timings)
    return {
        "model": name,
        "parameters": total,
        "local_parameters": local,
        "flops": flop_count,
        "latency_ms_mean": mean_latency,
        "latency_ms_std": statistics.pstdev(timings),
        "fps": 1000.0 / mean_latency,
        "inference_peak_bytes": inference_peak,
        "training_peak_bytes": training_peak,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [
        profile_model("dwtfreqnet_original", DWTFreqNet, device, args.warmup, args.repeats),
        profile_model("dwtfreqnet_wulle_a", DWTFreqNet_WULLE, device, args.warmup, args.repeats),
    ]
    assert results[1]["parameters"] < results[0]["parameters"]
    payload = {"input_shape": [1, 1, 256, 256], "device": str(device), "results": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
