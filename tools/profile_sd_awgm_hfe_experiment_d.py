"""Complexity profile for Original, WULLE-A, SD-AWGM and Experiment D HFE."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import (
    DWTFreqNet,
    HaarWaveletTransform,
    InverseHaarWaveletTransform,
)
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.DWTFreqNet_SingleDecoder_HFE import DWTFreqNet_SingleDecoder_HFE
from model.DWTFreqNet_WULLE import DWTFreqNet_WULLE


MODEL_NAMES = (
    "dwtfreqnet_original",
    "dwtfreqnet_wulle_a",
    "sd_awgm",
    "sd_awgm_hfe",
)


def make_model(name, mode):
    config = get_DWTFreqNet_config()
    if name == "dwtfreqnet_original":
        return DWTFreqNet(
            config, mode=mode, deepsuper=True, awgm_variant="awgm_original"
        )
    if name == "dwtfreqnet_wulle_a":
        return DWTFreqNet_WULLE(
            config, mode=mode, deepsuper=True, awgm_variant="awgm_original"
        )
    if name == "sd_awgm":
        return DWTFreqNet_SingleDecoder(
            config, mode=mode, deepsuper=True, sd_variant="sd_awgm"
        )
    if name == "sd_awgm_hfe":
        return DWTFreqNet_SingleDecoder_HFE(
            config, mode=mode, deepsuper=True
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
    sample = torch.randn(1, 1, 256, 256, device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    hfe_parameters = sum(
        parameter.numel()
        for parameter_name, parameter in model.named_parameters()
        if parameter_name.startswith("decoder_hfe")
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
            timings.append(1000 * (time.perf_counter() - start))
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
    loss = (
        sum(output.mean() for output in outputs)
        if isinstance(outputs, tuple)
        else outputs.mean()
    )
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [
        profile_one(name, device, args.warmup, args.repeats)
        for name in MODEL_NAMES
    ]
    by_name = {item["model"]: item for item in results}
    assert by_name["sd_awgm_hfe"]["hfe_parameters"] > 0
    assert by_name["sd_awgm_hfe"]["parameters"] > by_name["sd_awgm"]["parameters"]
    for name in ("sd_awgm", "sd_awgm_hfe"):
        assert by_name[name]["dwt_calls"] == 4
        assert by_name[name]["idwt_calls"] == 4
    payload = {
        "input_shape": [1, 1, 256, 256],
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "flops_note": (
            "THOP does not count torch.cdist and may omit direct attention "
            "matrix multiplications; latency and memory are measured separately."
        ),
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
