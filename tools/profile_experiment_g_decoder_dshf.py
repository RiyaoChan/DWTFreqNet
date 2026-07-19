"""RTX 3090 complexity, latency and memory profile for Experiment G0-G3."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF,
    EXPERIMENT_G_VARIANTS,
    initialize_experiment_g_model,
)
from train_one import init_weights


def make_model(name, mode):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF(
        get_DWTFreqNet_config(), decoder_variant=name, mode=mode, deepsuper=True
    )
    initialize_experiment_g_model(model, init_weights)
    return model


def count_prefix(model, prefixes):
    return sum(parameter.numel() for name, parameter in model.named_parameters() if name.startswith(prefixes))


def count_flops(model, sample):
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(sample,), verbose=False)
        return int(flops), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def profile_one(name, device, warmup, repeats):
    model = make_model(name, "test").to(device).eval(); model.record_statistics = False
    sample = torch.randn(1, 1, 256, 256, device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    module_parameters = {
        "lfss": count_prefix(model, ("lfss_blocks.",)),
        "decoder_dshf": count_prefix(model, ("decoder_hf_refiner",)),
        "semantic_gate": count_prefix(model, tuple(f"decoder_hf_refiner{stage}.semantic_gate" for stage in range(1, 5))),
    }
    flops, flops_error = count_flops(model, sample)
    with torch.no_grad(): model(sample)
    counts = dict(model.last_transform_counts)
    with torch.no_grad():
        for _ in range(warmup): model(sample)
    if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter(); model(sample)
            if device.type == "cuda": torch.cuda.synchronize()
            timings.append(1000.0 * (time.perf_counter() - start))
    inference_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    del model, sample
    if device.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)

    train_model = make_model(name, "train").to(device).train()
    train_input = torch.randn(2, 1, 256, 256, device=device)
    outputs = train_model(train_input); loss = sum(output.mean() for output in outputs); loss.backward()
    if device.type == "cuda": torch.cuda.synchronize()
    training_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    train_counts = dict(train_model.last_transform_counts)
    del train_model, train_input, outputs, loss
    if device.type == "cuda": torch.cuda.empty_cache()
    latency = statistics.mean(timings)
    return {
        "model": name, "parameters": parameters, "module_parameters": module_parameters,
        "flops": flops, "flops_error": flops_error,
        "latency_ms_mean": latency, "latency_ms_std": statistics.pstdev(timings),
        "fps": 1000.0 / latency, "inference_peak_bytes": inference_peak,
        "training_peak_bytes": training_peak, "dwt_calls": counts["dwt"],
        "idwt_calls": counts["idwt"], "training_dwt_calls": train_counts["dwt"],
        "training_idwt_calls": train_counts["idwt"],
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5); parser.add_argument("--repeats", type=int, default=20); parser.add_argument("--output", default="")
    args = parser.parse_args(); device = torch.device(args.device)
    results = [profile_one(name, device, args.warmup, args.repeats) for name in EXPERIMENT_G_VARIANTS]
    assert all(item["dwt_calls"] == item["idwt_calls"] == item["training_dwt_calls"] == item["training_idwt_calls"] == 4 for item in results)
    payload = {"input_shape": [1, 1, 256, 256], "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "precision": "FP32", "warmup": args.warmup, "repeats": args.repeats, "thop_note": "selective_scan may be omitted from THOP FLOPs", "results": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2); print(rendered)
    if args.output: Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__": main()
