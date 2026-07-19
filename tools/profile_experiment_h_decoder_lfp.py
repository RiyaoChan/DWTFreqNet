"""Parameters, THOP FLOPs, latency and memory for Experiment H."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP,
    EXPERIMENT_H_VARIANTS,
    initialize_experiment_h_model,
)
from model.decoder_lfp import LearnableDepthwiseGaussian
from train_one import init_weights, set_seed


NAMES = EXPERIMENT_H_VARIANTS
PAIRS = (
    ("h1_rawll_attention", "h1_decoder_attention"),
    ("h2_rawll_fixed_gaussian", "h2_decoder_fixed_gaussian"),
    ("h3_rawll_adaptive_gaussian", "h3_decoder_adaptive_gaussian"),
)


def make_model(name, mode):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP(
        get_DWTFreqNet_config(), lfp_variant=name, mode=mode, deepsuper=True
    )
    initialize_experiment_h_model(model, init_weights)
    return model


def count_prefix(model, prefixes):
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def count_gaussian_macs(module, inputs, output):
    del inputs
    batch, channels, height, width = output.shape
    operations = batch * channels * height * width * module.kernel_size ** 2
    module.total_ops += torch.DoubleTensor([operations])


def count_flops(model, sample):
    try:
        from thop import profile
    except ImportError as error:
        return None, f"thop unavailable: {error}"
    try:
        flops, _ = profile(
            model,
            inputs=(sample,),
            custom_ops={LearnableDepthwiseGaussian: count_gaussian_macs},
            verbose=False,
        )
        return int(flops), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def profile_one(name, device, warmup, repeats):
    set_seed(42)
    model = make_model(name, "test").to(device).eval()
    model.record_statistics = False
    sample = torch.randn(1, 1, 256, 256, device=device)
    groups = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "lfss": count_prefix(model, ("lfss_blocks.",)),
        "encoder": count_prefix(
            model,
            ("stem", "local_encoder", "dir_encoder", "stage_awgm", "lfss_blocks."),
        ),
        "decoder_body": count_prefix(
            model,
            ("align_", "decoder_fuse", "gt_conv", "out_head", "outconv"),
        ),
        "lfp_total": count_prefix(model, ("decoder_lfp",)),
        "lfp_attention": count_prefix(
            model, tuple(f"decoder_lfp{stage}.attention" for stage in range(1, 5))
        ),
        "lfp_gaussian": count_prefix(
            model, tuple(f"decoder_lfp{stage}.gaussian" for stage in range(1, 5))
        ),
        "lfp_adaptive_threshold": count_prefix(
            model,
            tuple(f"decoder_lfp{stage}.threshold_predictor" for stage in range(1, 5)),
        ),
    }
    flops, flops_error = count_flops(model, sample)
    with torch.no_grad():
        model(sample)
        for _ in range(warmup):
            model(sample)
    counts = dict(model.last_transform_counts)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(1000.0 * (time.perf_counter() - start))
    inference_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    del model, sample
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(42)
    train_model = make_model(name, "train").to(device).train()
    train_input = torch.randn(1, 1, 256, 256, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    outputs = train_model(train_input)
    loss = sum(output.float().mean() for output in outputs)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    train_counts = dict(train_model.last_transform_counts)
    del train_model, train_input, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_latency = statistics.mean(timings)
    return {
        "model": name,
        "parameters": groups,
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.repeats < 20:
        raise ValueError("Experiment H requires at least 20 profile repeats")
    device = torch.device(args.device)
    results = [profile_one(name, device, args.warmup, args.repeats) for name in NAMES]
    indexed = {result["model"]: result for result in results}
    for result in results:
        assert result["dwt_calls"] == result["idwt_calls"] == 4
        assert result["training_dwt_calls"] == result["training_idwt_calls"] == 4
    for left, right in PAIRS:
        assert indexed[left]["parameters"]["total"] == indexed[right]["parameters"]["total"]
        assert indexed[left]["parameters"]["lfp_total"] == indexed[right]["parameters"]["lfp_total"]
    payload = {
        "input_shape": [1, 1, 256, 256],
        "precision": "FP32",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "thop_note": (
            "Gaussian depthwise 3x3 MACs use an explicit THOP handler; "
            "selective_scan may still be omitted by THOP."
        ),
        "paired_parameter_counts_equal": True,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
