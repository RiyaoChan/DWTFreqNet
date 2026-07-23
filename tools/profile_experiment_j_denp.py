"""Parameters, FLOPs, latency, memory and DENP runtime components."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP import (
    DENP_STAGE_CHANNELS,
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP,
    EXPERIMENT_J_VARIANTS,
    initialize_experiment_j_model,
)
from model.decoder_denp import LearnableBandGaussian
from train_one import init_weights, set_seed


def make_model(variant, mode):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP(
        get_DWTFreqNet_config(), denp_variant=variant, mode=mode, deepsuper=True
    )
    initialize_experiment_j_model(model, init_weights)
    return model


def count_prefix(model, prefixes):
    return sum(parameter.numel() for name, parameter in model.named_parameters()
               if name.startswith(prefixes))


def count_gaussian_macs(module, inputs, output):
    del inputs
    batch, channels, height, width = output.shape
    module.total_ops += torch.DoubleTensor([
        batch * channels * height * width * module.kernel_size ** 2
    ])


def count_flops(model, sample):
    try:
        from thop import profile
    except ImportError as error:
        return None, f"thop unavailable: {error}"
    try:
        flops, _ = profile(
            model, inputs=(sample,),
            custom_ops={LearnableBandGaussian: count_gaussian_macs}, verbose=False,
        )
        return int(flops), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def timed(callable_, device, warmup, repeats):
    with torch.no_grad():
        for _ in range(warmup):
            callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            callable_()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(1000.0 * (time.perf_counter() - start))
    return statistics.mean(timings), statistics.pstdev(timings)


def component_timings(model, device, warmup, repeats, model_latency):
    if not model.use_denp:
        return {name: {"latency_ms": 0.0, "model_latency_fraction": 0.0}
                for name in ("mad", "compactness", "gaussian", "denp_complete")}
    bands, raw_lows, decoder_lows = {}, {}, {}
    raw_low_channels = {1: 32, 2: 64, 3: 128, 4: 256}
    for stage, channels in DENP_STAGE_CHANNELS.items():
        spatial = 256 // (2 ** stage)
        bands[stage] = tuple(torch.randn(1, channels, spatial, spatial, device=device)
                             for _ in range(3))
        raw_lows[stage] = torch.randn(
            1, raw_low_channels[stage], spatial, spatial, device=device
        )
        decoder_lows[stage] = torch.randn(1, channels, spatial, spatial, device=device)

    calls = {
        "mad": lambda: [
            getattr(model, f"decoder_denp{stage}").noise_estimators[band].robust_scale(
                bands[stage][index]
            )
            for stage in range(1, 5) for index, band in enumerate(("H", "V", "D"))
        ],
        "gaussian": lambda: [
            getattr(model, f"decoder_denp{stage}").gaussians[band](bands[stage][index])
            for stage in range(1, 5) for index, band in enumerate(("H", "V", "D"))
        ],
        "denp_complete": lambda: [
            getattr(model, f"decoder_denp{stage}")(
                raw_lows[stage], decoder_lows[stage], *bands[stage]
            ) for stage in range(1, 5)
        ],
    }
    compactness_calls = []
    for stage in range(1, 5):
        processor = getattr(model, f"decoder_denp{stage}")
        if hasattr(processor, "raw_compactness"):
            compactness_calls.append((processor.raw_compactness, raw_lows[stage]))
        if hasattr(processor, "decoder_compactness"):
            compactness_calls.append((processor.decoder_compactness, decoder_lows[stage]))
    calls["compactness"] = lambda: [module(tensor) for module, tensor in compactness_calls]

    results = {}
    for name, callable_ in calls.items():
        if name == "compactness" and not compactness_calls:
            latency, deviation = 0.0, 0.0
        else:
            latency, deviation = timed(callable_, device, warmup, repeats)
        results[name] = {
            "latency_ms": latency,
            "latency_ms_std": deviation,
            "model_latency_fraction": latency / model_latency if model_latency else None,
        }
    return results


def profile_one(variant, device, warmup, repeats):
    set_seed(42)
    model = make_model(variant, "test").to(device).eval()
    model.record_statistics = False
    sample = torch.randn(1, 1, 256, 256, device=device)
    groups = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "lfss": count_prefix(model, ("lfss_blocks.",)),
        "encoder": count_prefix(model, (
            "stem", "local_encoder", "dir_encoder", "stage_awgm", "lfss_blocks."
        )),
        "decoder_body": count_prefix(model, (
            "align_", "decoder_fuse", "gt_conv", "out_head", "outconv"
        )),
        "denp_total": count_prefix(model, ("decoder_denp",)),
        "denp_noise": count_prefix(model, tuple(
            f"decoder_denp{stage}.noise_estimators" for stage in range(1, 5)
        )),
        "denp_gaussian": count_prefix(model, tuple(
            f"decoder_denp{stage}.gaussians" for stage in range(1, 5)
        )),
        "denp_compactness": count_prefix(model, tuple(
            f"decoder_denp{stage}.raw_compactness" for stage in range(1, 5)
        ) + tuple(
            f"decoder_denp{stage}.decoder_compactness" for stage in range(1, 5)
        )),
        "denp_reliability": count_prefix(model, tuple(
            f"decoder_denp{stage}.gamma_" for stage in range(1, 5)
        )),
    }
    flops, flops_error = count_flops(model, sample)
    with torch.no_grad():
        model(sample)
    counts = dict(model.last_transform_counts)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mean_latency, std_latency = timed(lambda: model(sample), device, warmup, repeats)
    inference_peak = (torch.cuda.max_memory_allocated(device)
                      if device.type == "cuda" else None)
    components = component_timings(model, device, warmup, repeats, mean_latency)
    del model, sample
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(42)
    train_model = make_model(variant, "train").to(device).train()
    train_input = torch.randn(1, 1, 256, 256, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    outputs = train_model(train_input)
    loss = sum(output.float().mean() for output in outputs)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_peak = (torch.cuda.max_memory_allocated(device)
                     if device.type == "cuda" else None)
    training_counts = dict(train_model.last_transform_counts)
    del train_model, train_input, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "variant": variant,
        "parameters": groups,
        "flops": flops,
        "flops_error": flops_error,
        "latency_ms_mean": mean_latency,
        "latency_ms_std": std_latency,
        "fps": 1000.0 / mean_latency,
        "inference_peak_bytes": inference_peak,
        "training_peak_bytes": training_peak,
        "component_runtime": components,
        "dwt_calls": counts["dwt"],
        "idwt_calls": counts["idwt"],
        "training_dwt_calls": training_counts["dwt"],
        "training_idwt_calls": training_counts["idwt"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.repeats < 20:
        raise ValueError("Experiment J requires at least 20 profile repeats")
    device = torch.device(args.device)
    results = [profile_one(variant, device, args.warmup, args.repeats)
               for variant in EXPERIMENT_J_VARIANTS]
    indexed = {result["variant"]: result for result in results}
    for result in results:
        assert result["dwt_calls"] == result["idwt_calls"] == 4
        assert result["training_dwt_calls"] == result["training_idwt_calls"] == 4
    assert (indexed["j2_rawll_compactness"]["parameters"]["total"]
            == indexed["j2_decoder_compactness"]["parameters"]["total"])
    assert (indexed["j3_dual_evidence_reliability"]["parameters"]["total"]
            - indexed["j3_dual_evidence_fixed"]["parameters"]["total"] == 24)
    payload = {
        "input_shape": [1, 1, 256, 256],
        "precision": "FP32",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "thop_note": (
            "DENP Gaussian 3x3 depthwise MACs use an explicit THOP handler; "
            "median, comparisons and selective_scan may be omitted by THOP."
        ),
        "component_note": (
            "MAD, compactness and Gaussian times are isolated four-stage microbenchmarks; "
            "their fractions are diagnostic ratios to end-to-end latency, not additive attribution."
        ),
        "j2_parameter_counts_equal": True,
        "j3_reliability_extra_parameters": 24,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
