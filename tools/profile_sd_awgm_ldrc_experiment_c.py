"""Complexity profile for Experiment C and its three reference models."""

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
from model.DWTFreqNet_SingleDecoder_LDRC import DWTFreqNet_SingleDecoder_LDRC
from model.DWTFreqNet_WULLE import DWTFreqNet_WULLE


MODEL_NAMES = (
    "dwtfreqnet_original",
    "dwtfreqnet_wulle_a",
    "sd_awgm",
    "sd_awgm_ldrc",
)


def make_model(name, mode):
    config = get_DWTFreqNet_config()
    if name == "dwtfreqnet_original":
        return DWTFreqNet(config, mode=mode, deepsuper=True, awgm_variant="awgm_original")
    if name == "dwtfreqnet_wulle_a":
        return DWTFreqNet_WULLE(
            config, mode=mode, deepsuper=True, awgm_variant="awgm_original"
        )
    if name == "sd_awgm":
        return DWTFreqNet_SingleDecoder(
            config, mode=mode, deepsuper=True, sd_variant="sd_awgm"
        )
    return DWTFreqNet_SingleDecoder_LDRC(config, mode=mode, deepsuper=True)


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
            handles.append(module.register_forward_hook(
                lambda _m, _i, _o: counts.__setitem__("dwt", counts["dwt"] + 1)
            ))
        elif isinstance(module, InverseHaarWaveletTransform):
            handles.append(module.register_forward_hook(
                lambda _m, _i, _o: counts.__setitem__("idwt", counts["idwt"] + 1)
            ))
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
    ldrc_parameters = sum(
        parameter.numel()
        for parameter_name, parameter in model.named_parameters()
        if parameter_name.startswith("encoder_ldrc")
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

    train_model = make_model(name, "train").to(device).train()
    train_input = torch.randn(1, 1, 256, 256, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    outputs = train_model(train_input)
    loss = (
        sum(output.mean() for output in outputs)
        if isinstance(outputs, tuple) else outputs.mean()
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
        "ldrc_parameters": ldrc_parameters,
        "flops": flops,
        "latency_ms_mean": mean_latency,
        "latency_ms_std": statistics.pstdev(timings),
        "fps": 1000.0 / mean_latency,
        "inference_peak_bytes": inference_peak,
        "training_peak_bytes": training_peak,
        "dwt_calls": counts["dwt"],
        "idwt_calls": counts["idwt"],
    }


def attention_matrix_sizes():
    shapes = {
        "ldrc4": {"sam": [256, 256], "cam": [256, 9216]},
        "ldrc3": {"sam": [1024, 1024], "cam": [1024, 8448]},
        "ldrc2": {"sam": [4096, 4096], "cam": [4096, 5376]},
        "ldrc1": {"sam": [4096, 4096], "cam": [4096, 5376]},
    }
    head_count = 4
    total_elements = 0
    for stage in shapes.values():
        for shape in stage.values():
            total_elements += head_count * shape[0] * shape[1]
    return {
        "head_count": head_count,
        "per_head_shapes": shapes,
        "batch1_total_attention_elements_all_heads": total_elements,
        "batch1_fp32_attention_mib": total_elements * 4 / (1024 ** 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [profile_one(name, device, args.warmup, args.repeats) for name in MODEL_NAMES]
    by_name = {item["model"]: item for item in results}
    for name in ("sd_awgm", "sd_awgm_ldrc"):
        assert by_name[name]["dwt_calls"] == 4
        assert by_name[name]["idwt_calls"] == 4
    payload = {
        "input_shape": [1, 1, 256, 256],
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "attention_matrices": attention_matrix_sizes(),
        "results": results,
        "sd_awgm_to_ldrc_increment": {
            "parameters": (
                by_name["sd_awgm_ldrc"]["parameters"]
                - by_name["sd_awgm"]["parameters"]
            ),
            "flops": (
                None
                if by_name["sd_awgm_ldrc"]["flops"] is None
                else by_name["sd_awgm_ldrc"]["flops"]
                - by_name["sd_awgm"]["flops"]
            ),
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
