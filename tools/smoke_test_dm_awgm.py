import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import (
    AWGM_VARIANTS,
    DiagonalIndexCache,
    DWTFreqNet,
    WaveletEightDirectionAWGM,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test DWTFreqNet AWGM variants")
    parser.add_argument("--awgm_variant", choices=AWGM_VARIANTS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--timing-iters", type=int, default=5)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_diagonal_indices():
    expected = {
        "nwse": [0, 4, 8, 1, 5, 2, 3, 7, 6],
        "senw": [6, 7, 3, 2, 5, 1, 8, 4, 0],
        "nesw": [2, 4, 6, 1, 3, 0, 5, 7, 8],
        "swne": [8, 7, 5, 0, 3, 1, 6, 4, 2],
    }
    exact = DiagonalIndexCache.build(3, 3, order="concat")
    for direction, index in expected.items():
        if exact["idx_" + direction].tolist() != index:
            raise AssertionError("Unexpected 3x3 {} index".format(direction))
    for height, width in ((3, 3), (4, 4), (3, 5), (5, 3),
                          (16, 16), (32, 32), (64, 64)):
        for order in ("concat", "snake"):
            cache = DiagonalIndexCache.build(height, width, order=order)
            reference = torch.arange(height * width)
            for direction in expected:
                index = cache["idx_" + direction]
                if not torch.equal(torch.sort(index.cpu()).values, reference):
                    raise AssertionError("Incomplete diagonal permutation")
                restored = index.index_select(0, cache["inv_" + direction].cpu())
                if not torch.equal(restored, reference):
                    raise AssertionError("Broken inverse diagonal permutation")
    return True


def validate_sharing(module):
    if not isinstance(module, WaveletEightDirectionAWGM):
        return {"checked": False}
    axial = module.axial_branch
    diagonal = module.diagonal_branch
    axial_ids = {id(axial.get_mamba(direction)) for direction in axial.DIRECTIONS}
    diagonal_ids = {
        id(diagonal.get_mamba(direction)) for direction in diagonal.directions
    }
    all_ids = axial_ids | diagonal_ids
    if len(all_ids) != module.mamba_instance_count:
        raise AssertionError("Mamba instance count does not match shared objects")
    expected_counts = {
        "independent_8": 8,
        "pair_shared_4": 4,
        "subband_shared_3": 3,
        "axial_diag_shared_2": 2,
        "all_shared_1": 1,
    }
    expected = expected_counts[module.share_mode]
    if module.diag_directions == 2 and module.share_mode == "independent_8":
        expected = 6
    if module.diag_directions == 2 and module.share_mode == "pair_shared_4":
        expected = 3
    if len(all_ids) != expected:
        raise AssertionError(
            "Expected {} Mamba instances, found {}".format(expected, len(all_ids))
        )
    return {
        "checked": True,
        "share_mode": module.share_mode,
        "mamba_instances": len(all_ids),
        "diagonal_directions": module.diag_directions,
        "diagonal_order": module.diag_order,
        "direction_embedding": module.use_direction_embedding,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    index_tests_passed = validate_diagonal_indices()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = DWTFreqNet(
        get_DWTFreqNet_config(),
        mode="train",
        deepsuper=True,
        awgm_variant=args.awgm_variant,
        awgm_allow_fallback=args.allow_fallback,
    ).to(device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    awgm_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("wave_att_")
    )

    flops = None
    if not args.skip_flops:
        profile_input = torch.randn(
            1, 1, args.input_size, args.input_size, device=device
        )
        flops, _ = profile(model, inputs=(profile_input,), verbose=False)

    model.train()
    image = torch.randn(
        args.batch_size, 1, args.input_size, args.input_size, device=device
    )
    target = torch.rand(
        args.batch_size, 1, args.input_size, args.input_size, device=device
    )
    outputs = model(image)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    expected_shape = tuple(target.shape)
    output_shapes = [tuple(output.shape) for output in outputs]
    if any(shape != expected_shape for shape in output_shapes):
        raise AssertionError(
            f"Output shapes {output_shapes} do not match target {expected_shape}"
        )
    if not all(torch.isfinite(output).all() for output in outputs):
        raise FloatingPointError("Non-finite model output")

    criterion = nn.BCELoss()
    loss = sum(criterion(output, target) for output in outputs)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise FloatingPointError("Non-finite gradient")
    nonzero_mamba_gradient = None
    wave_module = model.wave_att_input_t
    sharing = validate_sharing(wave_module)
    if isinstance(wave_module, WaveletEightDirectionAWGM):
        unique_mixers = {
            id(wave_module.axial_branch.get_mamba(direction)):
                wave_module.axial_branch.get_mamba(direction)
            for direction in wave_module.axial_branch.DIRECTIONS
        }
        unique_mixers.update({
            id(wave_module.diagonal_branch.get_mamba(direction)):
                wave_module.diagonal_branch.get_mamba(direction)
            for direction in wave_module.diagonal_branch.directions
        })
        mixer_gradients = [
            parameter.grad
            for mixer in unique_mixers.values()
            for parameter in mixer.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        nonzero_mamba_gradient = bool(
            mixer_gradients
            and all(torch.isfinite(gradient).all() for gradient in mixer_gradients)
            and any(gradient.abs().sum().item() > 0 for gradient in mixer_gradients)
        )
        if not nonzero_mamba_gradient:
            raise AssertionError("W8M Mamba gradients are missing, zero, or non-finite")

    checkpoint_buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    checkpoint = torch.load(checkpoint_buffer, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    checkpoint_roundtrip = True

    model.eval()
    model.mode = "test"
    timing_input = torch.randn(1, 1, args.input_size, args.input_size, device=device)
    with torch.no_grad():
        for _ in range(2):
            model(timing_input)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.timing_iters):
            prediction = model(timing_input)
        synchronize(device)
        elapsed = time.perf_counter() - start

    result = {
        "variant": args.awgm_variant,
        "backends": model.awgm_backends,
        "output_shape": list(prediction.shape),
        "loss": float(loss.item()),
        "finite_gradients": finite_gradients,
        "nonzero_mamba_gradient": nonzero_mamba_gradient,
        "diagonal_index_tests": index_tests_passed,
        "sharing": sharing,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "parameters": total_params,
        "awgm_parameters": awgm_params,
        "thop_flops": flops,
        "flops_note": "THOP does not count custom selective-scan/DCN kernels exactly.",
        "inference_ms_per_image": elapsed * 1000 / args.timing_iters,
        "fps": args.timing_iters / elapsed,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "device": str(device),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
