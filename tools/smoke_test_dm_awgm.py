import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import AWGM_VARIANTS, DWTFreqNet


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


def main():
    args = parse_args()
    device = torch.device(args.device)
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
        "parameters": total_params,
        "awgm_parameters": awgm_params,
        "thop_flops": flops,
        "flops_note": "THOP does not count custom selective-scan/DCN kernels exactly.",
        "inference_ms_per_image": elapsed * 1000 / args.timing_iters,
        "device": str(device),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
