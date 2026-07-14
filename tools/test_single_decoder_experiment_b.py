"""Structural, shape, bypass and gradient tests for Experiment B."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import check_haar_direction_correspondence
from model.DWTFreqNet_SingleDecoder import (
    DWTFreqNet_SingleDecoder,
    SINGLE_DECODER_VARIANTS,
)


def build(variant, mode, device):
    return DWTFreqNet_SingleDecoder(
        get_DWTFreqNet_config(), mode=mode, deepsuper=True, sd_variant=variant
    ).to(device)


def has_nonzero_gradient(model, prefix):
    return any(
        name.startswith(prefix)
        and parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true", help="Use required 2x256 shape input")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    batch, size = (2, 256) if args.full else (1, 64)
    expected_shapes = {
        "X0": (batch, 32, size, size),
        "E1": (batch, 64, size // 2, size // 2),
        "E2": (batch, 128, size // 4, size // 4),
        "E3": (batch, 256, size // 8, size // 8),
        "E4": (batch, 256, size // 16, size // 16),
        "L3": (batch, 256, size // 8, size // 8),
        "L2": (batch, 128, size // 4, size // 4),
        "L1": (batch, 64, size // 2, size // 2),
        "L0": (batch, 32, size, size),
    }
    parameter_counts = {}
    bypass_results = {}

    for variant in SINGLE_DECODER_VARIANTS:
        model = build(variant, "test", device).eval()
        model.debug_tensors = True
        assert all(
            not hasattr(model, name) for name in model.FORBIDDEN_MODULE_NAMES
        )
        with torch.no_grad():
            output = model(torch.randn(batch, 1, size, size, device=device))
        assert tuple(output.shape) == (batch, 1, size, size)
        assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
        assert {key: model.last_shapes[key] for key in expected_shapes} == expected_shapes

        if model.use_pyramid:
            pyramid_channels = (64, 128, 256, 256)
            for stage, channels in enumerate(pyramid_channels, start=1):
                spatial = size // (2 ** stage)
                for direction in ("H", "V", "D"):
                    assert model.last_shapes[f"P{stage}{direction}"] == (
                        batch, channels, spatial, spatial
                    )

        if not model.use_awgm:
            for stage in range(1, 5):
                assert torch.equal(
                    model.last_debug["A"][stage],
                    model.last_debug["A_guided"][stage],
                )
        else:
            assert any(
                not torch.equal(
                    model.last_debug["A"][stage],
                    model.last_debug["A_guided"][stage],
                )
                for stage in range(1, 5)
            )

        if not model.use_pyramid:
            for stage in range(1, 5):
                for direction in ("H", "V", "D"):
                    tensors = model.last_debug["coefficients"][(stage, direction)]
                    assert tensors["delta"] is None
                    assert torch.equal(tensors["aligned"], tensors["coefficient"])
        else:
            assert any(
                not torch.equal(
                    model.last_debug["coefficients"][(stage, direction)]["aligned"],
                    model.last_debug["coefficients"][(stage, direction)]["coefficient"],
                )
                for stage in range(1, 5)
                for direction in ("H", "V", "D")
            )

        parameter_counts[variant] = sum(p.numel() for p in model.parameters())
        bypass_results[variant] = {
            "awgm_enabled": model.use_awgm,
            "pyramid_enabled": model.use_pyramid,
            "dwt": model.last_transform_counts["dwt"],
            "idwt": model.last_transform_counts["idwt"],
        }
        del model, output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gradient_results = {}
    gradient_size = 128 if args.full else 64
    for variant in SINGLE_DECODER_VARIANTS:
        model = build(variant, "train", device).train()
        outputs = model(torch.randn(1, 1, gradient_size, gradient_size, device=device))
        assert len(outputs) == 6
        assert all(
            tuple(output.shape) == (1, 1, gradient_size, gradient_size)
            for output in outputs
        )
        sum(output.mean() for output in outputs).backward()
        required = [
            "stem", "local_encoder1", "local_encoder2", "local_encoder3",
            "local_encoder4", "decoder_fuse0", "decoder_fuse1",
            "decoder_fuse2", "decoder_fuse3", "align_H1", "align_V2",
            "align_D3", "align_H4", "gt_conv5", "gt_conv4", "gt_conv3",
            "gt_conv2", "out_head", "outconv",
        ]
        if model.use_awgm:
            required.extend([
                "dir_encoder1", "dir_encoder2", "dir_encoder3", "dir_encoder4",
                "stage_awgm1", "stage_awgm2", "stage_awgm3", "stage_awgm4",
            ])
        elif model.use_pyramid:
            required.extend([
                "dir_encoder1", "dir_encoder2", "dir_encoder3", "dir_encoder4",
            ])
        if model.use_pyramid:
            required.extend([
                "directional_pyramid", "delta_H1", "delta_V2", "delta_D3",
                "delta_H4", "beta_H1", "beta_V2", "beta_D3", "beta_H4",
            ])
        missing = [prefix for prefix in required if not has_nonzero_gradient(model, prefix)]
        assert not missing, f"{variant} missing gradients: {missing}"
        assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
        gradient_results[variant] = required
        del model, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    direction = check_haar_direction_correspondence(32, str(device))
    assert direction["band_response_orientation"] == {
        "H": "vertical", "V": "horizontal"
    }
    assert direction["routing_aligned"] is True

    print(json.dumps({
        "status": "passed",
        "device": str(device),
        "full": args.full,
        "shape_input": [batch, 1, size, size],
        "parameter_counts": parameter_counts,
        "bypass": bypass_results,
        "gradient_prefixes": gradient_results,
        "haar": direction,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
