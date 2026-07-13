"""Shape, identity, gradient and Haar-routing tests for Experiment C."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import check_haar_direction_correspondence
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.DWTFreqNet_SingleDecoder_LDRC import (
    DWTFreqNet_SingleDecoder_LDRC,
)


def has_nonzero_gradient(model, prefix):
    return any(
        name.startswith(prefix)
        and parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
    )


def build_new(mode, device):
    return DWTFreqNet_SingleDecoder_LDRC(
        get_DWTFreqNet_config(), mode=mode, deepsuper=True
    ).to(device)


def build_baseline(mode, device):
    return DWTFreqNet_SingleDecoder(
        get_DWTFreqNet_config(), mode=mode, deepsuper=True,
        sd_variant="sd_awgm",
    ).to(device)


def copy_shared_parameters(source, target):
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {
        name: tensor
        for name, tensor in source_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    result = target.load_state_dict(shared, strict=False)
    unexpected = [name for name in result.unexpected_keys if not name.startswith("encoder_ldrc")]
    assert not unexpected, unexpected
    return len(shared)


def expected_shapes(batch, size):
    return {
        "X0": (batch, 32, size, size),
        "E1": (batch, 64, size // 2, size // 2),
        "E2": (batch, 128, size // 4, size // 4),
        "E3": (batch, 256, size // 8, size // 8),
        "E4": (batch, 256, size // 16, size // 16),
        "R1": (batch, 128, size // 4, size // 4),
        "R2": (batch, 128, size // 4, size // 4),
        "R3": (batch, 128, size // 8, size // 8),
        "R4": (batch, 128, size // 16, size // 16),
        "T1": (batch, (size // 4) ** 2, 128),
        "T2": (batch, (size // 4) ** 2, 128),
        "T3": (batch, (size // 8) ** 2, 128),
        "T4": (batch, (size // 16) ** 2, 128),
        "E1e": (batch, 64, size // 2, size // 2),
        "E2e": (batch, 128, size // 4, size // 4),
        "E3e": (batch, 256, size // 8, size // 8),
        "E4e": (batch, 256, size // 16, size // 16),
        "L3": (batch, 256, size // 8, size // 8),
        "L2": (batch, 128, size // 4, size // 4),
        "L1": (batch, 64, size // 2, size // 2),
        "L0": (batch, 32, size, size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true", help="Run required 2x256 shape test")
    args = parser.parse_args()
    device = torch.device(args.device)
    batch, size = (2, 256) if args.full else (1, 64)

    train_model = build_new("train", device).train()
    with torch.no_grad():
        train_outputs = train_model(torch.randn(batch, 1, size, size, device=device))
    assert len(train_outputs) == 6
    assert all(tuple(output.shape) == (batch, 1, size, size) for output in train_outputs)
    assert train_model.last_transform_counts == {"dwt": 4, "idwt": 4}
    assert train_model.last_shapes == expected_shapes(batch, size)
    metadata = train_model.experiment_metadata
    assert metadata["sd_variant"] == "sd_awgm_ldrc"
    assert metadata["stage_wise_awgm"] is True
    assert metadata["directional_pyramid"] is False
    assert metadata["second_dwt"] is False
    assert metadata["ldrc"] is True
    assert metadata["mamba"] is False
    del train_model, train_outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_model = build_new("test", device).eval()
    with torch.no_grad():
        test_output = test_model(torch.randn(batch, 1, size, size, device=device))
    assert tuple(test_output.shape) == (batch, 1, size, size)
    assert test_model.last_transform_counts == {"dwt": 4, "idwt": 4}
    assert test_model.last_shapes == expected_shapes(batch, size)
    for key in (
        "ldrc_gamma1_mean", "ldrc_gamma4_mean",
        "ldrc_E1e_E1_norm_ratio", "ldrc_E4e_E4_norm_ratio",
        "ldrc_sam_output_norm", "ldrc_cam_output_norm", "ldrc_ffl_output_norm",
    ):
        assert key in test_model.last_sd_statistics, key
    del test_model, test_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    identity_size = 256 if args.full else 64
    baseline = build_baseline("test", device).eval()
    new_model = build_new("test", device).eval()
    copied = copy_shared_parameters(baseline, new_model)
    with torch.no_grad():
        for stage in range(1, 5):
            getattr(new_model.encoder_ldrc, f"gamma{stage}").zero_()
        identity_input = torch.randn(1, 1, identity_size, identity_size, device=device)
        baseline_output = baseline(identity_input)
        new_output = new_model(identity_input)
    torch.testing.assert_close(
        baseline_output, new_output, rtol=1e-5, atol=1e-6
    )
    identity_max_abs = float((baseline_output - new_output).abs().max().cpu())
    del baseline, new_model, identity_input, baseline_output, new_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gradient_size = 256 if args.full else 64
    gradient_model = build_new("train", device).train()
    outputs = gradient_model(
        torch.randn(1, 1, gradient_size, gradient_size, device=device)
    )
    sum(output.mean() for output in outputs).backward()
    required = [
        "stem",
        "local_encoder1", "local_encoder2", "local_encoder3", "local_encoder4",
        "dir_encoder1", "dir_encoder2", "dir_encoder3", "dir_encoder4",
        "stage_awgm1", "stage_awgm2", "stage_awgm3", "stage_awgm4",
        "align_H1", "align_V1", "align_D1",
        "align_H2", "align_V2", "align_D2",
        "align_H3", "align_V3", "align_D3",
        "align_H4", "align_V4", "align_D4",
        "encoder_ldrc.proj1", "encoder_ldrc.proj2",
        "encoder_ldrc.proj3", "encoder_ldrc.proj4",
        "encoder_ldrc.ldrc1", "encoder_ldrc.ldrc2",
        "encoder_ldrc.ldrc3", "encoder_ldrc.ldrc4",
        "encoder_ldrc.back1", "encoder_ldrc.back2",
        "encoder_ldrc.back3", "encoder_ldrc.back4",
        "encoder_ldrc.gamma1", "encoder_ldrc.gamma2",
        "encoder_ldrc.gamma3", "encoder_ldrc.gamma4",
        "decoder_fuse0", "decoder_fuse1", "decoder_fuse2", "decoder_fuse3",
        "gt_conv5", "gt_conv4", "gt_conv3", "gt_conv2", "out_head", "outconv",
    ]
    missing = [prefix for prefix in required if not has_nonzero_gradient(gradient_model, prefix)]
    assert not missing, f"Missing non-zero gradients: {missing}"
    assert gradient_model.last_transform_counts == {"dwt": 4, "idwt": 4}
    parameter_count = sum(parameter.numel() for parameter in gradient_model.parameters())
    ldrc_parameter_count = sum(
        parameter.numel()
        for name, parameter in gradient_model.named_parameters()
        if name.startswith("encoder_ldrc")
    )
    del gradient_model, outputs
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
        "parameters": parameter_count,
        "ldrc_parameters": ldrc_parameter_count,
        "shared_parameters_copied": copied,
        "identity_max_abs": identity_max_abs,
        "gradient_prefixes": required,
        "haar": direction,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
