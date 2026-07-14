"""Structure, matching, baseline-regression and gradient tests for Experiment D."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import check_haar_direction_correspondence
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.DWTFreqNet_SingleDecoder_HFE import (
    ChannelMatchedAttention,
    ChannelMatching,
    DWTFreqNet_SingleDecoder_HFE,
)


def build_hfe(mode, device):
    return DWTFreqNet_SingleDecoder_HFE(
        get_DWTFreqNet_config(), mode=mode, deepsuper=True
    ).to(device)


def build_baseline(mode, device):
    return DWTFreqNet_SingleDecoder(
        get_DWTFreqNet_config(), mode=mode, deepsuper=True, sd_variant="sd_awgm"
    ).to(device)


def has_nonzero_gradient(model, prefix):
    return any(
        name.startswith(prefix)
        and parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
    )


def copy_common_parameters(source, target):
    source_state = source.state_dict()
    target_state = target.state_dict()
    common = {
        name: tensor
        for name, tensor in source_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    missing, unexpected = target.load_state_dict(common, strict=False)
    assert not unexpected
    assert all(name.startswith("decoder_hfe") for name in missing)
    return len(common)


def test_matching(device, size):
    channels = 8
    matching = ChannelMatching().to(device)
    query = torch.randn(2, channels, size, size, device=device, requires_grad=True)
    candidate = torch.randn(
        2, channels, size, size, device=device, requires_grad=True
    )
    selected, indices = matching(query, candidate)
    assert matching.last_distance_shape == (2, channels, channels)
    assert tuple(indices.shape) == (2, channels)
    assert tuple(selected.shape) == tuple(query.shape)
    assert not indices.requires_grad
    assert int(indices.min()) >= 0 and int(indices.max()) < channels
    assert selected.requires_grad
    selected.sum().backward()
    assert candidate.grad is not None
    assert bool(torch.count_nonzero(candidate.grad).item())
    try:
        ChannelMatchedAttention(10, 3)
    except ValueError:
        divisibility_assertion = True
    else:
        divisibility_assertion = False
    assert divisibility_assertion
    return {
        "distance": list(matching.last_distance_shape),
        "indices": list(indices.shape),
        "selected": list(selected.shape),
        "selected_requires_grad": selected.requires_grad,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true", help="Use required 2x256 inputs")
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
        "HFE4_shared": (batch, 256, size // 16, size // 16),
        "HFE4_refined": (batch, 256, size // 16, size // 16),
        "HFE3_shared": (batch, 256, size // 8, size // 8),
        "HFE3_refined": (batch, 256, size // 8, size // 8),
        "HFE2_shared": (batch, 128, size // 4, size // 4),
        "HFE2_refined": (batch, 128, size // 4, size // 4),
        "HFE1_shared": (batch, 64, size // 2, size // 2),
        "HFE1_refined": (batch, 64, size // 2, size // 2),
        "L3": (batch, 256, size // 8, size // 8),
        "L2": (batch, 128, size // 4, size // 4),
        "L1": (batch, 64, size // 2, size // 2),
        "L0": (batch, 32, size, size),
    }

    shape_model = build_hfe("train", device).train()
    shape_model.debug_tensors = True
    with torch.no_grad():
        train_outputs = shape_model(
            torch.randn(batch, 1, size, size, device=device)
        )
    assert len(train_outputs) == 6
    assert all(
        tuple(output.shape) == (batch, 1, size, size) for output in train_outputs
    )
    assert shape_model.last_transform_counts == {"dwt": 4, "idwt": 4}
    assert {key: shape_model.last_shapes[key] for key in expected_shapes} == expected_shapes
    assert shape_model.experiment_metadata["sd_variant"] == "sd_awgm_hfe"
    assert shape_model.experiment_metadata["directional_pyramid"] is False
    assert shape_model.experiment_metadata["ldrc"] is False
    assert shape_model.experiment_metadata["mamba"] is False
    stage_matching_shapes = {}
    for stage, channels in ((1, 64), (2, 128), (3, 256), (4, 256)):
        refiner = getattr(shape_model, f"decoder_hfe{stage}")
        for branch, transformation in (
            ("attn", refiner.hfe.attn.matching),
            ("ffn", refiner.hfe.ffn.matching),
        ):
            assert transformation.matching.last_distance_shape == (
                batch,
                channels,
                channels,
            )
            assert tuple(transformation.last_indices.shape) == (batch, channels)
            stage_matching_shapes[f"stage{stage}_{branch}"] = {
                "distance": list(transformation.matching.last_distance_shape),
                "indices": list(transformation.last_indices.shape),
            }
    parameter_count = sum(parameter.numel() for parameter in shape_model.parameters())
    hfe_parameter_count = sum(
        parameter.numel()
        for name, parameter in shape_model.named_parameters()
        if name.startswith("decoder_hfe")
    )
    del shape_model, train_outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_model = build_hfe("test", device).eval()
    test_model.record_statistics = False
    with torch.no_grad():
        test_output = test_model(torch.randn(batch, 1, size, size, device=device))
    assert tuple(test_output.shape) == (batch, 1, size, size)
    del test_model, test_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    baseline = build_baseline("test", device).eval()
    hfe_model = build_hfe("test", device).eval()
    baseline.record_statistics = False
    hfe_model.record_statistics = False
    copied_parameters = copy_common_parameters(baseline, hfe_model)
    for stage in range(1, 5):
        refiner = getattr(hfe_model, f"decoder_hfe{stage}")
        refiner.beta_h.data.zero_()
        refiner.beta_v.data.zero_()
        refiner.beta_d.data.zero_()
    regression_input = torch.randn(batch, 1, size, size, device=device)
    with torch.no_grad():
        baseline_output = baseline(regression_input)
        hfe_output = hfe_model(regression_input)
    torch.testing.assert_close(
        baseline_output, hfe_output, rtol=1e-5, atol=1e-6
    )
    max_abs_error = float((baseline_output - hfe_output).abs().max().cpu())
    del baseline, hfe_model, baseline_output, hfe_output, regression_input
    if device.type == "cuda":
        torch.cuda.empty_cache()

    matching_result = test_matching(device, 8 if args.full else 4)

    gradient_model = build_hfe("train", device).train()
    gradient_batch = batch if args.full else 1
    gradient_size = size if args.full else 64
    gradient_outputs = gradient_model(
        torch.randn(
            gradient_batch, 1, gradient_size, gradient_size, device=device
        )
    )
    sum(output.mean() for output in gradient_outputs).backward()
    required = [
        "stem",
        "local_encoder1",
        "local_encoder2",
        "local_encoder3",
        "local_encoder4",
        "dir_encoder1",
        "dir_encoder2",
        "dir_encoder3",
        "dir_encoder4",
        "stage_awgm1",
        "stage_awgm2",
        "stage_awgm3",
        "stage_awgm4",
        "align_H1",
        "align_V1",
        "align_D1",
        "align_H2",
        "align_V2",
        "align_D2",
        "align_H3",
        "align_V3",
        "align_D3",
        "align_H4",
        "align_V4",
        "align_D4",
        "decoder_fuse0",
        "decoder_fuse1",
        "decoder_fuse2",
        "decoder_fuse3",
        "gt_conv5",
        "gt_conv4",
        "gt_conv3",
        "gt_conv2",
        "out_head",
        "outconv",
    ]
    for stage in range(1, 5):
        required.extend(
            [
                f"decoder_hfe{stage}.subband_fusion",
                f"decoder_hfe{stage}.hfe.attn",
                f"decoder_hfe{stage}.hfe.ffn",
                f"decoder_hfe{stage}.head_h",
                f"decoder_hfe{stage}.head_v",
                f"decoder_hfe{stage}.head_d",
                f"decoder_hfe{stage}.beta_h",
                f"decoder_hfe{stage}.beta_v",
                f"decoder_hfe{stage}.beta_d",
            ]
        )
    missing_gradients = [
        prefix for prefix in required if not has_nonzero_gradient(gradient_model, prefix)
    ]
    assert not missing_gradients, f"Missing nonzero gradients: {missing_gradients}"
    assert gradient_model.last_transform_counts == {"dwt": 4, "idwt": 4}

    direction = check_haar_direction_correspondence(32, str(device))
    assert direction["band_response_orientation"] == {
        "H": "vertical",
        "V": "horizontal",
    }
    assert direction["routing_aligned"] is True

    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "full": args.full,
                "shape_input": [batch, 1, size, size],
                "expected_shapes": {key: list(value) for key, value in expected_shapes.items()},
                "parameters": parameter_count,
                "hfe_parameters": hfe_parameter_count,
                "common_state_entries_copied": copied_parameters,
                "baseline_zero_beta_max_abs_error": max_abs_error,
                "matching": matching_result,
                "stage_matching_shapes": stage_matching_shapes,
                "gradient_prefixes": required,
                "dwt_idwt": gradient_model.last_transform_counts,
                "haar": direction,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
