"""Validation suite for Experiment D spatial ablations D5, D6 and D7."""

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.DWTFreqNet_SingleDecoder_HFE import (
    ChannelMatching,
    MatchingTransformation,
)
from model.DWTFreqNet_SingleDecoder_HFE_Ablation import (
    DirectFusionTransformation,
    LocalCorrelationGate,
    SoftCosineTopKMatching,
    SoftMatchingTransformation,
)
from model.DWTFreqNet_SingleDecoder_HFE_SpatialAblation import (
    CENTER_OFFSET_INDEX,
    DWTFreqNet_SingleDecoder_HFE_SpatialAblation,
    NeighborhoodCrossFrequencyFusion,
    OFFSETS_3X3,
    SPATIAL_HFE_ABLATION_VARIANTS,
    SPATIAL_STAGE_CONFIG,
    SamePositionConsistencyFusion,
    TargetAwareNeighborhoodFusion,
    aggregate_shifted_low,
    shift_feature,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--full", action="store_true", help="Use 2x256 inputs")
    return parser.parse_args()


def build_spatial(variant, mode, device):
    return DWTFreqNet_SingleDecoder_HFE_SpatialAblation(
        get_DWTFreqNet_config(),
        spatial_hfe_ablation=variant,
        mode=mode,
        deepsuper=True,
    ).to(device)


def build_d4(mode, device):
    from model.DWTFreqNet_SingleDecoder_HFE_Ablation import (
        DWTFreqNet_SingleDecoder_HFE_Ablation,
    )

    return DWTFreqNet_SingleDecoder_HFE_Ablation(
        get_DWTFreqNet_config(),
        hfe_ablation="d4_no_matching",
        mode=mode,
        deepsuper=True,
    ).to(device)


def build_d0(mode, device):
    return DWTFreqNet_SingleDecoder(
        get_DWTFreqNet_config(),
        mode=mode,
        deepsuper=True,
        sd_variant="sd_awgm",
    ).to(device)


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


def copy_direct_head(source, target):
    for name in ("gate", "value", "project"):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())


def has_nonzero_gradient(module, prefix):
    return any(
        name.startswith(prefix)
        and parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in module.named_parameters()
    )


def test_shift_directions(device):
    coordinate = torch.arange(25, device=device).reshape(1, 1, 5, 5).float()
    checks = {}
    for dy, dx in OFFSETS_3X3:
        shifted = shift_feature(coordinate, dy, dx)
        for y in range(1, 4):
            for x in range(1, 4):
                expected = coordinate[0, 0, y + dy, x + dx]
                torch.testing.assert_close(shifted[0, 0, y, x], expected)
        checks[f"{dy},{dx}"] = float(shifted[0, 0, 2, 2].cpu())
    return checks


def test_d5_fairness(device):
    d4 = DirectFusionTransformation(16).to(device)
    d5 = SamePositionConsistencyFusion(16, 4).to(device)
    copy_direct_head(d4, d5)
    high = torch.randn(2, 16, 9, 9, device=device)
    low = torch.randn(2, 16, 9, 9, device=device)
    d4_output, _ = d4(high, low)
    d5_output, info = d5(high, low)
    torch.testing.assert_close(
        d5.last_spatial_scale,
        torch.ones_like(d5.last_spatial_scale),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(d4_output, d5_output, atol=1e-6, rtol=1e-5)
    return {
        "spatial_scale_shape": info["spatial_scale_shape"],
        "spatial_scale_min": info["spatial_scale_min"],
        "spatial_scale_max": info["spatial_scale_max"],
        "max_abs_error_vs_d4": float(
            (d4_output - d5_output).detach().abs().max().cpu()
        ),
    }


def test_d6_center_degeneration(device):
    low = torch.randn(2, 16, 9, 9, device=device)
    high = torch.randn_like(low)
    attention = torch.zeros(2, 9, 9, 9, device=device)
    attention[:, CENTER_OFFSET_INDEX] = 1.0
    matched = aggregate_shifted_low(low, attention)
    torch.testing.assert_close(matched, low, atol=0, rtol=0)
    d4 = DirectFusionTransformation(16).to(device)
    d6 = NeighborhoodCrossFrequencyFusion(16, 4).to(device)
    copy_direct_head(d4, d6)
    d4_output, _ = d4(high, low)
    d6_center_output, _ = DirectFusionTransformation.forward(d6, high, matched)
    torch.testing.assert_close(d4_output, d6_center_output, atol=1e-6, rtol=1e-5)

    d6.record_statistics = True
    output, info = d6(high, low)
    torch.testing.assert_close(
        d6.last_attention.sum(dim=1),
        torch.ones_like(d6.last_attention[:, 0]),
        atol=1e-6,
        rtol=1e-5,
    )
    return {
        "attention_shape": info["attention_shape"],
        "center_degeneration_max_abs_error": float(
            (d4_output - d6_center_output).detach().abs().max().cpu()
        ),
        "attention_sum_error": float(
            (d6.last_attention.sum(dim=1) - 1.0).abs().max().cpu()
        ),
        "output_shape": tuple(output.shape),
    }


def test_d7_uniform_prior(device):
    d6 = NeighborhoodCrossFrequencyFusion(16, 4).to(device)
    d7 = TargetAwareNeighborhoodFusion(16, 4).to(device)
    d7.q_proj.load_state_dict(d6.q_proj.state_dict())
    d7.k_proj.load_state_dict(d6.k_proj.state_dict())
    d7.log_temperature.data.copy_(d6.log_temperature.data)
    copy_direct_head(d6, d7)
    high = torch.randn(2, 16, 9, 9, device=device)
    low = torch.randn_like(high)
    targetness = torch.full((2, 1, 9, 9), 0.5, device=device)
    d6_output, _ = d6(high, low)
    d7_output, info = d7(high, low, targetness)
    torch.testing.assert_close(d6_output, d7_output, atol=1e-5, rtol=1e-5)
    assert info["targetness_detached"] is True
    assert math.isclose(info["targetness_scale"], 1.0, abs_tol=1e-6)

    side_logit = torch.randn(2, 1, 9, 9, device=device, requires_grad=True)
    detached_targetness = torch.sigmoid(side_logit).detach()
    detached_output, _ = d7(high, low, detached_targetness)
    detached_output.mean().backward()
    assert side_logit.grad is None
    return {
        "uniform_prior_max_abs_error": float(
            (d6_output - d7_output).detach().abs().max().cpu()
        ),
        "targetness_scale": info["targetness_scale"],
        "targetness_detached": info["targetness_detached"],
    }


def assert_structure_and_metadata(models):
    expected_types = {
        "d5_same_position": SamePositionConsistencyFusion,
        "d6_neighborhood": NeighborhoodCrossFrequencyFusion,
        "d7_target_neighborhood": TargetAwareNeighborhoodFusion,
    }
    expected_ids = {
        "d5_same_position": "D5",
        "d6_neighborhood": "D6",
        "d7_target_neighborhood": "D7",
    }
    forbidden = (
        SoftCosineTopKMatching,
        SoftMatchingTransformation,
        LocalCorrelationGate,
        ChannelMatching,
        MatchingTransformation,
    )
    metadata = {}
    for variant, model in models.items():
        for stage in range(1, 5):
            refiner = getattr(model, f"decoder_hfe{stage}")
            assert isinstance(refiner.hfe.attn.relation, expected_types[variant])
            assert isinstance(refiner.hfe.ffn.relation, expected_types[variant])
            assert refiner.hfe.attn.relation is not refiner.hfe.ffn.relation
        for module in model.modules():
            assert not isinstance(module, forbidden)
        item = model.experiment_metadata
        assert item["ablation_id"] == expected_ids[variant]
        assert item["explicit_channel_matching"] is False
        assert item["channel_similarity_matrix"] is False
        assert item["channel_candidate_selection"] is False
        if variant == "d7_target_neighborhood":
            assert item["targetness_prior_detached"] is True
            assert item["side_head_mapping"] == {
                "stage4": "gt_conv5",
                "stage3": "gt_conv4",
                "stage2": "gt_conv3",
                "stage1": "gt_conv2",
            }
        metadata[variant] = item
    return metadata


def assert_forward(variant, device, batch, size):
    model = build_spatial(variant, "train", device).eval()
    model.debug_tensors = True
    original_cdist, original_topk = torch.cdist, torch.topk

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Global matching operators are forbidden in D5-D7")

    torch.cdist = forbidden
    torch.topk = forbidden
    try:
        with torch.no_grad():
            outputs = model(torch.randn(batch, 1, size, size, device=device))
    finally:
        torch.cdist, torch.topk = original_cdist, original_topk
    assert len(outputs) == 6
    assert all(tuple(item.shape) == (batch, 1, size, size) for item in outputs)
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    relation_shapes = {}
    for stage in range(1, 5):
        expected_hw = size // (2 ** stage)
        relation = getattr(model, f"decoder_hfe{stage}").hfe.attn.relation
        info = relation.last_info
        if variant == "d5_same_position":
            expected = (batch, 1, expected_hw, expected_hw)
            assert info["spatial_scale_shape"] == expected
            torch.testing.assert_close(
                relation.last_spatial_scale,
                torch.ones_like(relation.last_spatial_scale),
                atol=1e-6,
                rtol=0,
            )
        else:
            expected = (batch, 9, expected_hw, expected_hw)
            assert info["attention_shape"] == expected
            torch.testing.assert_close(
                relation.last_attention.sum(dim=1),
                torch.ones_like(relation.last_attention[:, 0]),
                atol=1e-6,
                rtol=1e-5,
            )
        relation_shapes[f"stage{stage}"] = expected
    if variant == "d7_target_neighborhood":
        assert model.last_targetness_requires_grad == {
            1: False, 2: False, 3: False, 4: False
        }
    parameters = sum(parameter.numel() for parameter in model.parameters())
    relation_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".relation." in name and name.startswith("decoder_hfe")
    )
    del model, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "parameters": parameters,
        "relation_parameters": relation_parameters,
        "relation_shapes": relation_shapes,
        "cdist_forbidden": True,
        "topk_forbidden": True,
    }


def assert_zero_beta_regression(variant, device, size):
    baseline = build_d0("test", device).eval()
    model = build_spatial(variant, "test", device).eval()
    baseline.record_statistics = False
    model.record_statistics = False
    copied = copy_common_parameters(baseline, model)
    for stage in range(1, 5):
        refiner = getattr(model, f"decoder_hfe{stage}")
        refiner.beta_h.data.zero_()
        refiner.beta_v.data.zero_()
        refiner.beta_d.data.zero_()
    sample = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        expected = baseline(sample)
        observed = model(sample)
    torch.testing.assert_close(expected, observed, atol=1e-6, rtol=1e-5)
    result = {
        "common_state_entries": copied,
        "max_abs_error": float((expected - observed).abs().max().cpu()),
    }
    del baseline, model, sample, expected, observed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _gradient_pass(model, sample):
    outputs = model(sample)
    loss = sum(output.float().square().mean() for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    return outputs, loss


def assert_gradients_and_amp(variant, device, batch, size):
    model = build_spatial(variant, "train", device).train()
    sample = torch.randn(batch, 1, size, size, device=device)
    if variant == "d5_same_position":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        _, first_loss = _gradient_pass(model, sample)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del first_loss
    outputs, loss = _gradient_pass(model, sample)
    required = ["stem", "stage_awgm1", "decoder_fuse0", "out_head"]
    for stage in range(1, 5):
        base = f"decoder_hfe{stage}"
        required.extend(
            [
                f"{base}.subband_fusion",
                f"{base}.hfe.attn",
                f"{base}.hfe.ffn",
                f"{base}.head_h",
                f"{base}.head_v",
                f"{base}.head_d",
                f"{base}.beta_h",
                f"{base}.beta_v",
                f"{base}.beta_d",
            ]
        )
        for branch in ("attn", "ffn"):
            relation = f"{base}.hfe.{branch}.relation"
            required.extend(
                [f"{relation}.q_proj", f"{relation}.k_proj",
                 f"{relation}.gate", f"{relation}.value",
                 f"{relation}.project"]
            )
            if variant == "d5_same_position":
                required.extend(
                    [f"{relation}.low_response", f"{relation}.spatial_gate"]
                )
            else:
                required.append(f"{relation}.log_temperature")
            if variant == "d7_target_neighborhood":
                required.append(f"{relation}.raw_targetness_scale")
    if variant == "d7_target_neighborhood":
        required.extend(["gt_conv5", "gt_conv4", "gt_conv3", "gt_conv2"])
    missing = [prefix for prefix in required if not has_nonzero_gradient(model, prefix)]
    assert not missing, f"Missing nonzero gradients for {variant}: {missing}"
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not nonfinite, f"Non-finite gradients for {variant}: {nonfinite}"
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}

    amp = device.type == "cuda"
    if amp:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_outputs = model(sample)
            amp_loss = sum(item.float().square().mean() for item in amp_outputs)
        assert torch.isfinite(amp_loss)
        amp_loss.backward()
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        del amp_outputs, amp_loss
    result = {
        "required_gradient_prefixes": len(required),
        "autocast": amp,
        "loss": float(loss.detach().cpu()),
    }
    del model, sample, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    device = torch.device(args.device)
    batch, size = (2, 256) if args.full else (2, 64)
    shifts = test_shift_directions(device)
    d5_fairness = test_d5_fairness(device)
    d6_center = test_d6_center_degeneration(device)
    d7_uniform = test_d7_uniform_prior(device)

    models = {
        variant: build_spatial(variant, "test", device)
        for variant in SPATIAL_HFE_ABLATION_VARIANTS
    }
    metadata = assert_structure_and_metadata(models)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source = inspect.getsource(
        sys.modules[
            "model.DWTFreqNet_SingleDecoder_HFE_SpatialAblation"
        ]
    )
    assert "torch.cdist" not in source
    assert "torch.topk" not in source

    forwards = {
        variant: assert_forward(variant, device, batch, size)
        for variant in SPATIAL_HFE_ABLATION_VARIANTS
    }
    regressions = {
        variant: assert_zero_beta_regression(variant, device, size)
        for variant in SPATIAL_HFE_ABLATION_VARIANTS
    }
    gradients = {
        variant: assert_gradients_and_amp(variant, device, batch, size)
        for variant in SPATIAL_HFE_ABLATION_VARIANTS
    }
    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "full": args.full,
                "input_shape": [batch, 1, size, size],
                "offset_order": OFFSETS_3X3,
                "shift_checks": shifts,
                "d5_initial_degeneration": d5_fairness,
                "d6_center_degeneration": d6_center,
                "d7_uniform_prior_degeneration": d7_uniform,
                "metadata": metadata,
                "forwards": forwards,
                "zero_beta_regressions": regressions,
                "gradients_amp": gradients,
                "matching_modules_absent": True,
                "torch_cdist_forbidden": True,
                "torch_topk_forbidden": True,
                "dwt_idwt": {"dwt": 4, "idwt": 4},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
