"""Validation suite for Experiment D relation ablations D2, D3 and D4."""

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
from model.DWTFreqNet_SingleDecoder_HFE import MatchingTransformation
from model.DWTFreqNet_SingleDecoder_HFE_Ablation import (
    D2_STAGE_CONFIG,
    D3_STAGE_CONFIG,
    D4_STAGE_CONFIG,
    DirectFusionTransformation,
    DWTFreqNet_SingleDecoder_HFE_Ablation,
    LocalCorrelationGate,
    SoftCosineTopKMatching,
    SoftMatchingTransformation,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true", help="Use required 2x256 inputs")
    return parser.parse_args()


def build_model(ablation, mode, device):
    return DWTFreqNet_SingleDecoder_HFE_Ablation(
        get_DWTFreqNet_config(),
        hfe_ablation=ablation,
        mode=mode,
        deepsuper=True,
    ).to(device)


def build_baseline(mode, device):
    return DWTFreqNet_SingleDecoder(
        get_DWTFreqNet_config(),
        mode=mode,
        deepsuper=True,
        sd_variant="sd_awgm",
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


def module_signature(module):
    return {
        "type": type(module).__name__,
        "parameters": [
            (name, tuple(parameter.shape))
            for name, parameter in module.named_parameters()
        ],
    }


def count_trainable(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_soft_matching(device):
    channels, topk, spatial = 16, 8, 5
    module = SoftCosineTopKMatching(channels, topk, 0.1).to(device)
    query = torch.randn(2, channels, spatial, spatial, device=device, requires_grad=True)
    candidate = torch.randn(
        2, channels, spatial, spatial, device=device, requires_grad=True
    )
    matched, info = module(query, candidate)
    assert tuple(matched.shape) == tuple(query.shape)
    assert info["similarity_shape"] == (2, channels, channels)
    assert info["topk_indices_shape"] == (2, channels, topk)
    assert info["topk_weights_shape"] == (2, channels, topk)
    assert info["topk_weight_sum_error"] < 1e-6
    assert 0.03 <= info["temperature"] <= 1.0
    assert 0.0 <= info["normalized_matching_entropy"] <= 1.0 + 1e-6
    assert 1.0 <= info["effective_candidate_count"] <= topk + 1e-5
    matched.mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert candidate.grad is not None and torch.isfinite(candidate.grad).all()
    assert module.log_temperature.grad is not None

    candidate_flat = torch.randn(2, channels, 11, device=device)
    indices = torch.randint(0, channels, (2, channels, topk), device=device)
    weights = torch.softmax(
        torch.randn(2, channels, topk, device=device), dim=-1
    )
    dense = module.dense_aggregate(candidate_flat, indices, weights)
    gathered = module.gather_aggregate(candidate_flat, indices, weights)
    torch.testing.assert_close(dense, gathered, rtol=1e-6, atol=1e-6)
    return info


def test_local_gate(device):
    module = LocalCorrelationGate(16).to(device)
    high = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
    low = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
    output, info = module(high, low)
    assert tuple(output.shape) == tuple(high.shape)
    assert info["gate_shape"] == (2, 16, 8, 8)
    assert 0.0 <= info["gate_min"] <= info["gate_max"] <= 1.0
    assert math.isclose(
        info["low_selected_ratio"] + info["high_selected_ratio"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    output.mean().backward()
    assert high.grad is not None and torch.isfinite(high.grad).all()
    assert low.grad is not None and torch.isfinite(low.grad).all()
    return info


def test_direct_fusion(device):
    module = DirectFusionTransformation(16).to(device)
    high = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
    low = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
    output, info = module(high, low)
    assert tuple(output.shape) == tuple(high.shape)
    assert info["relation_mode"] == "direct_low_fusion"
    assert info["high_shape"] == info["low_shape"] == info["output_shape"]
    assert 0.0 <= info["gate_min"] <= info["gate_max"] <= 1.0
    assert all(
        math.isfinite(info[key])
        for key in (
            "gate_mean",
            "gate_std",
            "high_norm",
            "low_norm",
            "output_norm",
            "low_high_norm_ratio",
        )
    )
    output.mean().backward()
    assert high.grad is not None and torch.isfinite(high.grad).all()
    assert low.grad is not None and torch.isfinite(low.grad).all()
    for name in ("gate", "value", "project"):
        assert has_nonzero_gradient(module, name)
    return info


def assert_variant_structure(d2, d3, d4):
    fairness = {}
    for stage in range(1, 5):
        d2_refiner = getattr(d2, f"decoder_hfe{stage}")
        assert isinstance(d2_refiner.hfe.attn.relation, SoftMatchingTransformation)
        assert isinstance(d2_refiner.hfe.ffn.relation, SoftMatchingTransformation)

        d3_refiner = getattr(d3, f"decoder_hfe{stage}")
        expected = LocalCorrelationGate if stage <= 2 else SoftMatchingTransformation
        assert isinstance(d3_refiner.hfe.attn.relation, expected)
        assert isinstance(d3_refiner.hfe.ffn.relation, expected)

        d4_refiner = getattr(d4, f"decoder_hfe{stage}")
        assert isinstance(d4_refiner.hfe.attn.relation, DirectFusionTransformation)
        assert isinstance(d4_refiner.hfe.ffn.relation, DirectFusionTransformation)
        assert D4_STAGE_CONFIG[stage]["mode"] == "direct_low_fusion"
        for branch in ("attn", "ffn"):
            d2_relation = getattr(d2_refiner.hfe, branch).relation
            d4_relation = getattr(d4_refiner.hfe, branch).relation
            d1_parameters = count_trainable(
                MatchingTransformation(D4_STAGE_CONFIG[stage]["channels"])
            )
            d2_parameters = count_trainable(d2_relation)
            d4_parameters = count_trainable(d4_relation)
            assert d1_parameters == d4_parameters
            assert d2_parameters == d4_parameters + 1
            fairness[f"stage{stage}_{branch}"] = {
                "d1_parameters": d1_parameters,
                "d2_parameters": d2_parameters,
                "d4_parameters": d4_parameters,
            }

    assert D2_STAGE_CONFIG[3] == D3_STAGE_CONFIG[3]
    assert D2_STAGE_CONFIG[4] == D3_STAGE_CONFIG[4]
    deep_signatures = {}
    for stage in (3, 4):
        d2_refiner = getattr(d2, f"decoder_hfe{stage}")
        d3_refiner = getattr(d3, f"decoder_hfe{stage}")
        for branch in ("attn", "ffn"):
            d2_relation = getattr(d2_refiner.hfe, branch).relation
            d3_relation = getattr(d3_refiner.hfe, branch).relation
            d2_signature = module_signature(d2_relation)
            d3_signature = module_signature(d3_relation)
            assert d2_signature == d3_signature
            assert d2_relation.matching.topk == d3_relation.matching.topk == 8
            assert torch.allclose(
                d2_relation.matching.log_temperature.detach().cpu(),
                d3_relation.matching.log_temperature.detach().cpu(),
            )
            deep_signatures[f"stage{stage}_{branch}"] = d2_signature
    forbidden_types = (
        SoftCosineTopKMatching,
        SoftMatchingTransformation,
        LocalCorrelationGate,
        MatchingTransformation,
    )
    for module in d4.modules():
        assert not isinstance(module, forbidden_types)
        class_name = type(module).__name__
        assert "ChannelMatching" not in class_name
        assert "MatchingTransformation" not in class_name
    return deep_signatures, fairness


def assert_model_forward(ablation, device, batch, size):
    model = build_model(ablation, "train", device).train()
    model.debug_tensors = True
    original_cdist = torch.cdist
    original_topk = torch.topk

    def forbidden_cdist(*_args, **_kwargs):
        raise AssertionError("torch.cdist is forbidden in D2/D3/D4")

    def forbidden_topk(*_args, **_kwargs):
        raise AssertionError("D4 must not call torch.topk")

    torch.cdist = forbidden_cdist
    if ablation == "d4_no_matching":
        torch.topk = forbidden_topk
    try:
        with torch.no_grad():
            outputs = model(torch.randn(batch, 1, size, size, device=device))
    finally:
        torch.cdist = original_cdist
        torch.topk = original_topk

    assert len(outputs) == 6
    assert all(tuple(item.shape) == (batch, 1, size, size) for item in outputs)
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
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
    assert {key: model.last_shapes[key] for key in expected_shapes} == expected_shapes

    relation_shapes = {}
    for stage, channels in ((1, 64), (2, 128), (3, 256), (4, 256)):
        refiner = getattr(model, f"decoder_hfe{stage}")
        for branch in ("attn", "ffn"):
            relation = getattr(refiner.hfe, branch).relation
            info = relation.last_info
            if isinstance(relation, SoftMatchingTransformation):
                assert info["similarity_shape"] == (batch, channels, channels)
                assert info["topk_indices_shape"] == (batch, channels, 8)
                assert info["topk_weights_shape"] == (batch, channels, 8)
            elif isinstance(relation, LocalCorrelationGate):
                assert info["gate_shape"] == (
                    batch,
                    channels,
                    size // (2 ** stage),
                    size // (2 ** stage),
                )
            else:
                expected = (
                    batch,
                    channels,
                    size // (2 ** stage),
                    size // (2 ** stage),
                )
                assert isinstance(relation, DirectFusionTransformation)
                assert info["relation_mode"] == "direct_low_fusion"
                assert info["high_shape"] == expected
                assert info["low_shape"] == expected
                assert info["output_shape"] == expected
            relation_shapes[f"stage{stage}_{branch}"] = info

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    hfe_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("decoder_hfe")
    )
    del model, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_model = build_model(ablation, "test", device).eval()
    test_model.record_statistics = False
    with torch.no_grad():
        output = test_model(torch.randn(batch, 1, size, size, device=device))
    assert tuple(output.shape) == (batch, 1, size, size)
    del test_model, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "parameters": parameter_count,
        "hfe_parameters": hfe_parameters,
        "relation_shapes": relation_shapes,
    }


def assert_zero_beta_regression(ablation, device, size):
    baseline = build_baseline("test", device).eval()
    model = build_model(ablation, "test", device).eval()
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
    torch.testing.assert_close(expected, observed, rtol=1e-5, atol=1e-6)
    max_error = float((expected - observed).abs().max().cpu())
    del baseline, model, sample, expected, observed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"common_state_entries": copied, "max_abs_error": max_error}


def assert_gradients_and_amp(ablation, device, batch, size):
    model = build_model(ablation, "train", device).train()
    for stage in range(1, 5):
        refiner = getattr(model, f"decoder_hfe{stage}")
        for beta in (refiner.beta_h, refiner.beta_v, refiner.beta_d):
            torch.testing.assert_close(
                beta.detach(), torch.full_like(beta.detach(), 1e-3)
            )
    sample = torch.randn(batch, 1, size, size, device=device)
    autocast_enabled = device.type == "cuda"
    outputs = model(sample)
    loss = sum(output.float().square().mean() for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()

    required = [
        "stem",
        "local_encoder1",
        "stage_awgm1",
        "decoder_fuse0",
        "out_head",
    ]
    for stage in range(1, 5):
        required.extend(
            [
                f"decoder_hfe{stage}.subband_fusion",
                f"decoder_hfe{stage}.hfe.attn",
                f"decoder_hfe{stage}.hfe.ffn",
                f"decoder_hfe{stage}.hfe.attn.relation",
                f"decoder_hfe{stage}.hfe.ffn.relation",
                f"decoder_hfe{stage}.head_h",
                f"decoder_hfe{stage}.head_v",
                f"decoder_hfe{stage}.head_d",
                f"decoder_hfe{stage}.beta_h",
                f"decoder_hfe{stage}.beta_v",
                f"decoder_hfe{stage}.beta_d",
            ]
        )
        if ablation == "d2_softcos_all" or (
            ablation == "d3_scaleaware" and stage >= 3
        ):
            required.extend(
                [
                    f"decoder_hfe{stage}.hfe.attn.relation.matching.log_temperature",
                    f"decoder_hfe{stage}.hfe.ffn.relation.matching.log_temperature",
                ]
            )
    missing = [prefix for prefix in required if not has_nonzero_gradient(model, prefix)]
    assert not missing, f"Missing nonzero gradients for {ablation}: {missing}"
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not nonfinite, f"Non-finite gradients for {ablation}: {nonfinite}"
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}

    loss_scale = 4096.0 if autocast_enabled else 1.0
    if autocast_enabled:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_outputs = model(sample)
            amp_loss = sum(
                output.float().square().mean() for output in amp_outputs
            )
        assert torch.isfinite(amp_loss)
        (amp_loss * loss_scale).backward()
        amp_nonfinite = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        assert not amp_nonfinite, (
            f"Non-finite AMP gradients for {ablation}: {amp_nonfinite}"
        )
        del amp_outputs, amp_loss

    del model, sample, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "required_prefixes": required,
        "autocast": autocast_enabled,
        "loss_scale": loss_scale,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    batch, size = (2, 256) if args.full else (1, 64)
    soft_info = test_soft_matching(device)
    gate_info = test_local_gate(device)
    direct_info = test_direct_fusion(device)

    torch.manual_seed(42)
    d2_structure = build_model("d2_softcos_all", "test", device)
    torch.manual_seed(42)
    d3_structure = build_model("d3_scaleaware", "test", device)
    torch.manual_seed(42)
    d4_structure = build_model("d4_no_matching", "test", device)
    deep_signatures, fairness = assert_variant_structure(
        d2_structure, d3_structure, d4_structure
    )
    assert d2_structure.experiment_metadata["ablation_id"] == "D2"
    assert d3_structure.experiment_metadata["ablation_id"] == "D3"
    d4_metadata = d4_structure.experiment_metadata
    assert d4_metadata["ablation_id"] == "D4"
    assert d4_metadata["explicit_channel_matching"] is False
    assert d4_metadata["channel_similarity_matrix"] is False
    assert d4_metadata["channel_candidate_selection"] is False
    assert d4_metadata["direct_fusion_uses_raw_low"] is True
    assert d4_metadata["hfe_topk"] is None
    assert d4_metadata["hfe_initial_temperature"] is None
    assert all(
        d4_metadata[f"stage{stage}_relation"] == "direct_low_fusion"
        for stage in range(1, 5)
    )
    del d2_structure, d3_structure, d4_structure
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source = inspect.getsource(
        sys.modules["model.DWTFreqNet_SingleDecoder_HFE_Ablation"]
    )
    assert "torch.cdist" not in source

    forwards = {
        ablation: assert_model_forward(ablation, device, batch, size)
        for ablation in (
            "d2_softcos_all",
            "d3_scaleaware",
            "d4_no_matching",
        )
    }
    regressions = {
        ablation: assert_zero_beta_regression(ablation, device, size)
        for ablation in (
            "d2_softcos_all",
            "d3_scaleaware",
            "d4_no_matching",
        )
    }
    gradients = {
        ablation: assert_gradients_and_amp(ablation, device, batch, size)
        for ablation in (
            "d2_softcos_all",
            "d3_scaleaware",
            "d4_no_matching",
        )
    }

    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "full": args.full,
                "input_shape": [batch, 1, size, size],
                "soft_matching": soft_info,
                "local_gate": gate_info,
                "direct_fusion": direct_info,
                "deep_stage_signatures": deep_signatures,
                "relation_parameter_fairness": fairness,
                "forwards": forwards,
                "zero_beta_regressions": regressions,
                "gradients_amp": gradients,
                "torch_cdist_forbidden": True,
                "d4_torch_topk_forbidden": True,
                "d4_matching_modules_absent": True,
                "dwt_idwt": {"dwt": 4, "idwt": 4},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
