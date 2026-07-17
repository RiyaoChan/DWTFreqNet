"""Structural, numerical, CUDA and real-data tests for Experiment F DSHF."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import TrainSetLoader
from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import Res_block, check_haar_direction_correspondence
from model.DWTFreqNet_SingleDecoder import DirectionalBandEncoder
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    lfss_initialization_max_difference,
    snapshot_lfss_special_parameters,
)
from model.DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM import (
    AdaptiveSparseSupportGate,
    CrossDirectionLocalConsistencyGate,
    DSHFBlock,
    DSHF_STAGE_CONFIG,
    DSHF_VARIANT_CONFIGS,
    DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM,
    DirectionalMultiScaleExtractor,
    EXPERIMENT_F_BASE_COMMIT,
    EXPERIMENT_F_VARIANTS,
    initialize_experiment_f_model,
)
from train_one import deep_supervision_loss, init_weights, set_seed


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_CHANNELS = (32, 64, 128, 256)
EXPECTED_ENCODED_CHANNELS = (64, 128, 256, 256)
FORBIDDEN_BASE_FILES = (
    "model/DWTFreqNet.py",
    "model/DWTFreqNet_SingleDecoder.py",
    "model/DWTFreqNet_SingleDecoder_LFSS_AWGM.py",
    "model/third_party/wavemamba_lfss.py",
    "dataset.py",
    "train_one.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--dataset-name", default="NUAA-SIRST")
    return parser.parse_args()


def build(variant, mode, device=None, initialize=True):
    model = DWTFreqNet_SingleDecoder_LFSS_DSHF_AWGM(
        get_DWTFreqNet_config(),
        hf_variant=variant,
        mode=mode,
        deepsuper=True,
    )
    if initialize:
        before = snapshot_lfss_special_parameters(model)
        initialize_experiment_f_model(model, init_weights)
        after = snapshot_lfss_special_parameters(model)
        assert lfss_initialization_max_difference(before, after) == 0.0
    return model if device is None else model.to(device)


def nonzero_gradient(model, prefix):
    return any(
        name.startswith(prefix)
        and parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
    )


def module_signature(module):
    return {
        "type": type(module).__name__,
        "parameters": {
            name: tuple(parameter.shape)
            for name, parameter in module.named_parameters()
        },
    }


def test_e1_baseline_regression():
    baseline = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(),
        encoder_variant="e1_lfss_resblock",
        mode="test",
        deepsuper=True,
    )
    assert baseline.ablation_id == "E1"
    assert baseline.post_awgm_encoder == "original_res_block"
    assert all(
        isinstance(getattr(baseline, f"dir_encoder{stage}"), DirectionalBandEncoder)
        for stage in range(1, 5)
    )
    assert all(
        isinstance(getattr(baseline, f"local_encoder{stage}"), Res_block)
        for stage in range(1, 5)
    )
    assert not any(isinstance(module, DSHFBlock) for module in baseline.modules())

    unchanged = None
    if (ROOT / ".git").exists() or subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    ).returncode == 0:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--quiet",
                EXPERIMENT_F_BASE_COMMIT,
                "--",
                *FORBIDDEN_BASE_FILES,
            ],
            capture_output=True,
            text=True,
        )
        unchanged = result.returncode == 0
        assert unchanged, "Experiment F modified an E1 baseline file"
    return {
        "ablation_id": baseline.ablation_id,
        "post_awgm_encoder": baseline.post_awgm_encoder,
        "base_files_unchanged": unchanged,
    }


def test_structure_and_initialization():
    results = {}
    for variant in EXPERIMENT_F_VARIANTS:
        model = build(variant, "train", initialize=False)
        before = snapshot_lfss_special_parameters(model)
        initialize_experiment_f_model(model, init_weights)
        after = snapshot_lfss_special_parameters(model)
        lfss_difference = lfss_initialization_max_difference(before, after)
        assert lfss_difference == 0.0
        config = DSHF_VARIANT_CONFIGS[variant]
        for stage in range(1, 5):
            block = getattr(model, f"dir_encoder{stage}")
            assert isinstance(block, DSHFBlock)
            assert block.channels == DSHF_STAGE_CONFIG[stage]["channels"]
            assert block.use_sparse_gate == config["use_sparse_gate"]
            assert block.use_cross_direction == config["use_cross_direction"]
            assert block.use_low_guidance == config["use_low_guidance"]
            assert isinstance(block.extract_h, DirectionalMultiScaleExtractor)
            assert isinstance(block.extract_v, DirectionalMultiScaleExtractor)
            assert isinstance(block.extract_d, DirectionalMultiScaleExtractor)
            if block.use_sparse_gate:
                for sparse in (block.sparse_h, block.sparse_v, block.sparse_d):
                    last = sparse.threshold_predictor[-1]
                    assert torch.count_nonzero(last.weight) == 0
                    assert torch.count_nonzero(last.bias) == 0
            if block.use_cross_direction:
                assert isinstance(
                    block.cross_direction, CrossDirectionLocalConsistencyGate
                )
                last = block.cross_direction.gate[-1]
                assert torch.count_nonzero(last.weight) == 0
                assert torch.count_nonzero(last.bias) == 0
        assert not any(
            isinstance(module, DirectionalBandEncoder) for module in model.modules()
        )
        for forbidden in (
            "ChannelMatching",
            "SoftCosineTopKMatching",
            "DecoderHFE",
            "DirectionalPyramid",
        ):
            assert not any(forbidden in type(module).__name__ for module in model.modules())
        results[variant] = {
            "lfss_initialization_max_abs_difference": lfss_difference,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    return results


def copy_shared_state(source, target):
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = []
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)
            copied.append(name)
    target.load_state_dict(target_state)
    return copied


def test_module_numerics(device):
    channels = 8
    bands = [torch.randn(2, channels, 16, 16, device=device) for _ in range(3)]

    f1 = DSHFBlock(channels, "f1_multiscale").to(device).eval()
    for extractor in (f1.extract_h, f1.extract_v, f1.extract_d):
        nn.init.zeros_(extractor.fuse[0].weight)
        extractor.fuse[1].reset_running_stats()
        nn.init.ones_(extractor.fuse[1].weight)
        nn.init.zeros_(extractor.fuse[1].bias)
    with torch.no_grad():
        f1_outputs = f1(*bands)[:3]
    for output, raw in zip(f1_outputs, bands):
        torch.testing.assert_close(output, raw, atol=0.0, rtol=0.0)

    sparse = AdaptiveSparseSupportGate(channels).to(device).eval()
    sparse_input = torch.randn(2, channels, 16, 16, device=device)
    with torch.no_grad():
        sparse_output, sparse_debug = sparse(sparse_input)
    assert sparse_debug["support"].shape == sparse_input.shape
    assert sparse_debug["threshold"].shape == (2, channels, 1, 1)
    assert bool((sparse_debug["support"] > 0).all())
    assert bool((sparse_debug["support"] < 1).all())
    torch.testing.assert_close(
        sparse_debug["threshold_ratio"],
        torch.full_like(sparse_debug["threshold_ratio"], 0.5),
    )
    nonzero = sparse_input != 0
    assert torch.equal(torch.sign(sparse_output[nonzero]), torch.sign(sparse_input[nonzero]))

    f2 = DSHFBlock(channels, "f2_sparse").to(device).eval()
    f3 = DSHFBlock(channels, "f3_cross_direction").to(device).eval()
    copied_f2_f3 = copy_shared_state(f2, f3)
    with torch.no_grad():
        output_f2 = f2(*bands)[:3]
        output_f3 = f3(*bands)[:3]
    for left, right in zip(output_f2, output_f3):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)

    f4 = DSHFBlock(channels, "f4_low_guided_full").to(device).eval()
    copied_f3_f4 = copy_shared_state(f3, f4)
    low = torch.randn_like(bands[0])
    with torch.no_grad():
        output_f3 = f3(*bands)[:3]
        output_f4 = f4(*bands, low_feature=low)[:3]
    for left, right in zip(output_f3, output_f4):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)

    for variant in ("f1_multiscale", "f2_sparse", "f3_cross_direction"):
        block = DSHFBlock(channels, variant).to(device).eval()
        with torch.no_grad():
            first = block(*bands, low_feature=torch.zeros_like(low))[:3]
            second = block(*bands, low_feature=torch.ones_like(low))[:3]
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)

    return {
        "f1_zero_residual_exact": True,
        "sparse_threshold_ratio": 0.5,
        "f2_f3_shared_parameters": len(copied_f2_f3),
        "f3_f4_shared_parameters": len(copied_f3_f4),
        "f3_initially_equals_f2": True,
        "f4_initially_equals_f3": True,
    }


def test_order_shapes_and_inputs(variant, device, batch, size):
    model = build(variant, "test", device=device).eval()
    model.debug_tensors = True
    events = []
    lfss_outputs = {}
    dshf_outputs = {}
    dshf_low_inputs = {}
    awgm_inputs = {}
    handles = []

    def lfss_hook(stage):
        def hook(_module, _inputs, output):
            events.append((stage, "lfss"))
            lfss_outputs[stage] = output.detach()
        return hook

    def dshf_pre_hook(stage):
        def hook(_module, inputs, kwargs):
            low_feature = kwargs.get("low_feature")
            dshf_low_inputs[stage] = (
                None if low_feature is None else low_feature.detach()
            )
            return None
        return hook

    def dshf_hook(stage):
        def hook(_module, _inputs, output):
            events.append((stage, "dshf"))
            dshf_outputs[stage] = tuple(item.detach() for item in output[:3])
        return hook

    def awgm_pre_hook(stage):
        def hook(_module, inputs):
            events.append((stage, "awgm"))
            awgm_inputs[stage] = tuple(item.detach() for item in inputs)
        return hook

    def post_hook(stage):
        def hook(_module, _inputs, _output):
            events.append((stage, "resblock"))
        return hook

    for stage in range(1, 5):
        handles.append(model.lfss_blocks[str(stage)].register_forward_hook(lfss_hook(stage)))
        handles.append(getattr(model, f"dir_encoder{stage}").register_forward_pre_hook(
            dshf_pre_hook(stage), with_kwargs=True
        ))
        handles.append(getattr(model, f"dir_encoder{stage}").register_forward_hook(dshf_hook(stage)))
        handles.append(getattr(model, f"stage_awgm{stage}").register_forward_pre_hook(awgm_pre_hook(stage)))
        handles.append(getattr(model, f"local_encoder{stage}").register_forward_hook(post_hook(stage)))

    sample = torch.randn(batch, 1, size, size, device=device)
    with torch.no_grad():
        output = model(sample)
    for handle in handles:
        handle.remove()

    assert events == [
        item
        for stage in range(1, 5)
        for item in (
            (stage, "lfss"),
            (stage, "dshf"),
            (stage, "awgm"),
            (stage, "resblock"),
        )
    ]
    assert tuple(output.shape) == (batch, 1, size, size)
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}

    for stage, (raw_channels, encoded_channels) in enumerate(
        zip(EXPECTED_STAGE_CHANNELS, EXPECTED_ENCODED_CHANNELS), start=1
    ):
        spatial = size // (2 ** stage)
        expected_raw = (batch, raw_channels, spatial, spatial)
        expected_encoded = (batch, encoded_channels, spatial, spatial)
        debug = model.last_debug["experiment_f"][stage]
        assert tuple(debug["raw_ll"].shape) == expected_raw
        assert tuple(debug["lfss_ll"].shape) == expected_raw
        assert tuple(debug["dshf"]["output_h"].shape) == expected_raw
        assert tuple(debug["encoded"].shape) == expected_encoded
        torch.testing.assert_close(awgm_inputs[stage][0], lfss_outputs[stage])
        for index in range(3):
            torch.testing.assert_close(awgm_inputs[stage][index + 1], dshf_outputs[stage][index])

        if variant == "f4_low_guided_full":
            torch.testing.assert_close(dshf_low_inputs[stage], lfss_outputs[stage])
            assert (
                dshf_low_inputs[stage] - debug["raw_ll"]
            ).abs().max().item() > 1e-7
        else:
            assert dshf_low_inputs[stage] is None

        for direction, key in (("H", "raw_h"), ("V", "raw_v"), ("D", "raw_d")):
            expected_aligned = getattr(model, f"align_{direction}{stage}")(debug[key])
            actual_aligned = model.last_debug["coefficients"][(stage, direction)]["aligned"]
            torch.testing.assert_close(expected_aligned, actual_aligned)

    payload = {
        "events": events,
        "output_shape": list(output.shape),
        "dwt_idwt": dict(model.last_transform_counts),
        "raw_decoder_coefficients_verified": True,
        "f4_low_source_verified": variant == "f4_low_guided_full",
    }
    del model, sample, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def check_required_gradients(model, variant):
    prefixes = [
        *(f"lfss_blocks.{stage}" for stage in range(1, 5)),
        *(f"stage_awgm{stage}" for stage in range(1, 5)),
        *(f"dir_encoder{stage}.extract_" for stage in range(1, 5)),
        *(f"local_encoder{stage}" for stage in range(1, 5)),
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
    if variant != "f1_multiscale":
        prefixes.extend(f"dir_encoder{stage}.sparse_" for stage in range(1, 5))
    if variant in ("f3_cross_direction", "f4_low_guided_full"):
        prefixes.extend(
            f"dir_encoder{stage}.cross_direction.gate.2" for stage in range(1, 5)
        )
    if variant == "f4_low_guided_full":
        prefixes.extend(
            f"dir_encoder{stage}.low_contrast.response" for stage in range(1, 5)
        )
    missing = [prefix for prefix in prefixes if not nonzero_gradient(model, prefix)]
    assert not missing, f"{variant} missing gradient prefixes: {missing}"
    return prefixes


def train_step(model, sample, target=None, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    outputs = model(sample)
    if target is None:
        loss = sum(output.float().mean() for output in outputs)
    else:
        loss = deep_supervision_loss(outputs, target, nn.BCELoss())
    assert torch.isfinite(loss)
    loss.backward()
    if optimizer is not None:
        optimizer.step()
    return outputs, loss


def test_forbidden_ops(variant, device, size):
    model = build(variant, "test", device=device).eval()
    sample = torch.randn(1, 1, size, size, device=device)
    original_cdist, original_topk = torch.cdist, torch.topk

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Forbidden global matching operation was called")

    torch.cdist = forbidden
    torch.topk = forbidden
    try:
        with torch.no_grad():
            output = model(sample)
        assert torch.isfinite(output).all()
    finally:
        torch.cdist, torch.topk = original_cdist, original_topk
    del model, sample, output
    if device.type == "cuda":
        torch.cuda.empty_cache()


def test_fp32_and_amp(variant, device, batch, size):
    model = build(variant, "train", device=device).train()
    sample = torch.randn(batch, 1, size, size, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    outputs, loss = train_step(model, sample, optimizer=optimizer)
    assert len(outputs) == 6
    assert all(tuple(output.shape) == (batch, 1, size, size) for output in outputs)
    if variant in ("f3_cross_direction", "f4_low_guided_full"):
        outputs, loss = train_step(model, sample, optimizer=optimizer)
    prefixes = check_required_gradients(model, variant)
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    del model, sample, outputs, loss, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    amp_passed = None
    if device.type == "cuda":
        model = build(variant, "train", device=device).train()
        sample = torch.randn(batch, 1, size, size, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(sample)
            loss = sum(output.float().mean() for output in outputs)
        assert all(torch.isfinite(output).all() for output in outputs)
        assert torch.isfinite(loss)
        loss.backward()
        amp_passed = True
        del model, sample, outputs, loss
        torch.cuda.empty_cache()
    return {"gradient_prefixes": prefixes, "amp": amp_passed}


def real_data_smoke(variant, dataset_dir, dataset_name, device):
    dataset = TrainSetLoader(
        dataset_dir, dataset_name, patch_size=256, img_norm_cfg=None
    )
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True
    )
    image, target = next(iter(loader))
    assert image.shape[0] == 4
    image = image.to(device)
    target = target.to(device)
    model = build(variant, "train", device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    outputs, loss = train_step(model, image, target=target, optimizer=optimizer)
    value = float(loss.detach().cpu())
    del model, image, target, outputs, loss, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def main():
    args = parse_args()
    set_seed(42)
    payload = {
        "status": "construction_passed",
        "base_commit": EXPERIMENT_F_BASE_COMMIT,
        "e1_regression": test_e1_baseline_regression(),
        "structure": test_structure_and_initialization(),
    }
    if args.construct_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    batch, size = (2, 256) if args.full else (1, 64)
    payload["device"] = str(device)
    payload["input_shape"] = [batch, 1, size, size]
    payload["module_numerics"] = test_module_numerics(device)
    payload["variants"] = {}
    for variant in EXPERIMENT_F_VARIANTS:
        payload["variants"][variant] = {
            "order_shapes_inputs": test_order_shapes_and_inputs(
                variant, device, batch, size
            ),
            "fp32_amp": test_fp32_and_amp(variant, device, batch, size),
        }
        test_forbidden_ops(variant, device, size)
        payload["variants"][variant]["forbidden_ops"] = "passed"
        if args.dataset_dir:
            payload["variants"][variant]["real_data_batch4_loss"] = real_data_smoke(
                variant, args.dataset_dir, args.dataset_name, device
            )
    direction = check_haar_direction_correspondence(32, str(device))
    assert direction["routing_aligned"] is True
    payload["haar"] = direction
    payload["status"] = "passed"
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
