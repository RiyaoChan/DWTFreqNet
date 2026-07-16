"""Structural, ordering, initialization, CUDA and real-data tests for Experiment E."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import TrainSetLoader
from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import Res_block, check_haar_direction_correspondence
from model.DWTFreqNet_SingleDecoder import DWTFreqNet_SingleDecoder
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    EXPERIMENT_E_VARIANTS,
    LFSS_STAGE_CONFIG,
    LowFrequencyTransition,
    initialize_experiment_e_model,
    lfss_initialization_max_difference,
    snapshot_lfss_special_parameters,
)
from model.third_party import wavemamba_lfss
from train_one import deep_supervision_loss, init_weights, set_seed


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_CHANNELS = (32, 64, 128, 256)
EXPECTED_ENCODED_CHANNELS = (64, 128, 256, 256)


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
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(),
        encoder_variant=variant,
        mode=mode,
        deepsuper=True,
    )
    if initialize:
        before = snapshot_lfss_special_parameters(model)
        initialize_experiment_e_model(model, init_weights)
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


def test_source_and_initialization():
    for name in ("LFSSBlock", "SS2D", "ffn", "SimpleGate"):
        cls = getattr(wavemamba_lfss, name)
        assert cls.__module__ == "model.third_party.wavemamba_lfss"
    assert wavemamba_lfss.WAVE_MAMBA_SOURCE_COMMIT == (
        "7e8c63f37af7640e228345c410c2e2165e216117"
    )
    assert (ROOT / "model/third_party/WAVE_MAMBA_NOTICE.md").is_file()
    assert (ROOT / "model/third_party/WAVE_MAMBA_LICENSE").is_file()

    results = {}
    for variant in EXPERIMENT_E_VARIANTS:
        model = build(variant, "train", initialize=False)
        before = snapshot_lfss_special_parameters(model)
        assert len(before) == 24
        initialize_experiment_e_model(model, init_weights)
        after = snapshot_lfss_special_parameters(model)
        difference = lfss_initialization_max_difference(before, after)
        assert difference == 0.0
        for stage in range(1, 5):
            adapter = model.lfss_blocks[str(stage)]
            block = adapter.block
            assert adapter.channels == LFSS_STAGE_CONFIG[stage]["channels"]
            assert torch.equal(block.skip_scale, torch.ones_like(block.skip_scale))
            assert torch.equal(block.skip_scale2, torch.ones_like(block.skip_scale2))
            assert block.self_attention.A_logs.dtype == torch.float32
            assert block.self_attention.Ds.dtype == torch.float32
            assert torch.isfinite(block.self_attention.dt_projs_bias).all()
            assert sum(p.numel() for p in adapter.parameters()) == sum(
                p.numel() for p in block.parameters()
            )
        for forbidden in (
            "lfss_gamma",
            "outer_residual",
            "residual_blend",
            "low_scale",
        ):
            assert not any(hasattr(module, forbidden) for module in model.modules())

        if variant == "e1_lfss_resblock":
            assert all(
                isinstance(getattr(model, f"local_encoder{stage}"), Res_block)
                for stage in range(1, 5)
            )
        else:
            assert all(
                isinstance(
                    getattr(model, f"local_encoder{stage}"),
                    LowFrequencyTransition,
                )
                for stage in range(1, 5)
            )
            low_path_names = [
                name
                for name, _ in model.named_modules()
                if name.startswith("local_encoder")
            ]
            assert not any("Res_block" in name for name in low_path_names)
        results[variant] = {
            "initialization_max_abs_difference": difference,
            "special_parameter_count": len(before),
        }
    return results


def module_signature(module):
    return {
        "type": type(module).__name__,
        "parameters": {
            name: tuple(parameter.shape)
            for name, parameter in module.named_parameters()
        },
    }


def test_decoder_identity():
    base = DWTFreqNet_SingleDecoder(
        get_DWTFreqNet_config(), mode="test", deepsuper=True, sd_variant="sd_awgm"
    )
    names = [
        *(f"align_{direction}{stage}" for stage in range(1, 5) for direction in "HVD"),
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
    base_signatures = {name: module_signature(getattr(base, name)) for name in names}
    for variant in EXPERIMENT_E_VARIANTS:
        model = build(variant, "test", initialize=False)
        assert {
            name: module_signature(getattr(model, name)) for name in names
        } == base_signatures
    return names


def test_order_shapes_and_inputs(variant, device, batch, size):
    model = build(variant, "test", device=device).eval()
    model.debug_tensors = True
    events = []
    lfss_outputs = {}
    awgm_inputs = {}
    handles = []

    def make_lfss_hook(stage):
        def hook(_module, _inputs, output):
            events.append((stage, "lfss"))
            lfss_outputs[stage] = output.detach()
        return hook

    def make_awgm_pre_hook(stage):
        def hook(_module, inputs):
            events.append((stage, "awgm"))
            awgm_inputs[stage] = inputs[0].detach()
        return hook

    def make_post_hook(stage):
        def hook(_module, _inputs, _output):
            events.append((stage, "post"))
        return hook

    for stage in range(1, 5):
        handles.append(model.lfss_blocks[str(stage)].register_forward_hook(
            make_lfss_hook(stage)
        ))
        handles.append(getattr(model, f"stage_awgm{stage}").register_forward_pre_hook(
            make_awgm_pre_hook(stage)
        ))
        handles.append(getattr(model, f"local_encoder{stage}").register_forward_hook(
            make_post_hook(stage)
        ))

    sample = torch.randn(batch, 1, size, size, device=device)
    with torch.no_grad():
        output = model(sample)
    for handle in handles:
        handle.remove()

    assert events == [
        item for stage in range(1, 5)
        for item in ((stage, "lfss"), (stage, "awgm"), (stage, "post"))
    ]
    assert tuple(output.shape) == (batch, 1, size, size)
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    for stage, (raw_channels, encoded_channels) in enumerate(
        zip(EXPECTED_STAGE_CHANNELS, EXPECTED_ENCODED_CHANNELS), start=1
    ):
        spatial = size // (2 ** stage)
        expected_raw = (batch, raw_channels, spatial, spatial)
        expected_encoded = (batch, encoded_channels, spatial, spatial)
        assert tuple(model.last_debug["A"][stage].shape) == expected_raw
        assert tuple(model.last_debug["A_lfss"][stage].shape) == expected_raw
        assert tuple(model.last_debug["A_guided"][stage].shape) == expected_raw
        assert tuple(model.last_debug["E"][stage].shape) == expected_encoded
        torch.testing.assert_close(awgm_inputs[stage], lfss_outputs[stage])
        difference = (
            awgm_inputs[stage] - model.last_debug["A"][stage]
        ).abs().max().item()
        assert difference > 1e-7

    payload = {
        "events": events,
        "output_shape": list(output.shape),
        "stage_shapes": {
            str(stage): {
                "raw_LL": list(model.last_debug["A"][stage].shape),
                "LFSS_LL": list(model.last_debug["A_lfss"][stage].shape),
                "guided_LL": list(model.last_debug["A_guided"][stage].shape),
                "encoded": list(model.last_debug["E"][stage].shape),
            }
            for stage in range(1, 5)
        },
    }
    del model, sample, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def check_required_gradients(model, variant):
    prefixes = [
        *(f"lfss_blocks.{stage}" for stage in range(1, 5)),
        *(f"stage_awgm{stage}" for stage in range(1, 5)),
        *(f"dir_encoder{stage}" for stage in range(1, 5)),
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
    missing = [prefix for prefix in prefixes if not nonzero_gradient(model, prefix)]
    assert not missing, f"{variant} missing gradient prefixes: {missing}"

    suffixes = (
        "A_logs",
        "Ds",
        "dt_projs_weight",
        "dt_projs_bias",
        "in_proj.weight",
        "out_proj.weight",
        "conv_blk.conv1.weight",
        "conv_blk.conv2.weight",
        "conv_blk.conv3.weight",
        "skip_scale",
        "skip_scale2",
    )
    missing_parameters = []
    for stage in range(1, 5):
        stage_parameters = dict(model.lfss_blocks[str(stage)].named_parameters())
        for suffix in suffixes:
            matches = [
                parameter
                for name, parameter in stage_parameters.items()
                if name.endswith(suffix)
            ]
            if not matches or not all(
                parameter.grad is not None
                and bool(torch.count_nonzero(parameter.grad).item())
                for parameter in matches
            ):
                missing_parameters.append(f"stage{stage}:{suffix}")
    assert not missing_parameters, (
        f"{variant} missing LFSS parameter gradients: {missing_parameters}"
    )
    return prefixes


def test_fp32_and_amp(variant, device, batch, size):
    model = build(variant, "train", device=device).train()
    sample = torch.randn(batch, 1, size, size, device=device)
    outputs = model(sample)
    assert len(outputs) == 6
    assert all(tuple(output.shape) == (batch, 1, size, size) for output in outputs)
    loss = sum(output.float().mean() for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    prefixes = check_required_gradients(model, variant)
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    del model, sample, outputs, loss
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
    optimizer.zero_grad(set_to_none=True)
    loss = deep_supervision_loss(model(image), target, nn.BCELoss())
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    value = float(loss.detach().cpu())
    del model, image, target, loss, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def main():
    args = parse_args()
    set_seed(42)
    initialization = test_source_and_initialization()
    decoder_modules = test_decoder_identity()
    payload = {
        "status": "construction_passed",
        "source_commit": wavemamba_lfss.WAVE_MAMBA_SOURCE_COMMIT,
        "initialization": initialization,
        "decoder_modules": decoder_modules,
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
    payload["variants"] = {}
    for variant in EXPERIMENT_E_VARIANTS:
        payload["variants"][variant] = {
            "order_and_shapes": test_order_shapes_and_inputs(
                variant, device, batch, size
            ),
            "fp32_amp": test_fp32_and_amp(variant, device, batch, size),
        }
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
