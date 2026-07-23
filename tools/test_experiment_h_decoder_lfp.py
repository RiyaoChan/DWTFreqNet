"""Regression, structural, numerical and real-data tests for Experiment H."""

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
from model.DWTFreqNet import check_haar_direction_correspondence
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    initialize_experiment_e_model,
)
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP import (
    DECODER_LFP_STAGE_CHANNELS,
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP,
    EXPERIMENT_H_VARIANTS,
    initialize_experiment_h_model,
)
from model.decoder_lfp import (
    AdaptiveLFPThreshold,
    DecoderLFPProcessor,
    LearnableDepthwiseGaussian,
    LFPSpatialAttention,
    NS_FPN_SOURCE_COMMIT,
)
from train_one import deep_supervision_loss, init_weights, set_seed


FORMAL_VARIANTS = EXPERIMENT_H_VARIANTS[1:]
PAIR_VARIANTS = (
    ("h1_rawll_attention", "h1_decoder_attention"),
    ("h2_rawll_fixed_gaussian", "h2_decoder_fixed_gaussian"),
    ("h3_rawll_adaptive_gaussian", "h3_decoder_adaptive_gaussian"),
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
    parser.add_argument("--output", default="")
    parser.add_argument("--checkpoint-dir", default="")
    return parser.parse_args()


def build(variant, mode, device=None, initialize=True):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP(
        get_DWTFreqNet_config(), lfp_variant=variant, mode=mode, deepsuper=True
    )
    if initialize:
        initialize_experiment_h_model(model, init_weights)
    return model if device is None else model.to(device)


def tensor_max_difference(left, right):
    return float((left.float() - right.float()).abs().max().cpu())


def module_signature(module):
    return {
        "type": type(module).__name__,
        "parameters": {
            name: tuple(parameter.shape)
            for name, parameter in module.named_parameters()
        },
    }


def test_core_modules():
    attention = LFPSpatialAttention()
    low = torch.randn(2, 11, 17, 19)
    attention_map, debug = attention(low)
    assert attention_map.shape == (2, 1, 17, 19)
    assert bool(((attention_map > 0.0) & (attention_map < 1.0)).all())
    expected = attention.sigmoid(
        attention.conv(torch.cat([low.mean(1, keepdim=True), low.amax(1, keepdim=True)], 1))
    )
    torch.testing.assert_close(attention_map, expected)
    assert torch.equal(debug["attention"], attention_map)

    gaussian = LearnableDepthwiseGaussian(6)
    assert abs(float(gaussian.sigma.detach()) - 1.0) < 1e-6
    kernel = gaussian.kernel()
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))
    torch.testing.assert_close(kernel, kernel.flip(-1))
    torch.testing.assert_close(kernel, kernel.flip(-2))
    impulse = torch.zeros(1, 6, 9, 9)
    impulse[:, :, 4, 4] = torch.arange(1, 7).view(1, 6)
    response = gaussian(impulse)
    for channel in range(6):
        torch.testing.assert_close(
            response[0, channel, 3:6, 3:6], kernel[0, 0] * (channel + 1)
        )
    assert gaussian.rho.numel() == 1

    adaptive = AdaptiveLFPThreshold(12)
    high = torch.randn(2, 12, 8, 8)
    threshold, threshold_debug = adaptive(high)
    torch.testing.assert_close(
        threshold_debug["threshold_ratio"],
        torch.full_like(threshold_debug["threshold_ratio"], 0.5),
    )
    torch.testing.assert_close(
        threshold, high.abs().mean((2, 3), keepdim=True) * 0.5
    )

    signed = torch.linspace(-2.0, 2.0, 6 * 8 * 8).reshape(1, 6, 8, 8)
    fixed = DecoderLFPProcessor(2, True, "fixed_hard")
    h, v, d = signed.chunk(3, dim=1)
    output = fixed(torch.randn(1, 5, 8, 8), h, v, d)
    fixed_debug = output[-1]
    assert set(torch.unique(fixed_debug["mask"]).tolist()).issubset({0.0, 1.0})
    expected_mask = (fixed_debug["modulated_high"].abs() < 0.5).float()
    torch.testing.assert_close(fixed_debug["mask"], expected_mask)
    expected_purified = (
        fixed_debug["modulated_high"] * (1.0 - expected_mask)
        + fixed_debug["gaussian_high"] * expected_mask
    )
    torch.testing.assert_close(fixed_debug["purified_high"], expected_purified)
    assert bool((fixed_debug["purified_high"] < 0).any())
    assert bool((fixed_debug["purified_high"] > 0).any())

    attention_only = DecoderLFPProcessor(2, False, "none")
    direct = attention_only(torch.randn(1, 5, 8, 8), h, v, d)[-1]
    nonzero = direct["aligned_high"] != 0
    assert torch.equal(
        torch.sign(direct["purified_high"][nonzero]),
        torch.sign(direct["aligned_high"][nonzero]),
    )

    return {
        "source_commit": NS_FPN_SOURCE_COMMIT,
        "attention_range": [float(attention_map.min()), float(attention_map.max())],
        "gaussian_sigma": float(gaussian.sigma),
        "gaussian_kernel_sum": float(kernel.sum()),
        "adaptive_initial_ratio": float(threshold_debug["threshold_ratio"].mean()),
    }


def test_h0_strict_regression(device, size):
    reference = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
        mode="test", deepsuper=True,
    )
    h0 = build("h0_e1_passthrough", "test", initialize=False)
    assert list(reference.state_dict()) == list(h0.state_dict())
    assert not any(name.startswith("decoder_lfp") for name, _ in h0.named_parameters())
    initialize_experiment_e_model(reference, init_weights)
    h0.load_state_dict(reference.state_dict(), strict=True)
    reference, h0 = reference.to(device).eval(), h0.to(device).eval()
    reference.debug_tensors = True
    h0.debug_tensors = True
    captures = {"reference": {}, "h0": {}}
    handles = []
    for label, model in (("reference", reference), ("h0", h0)):
        for stage in range(4):
            module = getattr(model, f"decoder_fuse{stage}")
            handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, label=label, stage=stage:
                captures[label].__setitem__(f"decoder_fuse{stage}_input", inputs[0].detach())
            ))
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, output, label=label, stage=stage:
                captures[label].__setitem__(f"decoder_fuse{stage}_output", output.detach())
            ))
    sample = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        reference_output = reference(sample)
        h0_output = h0(sample)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(h0_output, reference_output, rtol=0, atol=0)
    max_differences = {"output": tensor_max_difference(h0_output, reference_output)}
    for family in ("A", "A_lfss", "A_guided", "E"):
        for stage in reference.last_debug[family]:
            left = h0.last_debug[family][stage]
            right = reference.last_debug[family][stage]
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            max_differences[f"{family}{stage}"] = tensor_max_difference(left, right)
    for name, reference_tensor in captures["reference"].items():
        h0_tensor = captures["h0"][name]
        torch.testing.assert_close(h0_tensor, reference_tensor, rtol=0, atol=0)
        max_differences[name] = tensor_max_difference(h0_tensor, reference_tensor)
    assert h0.last_transform_counts == reference.last_transform_counts == {
        "dwt": 4, "idwt": 4,
    }
    del reference, h0, sample, reference_output, h0_output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return max_differences


def test_architecture_and_initialization():
    base = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
        mode="test", deepsuper=True,
    )
    preserved_names = [
        "stem",
        *(f"local_encoder{stage}" for stage in range(1, 5)),
        *(f"dir_encoder{stage}" for stage in range(1, 5)),
        *(f"stage_awgm{stage}" for stage in range(1, 5)),
        *(f"align_{direction}{stage}" for stage in range(1, 5) for direction in "HVD"),
        *(f"decoder_fuse{stage}" for stage in range(4)),
        "out_head", "outconv",
    ]
    signatures = {name: module_signature(getattr(base, name)) for name in preserved_names}
    parameter_counts = {}
    for variant in EXPERIMENT_H_VARIANTS:
        model = build(variant, "test", initialize=False)
        assert {name: module_signature(getattr(model, name)) for name in preserved_names} == signatures
        attention_before = {
            name: module.weight.detach().clone()
            for name, module in model.named_modules()
            if name.endswith("attention.conv")
        }
        initialize_experiment_h_model(model, init_weights)
        modules = dict(model.named_modules())
        for name, expected in attention_before.items():
            torch.testing.assert_close(modules[name].weight, expected, rtol=0, atol=0)
        for stage in range(1, 5):
            if model.use_gaussian:
                assert abs(float(getattr(model, f"decoder_lfp{stage}").gaussian.sigma.detach()) - 1.0) < 1e-6
            if model.threshold_mode == "adaptive_soft":
                final = getattr(model, f"decoder_lfp{stage}").threshold_predictor.predictor[-1]
                assert not torch.count_nonzero(final.weight)
                assert not torch.count_nonzero(final.bias)
        parameter_counts[variant] = sum(p.numel() for p in model.parameters())
    for left, right in PAIR_VARIANTS:
        assert parameter_counts[left] == parameter_counts[right]
    return {
        "preserved_modules": preserved_names,
        "parameter_counts": parameter_counts,
        "paired_parameter_counts_equal": True,
    }


def assert_no_forbidden_operations():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "model/decoder_lfp.py",
        root / "model/DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderLFP.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("torch.cdist", "torch.topk", ".topk("):
        assert forbidden not in combined


def assert_forbidden_modules_absent(model):
    forbidden = {
        "DecoderDSHFCore",
        "DecoderHFRefiner",
        "DecoderSemanticDirectionGate",
        "AdaptiveSparseSupportGate",
        "Targetness",
        "DirectionalTopDownPyramid",
    }
    found = {type(module).__name__ for module in model.modules()} & forbidden
    assert not found, f"Forbidden Experiment G/pyramid modules found: {sorted(found)}"


def assert_nonzero_finite_gradient(model, prefix):
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.requires_grad
    ]
    assert gradients, f"No trainable parameter matches {prefix}"
    assert all(gradient is not None for gradient in gradients), f"Missing gradient under {prefix}"
    assert all(torch.isfinite(gradient).all() for gradient in gradients), f"Non-finite gradient under {prefix}"
    assert any(bool(torch.count_nonzero(gradient).item()) for gradient in gradients), (
        f"All gradients are zero under {prefix}"
    )


def test_variant_forward(variant, device, batch, size, use_amp=False):
    model = build(variant, "train", device=device).train()
    assert_forbidden_modules_absent(model)
    model.debug_tensors = True
    sample = torch.randn(batch, 1, size, size, device=device)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with context:
        outputs = model(sample)
        loss = sum(output.float().mean() for output in outputs)
    assert len(outputs) == 6
    assert all(output.shape == (batch, 1, size, size) for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    debug = model.last_debug
    for stage in range(1, 5):
        lfp = debug["decoder_lfp"][stage]
        spatial = size // (2 ** stage)
        channels = DECODER_LFP_STAGE_CHANNELS[stage]
        assert lfp["aligned_high"].shape == (batch, 3 * channels, spatial, spatial)
        assert lfp["purified_high"].shape == lfp["aligned_high"].shape
        expected_low = (
            debug["A"][stage]
            if "rawll" in variant
            else (debug["E"][4] if stage == 4 else debug["L"][stage])
        )
        torch.testing.assert_close(lfp["low_source"], expected_low, rtol=0, atol=0)
        attention = lfp["attention"]
        assert attention.shape == (batch, 1, spatial, spatial)
        assert bool(((attention > 0.0) & (attention < 1.0)).all())
        torch.testing.assert_close(
            lfp["modulated_high"], lfp["aligned_high"] * attention
        )
        if variant.startswith("h1"):
            torch.testing.assert_close(lfp["purified_high"], lfp["modulated_high"])
            assert lfp["mask"] is None
        elif variant.startswith("h2"):
            torch.testing.assert_close(
                getattr(model, f"decoder_lfp{stage}").fixed_tau,
                torch.tensor(0.5, device=device), rtol=0, atol=0,
            )
            assert set(torch.unique(lfp["mask"]).tolist()).issubset({0.0, 1.0})
            torch.testing.assert_close(
                lfp["mask"], (lfp["modulated_high"].abs() < 0.5).to(lfp["mask"].dtype)
            )
        else:
            ratio = lfp["threshold_debug"]["threshold_ratio"]
            torch.testing.assert_close(ratio, torch.full_like(ratio, 0.5), rtol=0, atol=1e-6)
            threshold = lfp["threshold"]
            assert threshold.shape == (batch, 3 * channels, 1, 1)
            assert bool(((lfp["mask"] >= 0.0) & (lfp["mask"] <= 1.0)).all())
            assert bool(((lfp["mask"] > 0.0) & (lfp["mask"] < 1.0)).any())
            expected = (
                lfp["modulated_high"] * (1.0 - lfp["mask"])
                + lfp["gaussian_high"] * lfp["mask"]
            )
            torch.testing.assert_close(lfp["purified_high"], expected)
    required_prefixes = [
        *(f"decoder_lfp{stage}.attention" for stage in range(1, 5)),
        *(f"align_{direction}{stage}" for stage in range(1, 5) for direction in "HVD"),
        *(f"lfss_blocks.{stage}" for stage in range(1, 5)),
        *(f"dir_encoder{stage}" for stage in range(1, 5)),
        *(f"stage_awgm{stage}" for stage in range(1, 5)),
        *(f"local_encoder{stage}" for stage in range(1, 5)),
        *(f"decoder_fuse{stage}" for stage in range(4)),
        "gt_conv5", "gt_conv4", "gt_conv3", "gt_conv2", "out_head", "outconv",
    ]
    if variant.startswith(("h2", "h3")):
        required_prefixes.extend(f"decoder_lfp{stage}.gaussian" for stage in range(1, 5))
    if variant.startswith("h3"):
        required_prefixes.extend(f"decoder_lfp{stage}.threshold_predictor" for stage in range(1, 5))
    for prefix in required_prefixes:
        assert_nonzero_finite_gradient(model, prefix)
    assert not any(
        ".gaussian." in name for name, _ in model.named_parameters()
    ) if variant.startswith("h1") else True
    assert not any(
        ".threshold_predictor." in name for name, _ in model.named_parameters()
    ) if variant.startswith(("h1", "h2")) else True
    result = {
        "output_shapes": [list(output.shape) for output in outputs],
        "amp": bool(use_amp and device.type == "cuda"),
        "gradient_prefixes": required_prefixes,
        "dwt_calls": model.last_transform_counts["dwt"],
        "idwt_calls": model.last_transform_counts["idwt"],
    }
    del model, sample, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def test_adaptive_two_step(device, size):
    model = build("h3_rawll_adaptive_gaussian", "train", device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = torch.randn(2, 1, size, size, device=device)
    earlier_gradients = []
    final_gradients = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(sample)
        loss = sum(output.float().mean() for output in outputs)
        loss.backward()
        earlier = model.decoder_lfp1.threshold_predictor.predictor[0].weight.grad
        final = model.decoder_lfp1.threshold_predictor.predictor[-1].weight.grad
        earlier_gradients.append(0 if earlier is None else float(earlier.abs().sum()))
        final_gradients.append(0 if final is None else float(final.abs().sum()))
        optimizer.step()
    assert final_gradients[0] > 0.0
    assert earlier_gradients[1] > 0.0
    del model, optimizer, sample, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"earlier_gradients": earlier_gradients, "final_gradients": final_gradients}


def test_runtime_forbidden_ops(device, size):
    model = build("h3_decoder_adaptive_gaussian", "test", device=device).eval()
    sample = torch.randn(1, 1, size, size, device=device)
    original_cdist, original_topk = torch.cdist, torch.topk
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Forbidden operation called")
    torch.cdist, torch.topk = forbidden, forbidden
    try:
        with torch.no_grad():
            output = model(sample)
        assert torch.isfinite(output).all()
    finally:
        torch.cdist, torch.topk = original_cdist, original_topk
    del model, sample, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return True


def real_data_two_step(variant, dataset_dir, dataset_name, device, checkpoint_dir):
    dataset = TrainSetLoader(dataset_dir, dataset_name, patch_size=256, img_norm_cfg=None)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    iterator = iter(loader)
    model = build(variant, "train", device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(2):
        try:
            image, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image, target = next(iterator)
        assert image.shape[0] == 4
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = deep_supervision_loss(model(image), target, nn.BCELoss())
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / f"{variant}_two_step.pth.tar"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "variant": variant,
            "dataset": dataset_name,
            "steps": 2,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": losses,
        }, checkpoint_path)
        assert checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
    del model, optimizer, image, target, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"losses": losses, "checkpoint": str(checkpoint_path) if checkpoint_path else None}


def main():
    args = parse_args()
    set_seed(42)
    payload = {
        "status": "construction_passed",
        "core": test_core_modules(),
        "architecture": test_architecture_and_initialization(),
    }
    assert_no_forbidden_operations()
    if args.construct_only:
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    size = 256 if args.full else 64
    payload.update({
        "device": str(device),
        "h0_strict_regression": test_h0_strict_regression(device, size),
        "variants": {},
    })
    for variant in FORMAL_VARIANTS:
        payload["variants"][variant] = {
            "fp32": test_variant_forward(variant, device, 2, size),
            "amp": test_variant_forward(variant, device, 2, size, use_amp=True),
        }
        if args.dataset_dir:
            payload["variants"][variant]["real_data_batch4_two_step"] = real_data_two_step(
                variant, args.dataset_dir, args.dataset_name, device, args.checkpoint_dir
            )
    payload["adaptive_two_step"] = test_adaptive_two_step(device, size)
    payload["runtime_forbidden_ops"] = test_runtime_forbidden_ops(device, size)
    direction = check_haar_direction_correspondence(32, str(device))
    assert direction["routing_aligned"] is True
    payload["haar_direction"] = direction
    payload["status"] = "passed"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
