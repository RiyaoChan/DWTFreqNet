"""Regression, structure, gradient, CUDA/AMP and real-data tests for Experiment G."""

import argparse
import inspect
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
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM,
    snapshot_lfss_special_parameters,
)
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF import (
    DECODER_STAGE_CHANNELS,
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF,
    EXPERIMENT_G_VARIANTS,
    initialize_experiment_g_model,
)
from model.decoder_dshf import (
    AdaptiveSparseSupportGate,
    DecoderHFRefiner,
    DecoderSemanticDirectionGate,
    DirectionalMultiScaleExtractor,
    EXPERIMENT_F2_CORE_COMMIT,
)
from train_one import deep_supervision_loss, init_weights, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--dataset-name", default="NUAA-SIRST")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def build(variant, mode="test", device=None, initialize=True):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DecoderDSHF(
        get_DWTFreqNet_config(), decoder_variant=variant, mode=mode, deepsuper=True
    )
    if initialize:
        protected = snapshot_lfss_special_parameters(model)
        initialize_experiment_g_model(model, init_weights)
        after = snapshot_lfss_special_parameters(model)
        assert all(torch.equal(protected[key], after[key]) for key in protected)
    return model if device is None else model.to(device)


def build_e1(mode="test", device=None):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock", mode=mode,
        deepsuper=True,
    )
    return model if device is None else model.to(device)


def copy_common(source, target):
    target_state = target.state_dict()
    common = {key: value for key, value in source.state_dict().items() if key in target_state and target_state[key].shape == value.shape}
    result = target.load_state_dict(common, strict=False)
    assert not result.unexpected_keys


def test_g0_strict_regression(device, size):
    set_seed(42)
    reference = build_e1(device=device).eval()
    set_seed(7)
    g0 = build("g0_e1_passthrough", device=device, initialize=False).eval()
    assert tuple(reference.state_dict()) == tuple(g0.state_dict())
    g0.load_state_dict(reference.state_dict(), strict=True)
    assert not hasattr(g0, "decoder_hf_refiners")
    reference.debug_tensors = g0.debug_tensors = True
    sample = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        expected, actual = reference(sample), g0(sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert reference.last_shapes == g0.last_shapes
    for group in ("A", "A_lfss", "A_guided", "E"):
        for stage in range(1, 5):
            torch.testing.assert_close(
                g0.last_debug[group][stage], reference.last_debug[group][stage],
                rtol=0, atol=0,
            )
    return {"state_keys": len(g0.state_dict()), "exact_max_difference": 0.0}


def module_signature(module):
    return type(module).__name__, {name: tuple(value.shape) for name, value in module.named_parameters()}


def test_encoder_unchanged():
    reference = build_e1()
    names = ["stem", *(f"local_encoder{i}" for i in range(1, 5)), *(f"dir_encoder{i}" for i in range(1, 5)), *(f"stage_awgm{i}" for i in range(1, 5)), "lfss_blocks"]
    signatures = {name: module_signature(getattr(reference, name)) for name in names}
    for variant in EXPERIMENT_G_VARIANTS:
        model = build(variant, initialize=False)
        assert model.encoder_variant == "e1_lfss_resblock"
        assert {name: module_signature(getattr(model, name)) for name in names} == signatures
        assert all(isinstance(getattr(model, f"local_encoder{stage}"), Res_block) for stage in range(1, 5))
        assert not any(name.startswith("encoder_dshf") for name, _ in model.named_modules())
    return names


def test_core_source_and_controls():
    assert EXPERIMENT_F2_CORE_COMMIT == "3034408051d3742d80473650fe9d198fc37e48ab"
    expected = {
        "H": ((3, 1), (5, 1)),
        "V": ((1, 3), (1, 5)),
        "D": ((3, 3), (3, 3)),
    }
    for direction, kernels in expected.items():
        module = DirectionalMultiScaleExtractor(8, direction)
        assert module.branch1.kernel_size == kernels[0]
        assert module.branch2.kernel_size == kernels[1]
        assert module.branch1.groups == module.branch2.groups == 8
        if direction == "D":
            assert module.branch2.dilation == (2, 2)
    sparse = AdaptiveSparseSupportGate(8)
    output, debug = sparse(torch.randn(2, 8, 9, 9))
    assert output.shape == (2, 8, 9, 9)
    torch.testing.assert_close(debug["threshold_ratio"], torch.full_like(debug["threshold_ratio"], 0.5))
    assert bool(((debug["support"] > 0) & (debug["support"] < 1)).all())
    gate = DecoderSemanticDirectionGate(8).eval()
    with torch.no_grad():
        scales, _ = gate(*([torch.randn(2, 8, 9, 9)] * 4))
    for scale in scales:
        torch.testing.assert_close(scale, torch.ones_like(scale))
    for variant in EXPERIMENT_G_VARIANTS[1:]:
        model = build(variant)
        for stage, channels in DECODER_STAGE_CHANNELS.items():
            refiner = getattr(model, f"decoder_hf_refiner{stage}")
            assert refiner.channels == channels
            for beta in (refiner.beta_h, refiner.beta_v, refiner.beta_d):
                torch.testing.assert_close(beta, torch.full_like(beta, 1e-3))
    source = inspect.getsource(sys.modules["model.decoder_dshf"])
    assert ".cdist(" not in source and ".topk(" not in source
    return {"f2_core_commit": EXPERIMENT_F2_CORE_COMMIT, "beta_init": 1e-3}


def set_betas(model, value):
    for stage in range(1, 5):
        refiner = getattr(model, f"decoder_hf_refiner{stage}")
        for beta in (refiner.beta_h, refiner.beta_v, refiner.beta_d):
            nn.init.constant_(beta, value)


def test_initial_equivalences(device, size):
    set_seed(11)
    e1 = build_e1(device=device).eval()
    g1 = build("g1_decoder_dshf", device=device).eval()
    copy_common(e1, g1)
    set_betas(g1, 0.0)
    sample = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        reference, beta_zero = e1(sample), g1(sample)
    torch.testing.assert_close(beta_zero, reference, rtol=0, atol=0)

    set_betas(g1, 1e-3)
    g2 = build("g2_decoder_dshf_semantic", device=device).eval()
    copy_common(g1, g2)
    with torch.no_grad():
        out1, out2 = g1(sample), g2(sample)
    torch.testing.assert_close(out2, out1, rtol=1e-6, atol=1e-7)

    g3 = build("g3_decoder_dshf_targetness", device=device).eval()
    copy_common(g2, g3)
    g3.debug_tensors = True
    with torch.no_grad():
        out2, out3 = g2(sample), g3(sample)
    torch.testing.assert_close(out3, out2, rtol=1e-6, atol=1e-7)
    for stage in range(1, 5):
        targetness = g3.last_debug["decoder_dshf"][stage]["targetness"]
        assert targetness is not None and not targetness.requires_grad
    assert not any("targetness_head" in name for name, _ in g3.named_modules())

    refiner = DecoderHFRefiner(8).eval()
    negative = -torch.ones(2, 8, 8, 8)
    with torch.no_grad():
        restored = refiner(negative, negative, negative, negative)[0:3]
    assert all(bool((tensor < 0).all()) for tensor in restored)
    return {"beta_zero_e1": True, "g2_initial_g1": True, "g3_initial_g2": True, "sign_preserved": True}


def test_forward_and_gradients(variant, device, size, amp=False):
    model = build(variant, mode="train", device=device).train()
    batch = 2
    sample = torch.randn(batch, 1, size, size, device=device)
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else torch.autocast(device_type="cpu", enabled=False)
    with context:
        outputs = model(sample)
        loss = sum(output.float().mean() for output in outputs)
    assert len(outputs) == 6 and all(output.shape == sample.shape for output in outputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    assert any(name.startswith("decoder_hf_refiner") and parameter.grad is not None and torch.count_nonzero(parameter.grad) for name, parameter in model.named_parameters())
    return float(loss.detach().cpu())


def real_data_two_steps(variant, dataset_dir, dataset_name, device):
    dataset = TrainSetLoader(dataset_dir, dataset_name, patch_size=256, img_norm_cfg=None)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    model = build(variant, mode="train", device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    iterator = iter(loader)
    for _ in range(2):
        try:
            image, target = next(iterator)
        except StopIteration:
            iterator = iter(loader); image, target = next(iterator)
        assert image.shape[0] == 4
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = deep_supervision_loss(model(image), target, nn.BCELoss())
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    return losses


def main():
    args = parse_args(); set_seed(42)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    size = 256 if args.full else 64
    payload = {
        "device": str(device),
        "g0_regression": test_g0_strict_regression(device, size),
        "encoder_unchanged": test_encoder_unchanged(),
        "core": test_core_source_and_controls(),
        "equivalences": test_initial_equivalences(device, size),
        "variants": {},
    }
    for variant in EXPERIMENT_G_VARIANTS[1:]:
        payload["variants"][variant] = {
            "fp32_loss": test_forward_and_gradients(variant, device, size),
            "amp_loss": test_forward_and_gradients(variant, device, size, amp=True) if device.type == "cuda" else None,
        }
        if args.dataset_dir:
            payload["variants"][variant]["real_batch4_two_step_losses"] = real_data_two_steps(variant, args.dataset_dir, args.dataset_name, device)
        if device.type == "cuda": torch.cuda.empty_cache()
    direction = check_haar_direction_correspondence(32, str(device))
    assert direction["routing_aligned"] is True
    payload["haar"] = direction; payload["status"] = "passed"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output: Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
