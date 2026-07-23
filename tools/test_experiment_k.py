"""Regression, formula, CUDA/AMP and real-data tests for Experiment K v2."""

import argparse
import importlib.util
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
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP,
    initialize_experiment_j_model,
)
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_K import (
    DWTFreqNet_SingleDecoder_LFSS_AWGM_K,
    EXPERIMENT_K_VARIANTS,
    K_STAGE_CHANNELS,
    initialize_experiment_k_model,
)
from model.decoder_k_dose import (
    DoseCalibratedBandPurifier,
    GaussianRadialCompactness,
    build_gaussian_radial_offsets,
)
from train_one import deep_supervision_loss, init_weights, set_seed


FORMAL_VARIANTS = EXPERIMENT_K_VARIANTS[2:6]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--phase1-root", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--dataset-name", default="NUAA-SIRST")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def build_k(variant, mode="test", device=None, initialize=True):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_K(
        get_DWTFreqNet_config(), k_variant=variant, mode=mode, deepsuper=True
    )
    if initialize:
        initialize_experiment_k_model(model, init_weights)
    return model if device is None else model.to(device)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def maximum_difference(left, right):
    return float((left.float() - right.float()).abs().max().cpu())


def load_shared_state(source, target, source_prefix="", target_prefix=""):
    mapped = {}
    target_state = target.state_dict()
    for name, value in source.state_dict().items():
        mapped_name = target_prefix + name[len(source_prefix):] if name.startswith(source_prefix) else name
        if mapped_name in target_state and target_state[mapped_name].shape == value.shape:
            mapped[mapped_name] = value
    missing, unexpected = target.load_state_dict(mapped, strict=False)
    assert not unexpected
    return missing


def test_offsets(phase1_root=""):
    offsets = build_gaussian_radial_offsets()
    radii = offsets.norm(dim=-1)
    result = {
        "inner": int((radii <= 1.0).sum()),
        "outer": int((radii > 1.0).sum()),
        "max_radius": float(radii.max()),
        "phase1_max_abs_error": None,
    }
    assert result["inner"] == 14 and result["outer"] == 18
    assert abs(result["max_radius"] - 2.0) < 1e-12
    if phase1_root:
        path = Path(phase1_root) / "tools" / "phase1" / "common.py"
        spec = importlib.util.spec_from_file_location("experiment_k_phase1_common", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = torch.from_numpy(module.geometry_points("gaussian_radial", radius=2.0))
        error = float((offsets - expected).abs().max())
        assert error < 1e-6
        result["phase1_max_abs_error"] = error
    return result


def test_compactness_and_stop_gradient():
    module = GaussianRadialCompactness()
    low = torch.randn(2, 7, 17, 19, requires_grad=True)
    protection, debug = module(low)
    assert protection.shape == (2, 1, 17, 19)
    assert torch.isfinite(protection).all()
    assert bool(((protection >= 0.0) & (protection <= 1.0)).all())
    assert not protection.requires_grad
    assert low.grad is None
    flat = torch.ones(1, 3, 9, 9)
    flat_p, flat_debug = module(flat)
    assert float(flat_p.abs().max()) < 1e-6
    point = torch.zeros(1, 3, 9, 9)
    point[:, :, 4, 4] = 8.0
    point_p, point_debug = module(point)
    assert float(point_p[0, 0, 4, 4]) > float(flat_p[0, 0, 4, 4])
    return {
        "flat_center": float(flat_p[0, 0, 4, 4]),
        "point_center": float(point_p[0, 0, 4, 4]),
        "point_ratio": float(point_debug["ratio"][0, 0, 4, 4]),
        "source_requires_grad": protection.requires_grad,
    }


def test_gate_formula():
    processor = DoseCalibratedBandPurifier(5, protection_enabled=True)
    low = torch.randn(2, 6, 13, 15, requires_grad=True)
    bands = tuple(torch.randn(2, 5, 13, 15) for _ in range(3))
    outputs = processor(*bands, prior_low=low)
    debug = outputs[-1]
    assert abs(float(processor.alpha.mean()) - 0.05) < 1e-6
    assert abs(float(processor.rho) - 0.05) < 1e-6
    for band in "HVD":
        item = debug["bands"][band]
        expected = item["alpha"] * item["noise_confidence"] * (
            1.0 - debug["rho"] * debug["protection"]
        )
        torch.testing.assert_close(item["dose"], expected)
        torch.testing.assert_close(
            item["purified"], item["aligned"] + expected * item["gaussian_residual"]
        )
    sum(output.mean() for output in outputs[:3]).backward()
    assert low.grad is None
    assert processor.alpha_logits.grad is not None
    assert processor.prior_protection.rho_logit.grad is not None
    return {"alpha": processor.alpha.tolist(), "rho": float(processor.rho)}


def test_spatial_dose_override():
    processor = DoseCalibratedBandPurifier(
        5, protection_enabled=False, learnable_alpha=False, fixed_alpha=1.0
    ).eval()
    bands = tuple(torch.randn(1, 5, 9, 11) for _ in range(3))
    with torch.no_grad():
        full = processor(*bands, alpha_override=1.0)
        override = torch.full((1, 1, 9, 11), -1.0)
        override[:, :, 4, 5] = 0.0
        local = processor(
            *bands,
            alpha_override=1.0,
            spatial_dose_override={band: override for band in "HVD"},
        )
    center_differences = {}
    outside_differences = {}
    for index, band in enumerate("HVD"):
        item = local[-1]["bands"][band]
        assert float(item["dose"][0, 0, 4, 5]) == 0.0
        torch.testing.assert_close(
            local[index][:, :, 4, 5], bands[index][:, :, 4, 5], rtol=0, atol=0
        )
        mask = torch.ones((9, 11), dtype=torch.bool)
        mask[4, 5] = False
        torch.testing.assert_close(
            local[index][:, :, mask], full[index][:, :, mask], rtol=1e-6, atol=1e-7
        )
        center_differences[band] = maximum_difference(
            local[index][:, :, 4, 5], full[index][:, :, 4, 5]
        )
        outside_differences[band] = maximum_difference(
            local[index][:, :, mask], full[index][:, :, mask]
        )
        assert center_differences[band] > 0.0
    return {
        "center_difference": center_differences,
        "outside_difference": outside_differences,
    }


def test_architecture():
    counts = {}
    expected_sources = {
        "k0_e1_passthrough": {1: None, 2: None, 3: None, 4: None},
        "k1_j1_full_dose": {1: None, 2: None, 3: None, 4: None},
        "k2_dose_calibrated": {1: None, 2: None, 3: None, 4: None},
        "k3_gr_raw_all": {1: "raw_ll", 2: "raw_ll", 3: "raw_ll", 4: "raw_ll"},
        "k4_gr_lfss_s123": {1: "lfss_ll", 2: "lfss_ll", 3: "lfss_ll", 4: None},
        "k5_gr_guided_s123": {1: "guided_ll", 2: "guided_ll", 3: "guided_ll", 4: None},
    }
    for variant, source_map in expected_sources.items():
        model = build_k(variant)
        counts[variant] = parameter_count(model)
        assert model.k_source_map == source_map
        assert model.second_dwt is False and model.last_transform_counts == {"dwt": 0, "idwt": 0}
        if variant == "k0_e1_passthrough":
            assert not any(name.startswith("decoder_k") for name, _ in model.named_parameters())
            continue
        alpha_parameters = sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if ".alpha_logits" in name
        )
        assert alpha_parameters == (0 if variant == "k1_j1_full_dose" else 12)
        rho_parameters = sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if name.endswith("rho_logit")
        )
        assert rho_parameters == sum(source is not None for source in source_map.values())
        for stage in range(1, 5):
            assert getattr(model, f"decoder_k{stage}").channels == K_STAGE_CHANNELS[stage]
    return counts


def test_strict_regressions(device, size):
    set_seed(42)
    e1 = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
        mode="test", deepsuper=True,
    )
    initialize_experiment_e_model(e1, init_weights)
    k0 = build_k("k0_e1_passthrough", initialize=False)
    assert list(e1.state_dict()) == list(k0.state_dict())
    k0.load_state_dict(e1.state_dict(), strict=True)

    j1 = DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP(
        get_DWTFreqNet_config(), denp_variant="j1_bandwise_noise_calibrated",
        mode="test", deepsuper=True,
    )
    initialize_experiment_j_model(j1, init_weights)
    k1 = build_k("k1_j1_full_dose", initialize=False)
    mapped = {
        name.replace("decoder_denp", "decoder_k"): value
        for name, value in j1.state_dict().items()
    }
    k1.load_state_dict(mapped, strict=True)

    k2 = build_k("k2_dose_calibrated", initialize=False)
    load_shared_state(e1, k2)
    k2_j = build_k("k2_dose_calibrated", initialize=False)
    mapped_k2 = {
        name.replace("decoder_denp", "decoder_k"): value
        for name, value in j1.state_dict().items()
        if name.replace("decoder_denp", "decoder_k") in k2_j.state_dict()
    }
    k2_j.load_state_dict(mapped_k2, strict=False)
    k3 = build_k("k3_gr_raw_all", initialize=False)
    load_shared_state(k2, k3)

    models = [e1, k0, j1, k1, k2, k2_j, k3]
    for model in models:
        model.to(device).eval()
    k2.alpha_override = 0.0
    k2_j.alpha_override = 1.0
    k2_j.rho_override = 0.0
    k2.alpha_override = None
    k3.rho_override = 0.0
    sample = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        out_e1 = e1(sample)
        out_k0 = k0(sample)
        # Re-apply alpha=0 only for the E1-degradation pass.
        k2.alpha_override = 0.0
        out_k2_e1 = k2(sample)
        out_j1 = j1(sample)
        out_k1 = k1(sample)
        out_k2_j1 = k2_j(sample)
        k2.alpha_override = None
        out_k2 = k2(sample)
        out_k3 = k3(sample)
    for actual, expected in (
        (out_k0, out_e1), (out_k2_e1, out_e1),
        (out_k1, out_j1), (out_k2_j1, out_j1), (out_k3, out_k2),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = {
        "k0_to_e1": maximum_difference(out_k0, out_e1),
        "alpha0_to_e1": maximum_difference(out_k2_e1, out_e1),
        "k1_to_j1": maximum_difference(out_k1, out_j1),
        "alpha1_to_j1": maximum_difference(out_k2_j1, out_j1),
        "rho0_to_k2": maximum_difference(out_k3, out_k2),
    }
    del models, e1, k0, j1, k1, k2, k2_j, k3, sample
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def test_decoder_low_causal_source(device, size):
    model = build_k("k3_gr_raw_all", device=device).eval()
    model.debug_tensors = True
    model.k_source_map = {stage: "decoder_low" for stage in range(1, 5)}
    model.rho_override = 0.25
    sample = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        model(sample)
    result = {}
    for stage in range(1, 5):
        debug = model.last_debug["decoder_k"][stage]
        expected = (
            model.last_debug["E"][4]
            if stage == 4
            else model.last_debug["L"][stage]
        )
        assert debug["prior_source"] == "decoder_low"
        assert debug["prior_low"] is not None
        torch.testing.assert_close(debug["prior_low"], expected, rtol=0, atol=0)
        assert debug["prior_low"].shape[1] == K_STAGE_CHANNELS[stage]
        result[f"stage{stage}"] = list(debug["prior_low"].shape)
    del model, sample
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def assert_gradient(model, fragment):
    gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if fragment in name and parameter.requires_grad
    ]
    assert gradients, f"No trainable parameters matching {fragment}"
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)


def test_variant_forward(variant, device, size, use_amp=False):
    model = build_k(variant, mode="train", device=device).train()
    model.debug_tensors = True
    sample = torch.randn(2, 1, size, size, device=device)
    context = (
        torch.autocast("cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else torch.autocast(device.type, enabled=False)
    )
    with context:
        outputs = model(sample)
        loss = sum(output.float().mean() for output in outputs)
    assert len(outputs) == 6 and torch.isfinite(loss)
    loss.backward()
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    for stage in range(1, 5):
        debug = model.last_debug["decoder_k"][stage]
        spatial = size // (2 ** stage)
        for band in "HVD":
            item = debug["bands"][band]
            assert item["aligned"].shape == (2, K_STAGE_CHANNELS[stage], spatial, spatial)
            assert torch.isfinite(item["purified"]).all()
    for fragment in (
        "alpha_logits", "noise_estimators", "gaussians", "align_",
        "lfss_blocks.", "stage_awgm", "local_encoder", "decoder_fuse", "out_head",
    ):
        assert_gradient(model, fragment)
    if any(model.k_source_map.values()):
        assert_gradient(model, "rho_logit")
    result = {"amp": bool(use_amp and device.type == "cuda"), "loss": float(loss.detach().cpu())}
    del model, sample, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def test_two_steps(device, size):
    model = build_k("k5_gr_guided_s123", mode="train", device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = torch.randn(2, 1, size, size, device=device)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(sample)
        loss = sum(output.float().mean() for output in outputs)
        loss.backward()
        assert_gradient(model, "alpha_logits")
        assert_gradient(model, "rho_logit")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def real_data_two_step(variant, dataset_dir, dataset_name, device, checkpoint_dir):
    dataset = TrainSetLoader(dataset_dir, dataset_name, patch_size=256, img_norm_cfg=None)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    iterator = iter(loader)
    model = build_k(variant, mode="train", device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(2):
        try:
            image, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image, target = next(iterator)
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = deep_supervision_loss(model(image), target, nn.BCELoss())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    checkpoint = None
    if checkpoint_dir:
        checkpoint = Path(checkpoint_dir) / f"{variant}_two_step.pth.tar"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"variant": variant, "state_dict": model.state_dict(), "losses": losses}, checkpoint)
    return {"losses": losses, "checkpoint": str(checkpoint) if checkpoint else None}


def main():
    args = parse_args()
    set_seed(42)
    payload = {
        "offsets": test_offsets(args.phase1_root),
        "compactness": test_compactness_and_stop_gradient(),
        "gate": test_gate_formula(),
        "spatial_dose_override": test_spatial_dose_override(),
        "parameter_counts": test_architecture(),
        "status": "construction_passed",
    }
    if not args.construct_only:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        size = 256 if args.full else 64
        payload["device"] = str(device)
        payload["strict_regressions"] = test_strict_regressions(device, size)
        payload["decoder_low_causal_source"] = test_decoder_low_causal_source(
            device, size
        )
        payload["variants"] = {}
        for variant in FORMAL_VARIANTS:
            payload["variants"][variant] = {
                "fp32": test_variant_forward(variant, device, size),
                "amp": test_variant_forward(variant, device, size, use_amp=True),
            }
        payload["two_step_gradient"] = test_two_steps(device, size)
        if args.dataset_dir:
            payload["real_data_batch4_two_step"] = real_data_two_step(
                "k2_dose_calibrated", args.dataset_dir, args.dataset_name,
                device, args.checkpoint_dir,
            )
        direction = check_haar_direction_correspondence(32, str(device))
        assert direction["routing_aligned"] is True
        payload["haar_direction"] = direction
        payload["status"] = "passed"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
