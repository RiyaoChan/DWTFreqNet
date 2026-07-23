"""Regression, formula, CUDA/AMP and real-data tests for Experiment J."""

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
from model.DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP import (
    DENP_STAGE_CHANNELS,
    DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP,
    EXPERIMENT_J_VARIANTS,
    initialize_experiment_j_model,
)
from model.decoder_denp import (
    DENP_BANDS,
    DENPPurifier,
    LearnableBandGaussian,
    LowFrequencyCompactness,
    RobustBandNoiseEstimator,
)
from train_one import deep_supervision_loss, init_weights, set_seed


FORMAL_VARIANTS = EXPERIMENT_J_VARIANTS[1:]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--dataset-name", default="NUAA-SIRST")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def build(variant, mode, device=None, initialize=True):
    model = DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP(
        get_DWTFreqNet_config(), denp_variant=variant, mode=mode, deepsuper=True
    )
    if initialize:
        initialize_experiment_j_model(model, init_weights)
    return model if device is None else model.to(device)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def maximum_difference(left, right):
    return float((left.float() - right.float()).abs().max().cpu())


def test_robust_mad():
    set_seed(42)
    estimator = RobustBandNoiseEstimator()
    gaussian = torch.randn(4, 7, 256, 256) * 2.0
    sigma = estimator.robust_scale(gaussian)
    relative_error = float((sigma - 2.0).abs().mean() / 2.0)
    assert relative_error < 0.08

    contaminated = gaussian.clone()
    flat = contaminated.flatten()
    indices = torch.randperm(flat.numel())[: flat.numel() // 100]
    flat[indices] += 100.0
    clean_mad = float(estimator.robust_scale(gaussian).mean())
    contaminated_mad = float(estimator.robust_scale(contaminated).mean())
    clean_std = float(gaussian.std())
    contaminated_std = float(contaminated.std())
    mad_change = abs(contaminated_mad - clean_mad) / clean_mad
    std_change = abs(contaminated_std - clean_std) / clean_std
    assert mad_change < 0.08
    assert std_change > mad_change * 5.0

    finite_cases = {
        "zero": torch.zeros(2, 3, 17, 19),
        "constant": torch.full((2, 3, 17, 19), 4.0),
        "laplace": torch.distributions.Laplace(0.0, 1.0).sample((2, 3, 33, 35)),
    }
    for band in finite_cases.values():
        noise, debug = estimator(band)
        assert torch.isfinite(noise).all()
        assert torch.isfinite(debug["sigma_hat"]).all()
        assert not debug["sigma_hat"].requires_grad
    assert abs(float(estimator.lambda_value.detach()) - 1.0) < 1e-6
    with torch.no_grad():
        estimator.lambda_logit.fill_(100.0)
        assert 0.5 < float(estimator.lambda_value) <= 3.0
        estimator.lambda_logit.fill_(-100.0)
        assert 0.5 <= float(estimator.lambda_value) < 3.0
    estimator.reset_control_parameters()
    return {
        "gaussian_relative_error": relative_error,
        "clean_mad": clean_mad,
        "contaminated_mad": contaminated_mad,
        "mad_relative_change": mad_change,
        "clean_std": clean_std,
        "contaminated_std": contaminated_std,
        "std_relative_change": std_change,
    }


def test_noise_formula():
    estimator = RobustBandNoiseEstimator()
    band = torch.linspace(-3.0, 3.0, 2 * 5 * 13 * 15).reshape(2, 5, 13, 15)
    noise, debug = estimator(band)
    expected = torch.sigmoid(
        (debug["tau"] - band.abs()) / (0.15 * debug["tau"] + 1e-6)
    )
    torch.testing.assert_close(noise, expected, rtol=0, atol=0)
    ordered = torch.tensor([0.0, 0.5, 1.0, 2.0])
    tau = torch.tensor(1.0)
    confidence = torch.sigmoid((tau - ordered) / (0.15 * tau + 1e-6))
    assert bool(torch.all(confidence[:-1] > confidence[1:]))
    return {"confidence": confidence.tolist()}


def test_gaussian():
    gaussian = LearnableBandGaussian(4)
    assert abs(float(gaussian.sigma.detach()) - 1.0) < 1e-6
    kernel = gaussian.kernel()
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0))
    torch.testing.assert_close(kernel, kernel.flip(-1))
    torch.testing.assert_close(kernel, kernel.flip(-2))
    impulse = torch.zeros(1, 4, 9, 9)
    impulse[:, 2, 4, 4] = 3.0
    response = gaussian(impulse)
    assert not torch.count_nonzero(response[:, :2])
    assert not torch.count_nonzero(response[:, 3:])
    torch.testing.assert_close(response[0, 2, 3:6, 3:6], 3.0 * kernel[0, 0])
    with torch.no_grad():
        gaussian.sigma_logit.fill_(100.0)
        assert 0.5 < float(gaussian.sigma) <= 2.0
        gaussian.sigma_logit.fill_(-100.0)
        assert 0.5 <= float(gaussian.sigma) < 2.0
    gaussian.reset_control_parameters()

    processors = DENPPurifier(2, "j1_bandwise_noise_calibrated")
    bands = [torch.zeros(1, 2, 9, 9) for _ in range(3)]
    bands[0][:, :, 4, 4] = 1.0
    output = processors(None, None, *bands)
    assert torch.equal(output[1], bands[1])
    assert torch.equal(output[2], bands[2])
    return {"kernel": kernel.flatten().tolist(), "kernel_sum": float(kernel.sum())}


def test_compactness():
    module = LowFrequencyCompactness()
    shape = (1, 4, 33, 33)
    cases = {"flat": torch.zeros(shape), "point": torch.zeros(shape),
             "blob": torch.zeros(shape), "line": torch.zeros(shape),
             "ring": torch.zeros(shape), "noise": torch.randn(shape)}
    cases["point"][:, :, 16, 16] = 8.0
    cases["blob"][:, :, 15:18, 15:18] = 4.0
    cases["line"][:, :, 16, 8:25] = 4.0
    cases["ring"][:, :, 13:20, 13:20] = 4.0
    cases["ring"][:, :, 15:18, 15:18] = 0.0
    scores = {}
    for name, low in cases.items():
        protection, debug = module(low)
        assert torch.isfinite(protection).all()
        assert torch.isfinite(debug["compactness"]).all()
        scores[name] = {
            "compactness": float(debug["compactness"][0, 0, 16, 16]),
            "protection": float(protection[0, 0, 16, 16]),
        }
    assert scores["blob"]["compactness"] > scores["line"]["compactness"]
    assert scores["point"]["compactness"] > scores["line"]["compactness"]
    assert scores["line"]["compactness"] > scores["flat"]["compactness"]
    assert scores["flat"]["compactness"] > scores["ring"]["compactness"]
    compactness = torch.linspace(-1.0, 1.0, 101)
    protection = torch.sigmoid(module.slope.detach() * (compactness - module.threshold.detach()))
    assert bool(torch.all(protection[1:] > protection[:-1]))
    assert abs(float(module.slope.detach()) - 5.0) < 1e-6
    assert abs(float(module.threshold.detach())) < 1e-7
    return scores


def test_gate_formulas():
    expected_features = {
        "j1_bandwise_noise_calibrated": (False, False, False),
        "j2_rawll_compactness": (True, False, False),
        "j2_decoder_compactness": (False, True, False),
        "j3_dual_evidence_fixed": (True, True, False),
        "j3_dual_evidence_reliability": (True, True, True),
    }
    results = {}
    for variant, (raw_enabled, decoder_enabled, reliability) in expected_features.items():
        processor = DENPPurifier(3, variant)
        raw = torch.randn(2, 5, 11, 13)
        decoder = torch.randn(2, 7, 11, 13)
        bands = tuple(torch.randn(2, 3, 11, 13) for _ in DENP_BANDS)
        outputs = processor(raw, decoder, *bands)
        debug = outputs[-1]
        assert processor.use_raw_compactness == raw_enabled
        assert processor.use_decoder_compactness == decoder_enabled
        assert processor.use_reliability == reliability
        for index, band in enumerate(DENP_BANDS):
            item = debug["bands"][band]
            expected_mask = item["noise_confidence"]
            if raw_enabled:
                gamma = processor.gamma_raw[index] if reliability else 1.0
                expected_mask = expected_mask * (1.0 - debug["raw_protection"]).pow(gamma)
            if decoder_enabled:
                gamma = processor.gamma_decoder[index] if reliability else 1.0
                expected_mask = expected_mask * (1.0 - debug["decoder_protection"]).pow(gamma)
            torch.testing.assert_close(item["mask"], expected_mask)
            expected_output = ((1.0 - expected_mask) * item["aligned"]
                               + expected_mask * item["gaussian"])
            torch.testing.assert_close(item["purified"], expected_output)
            assert bool(((item["mask"] >= 0.0) & (item["mask"] <= 1.0)).all())
        results[variant] = float(debug["bands"]["H"]["mask"].mean())

    noise = torch.tensor([0.8])
    low, high = torch.tensor([0.1]), torch.tensor([0.9])
    assert float(noise * (1 - high)) < float(noise * (1 - low))
    assert float(noise * (1 - high) * (1 - high)) < float(noise * (1 - low) * (1 - low))
    return results


def test_architecture_and_initialization():
    counts = {}
    for variant in EXPERIMENT_J_VARIANTS:
        model = build(variant, "test")
        counts[variant] = parameter_count(model)
        assert model.second_dwt is False
        assert model.directional_pyramid is False
        assert model.ldrc is False
        assert model.channel_matching is False
        if variant == "j0_e1_passthrough":
            assert not any(name.startswith("decoder_denp") for name, _ in model.named_parameters())
            continue
        for stage in range(1, 5):
            processor = getattr(model, f"decoder_denp{stage}")
            assert processor.high_channels == DENP_STAGE_CHANNELS[stage]
            for band in DENP_BANDS:
                assert abs(float(processor.noise_estimators[band].lambda_value) - 1.0) < 1e-6
                assert abs(float(processor.gaussians[band].sigma) - 1.0) < 1e-6
            for name in ("raw_compactness", "decoder_compactness"):
                if hasattr(processor, name):
                    compactness = getattr(processor, name)
                    assert abs(float(compactness.slope) - 5.0) < 1e-6
                    assert abs(float(compactness.threshold)) < 1e-7
            if processor.use_reliability:
                torch.testing.assert_close(processor.gamma_raw, torch.ones(3))
                torch.testing.assert_close(processor.gamma_decoder, torch.ones(3))
    assert counts["j2_rawll_compactness"] == counts["j2_decoder_compactness"]
    assert counts["j3_dual_evidence_reliability"] - counts["j3_dual_evidence_fixed"] == 24
    return counts


def test_j0_strict_regression(device, size):
    reference = DWTFreqNet_SingleDecoder_LFSS_AWGM(
        get_DWTFreqNet_config(), encoder_variant="e1_lfss_resblock",
        mode="test", deepsuper=True,
    )
    j0 = build("j0_e1_passthrough", "test", initialize=False)
    assert list(reference.state_dict()) == list(j0.state_dict())
    initialize_experiment_e_model(reference, init_weights)
    j0.load_state_dict(reference.state_dict(), strict=True)
    reference, j0 = reference.to(device).eval(), j0.to(device).eval()
    reference.debug_tensors = j0.debug_tensors = True
    captures = {"reference": {}, "j0": {}}
    handles = []
    for label, model in (("reference", reference), ("j0", j0)):
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
        reference_output, j0_output = reference(sample), j0(sample)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(j0_output, reference_output, rtol=0, atol=0)
    differences = {"output": maximum_difference(j0_output, reference_output)}
    for family in ("A", "A_lfss", "A_guided", "E"):
        for stage, reference_tensor in reference.last_debug[family].items():
            actual = j0.last_debug[family][stage]
            torch.testing.assert_close(actual, reference_tensor, rtol=0, atol=0)
            differences[f"{family}{stage}"] = maximum_difference(actual, reference_tensor)
    for name, reference_tensor in captures["reference"].items():
        actual = captures["j0"][name]
        torch.testing.assert_close(actual, reference_tensor, rtol=0, atol=0)
        differences[name] = maximum_difference(actual, reference_tensor)
    assert j0.last_transform_counts == reference.last_transform_counts == {"dwt": 4, "idwt": 4}
    del reference, j0, sample, reference_output, j0_output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return differences


def assert_gradient(model, prefixes):
    gradients = [parameter.grad for name, parameter in model.named_parameters()
                 if name.startswith(prefixes) and parameter.requires_grad]
    assert gradients, f"No parameters for {prefixes}"
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)


def test_variant_forward(variant, device, size, use_amp=False):
    model = build(variant, "train", device=device).train()
    model.debug_tensors = True
    sample = torch.randn(2, 1, size, size, device=device)
    context = (torch.autocast("cuda", dtype=torch.float16)
               if use_amp and device.type == "cuda"
               else torch.autocast(device.type, enabled=False))
    with context:
        outputs = model(sample)
        loss = sum(output.float().mean() for output in outputs)
    assert len(outputs) == 6 and torch.isfinite(loss)
    assert all(output.shape == (2, 1, size, size) and torch.isfinite(output).all()
               for output in outputs)
    loss.backward()
    assert model.last_transform_counts == {"dwt": 4, "idwt": 4}
    for stage in range(1, 5):
        debug = model.last_debug["decoder_denp"][stage]
        expected_raw = model.last_debug["A"][stage]
        expected_decoder = model.last_debug["E"][4] if stage == 4 else model.last_debug["L"][stage]
        torch.testing.assert_close(debug["raw_low"], expected_raw, rtol=0, atol=0)
        torch.testing.assert_close(debug["decoder_low"], expected_decoder, rtol=0, atol=0)
        spatial = size // (2 ** stage)
        for band in DENP_BANDS:
            item = debug["bands"][band]
            assert item["aligned"].shape == (2, DENP_STAGE_CHANNELS[stage], spatial, spatial)
            assert bool((item["aligned"] < 0).any()) and bool((item["aligned"] > 0).any())
            assert torch.isfinite(item["purified"]).all()
    required = [
        "decoder_denp",
        "align_", "lfss_blocks.", "dir_encoder", "stage_awgm",
        "local_encoder", "decoder_fuse", "gt_conv", "out_head", "outconv",
    ]
    for prefix in required:
        assert_gradient(model, prefix)
    result = {"amp": bool(use_amp and device.type == "cuda"),
              "loss": float(loss.detach().cpu()), "dwt": 4, "idwt": 4}
    del model, sample, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def test_two_steps(device, size):
    model = build("j3_dual_evidence_reliability", "train", device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = torch.randn(2, 1, size, size, device=device)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(sample)
        loss = sum(output.float().mean() for output in outputs)
        assert torch.isfinite(loss)
        loss.backward()
        assert_gradient(model, "decoder_denp")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    del model, optimizer, sample, outputs, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return losses


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
        assert image.shape == (4, 1, 256, 256)
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = deep_supervision_loss(model(image), target, nn.BCELoss())
        assert torch.isfinite(loss)
        loss.backward()
        assert_gradient(model, "decoder_denp")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    checkpoint = None
    if checkpoint_dir:
        checkpoint = Path(checkpoint_dir) / f"{variant}_two_step.pth.tar"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"variant": variant, "dataset": dataset_name, "steps": 2,
                    "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "losses": losses}, checkpoint)
        assert checkpoint.stat().st_size > 0
    del model, optimizer, image, target, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"losses": losses, "checkpoint": str(checkpoint) if checkpoint else None}


def assert_no_forbidden_operations():
    root = Path(__file__).resolve().parents[1]
    combined = "\n".join((root / path).read_text(encoding="utf-8") for path in (
        "model/decoder_denp.py", "model/DWTFreqNet_SingleDecoder_LFSS_AWGM_DENP.py"
    ))
    for forbidden in ("torch.cdist", "torch.topk", ".topk(", "AdaptiveSparseSupportGate",
                      "DirectionalTopDownPyramid", "DecoderDSHFCore"):
        assert forbidden not in combined


def main():
    args = parse_args()
    set_seed(42)
    assert_no_forbidden_operations()
    payload = {
        "robust_mad": test_robust_mad(),
        "noise_formula": test_noise_formula(),
        "gaussian": test_gaussian(),
        "compactness": test_compactness(),
        "gate_formulas": test_gate_formulas(),
        "parameter_counts": test_architecture_and_initialization(),
        "status": "construction_passed",
    }
    if not args.construct_only:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        size = 256 if args.full else 64
        payload["device"] = str(device)
        payload["j0_strict_regression"] = test_j0_strict_regression(device, size)
        payload["variants"] = {}
        for variant in FORMAL_VARIANTS:
            payload["variants"][variant] = {
                "fp32": test_variant_forward(variant, device, size),
                "amp": test_variant_forward(variant, device, size, use_amp=True),
            }
            if args.dataset_dir:
                payload["variants"][variant]["real_data_batch4_two_step"] = real_data_two_step(
                    variant, args.dataset_dir, args.dataset_name, device, args.checkpoint_dir
                )
        payload["two_step_gradient"] = test_two_steps(device, size)
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
