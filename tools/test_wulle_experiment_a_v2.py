"""Validation suite for Experiment A v2's isolated WULLE model."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.Config import get_DWTFreqNet_config
from model.DWTFreqNet import DWTFreqNet, check_haar_direction_correspondence
from model.DWTFreqNet_WULLE import DWTFreqNet_WULLE


VARIANTS = (
    "awgm_original",
    "dm_awgm_no_dcn",
    "w8m_diag4_subband_shared",
    "w8m_diag4_axial_diag_shared",
)


def parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def has_nonzero_gradient(model, prefix):
    return any(
        name.startswith(prefix)
        and parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in model.named_parameters()
    )


def build(model_class, mode, variant, fallback, device):
    return model_class(
        get_DWTFreqNet_config(),
        mode=mode,
        deepsuper=True,
        awgm_variant=variant,
        awgm_allow_fallback=fallback,
    ).to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true", help="Run 2x256 shape and backward checks")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    original = build(DWTFreqNet, "test", "awgm_original", False, device)
    wulle = build(DWTFreqNet_WULLE, "test", "awgm_original", False, device)

    removed = wulle.REMOVED_LOCAL_NODES + wulle.REMOVED_GLOBAL_PROJECTIONS
    assert all(not hasattr(wulle, name) for name in removed)
    assert all(
        hasattr(wulle, name)
        for name in ("wulle_decoder1", "wulle_decoder2", "wulle_decoder3")
    )
    original_parameters = parameters(original)
    wulle_parameters = parameters(wulle)
    assert wulle_parameters < original_parameters

    direction = check_haar_direction_correspondence(size=32, device=str(device))
    assert direction["band_response_orientation"] == {
        "H": "vertical",
        "V": "horizontal",
    }
    assert direction["routing_aligned"] is True

    batch, size = (2, 256) if args.full else (1, 64)
    sample = torch.randn(batch, 1, size, size, device=device)
    wulle.eval()
    with torch.no_grad():
        output = wulle(sample)
    assert tuple(output.shape) == (batch, 1, size, size)
    if args.full:
        expected = {
            "E1": (2, 64, 128, 128), "E2": (2, 128, 64, 64),
            "E3": (2, 256, 32, 32), "E4": (2, 256, 16, 16),
            "D3": (2, 256, 32, 32), "D2": (2, 128, 64, 64),
            "D1": (2, 64, 128, 128), "G1_4": (2, 64, 128, 128),
            "G2_3": (2, 128, 64, 64), "G3_2": (2, 256, 32, 32),
        }
        assert {key: wulle.last_wulle_shapes[key] for key in expected} == expected

    del original, wulle, sample, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    variant_results = {}
    for variant in VARIANTS:
        model = build(DWTFreqNet_WULLE, "test", variant, True, device).eval()
        with torch.no_grad():
            prediction = model(torch.randn(1, 1, 64, 64, device=device))
        assert tuple(prediction.shape) == (1, 1, 64, 64)
        if variant.startswith("w8m_"):
            assert model.awgm_backends["haar_routing_aligned"] is True
        variant_results[variant] = model.awgm_backends
        del model, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gradient_prefixes = []
    if args.full:
        model = build(DWTFreqNet_WULLE, "train", "awgm_original", False, device)
        outputs = model(torch.randn(1, 1, 256, 256, device=device))
        assert len(outputs) == 6
        assert all(tuple(output.shape) == (1, 1, 256, 256) for output in outputs)
        sum(output.mean() for output in outputs).backward()
        gradient_prefixes = (
            "conv_wavelet_inchannel_local", "local_encoder1_1", "local_encoder2_1",
            "local_encoder3_1", "local_encoder4_1", "wulle_decoder1",
            "wulle_decoder2", "wulle_decoder3", "global_encoder1_1",
            "global_encoder2_3", "global_encoder3_2", "global_encoder4_1",
            "wave_att_input_t", "wave_att_f1", "wave_att_f2", "wave_att_f3",
            "TransTo_input", "TransTo1e", "TransTo2e", "TransTo3e", "outc_global",
        )
        missing_gradients = [
            prefix for prefix in gradient_prefixes
            if not has_nonzero_gradient(model, prefix)
        ]
        assert not missing_gradients, f"Missing gradients: {missing_gradients}"

    print(json.dumps({
        "status": "passed",
        "device": str(device),
        "full": args.full,
        "original_parameters": original_parameters,
        "wulle_parameters": wulle_parameters,
        "parameter_reduction": original_parameters - wulle_parameters,
        "parameter_reduction_percent": 100 * (original_parameters - wulle_parameters) / original_parameters,
        "haar": direction,
        "variants": variant_results,
        "gradient_prefixes_checked": list(gradient_prefixes),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
