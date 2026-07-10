import unittest
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.DWTFreqNet import (
    AxialFourDirectionMamba,
    DiagonalFourDirectionMamba,
    DiagonalIndexCache,
    WaveletEightDirectionAWGM,
    check_haar_direction_correspondence,
)


class CumulativeMixer(nn.Module):
    """Small stateful sequence mixer used to verify route ordering."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, sequence):
        return torch.cumsum(self.proj(sequence), dim=1)


class DiagonalIndexTests(unittest.TestCase):
    def test_concat_indices_for_three_by_three(self):
        cache = DiagonalIndexCache.build(3, 3, order="concat")
        self.assertEqual(cache["idx_nwse"].tolist(), [0, 4, 8, 1, 5, 2, 3, 7, 6])
        self.assertEqual(cache["idx_senw"].tolist(), [6, 7, 3, 2, 5, 1, 8, 4, 0])
        self.assertEqual(cache["idx_nesw"].tolist(), [2, 4, 6, 1, 3, 0, 5, 7, 8])
        self.assertEqual(cache["idx_swne"].tolist(), [8, 7, 5, 0, 3, 1, 6, 4, 2])

    def test_every_index_is_a_complete_permutation(self):
        for height, width in ((3, 3), (4, 4), (3, 5), (5, 3),
                              (16, 16), (32, 32), (64, 64)):
            for order in ("concat", "snake"):
                cache = DiagonalIndexCache.build(height, width, order=order)
                expected = list(range(height * width))
                for direction in ("nwse", "senw", "nesw", "swne"):
                    self.assertEqual(
                        sorted(cache["idx_" + direction].tolist()), expected
                    )

    def test_inverse_permutations_restore_original(self):
        original = torch.arange(15).reshape(1, 15, 1)
        cache = DiagonalIndexCache.build(3, 5, order="snake")
        for direction in ("nwse", "senw", "nesw", "swne"):
            permuted = original.index_select(1, cache["idx_" + direction])
            restored = permuted.index_select(1, cache["inv_" + direction])
            self.assertTrue(torch.equal(restored, original))


class HaarDirectionCorrespondenceTests(unittest.TestCase):
    def test_raw_haar_h_v_response_orientation(self):
        result = check_haar_direction_correspondence(size=32)
        self.assertEqual(
            result["band_response_orientation"],
            {"H": "vertical", "V": "horizontal"},
        )
        self.assertEqual(result["responses"]["horizontal_line"]["H"], 0.0)
        self.assertGreater(result["responses"]["horizontal_line"]["V"], 0.0)
        self.assertGreater(result["responses"]["vertical_line"]["H"], 0.0)
        self.assertEqual(result["responses"]["vertical_line"]["V"], 0.0)


class SharingAndGradientTests(unittest.TestCase):
    def test_axial_diagonal_mode_has_two_shared_mambas(self):
        module = WaveletEightDirectionAWGM(
            8, share_mode="axial_diag_shared_2", allow_fallback=True
        )
        axial = module.axial_branch
        diagonal = module.diagonal_branch
        for direction in axial.DIRECTIONS:
            self.assertIs(axial.axial_mamba, axial.get_mamba(direction))
        for direction in diagonal.directions:
            self.assertIs(diagonal.diag_mamba, diagonal.get_mamba(direction))
        self.assertEqual(module.mamba_instance_count, 2)

    def test_all_shared_mode_has_one_mamba(self):
        module = WaveletEightDirectionAWGM(
            8, share_mode="all_shared_1", allow_fallback=True
        )
        mixers = [
            module.axial_branch.get_mamba(direction)
            for direction in module.axial_branch.DIRECTIONS
        ]
        mixers.extend(
            module.diagonal_branch.get_mamba(direction)
            for direction in module.diagonal_branch.directions
        )
        self.assertTrue(all(mixer is mixers[0] for mixer in mixers))
        self.assertEqual(module.mamba_instance_count, 1)

    def test_shared_route_outputs_differ_and_gradients_accumulate(self):
        dim = 4
        mixer = CumulativeMixer(dim)
        axial = AxialFourDirectionMamba(
            dim,
            share_mode="all_shared_1",
            shared_mamba=mixer,
            shared_backend="test_cumulative",
        )
        diagonal = DiagonalFourDirectionMamba(
            dim,
            share_mode="all_shared_1",
            shared_mamba=mixer,
            shared_backend="test_cumulative",
        )
        horizontal = torch.randn(2, dim, 4, 4, requires_grad=True)
        vertical = torch.randn(2, dim, 4, 4, requires_grad=True)
        h_out, v_out, axial_routes = axial(
            horizontal, vertical, return_routes=True
        )
        d_out, diagonal_routes = diagonal(horizontal, return_routes=True)
        self.assertFalse(torch.allclose(axial_routes["lr"], axial_routes["rl"]))
        self.assertFalse(
            torch.allclose(diagonal_routes["nwse"], diagonal_routes["nesw"])
        )
        (h_out.mean() + v_out.mean() + d_out.mean()).backward()
        gradient = mixer.proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
