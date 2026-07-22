"""Acceptance tests for Phase 1 offline task-prior validation."""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.phase1.common import (
    bilinear_sample,
    bootstrap_effect_ci,
    build_candidate_catalog,
    calibrate_haar,
    center_perturbations,
    connected_components,
    elliptical_gaussian,
    fit_gaussian_patch,
    geometry_points,
    paired_consistency,
    quadrant_consistency,
)


class Phase1CommonTests(unittest.TestCase):
    def test_connected_components_are_eight_connected(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2, 2] = True
        mask[3, 3] = True
        _, components = connected_components(mask)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["area"], 2)

    def test_hard_negatives_respect_exclusion(self):
        image = np.zeros((64, 64), dtype=float)
        mask = np.zeros_like(image, dtype=bool)
        mask[29:34, 29:34] = True
        image[mask] = 10
        image[8, 8] = 9
        image[50, 45] = 8
        candidates, exclusion = build_candidate_catalog(
            image, mask, "synthetic", "train", "sample", hard_per_target=2,
            easy_per_target=1, seed=42,
        )
        hard = [row for row in candidates if row["sample_type"] == "hard_negative"]
        self.assertGreaterEqual(len(hard), 1)
        for row in hard:
            self.assertFalse(exclusion[round(row["center_y"]), round(row["center_x"])])

    def test_gaussian_parameter_recovery(self):
        yy, xx = np.indices((41, 41), dtype=float)
        truth = np.array([7.0, 20.7, 19.2, 4.5, 2.2, 0.35, 0.1])
        rng = np.random.default_rng(42)
        patch = elliptical_gaussian(truth, xx, yy) + rng.normal(0, 0.03, xx.shape)
        result = fit_gaussian_patch(patch, expected_center=(20.7, 19.2))
        self.assertTrue(result["fit_success"])
        self.assertLess(abs(result["sigma_major"] - 4.5), 0.5)
        self.assertLess(abs(result["sigma_minor"] - 2.2), 0.4)
        self.assertLess(result["fitted_center_offset_to_reference"], 0.4)
        self.assertGreater(result["R2"], 0.98)
        self.assertGreaterEqual(result["radial_monotonicity"], 0.75)

    def test_pair_consistency_formula(self):
        same = paired_consistency(np.array([1.0, -2.0]), np.array([1.0, -2.0]), 1)
        opposite = paired_consistency(np.array([1.0, -2.0]), np.array([-1.0, 2.0]), -1)
        wrong = paired_consistency(np.array([1.0, -2.0]), np.array([-1.0, 2.0]), 1)
        self.assertAlmostEqual(same, 1.0, places=6)
        self.assertAlmostEqual(opposite, 1.0, places=6)
        self.assertLess(wrong, same)

    def test_quadrant_template(self):
        values = np.array([[2.0, -1.9, 2.1, -2.0]])
        result = quadrant_consistency(values, [1, -1, 1, -1])
        self.assertGreater(result["cosine"], 0.99)
        self.assertEqual(result["sign_agreement"], 1.0)
        self.assertGreater(result["amplitude_balance"], 0.8)

    def test_all_sampling_geometries_are_fair(self):
        for radius in (2, 3, 4, 5):
            for geometry in ("grid", "ring", "spiral", "random", "gaussian_radial"):
                points = geometry_points(geometry, radius, 32, seed=42)
                self.assertEqual(points.shape, (32, 2))
                self.assertAlmostEqual(np.linalg.norm(points, axis=1).max(), radius, places=7)

    def test_bilinear_sampling(self):
        feature = np.arange(16, dtype=float).reshape(4, 4)
        sampled = bilinear_sample(feature, [[1.5, 1.5]])
        self.assertAlmostEqual(sampled[0, 0], 7.5, places=6)

    def test_center_perturbations_include_required_magnitudes(self):
        perturbations = center_perturbations(42)
        magnitudes = {row["magnitude"] for row in perturbations}
        self.assertEqual(magnitudes, {0.0, 0.5, 1.0, 2.0})
        self.assertTrue(any(row["name"].startswith("random") for row in perturbations))

    def test_bootstrap_is_reproducible(self):
        x = np.arange(20, dtype=float)
        y = np.arange(20, dtype=float) - 1
        self.assertEqual(
            bootstrap_effect_ci(x, y, repeats=50, seed=7),
            bootstrap_effect_ci(x, y, repeats=50, seed=7),
        )

    def test_direction_calibration_is_explicit(self):
        calibration = calibrate_haar(size=32, device="cpu")
        self.assertIn(calibration["band_response_orientation"]["H"], ("horizontal", "vertical"))
        self.assertNotEqual(
            calibration["band_response_orientation"]["H"],
            calibration["band_response_orientation"]["V"],
        )
        self.assertEqual(len(calibration["d_quadrant_template"]), 4)
        self.assertTrue(calibration["routing_aligned"])

    def test_model_directory_is_unchanged_from_phase1_base(self):
        root = Path(__file__).resolve().parents[2]
        if not (root / ".git").exists():
            self.skipTest("deployment copy has no Git metadata; verify with directory diff")
        output = subprocess.check_output(
            ["git", "diff", "8cfd7a97bd460b07efbad28ca7b709d7277cdd1b", "--", "model/"],
            cwd=root, text=True,
        )
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
