# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.math_utils.weights (Weights) — the pure blendShape /
shape-key morph weight math shared by mayatk's and blendertk's
``blendshape_animator`` (they call ``Weights`` directly). Pinned DCC-free here in
its home; the DCC suites cover the animator that composes it.
"""
import unittest

from pythontk import Weights


class WeightsTest(unittest.TestCase):
    def test_round_weight_to_precision(self):
        self.assertEqual(Weights.round_weight(0.123456), 0.123)
        self.assertEqual(Weights.round_weight(1.0), 1.0)

    def test_frame_to_weight_clamps_and_interpolates(self):
        self.assertEqual(Weights.frame_to_weight(0, 10, 20), 0.0)   # below start
        self.assertEqual(Weights.frame_to_weight(10, 10, 20), 0.0)  # at start
        self.assertEqual(Weights.frame_to_weight(30, 10, 20), 1.0)  # above end
        self.assertEqual(Weights.frame_to_weight(20, 10, 20), 1.0)  # at end
        self.assertEqual(Weights.frame_to_weight(15, 10, 20), 0.5)  # midpoint

    def test_generate_weights_count_is_stable_across_endpoints(self):
        # Regression (mirrors mayatk's bug6): len == count regardless of include_endpoints.
        for n in (1, 2, 3, 5):
            self.assertEqual(len(Weights.generate_weights(n, include_endpoints=False)), n)
            self.assertEqual(len(Weights.generate_weights(n, include_endpoints=True)), n)

    def test_generate_weights_endpoints_exact_when_included(self):
        w = Weights.generate_weights(5, weight_range=(0.0, 1.0), include_endpoints=True)
        self.assertEqual(w[0], 0.0)
        self.assertEqual(w[-1], 1.0)

    def test_generate_weights_strictly_inside_when_excluded(self):
        w = Weights.generate_weights(3, weight_range=(0.0, 1.0), include_endpoints=False)
        self.assertTrue(all(0.0 < x < 1.0 for x in w))
        self.assertEqual(w, [0.25, 0.5, 0.75])

    def test_generate_weights_empty_for_nonpositive_count(self):
        self.assertEqual(Weights.generate_weights(0), [])
        self.assertEqual(Weights.generate_weights(-3), [])

    def test_generate_weights_single_with_endpoints(self):
        self.assertEqual(
            Weights.generate_weights(1, weight_range=(0.2, 0.8), include_endpoints=True),
            [0.2],
        )


if __name__ == "__main__":
    unittest.main()
