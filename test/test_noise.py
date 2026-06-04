#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk BandLimitedNoise.

Run with:
    python -m pytest test_noise.py -v
    python test_noise.py
"""
import unittest

from pythontk import BandLimitedNoise

from conftest import BaseTestCase


class BandLimitedNoiseTest(BaseTestCase):
    def test_deterministic_from_seed(self):
        a = BandLimitedNoise(seed=7)
        b = BandLimitedNoise(seed=7)
        self.assertEqual(a.at(0.3, 0.6), b.at(0.3, 0.6))

    def test_seed_changes_field(self):
        a = BandLimitedNoise(seed=1)
        b = BandLimitedNoise(seed=2)
        self.assertNotAlmostEqual(a.at(0.3, 0.6), b.at(0.3, 0.6), places=6)

    def test_is_coherent_not_white_noise(self):
        # Band-limited: a tiny step changes the value only a little (white
        # per-vertex noise would jump arbitrarily), while still varying.
        n = BandLimitedNoise(seed=0)
        vals = [n.at(i / 500.0, 0.5) for i in range(501)]
        steps = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
        self.assertGreater(max(vals) - min(vals), 0.05)
        self.assertLess(max(steps), 0.1)

    def test_amplitude_roughly_bounded(self):
        # Normalized amplitudes -> magnitude stays ~[-1, 1] regardless of octaves.
        n = BandLimitedNoise(seed=3, octaves=6)
        peak = max(abs(n.at(i / 200.0, j / 200.0))
                   for i in range(201) for j in range(0, 201, 25))
        self.assertLessEqual(peak, 1.0 + 1e-9)

    def test_periodic_wraps_across_u_seam(self):
        n = BandLimitedNoise(seed=4, u_periodic=True)
        for v in (0.0, 0.5, 1.0):
            self.assertAlmostEqual(n.at(0.0, v), n.at(1.0, v), places=6)

    def test_non_periodic_does_not_wrap(self):
        n = BandLimitedNoise(seed=4, u_periodic=False)
        # Almost certainly discontinuous across the seam for fractional cycles.
        self.assertNotAlmostEqual(n.at(0.0, 0.5), n.at(1.0, 0.5), places=3)


if __name__ == "__main__":
    unittest.main(exit=False)
