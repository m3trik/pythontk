#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk ProgressionCurves.

Run with:
    python -m pytest test_progression.py -v
    python test_progression.py
"""
import unittest

from pythontk.math_utils.progression import ProgressionCurves

from conftest import BaseTestCase


class ProgressionCurvesTest(BaseTestCase):
    """Tests for ProgressionCurves class."""

    def test_normalize_position(self):
        """Test position normalization."""
        self.assertEqual(ProgressionCurves._normalize_position(0, 5), 0.2)
        self.assertEqual(ProgressionCurves._normalize_position(4, 5), 1.0)

    def test_normalize_position_zero_count(self):
        """Test normalization with zero count."""
        self.assertEqual(ProgressionCurves._normalize_position(0, 0), 0.0)

    def test_linear(self):
        """Test linear progression."""
        self.assertEqual(ProgressionCurves.linear(0.0), 0.0)
        self.assertEqual(ProgressionCurves.linear(0.5), 0.5)
        self.assertEqual(ProgressionCurves.linear(1.0), 1.0)

    def test_exponential(self):
        """Test exponential progression."""
        self.assertEqual(ProgressionCurves.exponential(0.0, weight_curve=2.0), 0.0)
        self.assertEqual(ProgressionCurves.exponential(1.0, weight_curve=2.0), 1.0)
        self.assertEqual(ProgressionCurves.exponential(0.5, weight_curve=2.0), 0.25)

    def test_logarithmic(self):
        """Test logarithmic progression."""
        result = ProgressionCurves.logarithmic(0.5, weight_curve=1.0)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_logarithmic_zero_curve(self):
        """Test logarithmic with zero curve returns linear."""
        self.assertEqual(ProgressionCurves.logarithmic(0.5, weight_curve=0.0), 0.5)

    def test_sine(self):
        """Test sine progression."""
        self.assertEqual(ProgressionCurves.sine(0.0), 0.0)
        self.assertAlmostEqual(ProgressionCurves.sine(1.0), 1.0, places=5)

    def test_ease_in(self):
        """Test ease-in progression."""
        self.assertEqual(ProgressionCurves.ease_in(0.0), 0.0)
        self.assertEqual(ProgressionCurves.ease_in(1.0), 1.0)
        # With curve > 1, middle value should be less than linear
        self.assertLess(ProgressionCurves.ease_in(0.5, weight_curve=2.0), 0.5)

    def test_ease_out(self):
        """Test ease-out progression."""
        self.assertEqual(ProgressionCurves.ease_out(0.0), 0.0)
        self.assertEqual(ProgressionCurves.ease_out(1.0), 1.0)
        # With curve > 1, middle value should be greater than linear
        self.assertGreater(ProgressionCurves.ease_out(0.5, weight_curve=2.0), 0.5)

    def test_ease_in_out(self):
        """Test ease-in-out progression."""
        self.assertEqual(ProgressionCurves.ease_in_out(0.0), 0.0)
        self.assertEqual(ProgressionCurves.ease_in_out(1.0), 1.0)
        # At x=0.5 with default weight_curve=1.0, result depends on implementation
        result = ProgressionCurves.ease_in_out(0.5)
        self.assertIsInstance(result, float)

    def test_smooth_step(self):
        """Test smooth step progression."""
        self.assertEqual(ProgressionCurves.smooth_step(0.0), 0.0)
        self.assertAlmostEqual(ProgressionCurves.smooth_step(1.0), 1.0, places=5)
        # Smooth step at 0.5 produces 0.5 for the base curve (3*0.25 - 2*0.125 = 0.5)
        # but then applies weight_curve power transformation
        result = ProgressionCurves.smooth_step(0.5)
        self.assertIsInstance(result, float)

    def test_bounce(self):
        """Test bounce progression."""
        result = ProgressionCurves.bounce(0.5, weight_curve=1.0)
        self.assertIsInstance(result, float)

    def test_elastic(self):
        """Test elastic progression."""
        self.assertEqual(ProgressionCurves.elastic(0.0), 0.0)
        self.assertEqual(ProgressionCurves.elastic(1.0), 1.0)

    def test_weighted(self):
        """Test weighted progression."""
        result = ProgressionCurves.weighted(0.5, weight_curve=1.0, weight_bias=0.5)
        self.assertIsInstance(result, float)

    def test_calculate_progression_factor_linear(self):
        """Test calculate_progression_factor with linear mode."""
        factor = ProgressionCurves.calculate_progression_factor(
            0, 5, calculation_mode="linear"
        )
        self.assertEqual(factor, 0.2)

    def test_calculate_progression_factor_exponential(self):
        """Test calculate_progression_factor with exponential mode."""
        factor = ProgressionCurves.calculate_progression_factor(
            0, 5, weight_curve=2.0, calculation_mode="exponential"
        )
        self.assertAlmostEqual(factor, 0.04, places=5)  # 0.2^2

    def test_calculate_progression_factor_invalid_mode(self):
        """Test calculate_progression_factor with invalid mode falls back to linear."""
        factor = ProgressionCurves.calculate_progression_factor(
            0, 5, calculation_mode="invalid_mode"
        )
        self.assertEqual(factor, 0.2)  # Falls back to linear

    def test_get_curve_function(self):
        """Test get_curve_function returns correct function."""
        func = ProgressionCurves.get_curve_function("linear")
        self.assertEqual(func, ProgressionCurves.linear)

        func = ProgressionCurves.get_curve_function("exponential")
        self.assertEqual(func, ProgressionCurves.exponential)

    def test_get_curve_function_invalid(self):
        """Test get_curve_function returns linear for invalid name."""
        func = ProgressionCurves.get_curve_function("invalid")
        self.assertEqual(func, ProgressionCurves.linear)

    def test_generate_curve_samples(self):
        """Test generate_curve_samples."""
        samples = ProgressionCurves.generate_curve_samples("linear", num_samples=10)

        self.assertEqual(len(samples), 10)
        self.assertEqual(samples[0], 0.0)
        self.assertEqual(samples[-1], 1.0)

    def test_generate_curve_samples_exponential(self):
        """Test generate_curve_samples with exponential."""
        samples = ProgressionCurves.generate_curve_samples(
            "exponential", num_samples=5, weight_curve=2.0
        )

        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0], 0.0)
        self.assertEqual(samples[-1], 1.0)

    def test_calculation_modes_list(self):
        """Test that all calculation modes are valid."""
        for mode in ProgressionCurves.CALCULATION_MODES:
            func = ProgressionCurves.get_curve_function(mode)
            self.assertIsNotNone(func)
            # Test that function is callable
            result = func(0.5)
            self.assertIsInstance(result, float)

    def test_ease_in_out_continuous_and_monotonic(self):
        """Regression: ease_in_out must be continuous/monotonic (was discontinuous).

        Previously the coefficient was inverted, so the midpoint snapped
        backward: ease_in_out(0.5) returned 0.0 and ease_in_out(0.49) returned
        0.98. With default weight_curve=1.0 the curve should reduce to identity.
        """
        # Midpoint no longer collapses to 0.0.
        self.assertAlmostEqual(ProgressionCurves.ease_in_out(0.5), 0.5, places=7)
        # Default weight_curve=1.0 => identity, so it is monotonically increasing.
        self.assertAlmostEqual(ProgressionCurves.ease_in_out(0.49), 0.49, places=7)
        self.assertAlmostEqual(ProgressionCurves.ease_in_out(0.51), 0.51, places=7)
        # No backward jump across the midpoint.
        samples = [ProgressionCurves.ease_in_out(i / 100.0) for i in range(101)]
        for prev, curr in zip(samples, samples[1:]):
            self.assertLessEqual(prev, curr + 1e-9)
        # weight_curve=2.0 (easeInOutQuad) is also continuous at the midpoint.
        self.assertAlmostEqual(
            ProgressionCurves.ease_in_out(0.5, weight_curve=2.0), 0.5, places=7
        )

    def test_generate_curve_samples_single_sample(self):
        """Regression: num_samples=1 must not raise ZeroDivisionError."""
        samples = ProgressionCurves.generate_curve_samples("linear", num_samples=1)
        self.assertEqual(samples, [ProgressionCurves.linear(0.0)])

    def test_generate_curve_samples_zero_samples(self):
        """Regression: num_samples<=0 returns an empty list, no crash."""
        self.assertEqual(
            ProgressionCurves.generate_curve_samples("linear", num_samples=0), []
        )


if __name__ == "__main__":
    unittest.main(exit=False)
