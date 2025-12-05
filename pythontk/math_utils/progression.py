#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Union, List

# From this package:
from pythontk.math_utils._math_utils import MathUtils


class ProgressionCurves:
    """A collection of mathematical progression curves for animations and transformations.

    This class provides various easing and interpolation functions that can be used
    individually or through the main calculate_progression_factor method.
    """

    # Available calculation modes for easy reference
    CALCULATION_MODES = [
        "linear",
        "exponential",
        "logarithmic",
        "sine",
        "ease_in",
        "ease_out",
        "ease_in_out",
        "smooth_step",
        "bounce",
        "elastic",
        "weighted",
    ]

    @staticmethod
    def _normalize_position(index: int, total_count: int) -> float:
        """Convert index to normalized position (0.0 to 1.0).

        Parameters:
            index (int): Current index in the progression (0-based)
            total_count (int): Total number of steps in the progression

        Returns:
            float: Normalized position value
        """
        if total_count <= 0:
            return 0.0
        # Use 1-based indexing for consistency with common animation workflows
        return (index + 1) / total_count

    @staticmethod
    def linear(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float:
        """Simple linear progression f(x) = x.

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Not used for linear progression
            weight_bias (float): Not used for linear progression

        Returns:
            float: Linear progression value
        """
        return x

    @staticmethod
    def exponential(
        x: float, weight_curve: float = 1.0, weight_bias: float = 0.5
    ) -> float:
        """Exponential progression f(x) = x^curve.

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength - affects steepness
            weight_bias (float): Not used for exponential progression

        Returns:
            float: Exponential progression value
        """
        return x**weight_curve

    @staticmethod
    def logarithmic(
        x: float, weight_curve: float = 1.0, weight_bias: float = 0.5
    ) -> float:
        """Logarithmic progression for smooth acceleration.

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength
            weight_bias (float): Not used for logarithmic progression

        Returns:
            float: Logarithmic progression value
        """
        if weight_curve <= 0:
            return x
        exp_curve = math.exp(weight_curve)
        return math.log(1 + x * (exp_curve - 1)) / weight_curve

    @staticmethod
    def sine(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float:
        """Sine-based progression for smooth curves.

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength
            weight_bias (float): Not used for sine progression

        Returns:
            float: Sine progression value
        """
        sine_val = math.sin(x * math.pi / 2)
        return sine_val**weight_curve if weight_curve > 0 else sine_val

    @staticmethod
    def ease_in(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float:
        """Ease-in: slow start, fast end (acceleration).

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength - higher values = stronger acceleration
            weight_bias (float): Not used for ease_in progression

        Returns:
            float: Ease-in progression value
        """
        return x**weight_curve

    @staticmethod
    def ease_out(
        x: float, weight_curve: float = 1.0, weight_bias: float = 0.5
    ) -> float:
        """Ease-out: fast start, slow end (deceleration).

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength - higher values = stronger deceleration
            weight_bias (float): Not used for ease_out progression

        Returns:
            float: Ease-out progression value
        """
        return 1 - (1 - x) ** weight_curve

    @staticmethod
    def ease_in_out(
        x: float, weight_curve: float = 1.0, weight_bias: float = 0.5
    ) -> float:
        """Ease-in-out: slow start and end, fast middle (S-curve).

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength
            weight_bias (float): Not used for ease_in_out progression

        Returns:
            float: Ease-in-out progression value
        """
        if x < 0.5:
            return 2 * (x**weight_curve) / (2 ** (weight_curve - 1))
        else:
            return 1 - 2 * ((1 - x) ** weight_curve) / (2 ** (weight_curve - 1))

    @staticmethod
    def smooth_step(
        x: float, weight_curve: float = 1.0, weight_bias: float = 0.5
    ) -> float:
        """Smooth step function using Hermite interpolation (3x² - 2x³).

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength (scaled for smooth step)
            weight_bias (float): Not used for smooth_step progression

        Returns:
            float: Smooth step progression value
        """
        smooth_x = 3 * x * x - 2 * x * x * x
        return smooth_x ** (weight_curve / 4)  # Adjust curve scaling for smooth step

    @staticmethod
    def bounce(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float:
        """Bouncing effect using sine waves.

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Bounce frequency
            weight_bias (float): Amplitude control (inverted)

        Returns:
            float: Bounce progression value
        """
        bounce_freq = weight_curve
        amplitude = 1 - weight_bias  # Use bias as amplitude control
        return x + amplitude * math.sin(x * bounce_freq * math.pi)

    @staticmethod
    def elastic(x: float, weight_curve: float = 1.0, weight_bias: float = 0.5) -> float:
        """Elastic effect with overshoot.

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Elastic frequency
            weight_bias (float): Decay control

        Returns:
            float: Elastic progression value
        """
        if x == 0 or x == 1:
            return x
        elastic_freq = weight_curve
        decay = weight_bias * 2  # Use bias as decay control
        return x + (1 - decay) * math.sin(x * elastic_freq * math.pi) * (1 - x)

    @staticmethod
    def weighted(
        x: float, weight_curve: float = 1.0, weight_bias: float = 0.5
    ) -> float:
        """Advanced weighted progression with bias control.

        Parameters:
            x (float): Normalized position (0.0 to 1.0)
            weight_curve (float): Curve strength
            weight_bias (float): Bias factor (0.0 to 1.0) - affects curve shape

        Returns:
            float: Weighted progression value
        """
        weight_factor = (
            2 * (weight_bias - 0.5) if weight_bias >= 0.5 else 2 * (0.5 - weight_bias)
        )

        # Calculate base curve
        if weight_bias >= 0.5:
            base_curve = x**weight_curve
        else:
            base_curve = 1 - (1 - x) ** weight_curve

        # Lerp between linear and curved progression
        return MathUtils.lerp(x, base_curve, weight_factor)

    @classmethod
    def calculate_progression_factor(
        cls,
        index: int,
        total_count: int,
        weight_bias: float = 0.5,
        weight_curve: float = 1.0,
        calculation_mode: str = "linear",
    ) -> float:
        """Calculate a progression factor using various mathematical functions.

        This utility provides different progression curves for animations, transformations,
        and other applications that need non-linear interpolation between values.

        Parameters:
            index (int): Current index in the progression (0-based)
            total_count (int): Total number of steps in the progression
            weight_bias (float): Bias factor (0.0 to 1.0) - affects curve shape for some modes
            weight_curve (float): Curve strength - affects steepness of progression
            calculation_mode (str): Type of progression curve to use

        Available calculation modes:
            - "linear": Simple linear progression f(x) = x
            - "exponential": Exponential curve f(x) = x^curve
            - "logarithmic": Logarithmic curve for smooth acceleration
            - "sine": Sine-based progression for smooth curves
            - "ease_in": Slow start, fast end (acceleration)
            - "ease_out": Fast start, slow end (deceleration)
            - "ease_in_out": Slow start and end, fast middle (S-curve)
            - "smooth_step": Hermite interpolation (3x² - 2x³)
            - "bounce": Bouncing effect using sine waves
            - "elastic": Elastic overshoot effect
            - "weighted": Advanced weighted progression with bias control

        Returns:
            float: Progression factor (typically 0.0 to 1.0, some modes may exceed range)

        Examples:
            # Linear progression for 5 steps
            for i in range(5):
                factor = ProgressionCurves.calculate_progression_factor(i, 5, calculation_mode="linear")

            # Ease-in curve with strong acceleration
            factor = ProgressionCurves.calculate_progression_factor(2, 10, weight_curve=3.0, calculation_mode="ease_in")

            # Bounce effect with custom frequency
            factor = ProgressionCurves.calculate_progression_factor(3, 8, weight_curve=2.0, calculation_mode="bounce")
        """
        x = cls._normalize_position(index, total_count)

        # Get the appropriate curve method
        curve_method = getattr(cls, calculation_mode, cls.linear)
        return curve_method(x, weight_curve, weight_bias)

    @classmethod
    def get_curve_function(cls, calculation_mode: str):
        """Get the curve function by name.

        Parameters:
            calculation_mode (str): Name of the curve function

        Returns:
            callable: The curve function, or linear if not found
        """
        return getattr(cls, calculation_mode, cls.linear)

    @classmethod
    def generate_curve_samples(
        cls,
        calculation_mode: str,
        num_samples: int = 100,
        weight_bias: float = 0.5,
        weight_curve: float = 1.0,
    ) -> List[float]:
        """Generate a list of samples from a curve for visualization or analysis.

        Parameters:
            calculation_mode (str): Type of progression curve
            num_samples (int): Number of samples to generate
            weight_bias (float): Bias factor for the curve
            weight_curve (float): Curve strength

        Returns:
            List[float]: List of curve values
        """
        curve_func = cls.get_curve_function(calculation_mode)
        return [
            curve_func(i / (num_samples - 1), weight_curve, weight_bias)
            for i in range(num_samples)
        ]


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ...

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
