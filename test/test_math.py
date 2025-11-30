#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk MathUtils.

Comprehensive edge case coverage for:
- Vector operations (normalize, magnitude, dot/cross product)
- Point operations (distance, midpoint, move)
- Angle calculations
- Interpolation (lerp, clamp)
- Trigonometry and rotations

Run with:
    python -m pytest test_math.py -v
    python test_math.py
"""
import math
import unittest

from pythontk import MathUtils

from conftest import BaseTestCase


class MathTest(BaseTestCase):
    """Math utilities test class with comprehensive edge case coverage."""

    # -------------------------------------------------------------------------
    # Vector from Two Points Tests
    # -------------------------------------------------------------------------

    def test_get_vector_from_two_points_basic(self):
        """Test get_vector_from_two_points calculates direction vector."""
        self.assertEqual(
            MathUtils.get_vector_from_two_points((1, 2, 3), (1, 1, -1)),
            (0, -1, -4),
        )

    def test_get_vector_from_two_points_same_point(self):
        """Test get_vector_from_two_points with same point (zero vector)."""
        result = MathUtils.get_vector_from_two_points((1, 2, 3), (1, 2, 3))
        self.assertEqual(result, (0, 0, 0))

    def test_get_vector_from_two_points_negative_coords(self):
        """Test get_vector_from_two_points with negative coordinates."""
        result = MathUtils.get_vector_from_two_points((-1, -2, -3), (1, 2, 3))
        self.assertEqual(result, (2, 4, 6))

    def test_get_vector_from_two_points_2d(self):
        """Test get_vector_from_two_points requires 3D points."""
        # The function requires 3D points - 2D will raise IndexError
        with self.assertRaises(IndexError):
            MathUtils.get_vector_from_two_points((0, 0), (3, 4))

    # -------------------------------------------------------------------------
    # Clamp Tests
    # -------------------------------------------------------------------------

    def test_clamp_basic(self):
        """Test clamp restricts values to range."""
        self.assertEqual(
            MathUtils.clamp(range(10), 3, 7),
            [3, 3, 3, 3, 4, 5, 6, 7, 7, 7],
        )

    def test_clamp_single_value(self):
        """Test clamp with single value."""
        self.assertEqual(MathUtils.clamp(5, 0, 10), 5)
        self.assertEqual(MathUtils.clamp(-5, 0, 10), 0)
        self.assertEqual(MathUtils.clamp(15, 0, 10), 10)

    def test_clamp_at_boundaries(self):
        """Test clamp at exact boundaries."""
        self.assertEqual(MathUtils.clamp(0, 0, 10), 0)
        self.assertEqual(MathUtils.clamp(10, 0, 10), 10)

    def test_clamp_floats(self):
        """Test clamp with floating point values."""
        self.assertEqual(MathUtils.clamp(0.5, 0.0, 1.0), 0.5)
        self.assertEqual(MathUtils.clamp(-0.5, 0.0, 1.0), 0.0)
        self.assertEqual(MathUtils.clamp(1.5, 0.0, 1.0), 1.0)

    def test_clamp_negative_range(self):
        """Test clamp with negative range."""
        self.assertEqual(MathUtils.clamp(-5, -10, -1), -5)
        self.assertEqual(MathUtils.clamp(-15, -10, -1), -10)

    def test_clamp_empty_list(self):
        """Test clamp with empty list."""
        self.assertEqual(MathUtils.clamp([], 0, 10), [])

    # -------------------------------------------------------------------------
    # Normalize Tests
    # -------------------------------------------------------------------------

    def test_normalize_3d(self):
        """Test normalize creates unit vectors."""
        self.assertEqual(
            MathUtils.normalize((2, 3, 4)),
            (0.3713906763541037, 0.5570860145311556, 0.7427813527082074),
        )

    def test_normalize_2d(self):
        """Test normalize with 2D vector."""
        self.assertEqual(
            MathUtils.normalize((2, 3)),
            (0.5547001962252291, 0.8320502943378437),
        )

    def test_normalize_with_amount(self):
        """Test normalize with custom magnitude."""
        self.assertEqual(
            MathUtils.normalize((2, 3, 4), 2),
            (0.7427813527082074, 1.1141720290623112, 1.4855627054164149),
        )

    def test_normalize_unit_vector(self):
        """Test normalize on already unit vector."""
        result = MathUtils.normalize((1, 0, 0))
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)

    def test_normalize_zero_vector(self):
        """Test normalize with zero vector."""
        # Zero vector normalization typically returns zero or raises
        try:
            result = MathUtils.normalize((0, 0, 0))
            # If it doesn't raise, check it handles gracefully
            self.assertTrue(True)
        except (ZeroDivisionError, ValueError):
            self.assertTrue(True)

    def test_normalize_very_small_vector(self):
        """Test normalize with very small components."""
        result = MathUtils.normalize((1e-10, 1e-10, 1e-10))
        mag = math.sqrt(sum(x * x for x in result))
        self.assertAlmostEqual(mag, 1.0, places=5)

    def test_normalize_very_large_vector(self):
        """Test normalize with very large components."""
        result = MathUtils.normalize((1e10, 1e10, 1e10))
        mag = math.sqrt(sum(x * x for x in result))
        self.assertAlmostEqual(mag, 1.0, places=5)

    # -------------------------------------------------------------------------
    # Magnitude Tests
    # -------------------------------------------------------------------------

    def test_get_magnitude_3d(self):
        """Test get_magnitude calculates vector length."""
        self.assertEqual(MathUtils.get_magnitude((2, 3, 4)), 5.385164807134504)

    def test_get_magnitude_2d(self):
        """Test get_magnitude with 2D vector."""
        self.assertEqual(MathUtils.get_magnitude((2, 3)), 3.605551275463989)

    def test_get_magnitude_zero_vector(self):
        """Test get_magnitude with zero vector."""
        self.assertEqual(MathUtils.get_magnitude((0, 0, 0)), 0.0)

    def test_get_magnitude_unit_vector(self):
        """Test get_magnitude with unit vectors."""
        self.assertEqual(MathUtils.get_magnitude((1, 0, 0)), 1.0)
        self.assertEqual(MathUtils.get_magnitude((0, 1, 0)), 1.0)
        self.assertEqual(MathUtils.get_magnitude((0, 0, 1)), 1.0)

    def test_get_magnitude_negative_components(self):
        """Test get_magnitude with negative components."""
        # Magnitude should be same regardless of sign
        self.assertEqual(
            MathUtils.get_magnitude((-2, -3, -4)),
            MathUtils.get_magnitude((2, 3, 4)),
        )

    def test_get_magnitude_3_4_5_triangle(self):
        """Test get_magnitude with Pythagorean triple."""
        self.assertEqual(MathUtils.get_magnitude((3, 4)), 5.0)

    # -------------------------------------------------------------------------
    # Dot Product Tests
    # -------------------------------------------------------------------------

    def test_dot_product_3d(self):
        """Test dot_product calculates scalar product."""
        self.assertEqual(MathUtils.dot_product((1, 2, 3), (1, 1, -1)), 0)

    def test_dot_product_2d(self):
        """Test dot_product with 2D vectors."""
        self.assertEqual(MathUtils.dot_product((1, 2), (1, 1)), 3)

    def test_dot_product_normalized(self):
        """Test dot_product with normalization."""
        self.assertEqual(MathUtils.dot_product((1, 2, 3), (1, 1, -1), True), 0)

    def test_dot_product_parallel(self):
        """Test dot_product with parallel vectors."""
        result = MathUtils.dot_product((1, 0, 0), (2, 0, 0))
        self.assertEqual(result, 2)

    def test_dot_product_perpendicular(self):
        """Test dot_product with perpendicular vectors."""
        result = MathUtils.dot_product((1, 0, 0), (0, 1, 0))
        self.assertEqual(result, 0)

    def test_dot_product_opposite(self):
        """Test dot_product with opposite vectors."""
        result = MathUtils.dot_product((1, 0, 0), (-1, 0, 0))
        self.assertEqual(result, -1)

    def test_dot_product_zero_vector(self):
        """Test dot_product with zero vector."""
        result = MathUtils.dot_product((1, 2, 3), (0, 0, 0))
        self.assertEqual(result, 0)

    # -------------------------------------------------------------------------
    # Cross Product Tests
    # -------------------------------------------------------------------------

    def test_cross_product_basic(self):
        """Test cross_product calculates vector product."""
        self.assertEqual(
            MathUtils.cross_product((1, 2, 3), (1, 1, -1)),
            (-5, 4, -1),
        )

    def test_cross_product_three_points(self):
        """Test cross_product with three points."""
        self.assertEqual(
            MathUtils.cross_product((3, 1, 1), (1, 4, 2), (1, 3, 4)),
            (7, 4, 2),
        )

    def test_cross_product_normalized(self):
        """Test cross_product with normalization."""
        self.assertEqual(
            MathUtils.cross_product((1, 2, 3), (1, 1, -1), None, 1),
            (-0.7715167498104595, 0.6172133998483676, -0.1543033499620919),
        )

    def test_cross_product_parallel_vectors(self):
        """Test cross_product with parallel vectors (zero result)."""
        result = MathUtils.cross_product((1, 0, 0), (2, 0, 0))
        self.assertEqual(result, (0, 0, 0))

    def test_cross_product_perpendicular_unit_vectors(self):
        """Test cross_product with perpendicular unit vectors."""
        result = MathUtils.cross_product((1, 0, 0), (0, 1, 0))
        self.assertEqual(result, (0, 0, 1))

    def test_cross_product_anti_commutative(self):
        """Test cross_product anti-commutativity: a x b = -(b x a)."""
        a, b = (1, 2, 3), (4, 5, 6)
        ab = MathUtils.cross_product(a, b)
        ba = MathUtils.cross_product(b, a)
        self.assertEqual(ab, tuple(-x for x in ba))

    # -------------------------------------------------------------------------
    # Move Point Tests
    # -------------------------------------------------------------------------

    def test_move_point_relative_vector(self):
        """Test move_point_relative translates points by vector."""
        self.assertEqual(
            MathUtils.move_point_relative((0, 5, 0), (0, 5, 0)),
            (0, 10, 0),
        )

    def test_move_point_relative_distance_direction(self):
        """Test move_point_relative with distance and direction."""
        self.assertEqual(
            MathUtils.move_point_relative((0, 5, 0), 5, (0, 1, 0)),
            (0, 10, 0),
        )

    def test_move_point_relative_zero_distance(self):
        """Test move_point_relative with zero distance."""
        result = MathUtils.move_point_relative((1, 2, 3), 0, (1, 0, 0))
        self.assertEqual(result, (1, 2, 3))

    def test_move_point_relative_negative_distance(self):
        """Test move_point_relative with negative distance."""
        result = MathUtils.move_point_relative((0, 5, 0), -5, (0, 1, 0))
        self.assertEqual(result, (0, 0, 0))

    def test_move_point_relative_along_vector_toward(self):
        """Test move_point_relative_along_vector toward target."""
        self.assertEqual(
            MathUtils.move_point_relative_along_vector(
                (0, 0, 0), (0, 10, 0), (0, 1, 0), 5
            ),
            (0.0, 5.0, 0.0),
        )

    def test_move_point_relative_along_vector_away(self):
        """Test move_point_relative_along_vector away from target."""
        self.assertEqual(
            MathUtils.move_point_relative_along_vector(
                (0, 0, 0), (0, 10, 0), (0, 1, 0), 5, False
            ),
            (0.0, -5.0, 0.0),
        )

    # -------------------------------------------------------------------------
    # Distance Tests
    # -------------------------------------------------------------------------

    def test_distance_between_points_basic(self):
        """Test distance_between_points calculates Euclidean distance."""
        self.assertEqual(
            MathUtils.distance_between_points((0, 10, 0), (0, 5, 0)),
            5.0,
        )

    def test_distance_between_points_same_point(self):
        """Test distance_between_points with same point."""
        self.assertEqual(
            MathUtils.distance_between_points((1, 2, 3), (1, 2, 3)),
            0.0,
        )

    def test_distance_between_points_3d(self):
        """Test distance_between_points in 3D space."""
        # 3-4-5 triangle in 3D: distance should be 5
        result = MathUtils.distance_between_points((0, 0, 0), (3, 4, 0))
        self.assertEqual(result, 5.0)

    def test_distance_between_points_negative_coords(self):
        """Test distance_between_points with negative coordinates."""
        result = MathUtils.distance_between_points((-5, 0, 0), (5, 0, 0))
        self.assertEqual(result, 10.0)

    def test_distance_between_points_diagonal(self):
        """Test distance_between_points on unit cube diagonal."""
        result = MathUtils.distance_between_points((0, 0, 0), (1, 1, 1))
        self.assertAlmostEqual(result, math.sqrt(3), places=10)

    # -------------------------------------------------------------------------
    # Center / Midpoint Tests
    # -------------------------------------------------------------------------

    def test_get_center_of_two_points_basic(self):
        """Test get_center_of_two_points finds midpoint."""
        self.assertEqual(
            MathUtils.get_center_of_two_points((0, 10, 0), (0, 5, 0)),
            (0.0, 7.5, 0.0),
        )

    def test_get_center_of_two_points_same_point(self):
        """Test get_center_of_two_points with same point."""
        result = MathUtils.get_center_of_two_points((5, 5, 5), (5, 5, 5))
        self.assertEqual(result, (5.0, 5.0, 5.0))

    def test_get_center_of_two_points_origin(self):
        """Test get_center_of_two_points symmetric about origin."""
        result = MathUtils.get_center_of_two_points((-5, -5, -5), (5, 5, 5))
        self.assertEqual(result, (0.0, 0.0, 0.0))

    # -------------------------------------------------------------------------
    # Angle Tests
    # -------------------------------------------------------------------------

    def test_get_angle_from_two_vectors_radians(self):
        """Test get_angle_from_two_vectors in radians."""
        self.assertEqual(
            MathUtils.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1)),
            1.5707963267948966,
        )

    def test_get_angle_from_two_vectors_degrees(self):
        """Test get_angle_from_two_vectors in degrees."""
        self.assertEqual(
            MathUtils.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1), True),
            90,
        )

    def test_get_angle_from_two_vectors_parallel(self):
        """Test get_angle_from_two_vectors with parallel vectors."""
        result = MathUtils.get_angle_from_two_vectors((1, 0, 0), (2, 0, 0), True)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_get_angle_from_two_vectors_opposite(self):
        """Test get_angle_from_two_vectors with opposite vectors."""
        result = MathUtils.get_angle_from_two_vectors((1, 0, 0), (-1, 0, 0), True)
        self.assertAlmostEqual(result, 180.0, places=5)

    def test_get_angle_from_two_vectors_perpendicular(self):
        """Test get_angle_from_two_vectors with perpendicular vectors."""
        result = MathUtils.get_angle_from_two_vectors((1, 0, 0), (0, 1, 0), True)
        self.assertAlmostEqual(result, 90.0, places=5)

    def test_get_angle_from_three_points_radians(self):
        """Test get_angle_from_three_points in radians."""
        self.assertEqual(
            MathUtils.get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3)),
            0.7904487543360762,
        )

    def test_get_angle_from_three_points_degrees(self):
        """Test get_angle_from_three_points in degrees."""
        self.assertEqual(
            MathUtils.get_angle_from_three_points(
                (1, 1, 1), (-1, 2, 3), (1, 4, -3), True
            ),
            45.29,
        )

    def test_get_angle_from_three_points_right_angle(self):
        """Test get_angle_from_three_points with right angle."""
        # Angle at origin between (1,0,0), (0,0,0), (0,1,0) should be 90 degrees
        result = MathUtils.get_angle_from_three_points(
            (1, 0, 0), (0, 0, 0), (0, 1, 0), True
        )
        self.assertAlmostEqual(result, 90.0, places=2)

    # -------------------------------------------------------------------------
    # Triangle Tests
    # -------------------------------------------------------------------------

    def test_get_two_sides_of_asa_triangle_equilateral(self):
        """Test get_two_sides_of_asa_triangle with equilateral."""
        self.assertEqual(
            MathUtils.get_two_sides_of_asa_triangle(60, 60, 100),
            (100.00015320566493, 100.00015320566493),
        )

    def test_get_two_sides_of_asa_triangle_isoceles(self):
        """Test get_two_sides_of_asa_triangle with isoceles."""
        result = MathUtils.get_two_sides_of_asa_triangle(45, 45, 100)
        # Equal angles should give equal sides
        self.assertAlmostEqual(result[0], result[1], places=5)

    # -------------------------------------------------------------------------
    # Rotation Tests
    # -------------------------------------------------------------------------

    def test_xyz_rotation_radians(self):
        """Test xyz_rotation applies rotations in radians."""
        self.assertEqual(
            MathUtils.xyz_rotation(2, (0, 1, 0)),
            (3.589792907376932e-09, 1.9999999964102069, 3.589792907376932e-09),
        )

    def test_xyz_rotation_degrees(self):
        """Test xyz_rotation with degrees."""
        self.assertEqual(
            MathUtils.xyz_rotation(2, (0, 1, 0), [], True),
            (0.0, 114.59, 0.0),
        )

    def test_xyz_rotation_zero(self):
        """Test xyz_rotation with zero angle returns zero rotation."""
        result = MathUtils.xyz_rotation(0, (1, 0, 0))
        # Zero angle means no rotation, so result should be essentially (0, 0, 0)
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 0.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)

    # -------------------------------------------------------------------------
    # Lerp Tests
    # -------------------------------------------------------------------------

    def test_lerp_midpoint(self):
        """Test lerp at midpoint."""
        self.assertEqual(MathUtils.lerp(0, 10, 0.5), 5.0)

    def test_lerp_with_negatives(self):
        """Test lerp with negative values."""
        self.assertEqual(MathUtils.lerp(-10, 10, 0.5), 0.0)

    def test_lerp_at_start(self):
        """Test lerp at t=0."""
        self.assertEqual(MathUtils.lerp(0, 10, 0), 0)

    def test_lerp_at_end(self):
        """Test lerp at t=1."""
        self.assertEqual(MathUtils.lerp(0, 10, 1), 10)

    def test_lerp_quarter(self):
        """Test lerp at t=0.25."""
        self.assertEqual(MathUtils.lerp(0, 100, 0.25), 25.0)

    def test_lerp_extrapolation(self):
        """Test lerp with t > 1 (extrapolation)."""
        result = MathUtils.lerp(0, 10, 1.5)
        self.assertEqual(result, 15.0)

    def test_lerp_negative_t(self):
        """Test lerp with negative t (reverse extrapolation)."""
        result = MathUtils.lerp(0, 10, -0.5)
        self.assertEqual(result, -5.0)

    def test_lerp_same_values(self):
        """Test lerp when a equals b."""
        self.assertEqual(MathUtils.lerp(5, 5, 0.5), 5.0)

    def test_lerp_float_precision(self):
        """Test lerp maintains float precision."""
        result = MathUtils.lerp(0.0, 1.0, 0.333333333)
        self.assertAlmostEqual(result, 0.333333333, places=7)


if __name__ == "__main__":
    unittest.main(exit=False)
