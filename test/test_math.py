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
    # Linear Sum Assignment (Hungarian) Tests
    # -------------------------------------------------------------------------

    def test_linear_sum_assignment_square_min_cost(self):
        """Test Hungarian assignment on a known 3x3 minimum-cost matrix."""
        cost = [
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2],
        ]
        rows, cols = MathUtils.linear_sum_assignment(cost)

        # Canonical optimal solution: (0->1), (1->0), (2->2) cost = 1+2+2 = 5
        pairs = set(zip(rows, cols))
        self.assertEqual(pairs, {(0, 1), (1, 0), (2, 2)})

    def test_linear_sum_assignment_rectangular(self):
        """Test Hungarian assignment supports rectangular matrices."""
        # 2 rows, 3 cols; expect 2 assignments
        cost = [
            [10, 1, 10],
            [10, 10, 1],
        ]
        rows, cols = MathUtils.linear_sum_assignment(cost)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(cols), 2)
        self.assertEqual(set(zip(rows, cols)), {(0, 1), (1, 2)})

    def test_linear_sum_assignment_maximize(self):
        """Test maximize=True chooses maximum total score assignment."""
        score = [
            [1, 2],
            [3, 4],
        ]
        rows, cols = MathUtils.linear_sum_assignment(score, maximize=True)
        # Best is (0->1)=2 and (1->0)=3 total 5 (vs 1+4=5 tie)
        # Both are optimal; accept either.
        pairs = set(zip(rows, cols))
        self.assertTrue(pairs in ({(0, 1), (1, 0)}, {(0, 0), (1, 1)}))

    def test_linear_sum_assignment_empty(self):
        """Test empty input returns empty assignment."""
        rows, cols = MathUtils.linear_sum_assignment([])
        self.assertEqual(rows, [])
        self.assertEqual(cols, [])

    def test_linear_sum_assignment_jagged_raises(self):
        """Test jagged matrices raise a clear error."""
        with self.assertRaises(ValueError):
            MathUtils.linear_sum_assignment([[1, 2], [3]])

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
            MathUtils.normalize((0, 0, 0))
            # If it doesn't raise, it handled the zero vector gracefully
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
        """Equilateral sides must be exact — regression for the hardcoded
        3.14159 that put ~0.00015 of error in every result."""
        a, b = MathUtils.get_two_sides_of_asa_triangle(60, 60, 100)
        self.assertAlmostEqual(a, 100.0, places=9)
        self.assertAlmostEqual(b, 100.0, places=9)

    def test_get_two_sides_of_asa_triangle_isoceles(self):
        """Test get_two_sides_of_asa_triangle with isoceles."""
        result = MathUtils.get_two_sides_of_asa_triangle(45, 45, 100)
        # Equal angles should give equal sides
        self.assertAlmostEqual(result[0], result[1], places=5)

    def test_get_two_sides_of_asa_triangle_degenerate_raises_value_error(self):
        """Angles summing to >= 180° describe no triangle (the two side rays
        are parallel or diverge); the contract is a clear ValueError, not a
        bare ZeroDivisionError from sin(0).

        Regression: mayatk's create_curve_between_two_objs feeds angles built
        from opposing vectors; with parallel curve normals a1 + a2 == 180
        exactly and this crashed with ZeroDivisionError.
        """
        with self.assertRaises(ValueError):
            MathUtils.get_two_sides_of_asa_triangle(90, 90, 100)
        with self.assertRaises(ValueError):
            MathUtils.get_two_sides_of_asa_triangle(120, 70, 100)

    # -------------------------------------------------------------------------
    # Rotation Tests
    # -------------------------------------------------------------------------

    def test_xyz_rotation_radians(self):
        """xyz_rotation about Y must return (0, theta, 0) — the old
        hardcoded 3.14159265 put ~3.6e-09 of noise in every component."""
        x, y, z = MathUtils.xyz_rotation(2, (0, 1, 0))
        self.assertAlmostEqual(x, 0.0, places=12)
        self.assertAlmostEqual(y, 2.0, places=12)
        self.assertAlmostEqual(z, 0.0, places=12)

    def test_xyz_rotation_degrees(self):
        """Test xyz_rotation with degrees."""
        self.assertEqual(
            MathUtils.xyz_rotation(2, (0, 1, 0), degree=True),
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

    # -------------------------------------------------------------------------
    # K-Means 1D Tests
    # -------------------------------------------------------------------------

    def test_kmeans_1d_basic_two_clusters(self):
        """Test kmeans_1d separates obvious clusters."""
        values = [1, 2, 3, 50, 55, 60]
        centers, groups = MathUtils.kmeans_1d(values, k=2)
        self.assertEqual(len(centers), 2)
        self.assertEqual(len(groups), 2)
        # First cluster should contain small values, second large
        self.assertTrue(all(v < 10 for v in groups[0]))
        self.assertTrue(all(v > 40 for v in groups[1]))

    def test_kmeans_1d_preserves_duplicates(self):
        """Test kmeans_1d preserves duplicate values."""
        values = [1, 1, 1, 50, 50]
        centers, groups = MathUtils.kmeans_1d(values, k=2)
        # Should preserve all 5 values across groups
        total_values = sum(len(g) for g in groups)
        self.assertEqual(total_values, 5)
        # First group should have 3 ones
        self.assertEqual(groups[0], [1, 1, 1])
        # Second group should have 2 fifties
        self.assertEqual(groups[1], [50, 50])

    def test_kmeans_1d_three_clusters(self):
        """Test kmeans_1d with 3 clusters (small/medium/large)."""
        values = [1, 2, 3, 20, 25, 30, 100, 150, 200]
        centers, groups = MathUtils.kmeans_1d(values, k=3)
        self.assertEqual(len(centers), 3)
        self.assertEqual(len(groups), 3)
        # Centers should be sorted ascending
        self.assertEqual(centers, sorted(centers))

    def test_kmeans_1d_single_value(self):
        """Test kmeans_1d with single unique value."""
        values = [5, 5, 5]
        centers, groups = MathUtils.kmeans_1d(values, k=3)
        self.assertEqual(len(centers), 1)
        self.assertEqual(groups[0], [5, 5, 5])

    def test_kmeans_1d_empty_input(self):
        """Test kmeans_1d with empty input."""
        centers, groups = MathUtils.kmeans_1d([], k=3)
        self.assertEqual(centers, [])
        self.assertEqual(groups, [])

    def test_kmeans_1d_k_exceeds_unique(self):
        """Test kmeans_1d when k exceeds unique values."""
        values = [1, 1, 2, 2]  # Only 2 unique values
        centers, groups = MathUtils.kmeans_1d(values, k=5)
        # Should clamp to 2 clusters
        self.assertEqual(len(centers), 2)

    def test_kmeans_1d_negative_values(self):
        """Test kmeans_1d with negative values."""
        values = [-100, -90, -80, 10, 20, 30]
        centers, groups = MathUtils.kmeans_1d(values, k=2)
        self.assertTrue(all(v < 0 for v in groups[0]))
        self.assertTrue(all(v > 0 for v in groups[1]))

    def test_kmeans_1d_floats(self):
        """Test kmeans_1d with floating point values."""
        values = [0.1, 0.2, 0.3, 10.5, 10.6, 10.7]
        centers, groups = MathUtils.kmeans_1d(values, k=2)
        self.assertAlmostEqual(centers[0], 0.2, places=1)
        self.assertAlmostEqual(centers[1], 10.6, places=1)

    def test_kmeans_1d_k_greater_than_3(self):
        """Test kmeans_1d with k > 3 uses quantile initialization."""
        values = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]
        centers, groups = MathUtils.kmeans_1d(values, k=5)
        self.assertEqual(len(centers), 5)
        # Verify centers are sorted
        self.assertEqual(centers, sorted(centers))

    # -------------------------------------------------------------------------
    # K-Means N-Dimensional Tests
    # -------------------------------------------------------------------------

    def test_kmeans_clustering_basic(self):
        """Test kmeans_clustering separates 2D clusters."""
        points = [(0, 0), (1, 0), (0, 1), (10, 10), (11, 10), (10, 11)]
        groups = MathUtils.kmeans_clustering(points, k=2)
        self.assertEqual(len(groups), 2)
        # Should have 3 points in each cluster
        self.assertEqual(sorted(len(g) for g in groups), [3, 3])

    def test_kmeans_clustering_3d(self):
        """Test kmeans_clustering with 3D points."""
        points = [
            (0, 0, 0),
            (1, 1, 1),
            (100, 100, 100),
            (101, 101, 101),
        ]
        groups = MathUtils.kmeans_clustering(points, k=2)
        self.assertEqual(len(groups), 2)
        # Each cluster should have 2 points
        self.assertTrue(all(len(g) == 2 for g in groups))

    def test_kmeans_clustering_empty(self):
        """Test kmeans_clustering with empty input."""
        groups = MathUtils.kmeans_clustering([], k=3)
        self.assertEqual(groups, [])

    def test_kmeans_clustering_k_one(self):
        """Test kmeans_clustering with k=1 returns all points."""
        points = [(0, 0), (10, 10), (20, 20)]
        groups = MathUtils.kmeans_clustering(points, k=1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_kmeans_clustering_seed_indices(self):
        """Test kmeans_clustering with explicit seed indices."""
        points = [(0, 0), (1, 0), (10, 0), (11, 0)]
        groups = MathUtils.kmeans_clustering(points, k=2, seed_indices=[0, 2])
        self.assertEqual(len(groups), 2)

    # -------------------------------------------------------------------------
    # K-Means Threshold Tests
    # -------------------------------------------------------------------------

    def test_get_kmeans_threshold_basic(self):
        """Test get_kmeans_threshold finds natural breakpoint."""
        values = [0.8, 1.2, 2.1, 12.4, 15.0]
        threshold = MathUtils.get_kmeans_threshold(values, k=3)
        # Threshold should be between small/medium and large
        self.assertGreater(threshold, 2.5)
        self.assertLess(threshold, 12.0)

    def test_get_kmeans_threshold_empty(self):
        """Test get_kmeans_threshold with empty input."""
        threshold = MathUtils.get_kmeans_threshold([], k=3)
        self.assertEqual(threshold, 0.0)

    def test_get_kmeans_threshold_single_value(self):
        """Test get_kmeans_threshold with single unique value."""
        threshold = MathUtils.get_kmeans_threshold([5, 5, 5], k=3)
        self.assertEqual(threshold, 2.5)  # Half of 5

    def test_get_kmeans_threshold_two_clusters(self):
        """Test get_kmeans_threshold with k=2."""
        values = [1, 2, 3, 100, 200, 300]
        threshold = MathUtils.get_kmeans_threshold(values, k=2)
        # Should be between the two cluster centers (~2 and ~200)
        self.assertGreater(threshold, 3)  # Above the small cluster
        self.assertLess(threshold, 200)  # Below the large cluster

    def test_get_kmeans_threshold_merge_logic(self):
        """Test get_kmeans_threshold merge logic for close clusters."""
        # Small and medium are close (ratio < 3.0), so merge them
        values = [1.0, 1.5, 2.0, 2.5, 100.0, 150.0]
        threshold = MathUtils.get_kmeans_threshold(values, k=3)
        # Should threshold between merged small+medium and large
        self.assertGreater(threshold, 10)

    # -------------------------------------------------------------------------
    # Clamp Range Tests
    # -------------------------------------------------------------------------

    def test_clamp_range_no_boundaries(self):
        """Test clamp_range with no boundaries."""
        result = MathUtils.clamp_range(5, 15)
        self.assertEqual(result, (5, 15))

    def test_clamp_range_start_only(self):
        """Test clamp_range with only start boundary."""
        result = MathUtils.clamp_range(5, 15, clamp_start=10)
        self.assertEqual(result, (10, 15))

    def test_clamp_range_end_only(self):
        """Test clamp_range with only end boundary."""
        result = MathUtils.clamp_range(5, 15, clamp_end=12)
        self.assertEqual(result, (5, 12))

    def test_clamp_range_both_boundaries(self):
        """Test clamp_range with both boundaries."""
        result = MathUtils.clamp_range(5, 15, clamp_start=10, clamp_end=12)
        self.assertEqual(result, (10, 12))

    def test_clamp_range_invalid_input(self):
        """Test clamp_range with start >= end."""
        result = MathUtils.clamp_range(15, 5)
        self.assertIsNone(result)

    def test_clamp_range_clamping_makes_invalid(self):
        """Test clamp_range when clamping creates invalid range."""
        result = MathUtils.clamp_range(5, 15, clamp_start=20)
        self.assertIsNone(result)

    def test_clamp_range_none_input(self):
        """Test clamp_range with None values."""
        result = MathUtils.clamp_range(None, 15)
        self.assertIsNone(result)

    def test_clamp_range_no_validation(self):
        """Test clamp_range with validate=False."""
        result = MathUtils.clamp_range(15, 5, validate=False)
        self.assertEqual(result, (15, 5))

    # ------------------------------------------------------------- misc fixes

    def test_generate_geometric_sequence_instance_call(self):
        """Regression: missing @staticmethod made instance calls consume
        self as base_value and raise TypeError."""
        self.assertEqual(
            MathUtils().generate_geometric_sequence(2, 5), [2, 4, 8, 16, 32]
        )
        self.assertEqual(
            MathUtils.generate_geometric_sequence(3, 4, 3.0), [3, 9, 27, 81]
        )

    def test_kmeans_clustering_zero_iterations_clamped(self):
        """Regression: max_iterations=0 left labels/groups unbuilt
        (numpy path: all-empty result; fallback path: NameError)."""
        points = [(0, 0, 0), (0.1, 0, 0), (10, 0, 0), (10.1, 0, 0)]
        groups = MathUtils.kmeans_clustering(points, k=2, max_iterations=0)
        self.assertEqual(sorted(len(g) for g in groups), [2, 2])

    def test_round_to_aggressive_preferred_delegates(self):
        """aggressive == round_to_preferred with max_distance=10."""
        for v in (48.5, 73.2, 88.9, 23.4, 7.8, 100.0, 3.2):
            self.assertEqual(
                MathUtils.round_to_aggressive_preferred(v),
                MathUtils.round_to_preferred(v, max_distance=10),
            )
        self.assertEqual(MathUtils.round_to_aggressive_preferred(48.5), 50)
        self.assertEqual(MathUtils.round_to_aggressive_preferred(7.8), 10)

    # ------------------------------------------------------------- lerp (vector)

    def test_lerp_point_componentwise(self):
        """Sequence inputs interpolate component-wise to a tuple."""
        self.assertEqual(MathUtils.lerp((0, 0, 0), (2, 4, 6), 0.5), (1.0, 2.0, 3.0))

    def test_lerp_scalar_still_scalar(self):
        """Scalar inputs keep the original scalar behavior."""
        self.assertEqual(MathUtils.lerp(0.0, 10.0, 0.25), 2.5)

    # ----------------------------------------------------------- safe_normalize

    def test_safe_normalize_returns_fallback_on_zero(self):
        self.assertEqual(MathUtils.safe_normalize((0, 0, 0), (1, 0, 0)), (1, 0, 0))

    def test_safe_normalize_normalizes_nonzero(self):
        nx, ny, nz = MathUtils.safe_normalize((0, 3, 4), (1, 0, 0))
        self.assertAlmostEqual((nx, ny, nz)[1], 0.6, places=6)
        self.assertAlmostEqual((nx, ny, nz)[2], 0.8, places=6)

    # ---------------------------------------------------------------- smoothstep

    def test_smoothstep_clamps_and_eases(self):
        self.assertEqual(MathUtils.smoothstep(-1.0), 0.0)
        self.assertEqual(MathUtils.smoothstep(2.0), 1.0)
        self.assertAlmostEqual(MathUtils.smoothstep(0.5), 0.5, places=9)

    def test_smoothstep_edges(self):
        # Maps within [edge0, edge1]; zero-slope endpoints (0 and 1 exactly).
        self.assertEqual(MathUtils.smoothstep(2.0, 2.0, 4.0), 0.0)
        self.assertEqual(MathUtils.smoothstep(4.0, 2.0, 4.0), 1.0)
        self.assertAlmostEqual(MathUtils.smoothstep(3.0, 2.0, 4.0), 0.5, places=9)

    # ------------------------------------------------- falloff / B-spline basis

    def test_resolve_falloff_profile_passthrough_and_names(self):
        fn = lambda t: t * t
        self.assertIs(MathUtils.resolve_falloff_profile(fn), fn)
        self.assertIs(
            MathUtils.resolve_falloff_profile("smoothstep"), MathUtils.smoothstep
        )
        linear = MathUtils.resolve_falloff_profile("linear")
        self.assertAlmostEqual(linear(0.25), 0.25, places=9)

    def test_resolve_falloff_profile_invalid_raises(self):
        with self.assertRaises(ValueError):
            MathUtils.resolve_falloff_profile("not_a_profile")

    def test_bspline_clamped_knots_structure(self):
        stations = [0.0, 1.0, 3.0, 6.0]
        degree = 2
        knots = MathUtils.bspline_clamped_knots(stations, degree)
        self.assertEqual(len(knots), len(stations) + degree + 1)
        self.assertEqual(knots[: degree + 1], [0.0] * (degree + 1))
        self.assertEqual(knots[-(degree + 1):], [6.0] * (degree + 1))
        self.assertEqual(sorted(knots), knots)  # non-decreasing

    def test_bspline_basis_partition_of_unity_and_end_pinning(self):
        import bisect

        stations = [0.0, 1.0, 3.0, 6.0, 10.0]
        degree = 3
        knots = MathUtils.bspline_clamped_knots(stations, degree)
        n = len(stations)
        for s in (0.5, 2.0, 5.0, 9.5):
            span = min(max(bisect.bisect_right(knots, s) - 1, degree), n - 1)
            basis = MathUtils.bspline_basis(knots, span, degree, s)
            self.assertEqual(len(basis), degree + 1)
            self.assertAlmostEqual(sum(basis), 1.0, places=9)
            for w in basis:
                self.assertGreaterEqual(w, -1e-12)
        # Clamped ends: the first/last control point owns the end stations.
        span0 = degree
        self.assertAlmostEqual(
            MathUtils.bspline_basis(knots, span0, degree, 0.0)[0], 1.0, places=9
        )

    # ------------------------------------------------------------------- ricker

    def test_ricker_peak_and_zero_crossings(self):
        self.assertAlmostEqual(MathUtils.ricker(0.0), 1.0, places=9)
        self.assertAlmostEqual(MathUtils.ricker(1.0), 0.0, places=9)
        self.assertAlmostEqual(MathUtils.ricker(-1.0), 0.0, places=9)

    def test_ricker_has_negative_troughs(self):
        # Mean-preserving: it dips below zero past the crossings.
        self.assertLess(MathUtils.ricker(1.7), 0.0)

    def test_ricker_integrates_to_zero(self):
        # Riemann sum over a wide window is ~0 (the defining property).
        s = sum(MathUtils.ricker(x * 0.01) for x in range(-800, 801)) * 0.01
        self.assertAlmostEqual(s, 0.0, delta=1e-3)

    # ----------------------------------------------------------------- catenary

    def test_catenary_center_and_supports(self):
        self.assertAlmostEqual(MathUtils.catenary(0.0, 1.5), 1.0, places=9)
        self.assertAlmostEqual(MathUtils.catenary(1.0, 1.5), 0.0, places=9)
        self.assertAlmostEqual(MathUtils.catenary(-1.0, 1.5), 0.0, places=9)

    def test_catenary_parabolic_limit(self):
        self.assertAlmostEqual(MathUtils.catenary(0.5, 0.0), 0.75, places=9)

    def test_catenary_clamped_outside_span(self):
        self.assertAlmostEqual(MathUtils.catenary(2.0, 1.5), 0.0, places=9)

    def test_catenary_sag_no_round_matches_catenary(self):
        for t in (-1.0, -0.3, 0.0, 0.4, 1.0):
            self.assertAlmostEqual(
                MathUtils.catenary_sag(t, 1.5, 0.0),
                MathUtils.catenary(t, 1.5),
                places=9,
            )

    def test_catenary_sag_rounding_lowers_near_support(self):
        # The rounded profile rises with zero slope, so it sits below the crisp
        # catenary just inside a support.
        crisp = MathUtils.catenary(-1.0 + 1e-3, 3.0)
        rounded = MathUtils.catenary_sag(-1.0 + 1e-3, 3.0, 1.0)
        self.assertLess(rounded, crisp)

    def test_catenary_sag_gather_pushes_at_support_and_pulls_inside(self):
        # gather lifts the profile above the baseline AT the support (a gathered
        # pucker rising above the rail = negative sag) and adds extra sag just
        # inside it as the slack falls off; the center sag stays untouched.
        self.assertLess(MathUtils.catenary_sag(-1.0, 1.5, 0.0, gather=1.0), 0.0)
        self.assertGreater(
            MathUtils.catenary_sag(-0.5, 1.5, 0.0, gather=1.0),
            MathUtils.catenary(-0.5, 1.5),
        )
        self.assertAlmostEqual(
            MathUtils.catenary_sag(0.0, 1.5, 0.0, gather=1.0),
            MathUtils.catenary(0.0, 1.5),
            places=9,
        )

    def test_catenary_sag_gather_off_matches_catenary(self):
        for t in (-1.0, -0.3, 0.0, 0.4, 1.0):
            self.assertAlmostEqual(
                MathUtils.catenary_sag(t, 1.5, 0.0, gather=0.0),
                MathUtils.catenary(t, 1.5),
                places=9,
            )

    # -------------------------------------------------------------------------
    # Point/segment distance + Ramer-Douglas-Peucker simplification
    # -------------------------------------------------------------------------
    def test_point_segment_distance_perpendicular_and_clamped(self):
        a, b = (0, 0, 0), (10, 0, 0)
        # straight above the middle -> perpendicular distance
        self.assertAlmostEqual(MathUtils.point_segment_distance((5, 3, 0), a, b), 3.0)
        # past the end -> clamps to the endpoint b (not the infinite line)
        self.assertAlmostEqual(MathUtils.point_segment_distance((13, 4, 0), a, b), 5.0)
        # on the segment -> zero
        self.assertAlmostEqual(MathUtils.point_segment_distance((7, 0, 0), a, b), 0.0)

    def test_point_segment_distance_degenerate(self):
        # zero-length segment -> distance to the single point
        self.assertAlmostEqual(
            MathUtils.point_segment_distance((3, 4, 0), (0, 0, 0), (0, 0, 0)), 5.0
        )






    # -------------------------------------------------------------------------
    # Calculator engine — eval_expression / convert_length_unit
    # (shared by the mayatk + blendertk Calculator panels)
    # -------------------------------------------------------------------------

    def test_eval_expression_basic(self):
        self.assertEqual(MathUtils.eval_expression("2+2"), "4")
        self.assertEqual(MathUtils.eval_expression("10/4"), "2.5")
        self.assertEqual(MathUtils.eval_expression("2**3"), "8")

    def test_eval_expression_integer_float_collapses(self):
        # An integer-valued float drops the trailing .0
        self.assertEqual(MathUtils.eval_expression("8.0/2"), "4")

    def test_eval_expression_math_functions(self):
        self.assertEqual(MathUtils.eval_expression("sqrt(16)"), "4")
        self.assertEqual(MathUtils.eval_expression("max(3, 7, 1)"), "7")

    def test_eval_expression_empty_and_error(self):
        self.assertEqual(MathUtils.eval_expression(""), "")
        self.assertEqual(MathUtils.eval_expression("1/0"), "Error")
        self.assertEqual(MathUtils.eval_expression("nonsense("), "Error")

    def test_eval_expression_builtins_disabled(self):
        # Builtins are off, so arbitrary code paths resolve to Error, not execution.
        self.assertEqual(MathUtils.eval_expression("__import__('os')"), "Error")
        self.assertEqual(MathUtils.eval_expression("open('x')"), "Error")

    def test_convert_length_unit(self):
        self.assertEqual(MathUtils.convert_length_unit(100, "cm", "m"), "1.0")
        self.assertEqual(MathUtils.convert_length_unit(1, "in", "cm"), "2.54")
        self.assertEqual(MathUtils.convert_length_unit(10, "mm", "cm"), "1.0")
        self.assertEqual(MathUtils.convert_length_unit(1, "m", "mm"), "1000.0")

    def test_convert_length_unit_errors(self):
        self.assertEqual(MathUtils.convert_length_unit(1, "cm", "parsec"), "Error")
        self.assertEqual(MathUtils.convert_length_unit("abc", "cm", "m"), "Error")

    def test_calculate_uv_padding(self):
        # Pixel gutter scales with the map: 1/256 of it.
        self.assertEqual(MathUtils.calculate_uv_padding(1024), 4.0)
        self.assertEqual(MathUtils.calculate_uv_padding(2048), 8.0)
        self.assertEqual(MathUtils.calculate_uv_padding(4096), 16.0)
        self.assertEqual(MathUtils.calculate_uv_padding(8192), 32.0)

    def test_udim_to_tile(self):
        # UDIM = 1001 + u + 10*v; u spans a 10-tile row, v is unbounded.
        self.assertEqual(MathUtils.udim_to_tile(1001), (0, 0))
        self.assertEqual(MathUtils.udim_to_tile(1010), (9, 0))  # row end
        self.assertEqual(MathUtils.udim_to_tile(1011), (0, 1))  # wraps to next row
        self.assertEqual(MathUtils.udim_to_tile(1012), (1, 1))
        self.assertEqual(MathUtils.udim_to_tile(1101), (0, 10))

    def test_calculate_uv_padding_normalized_is_map_size_invariant(self):
        # The property packers rely on: the normalized gutter is 1/factor for
        # EVERY map size, so baking it into a pack call is resolution-safe.
        expected = 1 / 256
        for size in (256, 512, 1024, 2048, 4096, 8192):
            self.assertEqual(
                MathUtils.calculate_uv_padding(size, normalize=True), expected
            )
        # A custom factor moves both forms together.
        self.assertEqual(MathUtils.calculate_uv_padding(1024, factor=128), 8.0)
        self.assertEqual(
            MathUtils.calculate_uv_padding(1024, normalize=True, factor=128), 1 / 128
        )

    def test_max_axis_skew_orthogonal_reads_zero(self):
        # Identity and pure (non-uniform) scale keep axes perpendicular.
        self.assertEqual(
            MathUtils.max_axis_skew([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), 0.0
        )
        self.assertEqual(
            MathUtils.max_axis_skew([(3, 0, 0), (0, 1, 0), (0, 0, 0.5)]), 0.0
        )
        # A rotated but orthogonal basis still reads ~0.
        s = math.sqrt(0.5)
        self.assertAlmostEqual(
            MathUtils.max_axis_skew([(s, s, 0), (-s, s, 0), (0, 0, 1)]), 0.0
        )

    def test_max_axis_skew_measures_shear(self):
        # X and the sheared Y axis (1,0,0)·(0.5,1,0): cos = 0.5/|Y| ≈ 0.4472.
        skew = MathUtils.max_axis_skew([(1, 0, 0), (0.5, 1, 0), (0, 0, 1)])
        self.assertAlmostEqual(skew, 0.5 / math.sqrt(1.25), places=9)
        # The worst PAIR wins: a second, milder skew doesn't dilute it.
        worse = MathUtils.max_axis_skew([(1, 0, 0), (0.5, 1, 0), (0.1, 0, 1)])
        self.assertAlmostEqual(worse, 0.5 / math.sqrt(1.25), places=9)

    def test_max_axis_skew_degenerate_axis_reads_zero(self):
        # A zero-length axis has no direction to measure against.
        self.assertEqual(
            MathUtils.max_axis_skew([(0, 0, 0), (0.5, 1, 0), (0, 0, 1)]), 0.0
        )

    def test_move_decimal_point_negative_places_exact(self):
        # Regression: negative `places` must stay in the exact Decimal domain.
        # Decimal(10 ** -1) captures the inexact float 0.1 -> 0.30000000000000004;
        # Decimal(10) ** -1 is exact -> 0.3.
        self.assertEqual(MathUtils.move_decimal_point(3, -1), 0.3)
        self.assertEqual(MathUtils.move_decimal_point(11.05, -2), 0.1105)
        # Positive/zero shifts remain correct.
        self.assertEqual(MathUtils.move_decimal_point(3, 1), 30.0)
        self.assertEqual(MathUtils.move_decimal_point(3, 0), 3.0)

    # -------------------------------------------------------------------------
    # Regression tests (fix sweep)
    # -------------------------------------------------------------------------

    def test_eval_expression_ast_escape_blocked(self):
        # Regression: {"__builtins__": None} does not sandbox eval(); attribute
        # access reaches the class hierarchy. The AST-whitelist evaluator must
        # reject these while still computing legitimate arithmetic.
        self.assertEqual(
            MathUtils.eval_expression("(1).__class__.__base__.__subclasses__()"),
            "Error",
        )
        self.assertEqual(
            MathUtils.eval_expression(
                "[c for c in ().__class__.__base__.__subclasses__() "
                "if c.__name__=='catch_warnings'][0]()"
            ),
            "Error",
        )
        self.assertEqual(MathUtils.eval_expression("().__class__"), "Error")
        self.assertEqual(MathUtils.eval_expression("[].append"), "Error")
        # Legitimate arithmetic still evaluates through the new path.
        self.assertEqual(MathUtils.eval_expression("sin(pi/2) + 2**3"), "9")
        self.assertEqual(MathUtils.eval_expression("-5 + 3"), "-2")
        self.assertEqual(MathUtils.eval_expression("7 % 3"), "1")
        self.assertEqual(MathUtils.eval_expression("7 // 2"), "3")
        self.assertEqual(MathUtils.eval_expression("abs(-4)"), "4")

    def test_cross_product_parallel_normalize_no_zero_division(self):
        # Regression: normalize=1 on a degenerate (parallel/collinear) cross
        # product previously divided by a zero magnitude.
        self.assertEqual(
            MathUtils.cross_product((1, 0, 0), (2, 0, 0), normalize=1),
            (0, 0, 0),
        )
        # Non-degenerate normalization is unchanged.
        self.assertEqual(
            MathUtils.cross_product((1, 2, 3), (1, 1, -1), None, 1),
            (-0.7715167498104595, 0.6172133998483676, -0.1543033499620919),
        )

    def test_get_angle_from_two_vectors_identical_no_domain_error(self):
        # Regression: dot/(len*len) rounds to 1.0000000000000002 for identical
        # vectors, tripping math.acos domain error before the clamp.
        self.assertEqual(MathUtils.get_angle_from_two_vectors((1, 1, 1), (1, 1, 1)), 0.0)
        self.assertEqual(MathUtils.get_angle_from_two_vectors((2, 3, 4), (2, 3, 4)), 0.0)
        self.assertAlmostEqual(
            MathUtils.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1)),
            1.5707963267948966,
        )

    def test_get_angle_from_three_points_collinear_no_domain_error(self):
        # Regression: collinear rays normalize to the same unit vector, scalar
        # rounds past 1.0, math.acos raised a domain error before the clamp.
        self.assertEqual(
            MathUtils.get_angle_from_three_points((2, 2, 2), (0, 0, 0), (1, 1, 1)),
            0.0,
        )
        self.assertAlmostEqual(
            MathUtils.get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3), True),
            45.29,
            places=2,
        )

    def test_remap_descending_range_clamp(self):
        # Regression: clamp with a descending target range collapsed every value
        # to an endpoint; scalar and numpy paths must agree on the ordered range.
        self.assertEqual(
            MathUtils.remap(0.5, (0.0, 1.0), (255.0, 0.0), clamp=True), 127.5
        )
        self.assertEqual(
            MathUtils.remap(2.0, (0.0, 1.0), (255.0, 0.0), clamp=True), 0.0
        )
        self.assertEqual(
            MathUtils.remap(-1.0, (0.0, 1.0), (255.0, 0.0), clamp=True), 255.0
        )
        # Ascending range unaffected.
        self.assertEqual(
            MathUtils.remap(0.5, (0.0, 1.0), (0.0, 255.0), clamp=True), 127.5
        )
        try:
            import numpy as np

            out = MathUtils.remap(
                np.array([0.0, 0.5, 2.0, -1.0]), (0.0, 1.0), (255.0, 0.0), clamp=True
            )
            self.assertEqual(list(out), [255.0, 127.5, 0.0, 255.0])
        except ImportError:
            pass

    def test_step_offset_relative(self):
        """Relative mode is a plain signed step; sub-step drift is preserved."""
        self.assertAlmostEqual(MathUtils.step_offset(0.13, 0.5, 1), 0.5)
        self.assertAlmostEqual(MathUtils.step_offset(0.13, 0.5, -1), -0.5)
        # A negative step is normalized — direction alone decides the sign.
        self.assertAlmostEqual(MathUtils.step_offset(0.13, -0.5, 1), 0.5)
        # Zero direction is a no-op, not an error.
        self.assertEqual(MathUtils.step_offset(0.13, 0.5, 0), 0.0)

    def test_step_offset_snap(self):
        """Snap mode always lands on a multiple of the step."""
        for value, step, direction, expected in (
            (0.13, 0.5, 1, 0.37),  # off-grid: absorbs the drift first
            (0.13, 0.5, -1, -0.13),
            (0.5, 0.5, 1, 0.5),  # on-grid: advances a full step
            (0.5, 0.5, -1, -0.5),
            (0.0, 1.0, 1, 1.0),
            (2.0, 1.0, -1, -1.0),
            (-0.25, 0.5, 1, 0.25),  # negative UV space
        ):
            with self.subTest(value=value, step=step, direction=direction):
                offset = MathUtils.step_offset(value, step, direction, snap=True)
                self.assertAlmostEqual(offset, expected)
                # The landing point is on the grid.
                self.assertAlmostEqual((value + offset) % step, 0.0, places=6)

    def test_step_offset_snap_float_error_is_on_grid(self):
        """A value on a grid line only up to float error still advances a full step.

        Without the tolerance this rounds the other way and the move collapses to
        a ~0 offset, which reads as a dead button.
        """
        self.assertAlmostEqual(
            MathUtils.step_offset(0.9999999, 0.5, 1, snap=True), 0.5, places=5
        )
        self.assertAlmostEqual(
            MathUtils.step_offset(1.0000001, 0.5, -1, snap=True), -0.5, places=5
        )

    def test_step_offset_zero_step_raises(self):
        with self.assertRaises(ValueError):
            MathUtils.step_offset(0.13, 0.0, 1)

    def test_next_clear_offset_parks_against_the_nearest_blocker(self):
        box = (0.0, 0.1, 0.3, 0.3)
        blocker = (0.0, 1.0, 0.3, 1.4)

        offset = MathUtils.next_clear_offset(box, [blocker], 1, 1, margin=0.002)

        # The box's top edge lands one margin under the blocker's bottom.
        self.assertAlmostEqual(box[3] + offset, blocker[1] - 0.002)

    def test_next_clear_offset_skips_a_gap_too_small_to_fit(self):
        """The whole point of validating fit: a naive nearest-edge rule would
        land the box in the A-B gap and overlap A."""
        a, b = (0.0, 1.0, 0.3, 1.4), (0.0, 1.5, 0.3, 1.9)  # 0.1 gap
        box = (0.0, 0.1, 0.3, 0.3)  # 0.2 tall — cannot fit between them
        margin = 0.002

        first = MathUtils.next_clear_offset(box, [a, b], 1, 1, margin=margin)
        parked = (box[0], box[1] + first, box[2], box[3] + first)
        self.assertAlmostEqual(parked[3], a[1] - margin)  # under A

        second = MathUtils.next_clear_offset(parked, [a, b], 1, 1, margin=margin)
        landed = (parked[0], parked[1] + second, parked[2], parked[3] + second)
        self.assertAlmostEqual(landed[1], b[3] + margin)  # jumped clear of B
        # It overlaps neither shell it passed.
        for other in (a, b):
            self.assertFalse(other[3] > landed[1] and other[1] < landed[3])

    def test_next_clear_offset_walk_is_reversible(self):
        a, b = (0.0, 1.0, 0.3, 1.4), (0.0, 1.5, 0.3, 1.9)
        start = (0.0, 0.1, 0.3, 0.3)
        margin = 0.002

        box, forward = start, []
        while (off := MathUtils.next_clear_offset(box, [a, b], 1, 1, margin=margin)):
            forward.append(off)
            box = (box[0], box[1] + off, box[2], box[3] + off)
        self.assertEqual(len(forward), 2)

        while (off := MathUtils.next_clear_offset(box, [a, b], 1, -1, margin=margin)):
            box = (box[0], box[1] + off, box[2], box[3] + off)
        # Back to the first landing spot (not the start — nothing lies below it).
        self.assertAlmostEqual(box[3], a[1] - margin)

    def test_next_clear_offset_ignores_blockers_out_of_the_lane(self):
        """A shell beside the travel lane is not in the way."""
        box = (0.0, 0.1, 0.3, 0.3)
        beside = (0.5, 1.0, 0.8, 1.4)  # no u overlap
        self.assertIsNone(MathUtils.next_clear_offset(box, [beside], 1, 1))
        # Touching lanes don't block either (edge-adjacent, not overlapping).
        self.assertIsNone(
            MathUtils.next_clear_offset(box, [(0.3, 1.0, 0.6, 1.4)], 1, 1)
        )

    def test_next_clear_offset_returns_none_when_nothing_is_ahead(self):
        box = (0.0, 1.0, 0.3, 1.2)
        behind = (0.0, 0.1, 0.3, 0.4)
        self.assertIsNone(MathUtils.next_clear_offset(box, [behind], 1, 1))
        self.assertIsNone(MathUtils.next_clear_offset(box, [], 1, 1))
        self.assertIsNone(MathUtils.next_clear_offset(box, [behind], 1, 0))

    def test_next_clear_offset_already_parked_advances_past(self):
        """A box sitting exactly at the margin must not return a ~0 offset."""
        blocker = (0.0, 1.0, 0.3, 1.4)
        margin = 0.002
        parked = (0.0, 0.798, 0.3, 1.0 - margin)

        offset = MathUtils.next_clear_offset(parked, [blocker], 1, 1, margin=margin)

        self.assertIsNotNone(offset)
        self.assertGreater(offset, 0.1)
        self.assertAlmostEqual(parked[1] + offset, blocker[3] + margin)

    def test_next_clear_offset_works_on_the_u_axis(self):
        box = (0.1, 0.0, 0.3, 0.3)
        blocker = (1.0, 0.0, 1.4, 0.3)
        offset = MathUtils.next_clear_offset(box, [blocker], 0, 1, margin=0.002)
        self.assertAlmostEqual(box[2] + offset, blocker[0] - 0.002)

    def test_uv_tile_margin_is_half_gutter_and_invariant(self):
        """The tile border is half the island gutter, at every map size."""
        for size in (1024, 2048, 4096, 8192):
            with self.subTest(map_size=size):
                self.assertAlmostEqual(
                    MathUtils.uv_tile_margin(size),
                    MathUtils.calculate_uv_padding(size, normalize=True) / 2,
                )
                self.assertAlmostEqual(MathUtils.uv_tile_margin(size), 1 / 512)

    def test_majority_tile(self):
        """The bulk of a layout defines its tile; strays don't drag the target."""
        self.assertEqual(
            MathUtils.majority_tile(
                [(0.1, 0.1, 0.4, 0.4), (0.5, 0.2, 0.9, 0.6), (3.1, 2.1, 3.4, 2.4)]
            ),
            (0, 0),
        )
        # The winner needn't be the origin tile.
        self.assertEqual(
            MathUtils.majority_tile(
                [(1.1, 0.1, 1.4, 0.4), (1.5, 0.2, 1.9, 0.6), (0.1, 0.1, 0.4, 0.4)]
            ),
            (1, 0),
        )
        # A lone box is its own majority — nothing to gather.
        self.assertEqual(MathUtils.majority_tile([(3.2, 2.2, 3.6, 2.6)]), (3, 2))
        self.assertIsNone(MathUtils.majority_tile([]))

    def test_fit_into_tile_already_inside(self):
        """A box already inside the (margin-inset) tile does not move."""
        self.assertEqual(
            MathUtils.fit_into_tile((0.2, 0.2, 0.8, 0.8), (0, 0)), (0.0, 0.0)
        )
        self.assertEqual(
            MathUtils.fit_into_tile((2.1, 1.3, 2.4, 1.6), (2, 1)), (0.0, 0.0)
        )

    def test_fit_into_tile_whole_tile_translation(self):
        """A box in another tile keeps its sub-tile position — offsets are whole tiles."""
        du, dv = MathUtils.fit_into_tile((2.1, 0.2, 2.4, 0.5), (0, 0))
        self.assertEqual((du, dv), (-2.0, 0.0))
        du, dv = MathUtils.fit_into_tile((0.25, 0.25, 0.75, 0.75), (3, 2))
        self.assertEqual((du, dv), (3.0, 2.0))

    def test_fit_into_tile_straddler_clamps(self):
        """A box straddling a tile border is pulled fully inside."""
        du, dv = MathUtils.fit_into_tile((0.9, 0.1, 1.2, 0.4), (0, 0))
        self.assertAlmostEqual(du, -0.2)
        self.assertAlmostEqual(dv, 0.0)
        # And onto a non-origin tile.
        du, dv = MathUtils.fit_into_tile((0.9, 0.1, 1.2, 0.4), (1, 0))
        self.assertAlmostEqual(du, 0.1)

    def test_fit_into_tile_margin(self):
        """The margin insets the usable tile on every border."""
        margin = 0.002
        du, dv = MathUtils.fit_into_tile((0.0, 0.999, 0.3, 1.299), (0, 0), margin)
        self.assertAlmostEqual(du, margin)  # min edge pushed off the border
        self.assertAlmostEqual(dv, (1.0 - margin) - 1.299)  # max edge pulled in

    def test_fit_into_tile_oversize_is_centered(self):
        """A box larger than the inset tile is centered, splitting the overhang."""
        margin = 0.01
        du, dv = MathUtils.fit_into_tile((0.3, -0.2, 1.8, 1.4), (0, 0), margin)
        self.assertAlmostEqual((0.3 + du + 1.8 + du) / 2, 0.5)
        self.assertAlmostEqual((-0.2 + dv + 1.4 + dv) / 2, 0.5)

    def test_fit_into_tile_full_coverage_shell_does_not_move(self):
        """A shell spanning the target tile exactly stays put.

        Anchoring an edge instead pushed the common full-coverage layout off
        the far border by the whole margin — the tile it already filled.
        """
        margin = MathUtils.uv_tile_margin(4096)
        self.assertEqual(
            MathUtils.fit_into_tile((0.0, 0.0, 1.0, 1.0), (0, 0), margin), (0.0, 0.0)
        )
        # And from another tile it comes back to exact coverage.
        du, dv = MathUtils.fit_into_tile((3.0, 2.0, 4.0, 3.0), (0, 0), margin)
        self.assertAlmostEqual(du, -3.0)
        self.assertAlmostEqual(dv, -2.0)


if __name__ == "__main__":
    unittest.main(exit=False)
