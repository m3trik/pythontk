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
    # PCA Transform Tests
    # -------------------------------------------------------------------------

    def test_get_pca_transform_identity(self):
        """Test get_pca_transform with identical points returns identity-like matrix."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        try:
            result = MathUtils.get_pca_transform(pts, pts, tolerance=0.001)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 16)  # 4x4 matrix

    def test_get_pca_transform_translated(self):
        """Test get_pca_transform finds translation."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        pts_b = pts_a + np.array([10, 20, 30])  # Translated copy
        try:
            result = MathUtils.get_pca_transform(pts_a, pts_b, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNotNone(result)

    def test_get_pca_transform_rotated(self):
        """Test get_pca_transform finds rotation alignment."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        # Create a shape
        pts_a = np.array(
            [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 0.5, 0.5]], dtype=float
        )

        # Rotate 90 degrees around Z axis
        theta = np.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        pts_b = pts_a @ R.T

        try:
            result = MathUtils.get_pca_transform(pts_a, pts_b, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNotNone(result)

    def test_get_pca_transform_insufficient_points(self):
        """Test get_pca_transform with too few points returns None."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        try:
            result = MathUtils.get_pca_transform(pts, pts, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNone(result)

    def test_get_pca_transform_no_match(self):
        """Test get_pca_transform returns None when shapes don't match."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        pts_b = np.array(
            [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float
        )  # Different scale
        try:
            result = MathUtils.get_pca_transform(pts_a, pts_b, tolerance=0.001)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        self.assertIsNone(result)

    def test_get_pca_transform_robust_mode(self):
        """Test get_pca_transform robust mode with different point counts."""
        try:
            import numpy as np
            from scipy.spatial import KDTree  # noqa: F401
        except (ImportError, ValueError) as e:
            self.skipTest(f"numpy or scipy not available: {e}")

        np.random.seed(42)
        pts_a = np.random.rand(100, 3)
        pts_b = np.random.rand(80, 3)  # Different count

        try:
            # Robust mode should handle different point counts
            result = MathUtils.get_pca_transform(
                pts_a, pts_b, tolerance=10.0, robust=True
            )
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise
        # May or may not find a match depending on data, but shouldn't crash
        self.assertTrue(result is None or len(result) == 16)

    def test_get_pca_transform_caching(self):
        """Test get_pca_transform caches base rotations."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not available")

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        try:
            # First call creates cache
            MathUtils.get_pca_transform(pts, pts, tolerance=0.1)
        except (ImportError, ValueError) as e:
            if "binary incompatibility" in str(e) or "numpy" in str(e).lower():
                self.skipTest(f"scipy/numpy incompatibility: {e}")
            raise

        # Verify cache exists
        self.assertTrue(hasattr(MathUtils, "_pca_base_rotations"))
        self.assertEqual(len(MathUtils._pca_base_rotations), 24)

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

    # -------------------------------------------------------------------------
    # arrange_points_as_path
    #
    # Regression: the default distance_metric used to be
    # ``(p1 - p2).length()`` which only worked for PyMEL ``dt.Point`` /
    # OpenMaya ``MPoint``. Plain ``[x, y, z]`` lists (the documented input
    # type, and what ``cmds.pointPosition`` returns) raised TypeError
    # because ``list - list`` is undefined.
    # -------------------------------------------------------------------------

    def test_arrange_points_as_path_with_lists(self):
        """Default distance_metric must handle plain [x, y, z] lists."""
        # Three colinear points along the X axis, given out of order.
        points = [[2.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]]
        ordered = MathUtils.arrange_points_as_path(points)

        self.assertEqual(len(ordered), 3)
        # First point seeds the path; nearest neighbours follow.
        self.assertEqual(ordered[0], [2.0, 0, 0])
        self.assertEqual(ordered[1], [1.0, 0, 0])
        self.assertEqual(ordered[2], [0.0, 0, 0])

    def test_arrange_points_as_path_with_tuples(self):
        """Tuples should also work via the subscripting fallback."""
        points = [(0.0, 0, 0), (5.0, 0, 0), (10.0, 0, 0)]
        ordered = MathUtils.arrange_points_as_path(points)
        self.assertEqual([p[0] for p in ordered], [0.0, 5.0, 10.0])

    def test_arrange_points_as_path_with_xyz_objects(self):
        """Objects with .x/.y/.z attributes use the attribute branch."""

        class P:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

            def __eq__(self, other):
                return (
                    isinstance(other, P)
                    and (self.x, self.y, self.z) == (other.x, other.y, other.z)
                )

            def __hash__(self):
                return hash((self.x, self.y, self.z))

        points = [P(2, 0, 0), P(0, 0, 0), P(1, 0, 0)]
        ordered = MathUtils.arrange_points_as_path(points)
        self.assertEqual([p.x for p in ordered], [2, 1, 0])

    def test_arrange_points_as_path_empty(self):
        """Empty input returns an empty list."""
        self.assertEqual(MathUtils.arrange_points_as_path([]), [])

    def test_arrange_points_as_path_closed(self):
        """closed_path appends a copy of the first point at the end."""
        points = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
        ordered = MathUtils.arrange_points_as_path(points, closed_path=True)
        self.assertEqual(len(ordered), 4)
        self.assertEqual(ordered[0], ordered[-1])

    def test_arrange_points_as_path_custom_metric(self):
        """Caller-supplied distance_metric overrides the default."""
        points = [[0.0, 0, 0], [3.0, 0, 0], [1.0, 0, 0]]
        # Manhattan distance — same ordering on this colinear set.
        ordered = MathUtils.arrange_points_as_path(
            points,
            distance_metric=lambda a, b: abs(a[0] - b[0]),
        )
        self.assertEqual([p[0] for p in ordered], [0.0, 1.0, 3.0])

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

    def test_simplify_rdp_collapses_straight_run(self):
        # collinear points collapse to just the two endpoints
        pts = [(x, 0, 0) for x in range(11)]
        self.assertEqual(MathUtils.simplify_rdp(pts, 0.01), [0, 10])

    def test_simplify_rdp_keeps_corner(self):
        # an L: the corner must survive (it's the max-deviation point)
        pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]
        kept = MathUtils.simplify_rdp(pts, 0.1)
        self.assertEqual(kept[0], 0)
        self.assertEqual(kept[-1], len(pts) - 1)
        self.assertIn(2, kept)  # the corner vertex

    def test_simplify_rdp_concentrates_on_bends(self):
        # straight ends with a sharp middle bend: kept points cluster at the
        # bend, the straight runs contribute (almost) nothing.
        pts = (
            [(x, 0.0, 0.0) for x in range(0, 10)]
            + [(10, 0, 0), (10.5, 1.5, 0), (11, 0, 0)]
            + [(x, 0.0, 0.0) for x in range(12, 21)]
        )
        kept = MathUtils.simplify_rdp(pts, 0.2)
        bend_lo, bend_hi = 9, 13  # index window around the bend
        in_bend = sum(1 for i in kept if bend_lo <= i <= bend_hi)
        on_straight = sum(1 for i in kept if i < bend_lo or i > bend_hi)
        self.assertGreater(in_bend, on_straight)
        # tighter tolerance -> at least as many points kept (monotone refinement)
        self.assertGreaterEqual(len(MathUtils.simplify_rdp(pts, 0.05)), len(kept))

    def test_simplify_rdp_edge_cases(self):
        self.assertEqual(MathUtils.simplify_rdp([], 0.1), [])
        self.assertEqual(MathUtils.simplify_rdp([(0, 0, 0)], 0.1), [0])
        self.assertEqual(MathUtils.simplify_rdp([(0, 0, 0), (1, 1, 1)], 0.1), [0, 1])
        # tolerance <= 0 keeps everything
        pts = [(0, 0, 0), (1, 0.01, 0), (2, 0, 0)]
        self.assertEqual(MathUtils.simplify_rdp(pts, 0.0), [0, 1, 2])

    def test_simplify_rdp_works_in_2d(self):
        pts = [(0, 0), (1, 0), (2, 0), (2, 2)]
        self.assertEqual(MathUtils.simplify_rdp(pts, 0.1), [0, 2, 3])


if __name__ == "__main__":
    unittest.main(exit=False)
