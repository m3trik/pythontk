#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk MathUtils.

Run with:
    python -m pytest test_math.py -v
    python test_math.py
"""
import unittest

from pythontk import MathUtils

from conftest import BaseTestCase


class MathTest(BaseTestCase):
    """Math utilities test class."""

    def test_get_vector_from_two_points(self):
        """Test get_vector_from_two_points calculates direction vector."""
        self.assertEqual(
            MathUtils.get_vector_from_two_points((1, 2, 3), (1, 1, -1)),
            (0, -1, -4),
        )

    def test_clamp(self):
        """Test clamp restricts values to range."""
        self.assertEqual(
            MathUtils.clamp(range(10), 3, 7),
            [3, 3, 3, 3, 4, 5, 6, 7, 7, 7],
        )

    def test_normalize(self):
        """Test normalize creates unit vectors."""
        self.assertEqual(
            MathUtils.normalize((2, 3, 4)),
            (0.3713906763541037, 0.5570860145311556, 0.7427813527082074),
        )
        self.assertEqual(
            MathUtils.normalize((2, 3)),
            (0.5547001962252291, 0.8320502943378437),
        )
        self.assertEqual(
            MathUtils.normalize((2, 3, 4), 2),
            (0.7427813527082074, 1.1141720290623112, 1.4855627054164149),
        )

    def test_get_magnitude(self):
        """Test get_magnitude calculates vector length."""
        self.assertEqual(MathUtils.get_magnitude((2, 3, 4)), 5.385164807134504)
        self.assertEqual(MathUtils.get_magnitude((2, 3)), 3.605551275463989)

    def test_dot_product(self):
        """Test dot_product calculates scalar product."""
        self.assertEqual(MathUtils.dot_product((1, 2, 3), (1, 1, -1)), 0)
        self.assertEqual(MathUtils.dot_product((1, 2), (1, 1)), 3)
        self.assertEqual(MathUtils.dot_product((1, 2, 3), (1, 1, -1), True), 0)

    def test_cross_product(self):
        """Test cross_product calculates vector product."""
        self.assertEqual(
            MathUtils.cross_product((1, 2, 3), (1, 1, -1)),
            (-5, 4, -1),
        )
        self.assertEqual(
            MathUtils.cross_product((3, 1, 1), (1, 4, 2), (1, 3, 4)),
            (7, 4, 2),
        )
        self.assertEqual(
            MathUtils.cross_product((1, 2, 3), (1, 1, -1), None, 1),
            (-0.7715167498104595, 0.6172133998483676, -0.1543033499620919),
        )

    def test_move_point_relative(self):
        """Test move_point_relative translates points."""
        self.assertEqual(
            MathUtils.move_point_relative((0, 5, 0), (0, 5, 0)),
            (0, 10, 0),
        )
        self.assertEqual(
            MathUtils.move_point_relative((0, 5, 0), 5, (0, 1, 0)),
            (0, 10, 0),
        )

    def test_move_point_relative_along_vector(self):
        """Test move_point_relative_along_vector moves points along vector."""
        self.assertEqual(
            MathUtils.move_point_relative_along_vector(
                (0, 0, 0), (0, 10, 0), (0, 1, 0), 5
            ),
            (0.0, 5.0, 0.0),
        )
        self.assertEqual(
            MathUtils.move_point_relative_along_vector(
                (0, 0, 0), (0, 10, 0), (0, 1, 0), 5, False
            ),
            (0.0, -5.0, 0.0),
        )

    def test_distance_between_points(self):
        """Test distance_between_points calculates Euclidean distance."""
        self.assertEqual(
            MathUtils.distance_between_points((0, 10, 0), (0, 5, 0)),
            5.0,
        )

    def test_get_center_of_two_points(self):
        """Test get_center_of_two_points finds midpoint."""
        self.assertEqual(
            MathUtils.get_center_of_two_points((0, 10, 0), (0, 5, 0)),
            (0.0, 7.5, 0.0),
        )

    def test_get_angle_from_two_vectors(self):
        """Test get_angle_from_two_vectors calculates angle between vectors."""
        self.assertEqual(
            MathUtils.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1)),
            1.5707963267948966,
        )
        self.assertEqual(
            MathUtils.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1), True),
            90,
        )

    def test_get_angle_from_three_points(self):
        """Test get_angle_from_three_points calculates angle at vertex."""
        self.assertEqual(
            MathUtils.get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3)),
            0.7904487543360762,
        )
        self.assertEqual(
            MathUtils.get_angle_from_three_points(
                (1, 1, 1), (-1, 2, 3), (1, 4, -3), True
            ),
            45.29,
        )

    def test_get_two_sides_of_asa_triangle(self):
        """Test get_two_sides_of_asa_triangle solves ASA triangle."""
        self.assertEqual(
            MathUtils.get_two_sides_of_asa_triangle(60, 60, 100),
            (100.00015320566493, 100.00015320566493),
        )

    def test_xyz_rotation(self):
        """Test xyz_rotation applies rotations."""
        self.assertEqual(
            MathUtils.xyz_rotation(2, (0, 1, 0)),
            (3.589792907376932e-09, 1.9999999964102069, 3.589792907376932e-09),
        )
        self.assertEqual(
            MathUtils.xyz_rotation(2, (0, 1, 0), [], True),
            (0.0, 114.59, 0.0),
        )

    def test_lerp(self):
        """Test lerp performs linear interpolation."""
        self.assertEqual(MathUtils.lerp(0, 10, 0.5), 5.0)
        self.assertEqual(MathUtils.lerp(-10, 10, 0.5), 0.0)
        self.assertEqual(MathUtils.lerp(0, 10, 0), 0)
        self.assertEqual(MathUtils.lerp(0, 10, 1), 10)


if __name__ == "__main__":
    unittest.main(exit=False)
