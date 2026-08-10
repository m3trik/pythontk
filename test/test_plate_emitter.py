# !/usr/bin/python
# coding=utf-8
"""Tests for the PlateEmitter primitive — the plate->area-light arithmetic.

Shared by mayatk and blendertk, so the cases that bit a production room are
pinned here once rather than twice in DCC-bound tests.
"""

import unittest


class TestPlateEmitter(unittest.TestCase):
    def _cls(self):
        from pythontk import PlateEmitter

        return PlateEmitter

    def _troffer(self, y=3.9, thickness=0.104):
        """A 3.72 x 0.72 m ceiling plate, Blender-style Z-up."""
        half = thickness / 2.0
        return (-1.862, -0.361, y - half), (1.862, 0.361, y + half)

    def test_thin_axis_becomes_the_normal_and_the_others_the_rectangle(self):
        mn, mx = self._troffer()
        plate = self._cls().from_bounds(mn, mx, toward=(0, 0, 0))
        self.assertEqual(plate.axis, 2)
        self.assertAlmostEqual(plate.size[0], 3.724, places=3)
        self.assertAlmostEqual(plate.size[1], 0.722, places=3)

    def test_a_ceiling_plate_aims_down_at_the_room(self):
        mn, mx = self._troffer()
        plate = self._cls().from_bounds(mn, mx, toward=(0, 0, 1.5))
        self.assertEqual(plate.normal, (0.0, 0.0, -1.0))

    def test_the_light_clears_the_plates_own_thickness(self):
        """Pushed from the CENTRE by offset alone it stays inside the housing."""
        mn, mx = self._troffer(y=3.9, thickness=0.104)
        plate = self._cls().from_bounds(mn, mx, toward=(0, 0, 0), offset=0.01)
        self.assertAlmostEqual(plate.position[2], 3.9 - 0.052 - 0.01, places=6)
        self.assertLess(plate.position[2], mn[2], "light is inside its own housing")

    def test_a_coplanar_reference_resolves_to_down_not_to_noise(self):
        """A ceiling grid's own centre lies in its plane; the sign there is noise."""
        for delta in (0.0, 0.0001, -0.0001):
            mn, mx = self._troffer(y=3.9)
            plate = self._cls().from_bounds(mn, mx, toward=(0, 0, 3.9 + delta))
            self.assertEqual(
                plate.normal, (0.0, 0.0, -1.0), f"flipped on a {delta} difference"
            )

    def test_a_zero_thickness_plane_still_resolves_to_down(self):
        """A lens modelled as a flat plane has no thickness to threshold against."""
        mn, mx = (-1.862, -0.361, 3.9), (1.862, 0.361, 3.9)
        plate = self._cls().from_bounds(mn, mx, toward=(0, 0, 3.9001))
        self.assertEqual(plate.normal, (0.0, 0.0, -1.0))

    def test_a_genuinely_displaced_reference_is_believed(self):
        """Far enough to be a real direction, not modelling noise."""
        mn, mx = self._troffer(y=3.9)
        plate = self._cls().from_bounds(mn, mx, toward=(0, 0, 10.0))
        self.assertEqual(plate.normal, (0.0, 0.0, 1.0))

    def test_a_wall_plate_aims_inward(self):
        mn, mx = (-0.05, -1.5, 0.0), (0.05, 1.5, 2.5)
        plate = self._cls().from_bounds(mn, mx, toward=(4.0, 0.0, 1.2))
        self.assertEqual(plate.axis, 0)
        self.assertEqual(plate.normal, (1.0, 0.0, 0.0))

    def test_maya_up_axis_is_y(self):
        """Y-up hosts resolve an ambiguous plate down about axis 1, not 2."""
        mn, mx = (-186.2, 385.5, -36.1), (186.2, 395.9, 36.1)
        plate = self._cls().from_bounds(mn, mx, toward=(0, 390.7, 0), up_axis=1)
        self.assertEqual(plate.axis, 1)
        self.assertEqual(plate.normal, (0.0, -1.0, 0.0))

    def test_no_reference_falls_back_to_the_up_axis(self):
        mn, mx = self._troffer()
        self.assertEqual(
            self._cls().from_bounds(mn, mx).normal, (0.0, 0.0, -1.0)
        )


class TestPlateEmitterFromPoints(unittest.TestCase):
    """The oriented solver — no bounding box, so rotation costs nothing."""

    def _cls(self):
        from pythontk import PlateEmitter

        return PlateEmitter

    @staticmethod
    def _rect(width, depth, angle=0.0, height=3.9):
        """Corners of a width x depth plate, rotated *angle* radians about Z."""
        import math

        cos, sin = math.cos(angle), math.sin(angle)
        corners = []
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                x, y = sx * width, sy * depth
                corners.append((x * cos - y * sin, x * sin + y * cos, height))
        return corners

    def test_an_axis_aligned_patch_matches_its_real_size(self):
        plate = self._cls().from_points(self._rect(3.72, 0.72), normal=(0, 0, -1))
        self.assertAlmostEqual(plate.size[0], 3.72, places=6)
        self.assertAlmostEqual(plate.size[1], 0.72, places=6)

    def test_a_rotated_patch_keeps_its_real_size(self):
        """The whole reason this solver exists: a box would inflate it ~1.4x."""
        import math

        plate = self._cls().from_points(
            self._rect(3.72, 0.72, angle=math.radians(45)), normal=(0, 0, -1)
        )
        self.assertAlmostEqual(plate.size[0], 3.72, places=5)
        self.assertAlmostEqual(plate.size[1], 0.72, places=5)

    def test_the_tangent_follows_the_rotation(self):
        import math

        plate = self._cls().from_points(
            self._rect(3.72, 0.72, angle=math.radians(30)), normal=(0, 0, -1)
        )
        # Long edge at 30 degrees; sign is free, so compare the absolute dot.
        expected = (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)
        dot = sum(a * b for a, b in zip(plate.tangent, expected))
        self.assertAlmostEqual(abs(dot), 1.0, places=5)

    def test_the_supplied_normal_is_the_emission_direction(self):
        """A face knows which way it faces — no toward-guessing needed."""
        plate = self._cls().from_points(self._rect(2.0, 1.0), normal=(0, 0, -1))
        self.assertAlmostEqual(plate.normal[2], -1.0, places=6)
        flipped = self._cls().from_points(self._rect(2.0, 1.0), normal=(0, 0, 1))
        self.assertAlmostEqual(flipped.normal[2], 1.0, places=6)

    def test_offset_moves_along_the_normal_with_no_thickness_term(self):
        """These points ARE the emitting surface, not a housing."""
        plate = self._cls().from_points(
            self._rect(2.0, 1.0, height=3.9), normal=(0, 0, -1), offset=0.05
        )
        self.assertAlmostEqual(plate.position[2], 3.9 - 0.05, places=6)

    def test_the_position_centres_on_the_rectangle_not_the_vertex_mean(self):
        """A subdivided half drags the centroid off the middle."""
        points = self._rect(4.0, 1.0, height=0.0)
        # Pile extra vertices onto the -X half.
        points += [(-2.0 + i * 0.1, 0.0, 0.0) for i in range(10)]
        plate = self._cls().from_points(points, normal=(0, 0, 1))
        self.assertAlmostEqual(plate.position[0], 0.0, places=5)

    def test_a_tilted_patch_reports_its_own_normal(self):
        import math

        angle = math.radians(30)
        # A plate tilted about X: normal leans out of straight-down.
        points = [
            (-1.0, -0.5 * math.cos(angle), 3.0 - 0.5 * math.sin(angle)),
            (1.0, -0.5 * math.cos(angle), 3.0 - 0.5 * math.sin(angle)),
            (-1.0, 0.5 * math.cos(angle), 3.0 + 0.5 * math.sin(angle)),
            (1.0, 0.5 * math.cos(angle), 3.0 + 0.5 * math.sin(angle)),
        ]
        normal = (0.0, math.sin(angle), -math.cos(angle))
        plate = self._cls().from_points(points, normal=normal)
        self.assertAlmostEqual(plate.size[0], 2.0, places=5)
        self.assertAlmostEqual(plate.size[1], 1.0, places=5)
        for got, want in zip(plate.normal, normal):
            self.assertAlmostEqual(got, want, places=6)

    def test_without_a_normal_the_sign_falls_back_to_down(self):
        plate = self._cls().from_points(self._rect(2.0, 1.0))
        self.assertAlmostEqual(plate.normal[2], -1.0, places=6)

    def test_no_points_is_an_error_not_a_silent_default(self):
        with self.assertRaises(ValueError):
            self._cls().from_points([])


if __name__ == "__main__":
    unittest.main()
