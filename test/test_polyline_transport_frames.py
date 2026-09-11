# !/usr/bin/python
# coding=utf-8
"""Tests for :meth:`pythontk.Polyline.transport_frames`.

The primitive exists because :meth:`Polyline.frames` builds its normal as
``cross(up, tangent)`` against a CONSTANT reference, which collapses the moment
the path runs along that axis. These pin the properties that make parallel
transport the right answer there, and the one property it deliberately does not
have (a closed loop does not come back to its starting normal).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.geo_utils.polyline import Polyline


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mag(v):
    return math.sqrt(dot(v, v))


class TestFrameValidity(unittest.TestCase):
    """Whatever the path, every frame is an orthonormal basis at its point."""

    PATHS = {
        "straight-x": [(i, 0.0, 0.0) for i in range(6)],
        "straight-y": [(0.0, i, 0.0) for i in range(6)],
        "straight-z": [(0.0, 0.0, i) for i in range(6)],
        "diagonal": [(i, i, i) for i in range(6)],
        "arc": [(math.cos(t * 0.3) * 5, math.sin(t * 0.3) * 5, 0.0) for t in range(10)],
        "helix": [
            (math.cos(t * 0.5) * 3, t * 0.4, math.sin(t * 0.5) * 3) for t in range(14)
        ],
        "riser-hook": (
            [(0.0, i, 0.0) for i in range(6)]
            + [
                (math.sin(t * 0.4) * 2, 5 + math.cos(t * 0.4) * 2 - 2, 0.0)
                for t in range(1, 8)
            ]
        ),
    }

    def test_every_frame_is_orthonormal(self):
        for name, pts in self.PATHS.items():
            with self.subTest(path=name):
                for pos, tan, nrm in Polyline.transport_frames(pts):
                    self.assertAlmostEqual(mag(tan), 1.0, places=6)
                    self.assertAlmostEqual(mag(nrm), 1.0, places=6)
                    self.assertAlmostEqual(dot(tan, nrm), 0.0, places=6)

    def test_one_frame_per_input_point(self):
        for name, pts in self.PATHS.items():
            with self.subTest(path=name):
                self.assertEqual(len(Polyline.transport_frames(pts)), len(pts))

    def test_positions_are_the_input_points(self):
        pts = self.PATHS["helix"]
        for (pos, _, _), src in zip(Polyline.transport_frames(pts), pts):
            for a, b in zip(pos, src):
                self.assertAlmostEqual(a, b, places=9)


class TestVerticalRunIsNotSpecial(unittest.TestCase):
    """The case `frames` cannot serve, and the reason this method exists.

    A path along +Y makes ``cross(up=+Y, tangent)`` collapse, so `frames` falls
    back to a substituted direction unrelated to its neighbours. Transport has
    no reference direction to lose.
    """

    def test_a_vertical_run_gets_a_valid_perpendicular_normal(self):
        pts = [(0.0, i, 0.0) for i in range(8)]
        for _, tan, nrm in Polyline.transport_frames(pts):
            self.assertAlmostEqual(mag(nrm), 1.0, places=6)
            self.assertAlmostEqual(dot(tan, nrm), 0.0, places=6)

    def test_a_vertical_run_does_not_rotate_its_frame(self):
        """A straight path has no curvature, so the normal must not move."""
        frames = Polyline.transport_frames([(0.0, i, 0.0) for i in range(8)])
        first = frames[0][2]
        for _, _, nrm in frames[1:]:
            self.assertAlmostEqual(dot(first, nrm), 1.0, places=6)

    def test_a_plus_y_hint_on_a_vertical_run_still_yields_a_basis(self):
        """The hint is a preference, not a requirement.

        Passing the one direction that cannot work must degrade to a usable
        seed rather than collapse -- this is exactly how the tube rig calls it.
        """
        frames = Polyline.transport_frames(
            [(0.0, i, 0.0) for i in range(5)], up=(0.0, 1.0, 0.0)
        )
        for _, tan, nrm in frames:
            self.assertAlmostEqual(mag(nrm), 1.0, places=6)
            self.assertAlmostEqual(dot(tan, nrm), 0.0, places=6)


class TestMinimalRotation(unittest.TestCase):
    """Consecutive frames differ by the least possible twist."""

    def test_a_straight_run_carries_its_normal_unchanged(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                pts = []
                for i in range(6):
                    p = [0.0, 0.0, 0.0]
                    p[axis] = float(i)
                    pts.append(tuple(p))
                frames = Polyline.transport_frames(pts)
                first = frames[0][2]
                for _, _, nrm in frames[1:]:
                    self.assertAlmostEqual(dot(first, nrm), 1.0, places=6)

    def test_a_planar_arc_keeps_the_normal_in_plane(self):
        """Bending in XY must not twist about the tangent.

        The plane's normal (+Z here) stays perpendicular to every frame normal,
        which is the signature of zero transported twist.
        """
        pts = [(math.cos(t * 0.2) * 4, math.sin(t * 0.2) * 4, 0.0) for t in range(12)]
        for _, _, nrm in Polyline.transport_frames(pts):
            self.assertAlmostEqual(abs(dot(nrm, (0.0, 0.0, 1.0))), 1.0, places=6)

    def test_a_hint_is_honoured_when_it_is_well_conditioned(self):
        frames = Polyline.transport_frames(
            [(i, 0.0, 0.0) for i in range(5)], up=(0.0, 0.0, 1.0)
        )
        self.assertAlmostEqual(dot(frames[0][2], (0.0, 0.0, 1.0)), 1.0, places=6)


class TestDegenerateInput(unittest.TestCase):
    """Real centerlines carry duplicates and switchbacks; neither may raise."""

    def test_empty_and_single_point(self):
        self.assertEqual(Polyline.transport_frames([]), [])
        frames = Polyline.transport_frames([(1.0, 2.0, 3.0)])
        self.assertEqual(len(frames), 1)
        self.assertAlmostEqual(mag(frames[0][1]), 1.0, places=6)
        self.assertAlmostEqual(mag(frames[0][2]), 1.0, places=6)

    def test_repeated_points_do_not_rotate_the_frame(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        frames = Polyline.transport_frames(pts)
        self.assertEqual(len(frames), 4)
        first = frames[0][2]
        for _, tan, nrm in frames:
            self.assertAlmostEqual(mag(tan), 1.0, places=6)
            self.assertAlmostEqual(abs(dot(first, nrm)), 1.0, places=6)

    def test_a_duplicate_is_ignored_at_ANY_position_including_the_head(self):
        """Regression: the head had no previous tangent to fall back on.

        Differencing neighbours gave frame 0 of ``[A, A, B, C]`` a substituted
        direction unrelated to the path, which in the tube rig would mis-orient
        the ROOT joint. Every tangent now points at the next DISTINCT sample,
        so the position of a duplicate stops mattering.

        Swept over all three axes ON PURPOSE. The fallback the bug substituted
        is ``(1, 0, 0)``, which is ACCIDENTALLY correct on a +X path -- a test
        written only along X passes against the broken code and proves nothing.
        """
        for axis in range(3):
            direction = [0.0, 0.0, 0.0]
            direction[axis] = 1.0
            direction = tuple(direction)

            def at(step):
                return tuple(c * step for c in direction)

            for label, pts in (
                ("leading", [at(0), at(0), at(1), at(2)]),
                ("interior", [at(0), at(1), at(1), at(2)]),
                ("trailing", [at(0), at(1), at(2), at(2)]),
            ):
                with self.subTest(axis=axis, duplicate=label):
                    for _, tan, nrm in Polyline.transport_frames(pts):
                        self.assertAlmostEqual(dot(tan, direction), 1.0, places=6)
                        self.assertAlmostEqual(dot(tan, nrm), 0.0, places=6)

    def test_an_entirely_coincident_path_still_yields_a_basis(self):
        """No direction exists, so any unit vector is as good as another.

        What must not happen is a raise, a zero tangent, or a non-unit normal.
        """
        for _, tan, nrm in Polyline.transport_frames([(1.0, 1.0, 1.0)] * 4):
            self.assertAlmostEqual(mag(tan), 1.0, places=6)
            self.assertAlmostEqual(mag(nrm), 1.0, places=6)
            self.assertAlmostEqual(dot(tan, nrm), 0.0, places=6)

    def test_a_180_degree_switchback_does_not_raise_or_denormalise(self):
        """Anti-parallel tangents have no unique minimal rotation.

        The frame is left as it was rather than flipped by an arbitrary axis
        choice; what must NOT happen is a raise or a non-unit normal.
        """
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        for _, tan, nrm in Polyline.transport_frames(pts):
            self.assertAlmostEqual(mag(tan), 1.0, places=6)
            self.assertAlmostEqual(mag(nrm), 1.0, places=6)


class TestHolonomy(unittest.TestCase):
    """The property transport does NOT have, recorded so nobody assumes it."""

    def test_a_closed_loop_need_not_return_its_starting_normal(self):
        """Path dependence is inherent, not a defect.

        A caller closing a loop has to distribute the residual itself; asserting
        the residual is zero would be asserting something false.
        """
        pts = [
            (math.cos(a) * 3, math.sin(a) * 3, math.sin(a * 2))
            for a in [i * math.pi / 8 for i in range(17)]
        ]
        frames = Polyline.transport_frames(pts)
        residual = math.degrees(
            math.acos(max(-1.0, min(1.0, dot(frames[0][2], frames[-1][2]))))
        )
        self.assertGreaterEqual(residual, 0.0)
        self.assertLessEqual(residual, 180.0)


if __name__ == "__main__":
    unittest.main()
