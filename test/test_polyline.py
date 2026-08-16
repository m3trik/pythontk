# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.polyline (Polyline) — the pure polyline/curve
geometry primitive (generate / measure / resample / frame). DCC-free; the
rail-surface primitive that frames it is pinned in test_rail_surface.py and the
curtain drape that consumes both in mayatk's test_curtain_drape.py (vendored
mayatk/blendertk, drift-guarded by extapps' test_vendor_sync.py).
"""

import math
import unittest
from unittest import mock

from pythontk.geo_utils.polyline import Polyline


class TestPolyline(unittest.TestCase):
    def test_default_line_is_straight_and_centered(self):
        pts, closed = Polyline.make(width=6.0)
        self.assertFalse(closed)
        self.assertAlmostEqual(pts[0][0], -3.0)
        self.assertAlmostEqual(pts[-1][0], 3.0)
        self.assertTrue(all(p[1] == 0.0 and abs(p[2]) < 1e-9 for p in pts))

    def test_curvature_bows_forward(self):
        pts, _ = Polyline.make(width=6.0, curvature=0.5)
        mid = pts[len(pts) // 2]
        self.assertGreater(mid[2], 0.0)

    def test_closed_makes_a_ring(self):
        pts, closed = Polyline.make(width=4.0, closed=True, segments=16)
        self.assertTrue(closed)
        radii = [math.hypot(p[0], p[2]) for p in pts]
        self.assertTrue(all(abs(r - 2.0) < 1e-6 for r in radii))

    def test_length_and_resample(self):
        pts, _ = Polyline.make(width=6.0, segments=24)
        self.assertAlmostEqual(Polyline.length(pts, False), 6.0, places=6)
        res = Polyline.resample(pts, 10)
        self.assertEqual(len(res), 10)
        gaps = [
            math.dist(res[i - 1], res[i]) for i in range(1, len(res))
        ]
        self.assertTrue(all(abs(g - gaps[0]) < 1e-6 for g in gaps))

    def test_frames_tangent_and_normal_are_unit_and_perpendicular(self):
        pts, _ = Polyline.make(width=6.0)
        frames = Polyline.frames(pts, 12, False)
        self.assertEqual(len(frames), 13)
        for pos, tan, normal in frames:
            self.assertAlmostEqual(math.hypot(*tan), 1.0, places=6)
            self.assertAlmostEqual(math.hypot(*normal), 1.0, places=6)
            dot = sum(a * b for a, b in zip(tan, normal))
            self.assertAlmostEqual(dot, 0.0, places=6)

    def test_custom_up_vector_reorients_the_normal(self):
        # A straight line along +X: world-Y up gives a normal along Z; flipping
        # the reference up to +Z must put the in-plane normal along Y instead.
        pts, _ = Polyline.make(width=6.0)
        z_up = Polyline.frames(pts, 4, False, up=(0.0, 0.0, 1.0))
        for _pos, _tan, normal in z_up:
            self.assertAlmostEqual(abs(normal[1]), 1.0, places=6)
            self.assertAlmostEqual(normal[2], 0.0, places=6)

    def test_frames_builds_the_arc_table_once(self):
        # frames() takes three samples per frame (position + two tangent
        # probes). Sampling through the one-shot point_at_arc would rebuild
        # the cumulative-length table on every one of them, making the cost
        # O(segments * len(points)) distance ops -- measured 127ms for a
        # 200-point rail at 200 segments. The table must be built once and
        # shared, so this pins the call count rather than a wall-clock time.
        pts = [(i * 0.37, (i % 7) * 0.11, (i % 13) * 0.05) for i in range(200)]
        with mock.patch.object(
            Polyline, "cumulative_lengths", wraps=Polyline.cumulative_lengths
        ) as spy:
            frames = Polyline.frames(pts, 64, False)

        self.assertEqual(len(frames), 65)
        self.assertEqual(
            spy.call_count, 1, f"arc table rebuilt {spy.call_count}x; expected once"
        )

    def test_point_at_interpolates_in_index_space(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 4.0, 0.0)]
        # t spans the two segments uniformly: 0.5 = end of the first segment.
        self.assertEqual(Polyline.point_at(pts, 0.0), [0.0, 0.0, 0.0])
        self.assertEqual(Polyline.point_at(pts, 0.5), [2.0, 0.0, 0.0])
        self.assertEqual(Polyline.point_at(pts, 0.75), [2.0, 2.0, 0.0])
        self.assertEqual(Polyline.point_at(pts, 1.0), [2.0, 4.0, 0.0])
        # t clamps to [0, 1]
        self.assertEqual(Polyline.point_at(pts, 2.0), [2.0, 4.0, 0.0])

    def test_point_at_arc_interpolates_in_arc_space(self):
        # Uneven segments: a short 2-unit leg then a long 4-unit leg (total 6).
        # Index space puts t=0.5 at the corner; arc space puts it 3 units
        # along, i.e. 1 unit UP the second leg.
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 4.0, 0.0)]
        self.assertEqual(Polyline.point_at_arc(pts, 0.0), [0.0, 0.0, 0.0])
        self.assertEqual(Polyline.point_at_arc(pts, 1.0), [2.0, 4.0, 0.0])
        for t, expected in ((1 / 6, [1.0, 0.0, 0.0]), (0.5, [2.0, 1.0, 0.0])):
            for got, want in zip(Polyline.point_at_arc(pts, t), expected):
                self.assertAlmostEqual(got, want, places=9)
        # ...and that is NOT where index space lands (the two must differ).
        self.assertNotEqual(
            Polyline.point_at_arc(pts, 0.5), Polyline.point_at(pts, 0.5)
        )
        # t clamps to [0, 1]
        self.assertEqual(Polyline.point_at_arc(pts, 2.0), [2.0, 4.0, 0.0])
        self.assertEqual(Polyline.point_at_arc(pts, -1.0), [0.0, 0.0, 0.0])

    def test_point_at_arc_degenerate_inputs(self):
        self.assertEqual(Polyline.point_at_arc([], 0.5), [])
        self.assertEqual(Polyline.point_at_arc([(1.0, 2.0, 3.0)], 0.5), [1.0, 2.0, 3.0])
        coincident = [(1.0, 2.0, 3.0)] * 4  # zero total length: no arc to walk
        self.assertEqual(Polyline.point_at_arc(coincident, 0.7), [1.0, 2.0, 3.0])

    def test_resample_with_arc_interpolation_is_evenly_spaced(self):
        # Uneven input spacing: index-space resampling clusters where the
        # source points cluster; arc-space resampling does not.
        pts = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (9.0, 0.0, 0.0)]
        arc = Polyline.resample(pts, 5, interpolation=Polyline.point_at_arc)
        gaps = [math.dist(arc[i - 1], arc[i]) for i in range(1, len(arc))]
        self.assertTrue(all(abs(g - gaps[0]) < 1e-9 for g in gaps))
        index = Polyline.resample(pts, 5)
        index_gaps = [math.dist(index[i - 1], index[i]) for i in range(1, len(index))]
        self.assertGreater(max(index_gaps) - min(index_gaps), 1.0)

    def test_cumulative_lengths_matches_length(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 4.0, 0.0)]
        self.assertEqual(Polyline.cumulative_lengths(pts), [0.0, 2.0, 6.0])
        self.assertAlmostEqual(
            Polyline.cumulative_lengths(pts)[-1], Polyline.length(pts), places=9
        )
        self.assertEqual(Polyline.cumulative_lengths([]), [0.0])
        self.assertEqual(Polyline.length([]), 0.0)

    def test_measurement_ignores_a_fourth_component(self):
        # DCC point types (om.MPoint and friends) carry a w component; every
        # measurement reads x/y/z by index, so w must not reach the distance.
        # A 4-length tuple stands in for MPoint so this stays DCC-free.
        flat = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
        homogeneous = [p + (1.0,) for p in flat]
        self.assertEqual(
            Polyline.cumulative_lengths(homogeneous),
            Polyline.cumulative_lengths(flat),
        )
        self.assertEqual(Polyline.length(homogeneous), Polyline.length(flat))
        self.assertEqual(
            Polyline.point_at_arc(homogeneous, 0.5), Polyline.point_at_arc(flat, 0.5)
        )
        # frames() is the DCC-facing entry (RailSurface -> curtain drape feeds
        # it om.MPoint), and it is the one that used to sample through
        # MathUtils.lerp -- which interpolates component-wise over the WHOLE
        # sequence, so a 4-component input came back as a 4-component position
        # carrying an interpolated w. Positions must be plain x/y/z.
        homogeneous_frames = Polyline.frames(homogeneous, 4, False)
        flat_frames = Polyline.frames(flat, 4, False)
        self.assertEqual(len(homogeneous_frames), len(flat_frames))
        for (h_pos, h_tan, h_nrm), (f_pos, f_tan, f_nrm) in zip(
            homogeneous_frames, flat_frames
        ):
            self.assertEqual(len(h_pos), 3, "w must not ride along on a position")
            self.assertEqual(list(h_pos), list(f_pos))
            self.assertEqual(tuple(h_tan), tuple(f_tan))
            self.assertEqual(tuple(h_nrm), tuple(f_nrm))

    def test_resample_reverse_and_offsets(self):
        pts, _ = Polyline.make(width=10.0)  # x from -5 to 5
        # reverse flips the order
        fwd = Polyline.resample(pts, 5)
        rev = Polyline.resample(pts, 5, reverse=True)
        self.assertEqual([p[0] for p in rev], [p[0] for p in fwd][::-1])
        # offsets trim the sampled span in from each end
        trimmed = Polyline.resample(pts, 5, start_offset=0.25, end_offset=0.25)
        self.assertGreater(trimmed[0][0], fwd[0][0])
        self.assertLess(trimmed[-1][0], fwd[-1][0])

    def test_resample_rejects_bad_offsets(self):
        pts, _ = Polyline.make(width=4.0)
        with self.assertRaises(ValueError):
            Polyline.resample(pts, 4, start_offset=0.6, end_offset=0.6)

    def test_smooth_averages_and_passes_through_below_window(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        # window_size <= 1 is a no-op (returns a copy of the input)
        self.assertEqual(Polyline.smooth(pts, 1), pts)
        # window of 2 = trailing moving average of the last two points
        sm = Polyline.smooth(pts, 2)
        self.assertEqual(len(sm), 3)
        self.assertEqual(sm[0], (0.0, 0.0, 0.0))
        self.assertEqual(sm[1], (1.0, 0.0, 0.0))
        self.assertEqual(sm[2], (3.0, 0.0, 0.0))


class TestOrderPoints(unittest.TestCase):
    # Regression: the default distance_metric used to be ``(p1 - p2).length()``
    # which only worked for PyMEL ``dt.Point`` / OpenMaya ``MPoint``. Plain
    # ``[x, y, z]`` lists (the documented input, and what ``cmds.pointPosition``
    # returns) raised TypeError because ``list - list`` is undefined.

    def test_order_points_with_lists(self):
        """Default distance_metric must handle plain [x, y, z] lists."""
        # Three colinear points along the X axis, given out of order.
        points = [[2.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]]
        ordered = Polyline.order_points(points)

        self.assertEqual(len(ordered), 3)
        # First point seeds the path; nearest neighbours follow.
        self.assertEqual(ordered[0], [2.0, 0, 0])
        self.assertEqual(ordered[1], [1.0, 0, 0])
        self.assertEqual(ordered[2], [0.0, 0, 0])

    def test_order_points_with_tuples(self):
        """Tuples should also work via the subscripting fallback."""
        points = [(0.0, 0, 0), (5.0, 0, 0), (10.0, 0, 0)]
        ordered = Polyline.order_points(points)
        self.assertEqual([p[0] for p in ordered], [0.0, 5.0, 10.0])

    def test_order_points_with_xyz_objects(self):
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
        ordered = Polyline.order_points(points)
        self.assertEqual([p.x for p in ordered], [2, 1, 0])

    def test_order_points_empty(self):
        """Empty input returns an empty list."""
        self.assertEqual(Polyline.order_points([]), [])

    def test_order_points_closed(self):
        """closed_path appends a copy of the first point at the end."""
        points = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
        ordered = Polyline.order_points(points, closed_path=True)
        self.assertEqual(len(ordered), 4)
        self.assertEqual(ordered[0], ordered[-1])

    def test_order_points_custom_metric(self):
        """Caller-supplied distance_metric overrides the default."""
        points = [[0.0, 0, 0], [3.0, 0, 0], [1.0, 0, 0]]
        # Manhattan distance — same ordering on this colinear set.
        ordered = Polyline.order_points(
            points,
            distance_metric=lambda a, b: abs(a[0] - b[0]),
        )
        self.assertEqual([p[0] for p in ordered], [0.0, 1.0, 3.0])

    def test_order_points_does_not_mutate_input(self):
        """Regression: the path walk used pop/remove on the caller's list,
        silently emptying it as a side effect."""
        points = [[0.0, 0, 0], [5.0, 0, 0], [1.0, 0, 0]]
        Polyline.order_points(points)
        self.assertEqual(points, [[0.0, 0, 0], [5.0, 0, 0], [1.0, 0, 0]])


class TestSimplify(unittest.TestCase):
    def test_simplify_collapses_straight_run(self):
        # collinear points collapse to just the two endpoints
        pts = [(x, 0, 0) for x in range(11)]
        self.assertEqual(Polyline.simplify(pts, 0.01), [0, 10])

    def test_simplify_keeps_corner(self):
        # an L: the corner must survive (it's the max-deviation point)
        pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 2, 0)]
        kept = Polyline.simplify(pts, 0.1)
        self.assertEqual(kept[0], 0)
        self.assertEqual(kept[-1], len(pts) - 1)
        self.assertIn(2, kept)  # the corner vertex

    def test_simplify_concentrates_on_bends(self):
        # straight ends with a sharp middle bend: kept points cluster at the
        # bend, the straight runs contribute (almost) nothing.
        pts = (
            [(x, 0.0, 0.0) for x in range(0, 10)]
            + [(10, 0, 0), (10.5, 1.5, 0), (11, 0, 0)]
            + [(x, 0.0, 0.0) for x in range(12, 21)]
        )
        kept = Polyline.simplify(pts, 0.2)
        bend_lo, bend_hi = 9, 13  # index window around the bend
        in_bend = sum(1 for i in kept if bend_lo <= i <= bend_hi)
        on_straight = sum(1 for i in kept if i < bend_lo or i > bend_hi)
        self.assertGreater(in_bend, on_straight)
        # tighter tolerance -> at least as many points kept (monotone refinement)
        self.assertGreaterEqual(len(Polyline.simplify(pts, 0.05)), len(kept))

    def test_simplify_edge_cases(self):
        self.assertEqual(Polyline.simplify([], 0.1), [])
        self.assertEqual(Polyline.simplify([(0, 0, 0)], 0.1), [0])
        self.assertEqual(Polyline.simplify([(0, 0, 0), (1, 1, 1)], 0.1), [0, 1])
        # tolerance <= 0 keeps everything
        pts = [(0, 0, 0), (1, 0.01, 0), (2, 0, 0)]
        self.assertEqual(Polyline.simplify(pts, 0.0), [0, 1, 2])

    def test_simplify_works_in_2d(self):
        pts = [(0, 0), (1, 0), (2, 0), (2, 2)]
        self.assertEqual(Polyline.simplify(pts, 0.1), [0, 2, 3])


class TestFromPointCloud(unittest.TestCase):
    @staticmethod
    def _tube_cloud(length=10.0, radius=1.0, rings=20, segs=8, axis=0):
        """A straight tube point cloud along *axis*: `rings` rings of `segs` verts each."""
        cloud = []
        for r in range(rings):
            t = r / (rings - 1)
            along = t * length
            for s in range(segs):
                a = 2 * math.pi * s / segs
                radial = [radius * math.cos(a), radius * math.sin(a)]
                p = [0.0, 0.0, 0.0]
                p[axis] = along
                others = [i for i in range(3) if i != axis]
                p[others[0]], p[others[1]] = radial
                cloud.append(p)
        return cloud

    def test_from_point_cloud_straight_tube(self):
        """A straight X-tube → an axis-aligned centerline on the tube's core (y≈z≈0)."""
        cloud = self._tube_cloud(length=10.0, radius=1.0, axis=0)
        cl = Polyline.from_point_cloud(cloud, count=6)
        self.assertEqual(len(cl), 6)
        # centered: y and z ~0 (the ring centers), spanning the tube length in x.
        self.assertTrue(all(abs(p[1]) < 1e-6 and abs(p[2]) < 1e-6 for p in cl))
        self.assertAlmostEqual(cl[0][0], 0.0, delta=0.6)
        self.assertAlmostEqual(cl[-1][0], 10.0, delta=0.6)
        # monotonic along x (ordered start->end)
        xs = [p[0] for p in cl]
        self.assertEqual(xs, sorted(xs))

    def test_from_point_cloud_auto_axis(self):
        """Auto-axis picks the longest extent (a Z-tube → centerline along Z)."""
        cloud = self._tube_cloud(length=8.0, radius=0.5, axis=2)
        cl = Polyline.from_point_cloud(cloud, count=5)
        self.assertEqual(len(cl), 5)
        self.assertTrue(all(abs(p[0]) < 1e-6 and abs(p[1]) < 1e-6 for p in cl))
        self.assertAlmostEqual(cl[-1][2] - cl[0][2], 8.0, delta=1.0)

    def test_from_point_cloud_accepts_xyz_objects(self):
        """Accepts .x/.y/.z objects (like Maya MPoint / bpy Vector), not just sequences."""
        class P:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        cloud = [P(*p) for p in self._tube_cloud(length=4.0, axis=0)]
        cl = Polyline.from_point_cloud(cloud, count=4)
        self.assertEqual(len(cl), 4)

    def test_from_point_cloud_degenerate(self):
        """Too few points / count<2 / zero extent → empty list (no crash)."""
        self.assertEqual(Polyline.from_point_cloud([], 5), [])
        self.assertEqual(Polyline.from_point_cloud([[0, 0, 0]], 5), [])
        self.assertEqual(Polyline.from_point_cloud([[0, 0, 0], [1, 0, 0]], 1), [])
        # all points coincident → zero extent on every axis
        self.assertEqual(Polyline.from_point_cloud([[1, 1, 1]] * 10, 5), [])


if __name__ == "__main__":
    unittest.main()
