# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.drape (CurtainDrape) — the pure draped-cloth
engine extracted from mayatk's curtain tool. The DCC adapters (mayatk
``CurtainMesh``, blendertk ``create_curtain``) only build a mesh from
:meth:`CurtainDrape.grid_points`, so the drape behavior is pinned here once,
DCC-free. (The rail geometry it consumes is tested in test_polyline.py; the
catenary/sag primitives themselves are MathUtils tests.)
"""

import unittest

from pythontk.geo_utils.drape import CurtainDrape
from pythontk.geo_utils.polyline import Polyline

STRAIGHT = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]


class TestCurtainDrape(unittest.TestCase):
    def test_rejects_degenerate_rail(self):
        with self.assertRaises(ValueError):
            CurtainDrape([(0.0, 0.0, 0.0)])

    def test_grid_shape_and_row_order(self):
        d = CurtainDrape(STRAIGHT, height=2.0)
        u_segs, v_segs, pts = d.grid_points()
        self.assertEqual(len(pts), (u_segs + 1) * (v_segs + 1))
        # Row 0 = hem (lowest), last row = rail (pinned at y=0 minus sag only).
        hem_y = max(p[1] for p in pts[: u_segs + 1])
        rail_y = min(p[1] for p in pts[-(u_segs + 1):])
        self.assertLess(hem_y, rail_y)

    def test_drop_matches_height_without_gravity(self):
        d = CurtainDrape(STRAIGHT, height=2.5, gravity=0.0, irregularity=0.0)
        u_segs, v_segs, pts = d.grid_points()
        top = pts[-(u_segs + 1):]
        hem = pts[: u_segs + 1]
        self.assertTrue(all(abs(p[1]) < 1e-9 for p in top))
        self.assertTrue(all(abs(p[1] + 2.5) < 1e-9 for p in hem))

    def test_gravity_sags_between_pinned_hang_points(self):
        d = CurtainDrape(
            STRAIGHT, height=2.0, hanging_points=4, gravity=0.5, irregularity=0.0
        )
        u_segs, v_segs, frames = d.prepare()
        ys = []
        for c in range(u_segs + 1):
            pos, tan, normal = frames[c]
            ys.append(d.drape(c / u_segs, 1.0, pos, tan, normal)[1])
        # Pinned at the hang points (u = 0, 1/3, 2/3, 1)...
        for hp in d._hang_points:
            col = round(hp * u_segs)
            self.assertAlmostEqual(ys[col], 0.0, places=2)
        # ...and sagging between them.
        self.assertLess(min(ys), -0.05)

    def test_more_gravity_sags_deeper(self):
        def deepest(g):
            d = CurtainDrape(STRAIGHT, gravity=g, irregularity=0.0)
            u_segs, _, frames = d.prepare()
            return min(
                d.drape(c / u_segs, 1.0, *frames[c])[1] for c in range(u_segs + 1)
            )

        self.assertLess(deepest(0.6), deepest(0.2))

    def test_fullness_drives_belly_depth(self):
        def z_range(fullness):
            d = CurtainDrape(STRAIGHT, fullness=fullness, irregularity=0.0)
            _, _, pts = d.grid_points()
            zs = [p[2] for p in pts]
            return max(zs) - min(zs)

        self.assertGreater(z_range(3.0), z_range(1.0) + 0.1)

    def test_hang_jitter_keeps_points_ordered_and_pinned_ends(self):
        d = CurtainDrape(STRAIGHT, hanging_points=8, hang_jitter=1.0, hang_seed=7)
        hp = d._hang_points
        self.assertEqual(hp[0], 0.0)
        self.assertEqual(hp[-1], 1.0)
        self.assertEqual(hp, sorted(hp))

    def test_closed_loop_adds_the_wrap_span(self):
        open_d = CurtainDrape(STRAIGHT, hanging_points=6)
        ring, closed = Polyline.make(width=6.0, closed=True)
        closed_d = CurtainDrape(ring, hanging_points=6, closed=closed)
        self.assertEqual(open_d.spans, 5)
        self.assertEqual(closed_d.spans, 6)

    def test_seeded_features_are_deterministic(self):
        a = CurtainDrape(STRAIGHT, creases=1.0, crease_seed=3, mid_folds=1.0,
                         mid_fold_seed=4, sway=1.0, sway_seed=5)
        b = CurtainDrape(STRAIGHT, creases=1.0, crease_seed=3, mid_folds=1.0,
                         mid_fold_seed=4, sway=1.0, sway_seed=5)
        a.prepare()
        b.prepare()
        self.assertEqual(a._creases, b._creases)
        self.assertEqual(a._midfolds, b._midfolds)
        self.assertEqual(a._sway, b._sway)

    def test_feature_seeds_change_the_pattern(self):
        a = CurtainDrape(STRAIGHT, creases=1.0, crease_seed=1)
        b = CurtainDrape(STRAIGHT, creases=1.0, crease_seed=2)
        a.prepare()
        b.prepare()
        self.assertNotEqual(a._creases, b._creases)

    def test_features_off_are_empty(self):
        d = CurtainDrape(STRAIGHT, irregularity=0.0)
        d.prepare()
        self.assertEqual(d._creases, [])
        self.assertEqual(d._midfolds, [])
        self.assertEqual(d._sway, [])
        self.assertIsNone(d._billow)

    def test_resolution_caps(self):
        d = CurtainDrape(STRAIGHT, density=10000.0, height=100.0)
        d._total_length = Polyline.length(d.rail, d.closed)
        u_segs, v_segs = d._resolve_resolution()
        self.assertLessEqual(u_segs, 4000)
        self.assertLessEqual(v_segs, 1000)


if __name__ == "__main__":
    unittest.main()
