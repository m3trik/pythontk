# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.geo_utils.rail_surface (RailSurface) — the general
rail-driven parametric-surface primitive. Covers grid shape/ordering, the
displacement seam, and rail framing. The curtain generator that composes it is
pinned separately in test_curtain.py.
"""
import unittest

from pythontk import RailSurface


STRAIGHT = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]


class TestRailSurface(unittest.TestCase):
    def test_rejects_degenerate_rail(self):
        with self.assertRaises(ValueError):
            RailSurface([(0.0, 0.0, 0.0)], 4, 4)

    def test_segment_counts_floor_at_one(self):
        rs = RailSurface(STRAIGHT, 0, -3)
        self.assertEqual(rs.u_segs, 1)
        self.assertEqual(rs.v_segs, 1)

    def test_frames_count_is_u_segs_plus_one(self):
        rs = RailSurface(STRAIGHT, 8, 3)
        self.assertEqual(len(rs.frames), 9)
        # each frame is (pos, tan, normal)
        pos, tan, normal = rs.frames[0]
        self.assertEqual(len(pos), 3)
        self.assertEqual(len(tan), 3)
        self.assertEqual(len(normal), 3)

    def test_length_of_straight_rail(self):
        self.assertAlmostEqual(RailSurface(STRAIGHT, 4, 4).length, 6.0)

    def test_grid_point_count_is_row_major_full_grid(self):
        rs = RailSurface(STRAIGHT, 5, 3)
        u, v, pts = rs.grid_points(lambda u, v, pos, tan, nrm: pos)
        self.assertEqual((u, v), (5, 3))
        self.assertEqual(len(pts), (5 + 1) * (3 + 1))

    def test_displace_receives_u_v_and_frame(self):
        rs = RailSurface(STRAIGHT, 2, 2)
        seen = []
        rs.grid_points(lambda u, v, pos, tan, nrm: seen.append((u, v)) or pos)
        # row-major: v outer (0, .5, 1), u inner (0, .5, 1)
        self.assertEqual(seen[0], (0.0, 0.0))
        self.assertEqual(seen[-1], (1.0, 1.0))
        # every u and v is a normalized 0..1 fraction
        self.assertTrue(all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in seen))

    def test_displace_output_is_used_verbatim(self):
        rs = RailSurface(STRAIGHT, 1, 1)
        _, _, pts = rs.grid_points(lambda u, v, pos, tan, nrm: (u, v, 7.0))
        # 2x2 grid, each vertex = (u, v, 7)
        self.assertIn((0.0, 0.0, 7.0), pts)
        self.assertIn((1.0, 1.0, 7.0), pts)
        self.assertTrue(all(z == 7.0 for _, _, z in pts))

    def test_vertical_drop_field(self):
        # A plain "hang straight down by (1-v)" field — the trivial curtain.
        rs = RailSurface(STRAIGHT, 3, 2)
        _, _, pts = rs.grid_points(
            lambda u, v, pos, tan, nrm: (pos[0], pos[1] - (1.0 - v), pos[2])
        )
        # hem row (v=0) drops a full unit; rail row (v=1) stays at rail height.
        hem = pts[0]          # row 0, col 0
        rail_row = pts[-1]    # last row, last col
        self.assertAlmostEqual(hem[1], -1.0)
        self.assertAlmostEqual(rail_row[1], 0.0)


if __name__ == "__main__":
    unittest.main()
