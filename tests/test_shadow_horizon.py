# !/usr/bin/python
# coding=utf-8
"""``ShadowHorizon`` / ``HeightFieldMap`` (``geo_utils/shadow_horizon.py``).

The bake (depth-peeled spans, the distance field, the pyramid), the PNG
encoding round trip, and the reference -- ``HeightFieldMap.alpha`` -- against
the exact projection on the props that broke the map this one replaced.
"""

import base64
import json
import math
import os
import sys
import time
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythontk.geo_utils.shadow_horizon import (  # noqa: E402
    DIST_FAR,
    INTERIOR_MARKER,
    HeightFieldMap,
    HorizonMap,
    ShadowHorizon,
    ShadowHorizonConformance,
)
from pythontk.img_utils._img_utils import ImgUtils  # noqa: E402


class HorizonCase(unittest.TestCase):
    """Fixtures: axis-aligned boxes standing on the ground of a Y-up frame."""

    _box = staticmethod(ShadowHorizonConformance._box)

    @classmethod
    def box(cls):
        """A 2 x 2 x 2 box on the ground, centred on the contact."""
        return [cls._box(-1, 1, 0, 2, -1, 1)]

    @classmethod
    def pole(cls):
        """A 5 cm pole, 2 m tall, standing at (0.7, 0.7)."""
        return [cls._box(0.675, 0.725, 0, 2, 0.675, 0.725)]

    @classmethod
    def table(cls):
        """A 1.2 x 0.8 top, 5 cm thick at 0.7 m, on four 5 cm legs."""
        parts = [cls._box(-0.6, 0.6, 0.7, 0.75, -0.4, 0.4)]
        for x, z in ((-0.55, -0.35), (0.5, 0.3), (-0.55, 0.3), (0.5, -0.35)):
            parts.append(cls._box(x, x + 0.05, 0, 0.7, z, z + 0.05))
        return parts

    @classmethod
    def chair(cls):
        """A seat at 0.4 m, a back to 0.9 m, four 5 cm legs."""
        parts = [
            cls._box(-0.25, 0.25, 0.40, 0.45, -0.25, 0.25),
            cls._box(-0.25, 0.25, 0.45, 0.90, 0.20, 0.25),
        ]
        for x in (-0.23, 0.18):
            for z in (-0.23, 0.18):
                parts.append(cls._box(x, x + 0.05, 0.0, 0.40, z, z + 0.05))
        return parts

    @classmethod
    def stool(cls):
        """A seat at 0.45 m on four 3 cm legs tied by two stretchers at
        0.15 m: two solid spans in one column with daylight between."""
        parts = [cls._box(-0.2, 0.2, 0.45, 0.48, -0.2, 0.2)]
        for x in (-0.19, 0.16):
            for z in (-0.19, 0.16):
                parts.append(cls._box(x, x + 0.03, 0.0, 0.45, z, z + 0.03))
        for z in (-0.19, 0.16):
            parts.append(cls._box(-0.16, 0.16, 0.15, 0.18, z, z + 0.03))
        return parts

    @classmethod
    def arch(cls):
        """Two posts and a lintel: a floating member over open ground."""
        return [
            cls._box(-0.65, -0.55, 0.0, 1.8, -0.05, 0.05),
            cls._box(0.55, 0.65, 0.0, 1.8, -0.05, 0.05),
            cls._box(-0.7, 0.7, 1.8, 1.95, -0.05, 0.05),
        ]

    @staticmethod
    def _extent(meshes):
        allp = np.concatenate([p for p, _ in meshes])
        mn, mx = allp.min(0), allp.max(0)
        return 0.5 * math.hypot(mx[0] - mn[0], mx[2] - mn[2]), float(mx[1])


class TestHeightSpans(HorizonCase):
    """ImgUtils.rasterize_height_spans -- the bake's solid columns."""

    def test_two_parts_meeting_a_float_apart_bake_one_span(self):
        """Bug: an awning's post and slab, two boxes in one mesh, meet where
        the post's top is the slab's bottom; an engine's import of one DCC's
        delivery put them one float32 step apart (1.0999999 and 1.1, measured
        2026-11-08), and the bake kept the 1e-7 gap as daylight: a second span
        over the post the delivered map (baked where they coincide) has not.
        A gap no wider than one step of the map's 16-bit height code is none
        -- it encodes as touching -- so the map is one span there, as with
        the faces coincident. Fixed: 2026-11-08"""
        under = float(np.nextafter(np.float32(1.1), np.float32(0.0)))
        for post_top in (1.1, under):
            with self.subTest(post_top=post_top):
                awning = [
                    self._box(-0.06, 0.06, 0.0, post_top, -0.06, 0.06),
                    self._box(-0.3, 0.5, 1.1, 1.18, -0.2, 0.2),
                ]
                hmap = ShadowHorizon.bake(awning, up=1, size=64, spans=2)
                a0, a1, b0, b1 = hmap.bounds
                ix = int((0.0 - a0) / (a1 - a0) * hmap.size)
                iy = int((0.0 - b0) / (b1 - b0) * hmap.size)
                self.assertTrue(np.isnan(hmap.hi[1, iy, ix]), "one span over the post")
                self.assertEqual(float(hmap.lo[0, iy, ix]), 0.0)
                self.assertAlmostEqual(float(hmap.hi[0, iy, ix]), 1.18, places=4)
                self.assertTrue(np.isnan(hmap.hi[1]).all(), "nowhere a second span")

    def test_box_gives_a_solid_column(self):
        lo, hi, bounds = ImgUtils.rasterize_height_spans(
            self.box(), up=1, size=32, spans=2
        )
        self.assertTrue(np.isnan(hi[1]).all(), "one span per column")
        self.assertAlmostEqual(float(lo[0, 16, 16]), 0.0, places=6)
        self.assertAlmostEqual(float(hi[0, 16, 16]), 2.0, places=6)
        a0, a1, b0, b1 = bounds
        self.assertLess(a0, -1.0)
        self.assertGreater(a1, 1.0)

    def test_a_stretcher_under_a_seat_keeps_daylight(self):
        """The failure the horizon map could not represent: two solid spans
        in one column. The hull (one span) fills the gap between them."""
        lo, hi, _ = ImgUtils.rasterize_height_spans(
            self.stool(), up=1, size=64, spans=2
        )
        both = ~np.isnan(hi[1])
        self.assertGreater(int(both.sum()), 100, "columns with two spans")
        iy, ix = np.argwhere(both)[0]
        self.assertAlmostEqual(float(lo[0, iy, ix]), 0.15, places=6)
        self.assertAlmostEqual(float(hi[0, iy, ix]), 0.18, places=6)
        self.assertAlmostEqual(float(lo[1, iy, ix]), 0.45, places=6)
        self.assertAlmostEqual(float(hi[1, iy, ix]), 0.48, places=6)
        z_top, z_bot, mask, _ = ImgUtils.rasterize_height_fields(
            self.stool(), up=1, size=64
        )
        self.assertTrue(mask[iy, ix])
        self.assertAlmostEqual(float(z_bot[iy, ix]), 0.15, places=6)
        self.assertAlmostEqual(float(z_top[iy, ix]), 0.48, places=6)

    def test_the_hull_is_the_union_of_the_spans(self):
        lo, hi, _ = ImgUtils.rasterize_height_spans(
            self.chair(), up=1, size=64, spans=3
        )
        z_top, z_bot, mask, _ = ImgUtils.rasterize_height_fields(
            self.chair(), up=1, size=64
        )
        union_hi = np.fmax.reduce(hi, axis=0)  # NaN only where no span is
        union_lo = np.fmin.reduce(lo, axis=0)
        self.assertTrue(np.array_equal(mask, ~np.isnan(union_hi)))
        np.testing.assert_allclose(z_top[mask], union_hi[mask])
        np.testing.assert_allclose(z_bot[mask], union_lo[mask])

    def test_shared_edges_do_not_split_a_column(self):
        """The two triangles of a quad both report a pixel centre on their
        shared edge; paired blindly those duplicates made zero-thickness
        spans, and the box's centre column read as two slivers."""
        lo, hi, _ = ImgUtils.rasterize_height_spans(self.box(), up=1, size=16, spans=4)
        present = ~np.isnan(hi)
        self.assertEqual(int(present[1:].sum()), 0)
        solid = present[0]
        self.assertTrue((hi[0][solid] > 1.99).all())

    def test_a_ray_along_an_edge_two_triangles_share_counts_once(self):
        """Watertight: a box split on either diagonal bakes the same solid,
        every column of its footprint full height. Maya's polyCube split
        (the ShadowPlane sample's box) left 6 columns on its faces'
        diagonals with only a zero-thickness span at the top: the ray down
        the bottom face's shared edge missed both its triangles. Added:
        2026-11-07."""
        extents = (-30, 30, 0, 60, -30, 30)
        maps = [
            ShadowHorizon.bake([mesh], size=128, spans=2)
            for mesh in (
                ShadowHorizonConformance._box_flipped(*extents),
                self._box(*extents),
            )
        ]
        for hmap in maps:
            a0, a1, b0, b1 = hmap.bounds
            c = (np.arange(hmap.size) + 0.5) / hmap.size
            inside = (np.abs(b0 + c * (b1 - b0))[:, None] < 30) & (
                np.abs(a0 + c * (a1 - a0))[None, :] < 30
            )
            full = (hmap.lo[0] <= 0.0) & (hmap.hi[0] >= 60.0)
            self.assertEqual(np.argwhere(inside & ~full).tolist(), [])
        np.testing.assert_array_equal(maps[0].to_rgba(), maps[1].to_rgba())

    def test_a_point_on_a_shared_edge_or_vertex_is_one_triangles(self):
        """The fill rule: a pixel centre on an edge two triangles share, or
        on the vertex a fan shares, is covered by exactly one of them -- an
        inclusive test gave it to all of them (the duplicates removed after)
        or, a rounding below zero each side, to none. Added: 2026-11-07."""
        from pythontk.img_utils._rasterize import _ImgRasterizeInternal

        pts, tris = ShadowHorizonConformance._fan()
        x = (pts[:, 0] + 1.0) / 2.0 * 16
        y = (pts[:, 2] + 1.0) / 2.0 * 16
        yy, xx = np.mgrid[0:16, 0:16] + 0.5
        top = [t for t in tris if (pts[t, 1] == 1.0).all()]
        self.assertEqual(len(top), 8)
        covered = np.zeros((16, 16), int)
        for t in top:
            xs, ys = x[t], y[t]
            ccw = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0]) > 0
            covered += _ImgRasterizeInternal._covers(xs, ys, ccw, xx, yy)
        self.assertTrue((covered[4:12, 4:12] == 1).all(), covered[4:12, 4:12])
        self.assertLessEqual(int(covered.max()), 1)

    def test_thin_member_registers_at_pixel_resolution(self):
        lo, hi, _ = ImgUtils.rasterize_height_spans(self.pole(), up=1, size=16, spans=1)
        self.assertGreaterEqual(int((~np.isnan(hi[0])).sum()), 1)
        self.assertAlmostEqual(float(np.nanmax(hi[0])), 2.0, places=6)

    def test_buried_geometry_blocks_nothing(self):
        lo, hi, _ = ImgUtils.rasterize_height_spans(
            [self._box(-1, 1, -2, -0.5, -1, 1)], up=1, size=16
        )
        self.assertTrue(np.isnan(hi).all())

    def test_a_mesh_cut_by_the_ground_keeps_its_column_from_the_ground_up(self):
        """A box straddling the ground plane (the DCC rigs' own fixture, a
        cube centred on the origin) is solid from the ground to its top.
        Dropping its buried floor face -- as the old hull did -- left only
        the top face's crossing: a span of no thickness at the top, and a
        ray under it passed straight through."""
        lo, hi, _ = ImgUtils.rasterize_height_spans(
            [self._box(-1, 1, -1, 1, -1, 1)], up=1, size=16, spans=2
        )
        self.assertAlmostEqual(float(lo[0, 8, 8]), 0.0, places=6)
        self.assertAlmostEqual(float(hi[0, 8, 8]), 1.0, places=6)
        self.assertTrue(np.isnan(hi[1, 8, 8]))
        hmap = ShadowHorizon.bake([self._box(-1, 1, -1, 1, -1, 1)], up=1, size=32)
        self.assertAlmostEqual(hmap.height_scale, 1.0, places=3)
        self.assertEqual(float(hmap.alpha([[-2.0, 0.0, 0.0]], [6.0, 3.0, 0.0])[0]), 1.0)


class TestEncoding(HorizonCase):
    """The map's quantisation, its PNG image and the pyramid."""

    def test_rgba_round_trip_is_exact(self):
        hmap = ShadowHorizon.bake(self.stool(), up=1, size=64)
        img = hmap.to_rgba()
        self.assertEqual(img.shape, (64, 64 * hmap.tiles, 4))
        self.assertEqual(img.dtype, np.uint8)
        back = HeightFieldMap.from_rgba(
            img,
            size=hmap.size,
            spans=hmap.spans,
            bounds=hmap.bounds,
            ground=hmap.ground,
            up=hmap.up,
            height_scale=hmap.height_scale,
        )
        for name in ("lo", "hi", "dist", "near_lo", "near_hi", "interior"):
            a, b = getattr(hmap, name), getattr(back, name)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b), name)
            np.testing.assert_allclose(
                np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=1e-6, err_msg=name
            )

    def test_the_interior_field_rides_the_pyramid_tiles_free_half_under_a_marker(self):
        """The field the inner penumbra reads: every solid pixel's distance
        to the nearest empty one, in the pyramid tile's right half with the
        marker texel, so the layout, the encoding and the record stay as
        they were and a map baked before it still reads (as ``None``)."""
        hmap = ShadowHorizon.bake(self.box(), up=1, size=32)
        S, K = hmap.size, hmap.spans
        solid = ~np.isnan(hmap.hull()[1])
        self.assertTrue((hmap.interior[~solid] == 0.0).all())
        self.assertTrue((hmap.interior[solid] >= 1.0).all())
        # the box's centre is half its width from the nearest empty pixel
        a0, a1, _, _ = hmap.bounds
        half = 1.0 / ((a1 - a0) / S)
        self.assertAlmostEqual(float(hmap.interior.max()), half, delta=1.0)
        img = hmap.to_rgba()
        pyr = img[:, (K + 1) * S : (K + 2) * S]
        self.assertEqual(tuple(int(v) for v in pyr[S // 2, S // 2]), INTERIOR_MARKER)
        # the pyramid's own levels never reach the right half
        self.assertEqual(int(pyr[S // 2 + 1 :, S // 2 :].sum()), 0)
        old = hmap._replace(interior=None).to_rgba()
        self.assertEqual(int(old[:, (K + 1) * S + S // 2 :].sum()), 0)
        back = HeightFieldMap.from_rgba(
            old,
            size=S,
            spans=K,
            bounds=hmap.bounds,
            ground=hmap.ground,
            up=hmap.up,
            height_scale=hmap.height_scale,
        )
        self.assertIsNone(back.interior)

    def test_the_size_rounds_up_to_a_power_of_two(self):
        hmap = ShadowHorizon.bake(self.box(), up=1, size=100)
        self.assertEqual(hmap.size, 128)
        self.assertEqual(hmap.levels, 7)
        self.assertEqual(len(hmap.pyramid()), 7)

    def test_the_pyramid_bounds_every_read_the_march_makes_in_a_column_cell(self):
        """A column cell's hull covers each pixel's own spans and the
        nearest hull of every pixel in the cell's one-pixel ring (what the
        bilinear penumbra read reaches); a cell without a column carries no
        hull, and the distance flags exactly the cells that hold one."""
        hmap = ShadowHorizon.bake(self.chair(), up=1, size=64)
        S = hmap.size
        hull_lo, hull_hi = hmap.hull()
        own_lo = np.nan_to_num(hull_lo, nan=np.inf)
        own_hi = np.nan_to_num(hull_hi, nan=-np.inf)
        ring_lo = np.pad(hmap.near_lo, 1, constant_values=np.inf)
        ring_hi = np.pad(hmap.near_hi, 1, constant_values=-np.inf)
        for level, (lo, hi, dist) in enumerate(hmap.pyramid(), start=1):
            c = 1 << level
            n = S >> level
            column = hmap.dist.reshape(n, c, n, c).min(axis=(1, 3)) == 0
            np.testing.assert_array_equal(dist == 0, column, f"level {level}")
            self.assertTrue(np.isnan(lo[~column]).all() and np.isnan(hi[~column]).all())
            self.assertTrue(
                np.isfinite(lo[column]).all() and np.isfinite(hi[column]).all()
            )
            for iy, ix in zip(*np.nonzero(column)):
                rows, cols = slice(iy * c, (iy + 1) * c), slice(ix * c, (ix + 1) * c)
                ring_rows = slice(iy * c, (iy + 1) * c + 2)
                ring_cols = slice(ix * c, (ix + 1) * c + 2)
                floor_ = min(
                    own_lo[rows, cols].min(), ring_lo[ring_rows, ring_cols].min()
                )
                ceil_ = max(
                    own_hi[rows, cols].max(), ring_hi[ring_rows, ring_cols].max()
                )
                self.assertLessEqual(float(lo[iy, ix]), float(floor_) + 1e-6)
                self.assertGreaterEqual(float(hi[iy, ix]), float(ceil_) - 1e-6)

    def test_the_distance_field_is_the_euclidean_distance_to_the_nearest_column(self):
        hmap = ShadowHorizon.bake(self.pole(), up=1, size=32)
        solid = hmap.dist == 0
        self.assertTrue(solid.any())
        ys, xs = np.nonzero(solid)
        iy, ix = 3, 5
        expected = float(np.min(np.hypot(ys - iy, xs - ix)))
        self.assertAlmostEqual(float(hmap.dist[iy, ix]), expected, places=1)
        self.assertAlmostEqual(float(hmap.near_hi[iy, ix]), 2.0, delta=0.01)

    def test_an_elongated_footprint_measures_distance_in_the_smaller_pitch(self):
        """A 2:1 footprint: a column step counts two units, a row step one,
        so the penumbra's lateral clearance is the same length in frame
        units whichever way the ray leaves the column."""
        hmap = ShadowHorizon.bake(self.pole(), up=1, size=32, bounds=(-2, 2, -1, 1))
        self.assertAlmostEqual(hmap.aspect, 2.0, places=6)
        ys, xs = np.nonzero(hmap.dist == 0)
        iy = int(ys[len(ys) // 2])
        ix_max = int(xs[ys == iy].max())
        ix = int(xs[len(xs) // 2])
        iy_max = int(ys[xs == ix].max())
        self.assertAlmostEqual(float(hmap.dist[iy, ix_max + 3]), 6.0, places=1)
        self.assertAlmostEqual(float(hmap.dist[iy_max + 3, ix]), 3.0, places=1)

    def test_an_empty_map_reports_the_far_distance(self):
        hmap = ShadowHorizon.bake([self._box(-1, 1, -2, -0.5, -1, 1)], up=1, size=16)
        self.assertTrue((hmap.dist == DIST_FAR).all())
        self.assertTrue(np.isnan(hmap.hi).all())
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]])
        np.testing.assert_array_equal(hmap.alpha(pts, [5.0, 5.0, 5.0]), [0.0, 0.0])

    def test_the_record_block_spells_the_schema_the_readers_expect(self):
        """What a reader reads and nothing it computes: a map's pyramid levels
        are log2 of its size, and the one mapping and encoding a horizon map
        has are no field (2026-10-20)."""
        block = ShadowHorizon.record(
            texture="a_horizon.png",
            size=128,
            spans=2,
            bounds=(-1.2345678, 1.2345678, -0.5, 0.5),
            height_scale=2.0,
            frame_a=(1, 0, 0),
            frame_b=(0, 0, 1),
            rect=[1.0, 1.0, 0.0, 0.0],
        )
        self.assertEqual(
            sorted(block),
            sorted(
                [
                    "texture",
                    "size",
                    "spans",
                    "bounds",
                    "height_scale",
                    "frame_a",
                    "frame_b",
                    "rect",
                ]
            ),
        )
        self.assertEqual(block["bounds"], [-1.234568, 1.234568, -0.5, 0.5])
        self.assertEqual(block["frame_a"], [1, 0, 0])

    def test_the_retired_horizon_fields_warn_and_change_nothing(self):
        """``ShadowHorizon.record(levels=)``, ``MAPPING`` and ``ENCODING``
        (retired 2026-10-20, removed in 0.18.0): a map's levels are log2 of
        its size, and a horizon map has one mapping and one encoding, so the
        record states none. The parameter warns and is ignored; the constants
        warn and serve their values."""
        kwargs = dict(
            texture="a_horizon.png",
            size=128,
            spans=2,
            bounds=(-1.0, 1.0, -0.5, 0.5),
            height_scale=2.0,
            frame_a=(1, 0, 0),
            frame_b=(0, 0, 1),
            rect=[1.0, 1.0, 0.0, 0.0],
        )
        with self.assertWarns(DeprecationWarning):
            leveled = ShadowHorizon.record(levels=7, **kwargs)
        self.assertEqual(leveled, ShadowHorizon.record(**kwargs))
        for name in ("MAPPING", "ENCODING"):
            with self.subTest(name=name), self.assertWarns(DeprecationWarning):
                self.assertIsNotNone(getattr(ShadowHorizon, name))

    def test_horizon_map_is_the_old_name_for_one_release(self):
        self.assertIs(HorizonMap, HeightFieldMap)


class TestReference(HorizonCase):
    """HeightFieldMap.alpha against the geometry and the exact projection."""

    def test_a_pole_casts_one_shadow_on_the_true_bearing(self):
        hmap = ShadowHorizon.bake(self.pole(), up=1, size=64)
        light = np.array([-3.0, 4.0, 0.7])  # from -X, 4 m up
        # the shadow runs from the pole away from the light: +X
        on = np.array([[1.5, 0.0, 0.7], [2.5, 0.0, 0.7]])
        off = np.array([[1.5, 0.0, 0.2], [-0.5, 0.0, 0.7], [0.7, 0.0, 1.5]])
        np.testing.assert_array_equal(hmap.alpha(on, light), [1.0, 1.0])
        np.testing.assert_array_equal(hmap.alpha(off, light), [0.0, 0.0, 0.0])
        # the far end: the light through the pole's top (0.7, 2) lands at
        # x = 0.7 + 3.7 * 2 / 2 = 4.4
        self.assertEqual(float(hmap.alpha([[4.0, 0.0, 0.7]], light)[0]), 1.0)
        self.assertEqual(float(hmap.alpha([[4.8, 0.0, 0.7]], light)[0]), 0.0)

    def test_the_height_is_replaced_by_the_ground_plane(self):
        hmap = ShadowHorizon.bake(self.box(), up=1, size=32)
        light = np.array([6.0, 3.0, 0.0])
        lifted = np.array([[-2.0, 0.5, 0.0]])  # a plane lifted above the ground
        ground = np.array([[-2.0, 0.0, 0.0]])
        self.assertEqual(hmap.alpha(lifted, light)[0], hmap.alpha(ground, light)[0])
        self.assertEqual(float(hmap.alpha(ground, light)[0]), 1.0)

    def test_nothing_from_below_the_ground(self):
        hmap = ShadowHorizon.bake(self.box(), up=1, size=32)
        pts = np.array([[-2.0, 0.0, 0.0]])
        self.assertEqual(float(hmap.alpha(pts, [6.0, -1.0, 0.0])[0]), 0.0)
        self.assertEqual(float(hmap.alpha(pts, direction=[-1.0, 0.2, 0.0])[0]), 0.0)

    def test_directional_source_matches_a_far_positional_one(self):
        hmap = ShadowHorizon.bake(self.table(), up=1, size=64)
        rng = np.random.default_rng(3)
        pts = np.column_stack(
            [rng.uniform(-3, 3, 400), np.zeros(400), rng.uniform(-3, 3, 400)]
        )
        d = np.array([-0.6, -0.5, -0.3])
        d /= np.linalg.norm(d)
        far = hmap.alpha(pts, -d * 1e4)
        directional = hmap.alpha(pts, direction=d)
        self.assertGreater(float(far.mean()), 0.02)
        self.assertLess(float(np.abs(far - directional).mean()), 0.01)

    def test_a_source_size_straddles_the_hard_edge(self):
        """A round source's penumbra is centred on the hard shadow's edge:
        half the disc hidden there, the umbra eaten from inside as much as
        the shadow grows outside, and nothing under the box changes."""
        hmap = ShadowHorizon.bake(self.box(), up=1, size=64)
        light = np.array([6.0, 4.0, 0.5])
        # the light through the box's far top edge (-1, 2) lands at x = -8
        xs = np.linspace(-12.0, 0.0, 600)
        pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
        hard = hmap.alpha(pts, light)
        soft = hmap.alpha(pts, light, source_size=1.2)
        self.assertTrue(set(np.unique(hard)) <= {0.0, 1.0})
        self.assertEqual(float(hard[np.argmin(np.abs(xs + 7.5))]), 1.0)
        self.assertEqual(float(hard[np.argmin(np.abs(xs + 8.5))]), 0.0)
        between = (soft > 0.05) & (soft < 0.95)
        edge = xs[np.argmax(hard > 0.5)]  # the hard shadow's far end
        self.assertAlmostEqual(float(np.interp(edge, xs, soft)), 0.5, delta=0.1)
        self.assertGreater(int((between & (xs < edge)).sum()), 5, "outside")
        self.assertGreater(int((between & (xs > edge)).sum()), 5, "inside")
        self.assertTrue((soft[hard == 0.0] < 0.5).all())
        self.assertTrue((soft[hard == 1.0] >= 0.5 - 1e-6).all())
        # deep inside the umbra and under the box, nothing moves
        np.testing.assert_array_equal(soft[xs > -3.0], hard[xs > -3.0])

    def _disc_integral(self, hmap, points, toward, angle, n=96):
        """The physical answer: the share of a disc source of angular
        diameter *angle* hidden, as the mean of *n* hard rays spread evenly
        over the disc (a Fibonacci spiral) -- the same map, so the
        representation cancels and only the penumbra model is measured."""
        c = np.asarray(toward, float) / np.linalg.norm(toward)
        u = np.cross(c, [0.0, 0.0, 1.0] if abs(c[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(c, u)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        total = np.zeros(len(points))
        for i in range(n):
            r = 0.5 * angle * math.sqrt((i + 0.5) / n)
            d = c + math.tan(r) * (math.cos(i * golden) * u + math.sin(i * golden) * v)
            total += hmap.alpha(points, direction=-d / np.linalg.norm(d))
        return total / n

    def test_the_penumbra_matches_the_area_light_integral(self):
        """Low sun and high sun: the penumbra the reference draws against
        the disc-integrated hard shadow on the same map. The model it
        replaced -- clearance over the horizontal distance, a linear ramp
        outward from the hard edge -- scored 0.18-0.31 mean error over the
        band on this box and the chair and drew it 2-8x too narrow, worst
        under the high source; this one scores 0.05-0.09 (measured
        2026-10-10). A high sun is where 1 / cos^2 of the elevation bites."""
        hmap = ShadowHorizon.bake(self.box(), up=1, size=64)
        angle = math.radians(6.0)
        for elev in (30.0, 65.0):
            el = math.radians(elev)
            toward = np.array([math.cos(el), math.sin(el), 0.25])
            reach = 2.0 / math.tan(el)
            xs = np.linspace(-reach - 2.5, -0.5, 240)
            zs = np.linspace(-2.2, 2.2, 240)
            pts = np.concatenate(
                [
                    np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)]),
                    np.column_stack([np.full_like(zs, -1.5), np.zeros_like(zs), zs]),
                ]
            )
            truth = self._disc_integral(hmap, pts, toward, angle)
            got = hmap.alpha(pts, direction=-toward, source_angle=angle)
            band = (truth > 0.02) & (truth < 0.98)
            self.assertGreater(int(band.sum()), 10, f"a penumbra at {elev}")
            err = np.abs(got - truth)[band]
            self.assertLess(float(err.mean()), 0.12, f"band error at {elev}")
            # as wide as the truth's, within a third
            mine = int(((got > 0.02) & (got < 0.98)).sum())
            self.assertLess(abs(mine - int(band.sum())), 0.34 * band.sum(), elev)

    def test_a_member_thinner_than_its_penumbra_greys_out(self):
        """A 5 cm pole under a broad source: solid black at its foot, a grey
        smear with no umbra where its shadow is wider than the pole -- what
        a real one does, and what an outward-only ramp could not."""
        hmap = ShadowHorizon.bake(self.pole(), up=1, size=128)
        light = np.array([-3.0, 4.0, 0.7])
        xs = np.linspace(0.75, 4.4, 300)
        zs = np.linspace(0.3, 1.1, 161)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
        soft = hmap.alpha(pts, light, source_size=0.6).reshape(X.shape)
        across = soft.max(axis=1)  # darkest across the shadow, per x
        self.assertGreaterEqual(
            float(across[xs < 0.9].max()), 0.95, "black at the foot"
        )
        far = xs > 3.0
        self.assertLess(float(across[far].max()), 0.9, "no umbra far out")
        self.assertGreater(float(across[far].min()), 0.05, "still a shadow")

    def test_a_map_baked_before_the_interior_field_still_marches(self):
        """No field (a map from an earlier bake): the inner penumbra comes
        from heights alone -- the far edge still straddles, the umbra deep
        inside stays black, and the lit side is unchanged."""
        hmap = ShadowHorizon.bake(self.box(), up=1, size=64)
        old = hmap._replace(interior=None)
        light = np.array([6.0, 4.0, 0.5])
        xs = np.linspace(-12.0, 0.0, 600)
        pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
        a = old.alpha(pts, light, source_size=1.2)
        b = hmap.alpha(pts, light, source_size=1.2)
        self.assertTrue(((a >= 0.0) & (a <= 1.0)).all())
        np.testing.assert_array_equal(a[xs > -3.0], 1.0)
        lit = b < 0.5
        np.testing.assert_allclose(a[lit], b[lit], atol=1e-6)

    def test_blender_z_up_matches_maya_y_up(self):
        """The same prop baked in a Z-up frame (Blender) evaluates the same
        way once the points and the light are expressed in that frame."""
        y_up = ShadowHorizon.bake(self.chair(), up=1, size=64)
        swapped = [(pts[:, [0, 2, 1]], tris) for pts, tris in self.chair()]
        z_up = ShadowHorizon.bake(swapped, up=2, size=64)
        rng = np.random.default_rng(1)
        pts = np.column_stack(
            [rng.uniform(-2, 2, 300), np.zeros(300), rng.uniform(-2, 2, 300)]
        )
        light = np.array([3.0, 2.0, -1.0])
        a = y_up.alpha(pts, light)
        b = z_up.alpha(pts[:, [0, 2, 1]], light[[0, 2, 1]])
        self.assertGreater(float(a.mean()), 0.05)
        np.testing.assert_allclose(a, b, atol=1e-6)


class TestMeasure(HorizonCase):
    """The reference against the exact projection, and the bake's budget."""

    def test_fixtures_within_the_measured_bounds(self):
        """One-texel-tolerant disagreement at the defaults, ``samples=6``,
        ``size=192``, measured 2026-09-05: box 0.000, table 0.000, chair
        0.000, stool 0.000, arch 0.002 -- the raw scores 1.1 / 2.2 / 1.8 /
        2.3 / 4.8 % (edge registration on thin shadows). The horizon map
        this replaced scored 1.2 / 3.0 / 4.9 / 15 / 15 % tolerant. Asserted
        with headroom on both."""
        for name, meshes, bound_tol, bound_raw in (
            ("box", self.box(), 0.01, 0.03),
            ("table", self.table(), 0.01, 0.05),
            ("chair", self.chair(), 0.01, 0.05),
            ("stool", self.stool(), 0.01, 0.06),
            ("arch", self.arch(), 0.02, 0.09),
        ):
            radius, height = self._extent(meshes)
            hmap = ShadowHorizon.bake(meshes, up=1)
            score = ShadowHorizon.measure(
                hmap, meshes, samples=6, size=192, radius=radius, height=height
            )
            self.assertEqual(score["samples"], 6)
            self.assertLess(score["tolerant_mean"], bound_tol, f"{name}: {score}")
            self.assertLess(score["mean"], bound_raw, f"{name}: {score}")

    def test_adaptive_size_gates_on_the_tolerant_score(self):
        hmap, score = ShadowHorizon.bake_adaptive(
            self.chair(), up=1, threshold=0.05, measure_samples=3
        )
        self.assertIn(score["size"], ShadowHorizon.ADAPTIVE_SIZES)
        self.assertEqual(hmap.size, score["size"])
        if score["size"] != ShadowHorizon.ADAPTIVE_SIZES[-1]:
            self.assertLessEqual(score["tolerant_mean"], 0.05)

    def test_adaptive_below_the_smallest_rung_still_returns_a_map(self):
        with self.assertWarns(RuntimeWarning):
            hmap, score = ShadowHorizon.bake_adaptive(
                self.chair(),
                up=1,
                max_size=ShadowHorizon.ADAPTIVE_SIZES[0] - 1,
                measure_samples=2,
            )
        self.assertIsNotNone(hmap)
        self.assertEqual(hmap.size, ShadowHorizon.ADAPTIVE_SIZES[0])
        self.assertEqual(score["size"], ShadowHorizon.ADAPTIVE_SIZES[0])

    def test_adaptive_rejects_a_fixed_size_kwarg_by_name(self):
        with self.assertRaises(TypeError) as caught:
            ShadowHorizon.bake_adaptive(self.chair(), up=1, size=64)
        self.assertIn("max_size", str(caught.exception))

    def test_bake_time_stays_within_budget(self):
        """Guard against a pathological regression, not drift: the bake is
        a rasterisation plus a distance transform, measured 24-37 ms at the
        defaults on the fixtures (2026-09-05); the budget is 40x that."""
        t0 = time.perf_counter()
        ShadowHorizon.bake(self.chair(), up=1)
        self.assertLess(time.perf_counter() - t0, 1.5)


class TestShadowHorizonConformance(unittest.TestCase):
    """The golden cases a port of the bake and the evaluation is held to
    (``Conformance.cases("shadow_horizon")``)."""

    @classmethod
    def setUpClass(cls):
        cls.doc = ShadowHorizonConformance.cases(seed=2, per_kind=2)
        cls.by_name = {c["name"]: c for c in cls.doc["cases"]}

    @staticmethod
    def _image(m):
        raw = zlib.decompress(base64.b64decode(m["rgba"]))
        return np.frombuffer(raw, np.uint8).reshape(m["height"], m["width"], 4)

    @staticmethod
    def _delivered(m):
        return HeightFieldMap.from_rgba(
            TestShadowHorizonConformance._image(m),
            size=m["size"],
            spans=m["spans"],
            bounds=m["bounds"],
            ground=m["ground"],
            up=m["up"],
            height_scale=m["height_scale"],
        )

    def test_the_named_cases_cover_each_edge_the_bake_has(self):
        for name in (
            "single_triangle",
            "overhang",
            "interior_hole",
            "canvas_edge",
            "empty_outside",
            "empty_buried",
            "no_geometry",
        ):
            self.assertIn(name, self.by_name)
        baked = [c for c in self.doc["cases"] if "map" in c]
        self.assertEqual({c["map"]["spans"] for c in baked}, {1, 2, 3})
        self.assertIn(ShadowHorizon.DEFAULT_SPANS, {c["map"]["spans"] for c in baked})
        self.assertTrue(
            set(ShadowHorizon.ADAPTIVE_SIZES) <= {c["map"]["size"] for c in baked}
        )
        self.assertEqual({c["map"]["up"] for c in baked}, {1, 2})

    def test_each_image_is_the_bake_of_its_input(self):
        for c in self.doc["cases"]:
            if "map" not in c:
                continue
            with self.subTest(c["name"]):
                given = c["input"]
                meshes = [
                    (np.reshape(m["points"], (-1, 3)), np.reshape(m["triangles"], (-1, 3)))
                    for m in given["meshes"]
                ]
                hmap = ShadowHorizon.bake(
                    meshes,
                    ground=given["ground"],
                    up=given["up"],
                    size=given["size"],
                    spans=given["spans"],
                    bounds=given["bounds"],
                    padding=given["padding"],
                )
                np.testing.assert_array_equal(self._image(c["map"]), hmap.to_rgba())

    def test_each_alpha_is_the_delivered_maps(self):
        c = self.by_name["stool"]
        hmap = self._delivered(c["map"])
        for e in c["alpha"]:
            got = hmap.alpha(
                np.reshape(e["points"], (-1, 3)),
                e["light"],
                direction=e["direction"],
                source_size=e["source_size"],
                source_angle=e["source_angle"],
                intensity=e["intensity"],
            )
            self.assertEqual([float(v) for v in got], e["alpha"])
        # a soft and a hard source of each kind, all shadowing some point
        kinds = [(e["light"] is None, e["source_size"] + e["source_angle"] > 0) for e in c["alpha"]]
        self.assertEqual(len(set(kinds)), 4)
        self.assertTrue(all(max(e["alpha"]) > 0.0 for e in c["alpha"]))

    def test_an_empty_footprint_holds_no_column_and_casts_nothing(self):
        for name in ("empty_outside", "empty_buried"):
            with self.subTest(name):
                c = self.by_name[name]
                hmap = self._delivered(c["map"])
                self.assertTrue(np.isnan(hmap.hi).all())
                self.assertEqual(c["map"]["height_scale"], 1e-3)
                self.assertTrue(np.all(hmap.dist == DIST_FAR))
                self.assertTrue(all(v == 0.0 for e in c["alpha"] for v in e["alpha"]))

    def test_the_edge_ray_cases_pin_a_watertight_bake(self):
        """Either split of one box is one image, and every column of the
        fan's interior is solid from the ground to its top: a port whose
        fill rule leaks on a shared edge, or counts it twice, fails them.
        Added: 2026-11-07."""
        a, b = (self.by_name[n]["map"] for n in ("edge_rays_flipped", "edge_rays_split"))
        np.testing.assert_array_equal(self._image(a), self._image(b))
        fan = self._delivered(self.by_name["edge_rays_fan"]["map"])
        self.assertTrue((fan.lo[0][4:12, 4:12] == 0.0).all())
        self.assertTrue((fan.hi[0][4:12, 4:12] >= 1.0).all())

    def test_the_exact_cases_sit_on_their_ties(self):
        """Each pins one rule a port could spell otherwise and pass every
        other case (measured on the C# port: each mutant survived the rest):
        the lowest of equal gaps merged; a half code rounded to even; the
        lower row of two equally near columns; a ray leaving a pixel edge
        starting in the pixel it heads into; a sun within 1e-9 of the zenith
        casting nothing. Added: 2026-11-07."""
        boards = self._delivered(self.by_name["equal_gaps"]["map"])
        np.testing.assert_allclose(boards.lo[:, 8, 8], (0.25, 0.75), atol=1e-4)
        np.testing.assert_allclose(boards.hi[:, 8, 8], (0.625, 0.875), atol=1e-4)
        half = self.by_name["half_code"]["input"]["bounds"]
        self.assertEqual((half[1] - half[0]) / (half[3] - half[2]), 129 / 128)
        self.assertIn(1.0, self._delivered(self.by_name["half_code"]["map"]).dist)
        tie = self._delivered(self.by_name["row_tie"]["map"])
        self.assertEqual(np.argwhere(~np.isnan(tie.hi[0])).tolist(), [[6, 7], [9, 8]])
        self.assertEqual(float(tie.near_hi[8, 6]), 1.0)
        on_edge, zenith = self.by_name["march_edges"]["alpha"][-2:]
        self.assertEqual(on_edge["points"][0], 0.0)
        self.assertEqual(on_edge["alpha"] + zenith["alpha"], [0.0] * 4)

    def test_a_refused_input_carries_its_error_and_no_map(self):
        c = self.by_name["no_geometry"]
        self.assertEqual(c["error"], "ValueError")
        self.assertNotIn("map", c)
        self.assertNotIn("alpha", c)

    def test_a_source_under_the_ground_casts_nothing(self):
        below = self.by_name["box"]["alpha"][-1]
        self.assertLess(below["light"][1], 0.0)
        self.assertTrue(all(v == 0.0 for v in below["alpha"]))

    def test_the_document_stays_small_enough_to_generate_per_test_run(self):
        """Measured: 0.9 MB of JSON in ~3.5 s at the default draw; the whole
        images ride because deflated they cost 1 to 80 KB each."""
        self.assertLess(len(json.dumps(ShadowHorizonConformance.cases())), 3_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
