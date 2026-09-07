# !/usr/bin/python
# coding=utf-8
"""``ShadowHorizon`` / ``HeightFieldMap`` (``geo_utils/shadow_horizon.py``).

The bake (depth-peeled spans, the distance field, the pyramid), the PNG
encoding round trip, and the reference -- ``HeightFieldMap.alpha`` -- against
the exact projection on the props that broke the map this one replaced.
"""

import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythontk.geo_utils.shadow_horizon import (  # noqa: E402
    DIST_FAR,
    HeightFieldMap,
    HorizonMap,
    ShadowHorizon,
)
from pythontk.img_utils._img_utils import ImgUtils  # noqa: E402


class HorizonCase(unittest.TestCase):
    """Fixtures: axis-aligned boxes standing on the ground of a Y-up frame."""

    @staticmethod
    def _box(x0, x1, y0, y1, z0, z1):
        pts = np.array(
            [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
            dtype=float,
        )
        quads = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]
        tris = []
        for a, b, c, d in quads:
            tris += [(a, b, c), (a, c, d)]
        return pts, np.array(tris, dtype=np.int64)

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
        with np.errstate(all="ignore"):
            union_hi = np.nanmax(hi, axis=0)
            union_lo = np.nanmin(lo, axis=0)
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
        for name in ("lo", "hi", "dist", "near_lo", "near_hi"):
            a, b = getattr(hmap, name), getattr(back, name)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b), name)
            np.testing.assert_allclose(
                np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=1e-6, err_msg=name
            )

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
        block = ShadowHorizon.record(
            texture="a_horizon.png",
            size=128,
            spans=2,
            levels=7,
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
                    "mapping",
                    "encoding",
                    "size",
                    "spans",
                    "levels",
                    "bounds",
                    "height_scale",
                    "frame_a",
                    "frame_b",
                    "rect",
                ]
            ),
        )
        self.assertEqual(block["mapping"], ShadowHorizon.MAPPING)
        self.assertEqual(block["encoding"], ShadowHorizon.ENCODING)
        self.assertEqual(block["bounds"], [-1.234568, 1.234568, -0.5, 0.5])
        self.assertEqual(block["frame_a"], [1, 0, 0])

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

    def test_source_size_softens_the_edges_without_moving_the_umbra(self):
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
        self.assertGreater(int(between.sum()), 5, "a penumbra")
        np.testing.assert_array_equal(soft[hard == 1.0], 1.0, "the umbra is untouched")
        self.assertTrue((soft >= hard).all())
        # the penumbra lies beyond the umbra's far end, never under the box
        self.assertGreater(int(between[xs < -7.0].sum()), 5)
        self.assertEqual(int(between[xs > -3.0].sum()), 0)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
