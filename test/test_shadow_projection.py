# !/usr/bin/python
# coding=utf-8
"""Tests for the ShadowProjection primitive — the ground-shadow geometry the
mayatk / blendertk ShadowRig twins share.

The physics is pinned once here: a sun's shadow is height x cot(elevation)
long, a positional source's grows with perspective, an overhead source draws
the footprint, a floating target's shadow slides away from the light, and the
reach cap keeps a grazing source finite. The DCC tests then only have to show
their expression / drivers agree with this reference.
"""

import math
import unittest

import numpy as np

from pythontk import ImgUtils, ShadowProjection


class TestProject(unittest.TestCase):
    def test_sun_at_45_degrees_lands_one_height_away(self):
        """A parallel projection: a point 1 up, sun from -X at 45 deg -> +1 in x."""
        d = (1.0, -1.0, 0.0)  # shines toward +X and down
        ground, spread = ShadowProjection.project([(0, 1, 0)], direction=d)
        self.assertAlmostEqual(ground[0][0], 1.0, places=6)
        self.assertAlmostEqual(ground[0][1], 0.0, places=6)
        # spread = the ray length (sqrt 2 for a unit rise at 45 deg)
        self.assertAlmostEqual(spread[0], math.sqrt(2.0), places=6)

    def test_point_source_is_a_perspective_projection(self):
        """Light at (0, 4, 0): a point 2 up at x=1 lands at x=2 (t = 4/2)."""
        ground, spread = ShadowProjection.project([(1, 2, 0)], light=(0, 4, 0))
        self.assertAlmostEqual(ground[0][0], 2.0, places=6)
        self.assertAlmostEqual(spread[0], 1.0, places=6)  # t - 1

    def test_points_on_or_below_the_ground_stay_put(self):
        ground, spread = ShadowProjection.project(
            [(3, 0, 1), (3, -2, 1)], light=(0, 4, 0)
        )
        for row in ground:
            self.assertAlmostEqual(row[0], 3.0, places=6)
            self.assertAlmostEqual(row[1], 1.0, places=6)
        self.assertTrue((spread == 0).all())

    def test_a_point_level_with_the_source_is_capped(self):
        ground, _ = ShadowProjection.project(
            [(1, 4, 0)], light=(0, 4, 0), max_length=10.0
        )
        self.assertAlmostEqual(ground[0][0], 11.0, places=5)  # 1 + the cap

    def test_a_source_below_the_ground_casts_nothing(self):
        self.assertIsNone(ShadowProjection.project([(0, 1, 0)], light=(0, -1, 0)))
        self.assertIsNone(ShadowProjection.project([(0, 1, 0)], direction=(1, 0.2, 0)))

    def test_blender_z_up_matches_maya_y_up(self):
        """The same scene in Z-up projects to the same horizontal result."""
        y_up, _ = ShadowProjection.project([(1, 2, 3)], light=(0, 4, 0), up=1)
        z_up, _ = ShadowProjection.project([(1, 3, 2)], light=(0, 0, 4), up=2)
        np.testing.assert_allclose(y_up, z_up, atol=1e-9)


class TestModel(unittest.TestCase):
    R, H = 0.5, 2.0  # a 1-wide, 2-tall cylinder on the origin

    def test_sun_reach_is_height_times_cot_elevation(self):
        for deg in (30.0, 45.0, 60.0):
            rad = math.radians(deg)
            d = (math.cos(rad), -math.sin(rad), 0.0)
            m = ShadowProjection.model(
                (0, 0, 0), direction=d, radius=self.R, height=self.H
            )
            self.assertAlmostEqual(m.reach, self.H / math.tan(rad), places=3, msg=deg)
            self.assertAlmostEqual(m.k_base, 1.0, places=3)
            self.assertAlmostEqual(m.k_top, 1.0, places=3)
            self.assertAlmostEqual(m.width, 2 * self.R, places=3)
            self.assertAlmostEqual(m.bearing[0], 1.0, places=6)
            self.assertAlmostEqual(m.anchor[0], 0.0, places=6)
            self.assertFalse(m.overhead)

    def test_point_source_grows_the_head(self):
        """Light at (-4, 4, 0): the top disk projects at k_top = 4/2 = 2, the
        base at 1; the head lands dist x (k_top - k_base) = 4 further."""
        m = ShadowProjection.model((0, 0, 0), (-4, 4, 0), radius=self.R, height=self.H)
        self.assertAlmostEqual(m.k_base, 1.0, places=6)
        self.assertAlmostEqual(m.k_top, 2.0, places=6)
        self.assertAlmostEqual(m.reach, 4.0, places=6)
        self.assertAlmostEqual(m.near, -self.R, places=6)
        self.assertAlmostEqual(m.length, 4.0 + self.R * 3.0, places=6)
        self.assertAlmostEqual(m.width, 2.0 * self.R * 2.0, places=6)

    def test_overhead_source_draws_the_footprint(self):
        m = ShadowProjection.model((0, 0, 0), (0, 4, 0), radius=self.R, height=self.H)
        self.assertTrue(m.overhead)
        self.assertEqual(m.bearing, ShadowProjection.OVERHEAD_BEARING)
        self.assertAlmostEqual(m.reach, 0.0, places=6)
        # Perspective still grows the head: k_top = 4/2 = 2.
        self.assertAlmostEqual(m.width, 2.0 * self.R * 2.0, places=6)
        self.assertAlmostEqual(m.length, self.R * 3.0, places=6)

    def test_floating_target_slides_away_from_the_light(self):
        """Contact 2 up under a light at (-4, 10, 0): k = 10/8 -> the anchor
        lands 4 x 1.25 past the light, i.e. at x = +1."""
        m = ShadowProjection.model((0, 2, 0), (-4, 10, 0), radius=self.R, height=self.H)
        self.assertAlmostEqual(m.k_base, 1.25, places=6)
        self.assertAlmostEqual(m.anchor[0], 1.0, places=6)

    def test_reach_cap(self):
        rad = math.radians(2.0)
        d = (math.cos(rad), -math.sin(rad), 0.0)
        m = ShadowProjection.model(
            (0, 0, 0), direction=d, radius=self.R, height=self.H, max_stretch=3.0
        )
        self.assertAlmostEqual(m.reach, 3.0 * self.H, places=3)

    def test_fractions_round_trip_and_placement(self):
        m = ShadowProjection.model((1, 0, 2), (-3, 4, 2), radius=self.R, height=self.H)
        rect = (m.near + 0.1, m.near + 0.9 * m.length, -0.3 * m.width, 0.4 * m.width)
        f = ShadowProjection.fractions(rect, m)
        np.testing.assert_allclose(m.rect(f), rect, atol=1e-9)
        centre, du, dw = m.placement(f)
        self.assertAlmostEqual(du, rect[1] - rect[0], places=9)
        self.assertAlmostEqual(dw, rect[3] - rect[2], places=9)
        # The canvas centre, re-expressed in the frame, is the rect's centre.
        uw = ShadowProjection.to_frame([centre], m)[0]
        self.assertAlmostEqual(uw[0], 0.5 * (rect[0] + rect[1]), places=9)
        self.assertAlmostEqual(uw[1], 0.5 * (rect[2] + rect[3]), places=9)

    def test_across_is_the_planes_local_x(self):
        m = ShadowProjection.model((0, 0, 0), (0, 4, -4), radius=self.R, height=self.H)
        self.assertAlmostEqual(m.bearing[1], 1.0, places=6)  # away = +Z
        self.assertAlmostEqual(m.across[0], 1.0, places=6)  # across = +X


class TestRasterizeShadow(unittest.TestCase):
    """The rasterizer draws the true projection into a canvas it reports."""

    @staticmethod
    def _box(x0, x1, y0, y1, z0, z1):
        """A closed box as (points, tris)."""
        pts = np.array(
            [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
            dtype=float,
        )
        faces = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]
        tris = []
        for a, b, c, d in faces:
            tris += [(a, b, c), (a, c, d)]
        return pts, np.array(tris, dtype=np.int64)

    def _alpha(self, rgba):
        return rgba[:, :, 3].astype(float) / 255.0

    def test_overhead_source_draws_the_footprint_only(self):
        box = self._box(-1, 1, 0, 2, -1, 1)
        rgba, raster = ImgUtils.rasterize_shadow(
            [box], light=(0, 10, 0), size=64, blur_amount=0
        )
        a = self._alpha(rgba)
        self.assertTrue(raster.model.overhead)
        # The canvas is the top face's projection (k = 10/8) plus padding: a
        # 2.5-wide square, and the coverage fills most of it.
        u_lo, u_hi, w_lo, w_hi = raster.rect
        self.assertAlmostEqual(u_hi - u_lo, w_hi - w_lo, places=6)
        self.assertGreater(u_hi - u_lo, 2.5)
        self.assertLess(u_hi - u_lo, 3.0)
        self.assertGreater((a > 0.5).mean(), 0.6)

    def test_sun_stretches_the_shadow_away_from_the_light(self):
        """A 45 deg sun from -X on a 2-tall box: the shadow runs from the
        footprint's near edge (x=-1) to x = +1 + 2 = 3 — canvas length ~4."""
        box = self._box(-1, 1, 0, 2, -1, 1)
        rgba, raster = ImgUtils.rasterize_shadow(
            [box], direction=(1, -1, 0), size=64, blur_amount=0, padding=0.0
        )
        u_lo, u_hi, w_lo, w_hi = raster.rect
        self.assertAlmostEqual(u_lo, -1.0, places=3)  # near edge = footprint
        self.assertAlmostEqual(u_hi, 3.0, places=3)
        self.assertAlmostEqual(w_hi - w_lo, 2.0, places=3)  # no growth across
        a = self._alpha(rgba)
        self.assertGreater((a > 0.5).mean(), 0.9)  # a box's shadow fills its rect
        # The saved image's top row is the light-side (near) edge, bottom the tip.
        self.assertGreater(a[1].mean(), 0.5)
        self.assertGreater(a[-2].mean(), 0.5)

    def test_orientation_a_post_on_the_across_side_lands_right_of_centre(self):
        """A tall post standing on the +w side of a low slab: its long shadow
        shows up in the right-hand columns and the far (bottom) rows."""
        slab = self._box(-2, 2, 0, 0.2, -2, 2)
        post = self._box(1.5, 2, 0, 4, -0.25, 0.25)  # tall, at +X
        # Light from -Z: away = +Z (bearing (0,1)), across w = (1, 0) = +X.
        rgba, raster = ImgUtils.rasterize_shadow(
            [slab, post], light=(0, 4, -20), size=64, blur_amount=0
        )
        self.assertAlmostEqual(raster.model.bearing[1], 1.0, places=3)
        a = self._alpha(rgba)
        rows, cols = np.nonzero(a[40:, :] > 0.5)  # the far rows: the post's shadow
        self.assertTrue(len(cols))
        self.assertGreater(cols.mean(), 32)  # right of centre (= +X = +w)

    def test_penumbra_widens_away_from_the_contact(self):
        """A sized source blurs the tip more than the contact edge."""
        box = self._box(-1, 1, 0, 2, -1, 1)
        kw = dict(direction=(1, -1, 0), size=128, blur_amount=0, padding=0.1)
        sharp, _ = ImgUtils.rasterize_shadow([box], source_size=0.0, **kw)
        soft, raster = ImgUtils.rasterize_shadow([box], source_size=0.2, **kw)
        s = self._alpha(sharp)
        p = self._alpha(soft)
        self.assertGreater(raster.penumbra, 0.0)
        # More partial coverage with a sized source...

        def partial(arr):
            return ((arr > 0.05) & (arr < 0.95)).sum()

        self.assertGreater(partial(p), partial(s) * 3)
        # ...and the transition is wider at the far edge than at the near edge:
        # count partial pixels along the shadow's centre column in each half.
        col = p[:, 64]
        near_half, far_half = col[:64], col[64:]
        self.assertGreater(
            ((far_half > 0.05) & (far_half < 0.95)).sum(),
            ((near_half > 0.05) & (near_half < 0.95)).sum(),
        )

    def test_given_canvas_is_honoured(self):
        box = self._box(-1, 1, 0, 2, -1, 1)
        canvas = (-3.0, 5.0, -4.0, 4.0)
        rgba, raster = ImgUtils.rasterize_shadow(
            [box], direction=(1, -1, 0), size=64, canvas=canvas
        )
        self.assertEqual(raster.rect, canvas)
        np.testing.assert_allclose(
            raster.model.rect(raster.fractions), canvas, atol=1e-9
        )
        # Drawn small inside the big canvas: the corners stay empty.
        a = self._alpha(rgba)
        self.assertEqual(a[0, 0], 0.0)
        self.assertGreater((a > 0.5).mean(), 0.05)

    def test_given_model_inputs_define_the_frame(self):
        """The DCC passes the contact and constants its expression reads; the
        fractions are then measured in that frame (the rect itself is the
        same shadow either way)."""
        box = self._box(-1, 1, 0, 2, -1, 1)
        kw = dict(direction=(1, -1, 0), size=32, padding=0.0)
        _, own = ImgUtils.rasterize_shadow([box], **kw)
        _, given = ImgUtils.rasterize_shadow(
            [box], contact=(0.5, 0.0, 0.0), radius=2.0, height=3.0, **kw
        )
        self.assertAlmostEqual(given.model.anchor[0], 0.5, places=6)
        self.assertAlmostEqual(given.model.length, 3.0 + 4.0, places=4)
        # The same canvas in world space, expressed against a different model.
        (cx, cz), du, dw = own.model.placement(own.fractions)
        (gx, gz), gu, gw = given.model.placement(given.fractions)
        self.assertAlmostEqual(gx, cx, places=6)
        self.assertAlmostEqual(gz, cz, places=6)
        self.assertAlmostEqual(gu, du, places=6)
        self.assertAlmostEqual(gw, dw, places=6)
        self.assertNotEqual(given.fractions, own.fractions)

    def test_penumbra_is_capped_for_a_point_level_with_the_source(self):
        """A tall box whose top sits at the lamp's height: the spread there is
        unbounded, and the canvas must not blow up with it."""
        box = self._box(-1, 1, 0, 4, -1, 1)
        _, raster = ImgUtils.rasterize_shadow(
            [box], light=(0, 4, 0), size=32, source_size=1.0, max_stretch=3.0
        )
        self.assertLessEqual(raster.penumbra, 3.0 * 4.0 + 1e-6)
        u_lo, u_hi, w_lo, w_hi = raster.rect
        self.assertLess(u_hi - u_lo, 60.0)

    def test_source_below_the_ground_is_transparent(self):
        box = self._box(-1, 1, 0, 2, -1, 1)
        rgba, raster = ImgUtils.rasterize_shadow([box], light=(0, -3, 0), size=32)
        self.assertEqual(rgba[:, :, 3].max(), 0)
        self.assertEqual(len(raster.rect), 4)

    def test_contact_falloff_is_optional(self):
        box = self._box(-1, 1, 0, 2, -1, 1)
        kw = dict(direction=(1, -1, 0), size=64, blur_amount=0)
        flat, _ = ImgUtils.rasterize_shadow([box], uniform_alpha=True, **kw)
        faded, _ = ImgUtils.rasterize_shadow([box], uniform_alpha=False, **kw)
        f, d = self._alpha(flat), self._alpha(faded)
        self.assertLess(f[f > 0].std(), 0.05)
        self.assertGreater(d[d > 0].std(), 0.1)

    def test_blender_z_up_gives_the_same_canvas(self):
        box_y = self._box(-1, 1, 0, 2, -1, 1)
        pts, tris = box_y
        box_z = (pts[:, [0, 2, 1]], tris)  # swap y/z -> Z-up
        _, ry = ImgUtils.rasterize_shadow([box_y], direction=(1, -1, 0), size=32, up=1)
        _, rz = ImgUtils.rasterize_shadow([box_z], direction=(1, 0, -1), size=32, up=2)
        np.testing.assert_allclose(ry.rect, rz.rect, atol=1e-6)
        np.testing.assert_allclose(ry.fractions, rz.fractions, atol=1e-6)


class TestCanvasAttachment(unittest.TestCase):
    """The reported gap: a grounded target's shadow drifted away from its feet
    as the light lowered, while an overhead light drew it attached. A canvas
    stamped as fractions of the model's LENGTH slides with that length; the
    physics pins the near edge to the footprint (ground points project onto
    themselves at any light height) and only the far edge follows the top's
    projection, so the stamp must be measured against the base disk and the
    top disk separately."""

    R = math.hypot(2.0, 2.0) / 2.0  # a 2 x 2 x 2 box's footprint radius
    H = 2.0

    @staticmethod
    def _corners():
        return np.array(
            [(x, y, z) for x in (-1.0, 1.0) for y in (0.0, 2.0) for z in (-1.0, 1.0)]
        )

    def _rect(self, light):
        model = ShadowProjection.model((0, 0, 0), light, radius=self.R, height=self.H)
        ground, _ = ShadowProjection.project(self._corners(), light)
        uw = ShadowProjection.to_frame(ground, model)
        return (uw[:, 0].min(), uw[:, 0].max(), uw[:, 1].min(), uw[:, 1].max()), model

    def test_grounded_canvas_stays_attached_to_the_footprint(self):
        """Rasterized under a high light at +X, then re-placed under a low one:
        the canvas's near edge stays at the box's near face (u = -1) and its far
        edge lands where the top's far corner projects (reach + 1 x k_top)."""
        rect_high, high = self._rect((6.0, 20.0, 0.0))
        stamp = ShadowProjection.fractions(rect_high, high)
        rect_low, low = self._rect((6.0, 4.0, 0.0))
        self.assertAlmostEqual(low.k_top, 2.0, places=6)  # no cap in play
        placed = low.rect(stamp)
        self.assertAlmostEqual(rect_low[0], -1.0, places=6)  # the near face, exact
        self.assertAlmostEqual(placed[0], rect_low[0], places=6)
        self.assertAlmostEqual(placed[1], rect_low[1], places=6)
        # The overhead-ish rect it was stamped from was much shorter.
        self.assertLess(rect_high[1] - rect_high[0], 0.5 * (rect_low[1] - rect_low[0]))

    def test_floating_canvas_scales_with_the_base_disk(self):
        """A target 1 up: its footprint projects at k_base = 4/3 under a light
        at height 4 — the canvas's near edge follows the projected footprint."""
        model = ShadowProjection.model((0, 1, 0), (6, 4, 0), radius=self.R, height=1.0)
        self.assertAlmostEqual(model.k_base, 4.0 / 3.0, places=6)
        stamp = (-1.0 / self.R, 1.0 / self.R, -0.5, 0.5)  # the box's own extents
        rect = model.rect(stamp)
        self.assertAlmostEqual(rect[0], -1.0 * model.k_base, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
