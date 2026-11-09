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

from pythontk import (
    ImgUtils,
    ShadowConformance,
    ShadowHorizonConformance,
    ShadowProjection,
)


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

    _box = staticmethod(ShadowHorizonConformance._box)

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
    """The reported gap, twice: a grounded target's shadow drifted away from
    its feet as the light lowered. A canvas stamped as fractions of the
    model's LENGTH slid with that length; one stamped in footprint radii
    pinned the canvas's back edge but let the drawn feet slide forward as
    the far edge grew. Ground points project onto themselves at any light
    height, so the anchor -- the feet -- must keep its place in the texture:
    the near edge is stamped against the far edge, and the far edge in
    top-disk radii from where the head lands."""

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

    @staticmethod
    def _feet(rect):
        """Where the anchor (u = 0) sits along the canvas, as a fraction."""
        return (0.0 - rect[0]) / (rect[1] - rect[0])

    def test_grounded_feet_keep_their_place_as_the_light_lowers(self):
        """Rasterized under a high light at +X, then re-placed under a low one:
        the far edge lands where the top's far corner projects (reach + 1 x
        k_top) and the anchor sits at the same fraction of the canvas as it
        did in the texture -- the drawn feet stay under the box's feet. The
        part behind the feet stretches with the reach (the accepted cost: it
        only bridges the moves between two renders)."""
        rect_high, high = self._rect((6.0, 20.0, 0.0))
        stamp = ShadowProjection.fractions(rect_high, high)
        rect_low, low = self._rect((6.0, 4.0, 0.0))
        self.assertAlmostEqual(low.k_top, 2.0, places=6)  # no cap in play
        placed = low.rect(stamp)
        self.assertAlmostEqual(placed[1], rect_low[1], places=6)
        self.assertAlmostEqual(self._feet(placed), self._feet(rect_high), places=9)
        # The overhead-ish rect it was stamped from was much shorter, and
        # the canvas's back edge (not the feet) is what moved to keep them.
        self.assertLess(rect_high[1] - rect_high[0], 0.5 * (rect_low[1] - rect_low[0]))
        self.assertLess(placed[0], rect_low[0])
        # The near-edge stamp of old (the near face pinned, the far edge on
        # the head) would have put the feet 0.36 of the canvas -- 2.2 units
        # -- ahead of the box's feet here.
        drifted = (rect_low[0], placed[1])
        self.assertGreater(
            self._feet(rect_high) * (drifted[1] - drifted[0]) + drifted[0], 2.0
        )

    def test_floating_feet_keep_their_place_too(self):
        """A target 1 up: its footprint projects at k_base = 4/3 under a light
        at height 4 and the anchor slides away from the light -- the canvas
        keeps the anchor at its stamped fraction, whatever the base factor."""
        model = ShadowProjection.model((0, 1, 0), (6, 4, 0), radius=self.R, height=1.0)
        self.assertAlmostEqual(model.k_base, 4.0 / 3.0, places=6)
        drawn = (-1.0, 3.0, -1.0, 1.0)  # a canvas with the feet a quarter in
        stamp = ShadowProjection.fractions(drawn, model)
        np.testing.assert_allclose(model.rect(stamp), drawn, atol=1e-9)
        higher = ShadowProjection.model((0, 1, 0), (6, 9, 0), radius=self.R, height=1.0)
        self.assertAlmostEqual(self._feet(higher.rect(stamp)), 0.25, places=9)


def _producer_fade(C, L, ground, radius, height, max_stretch, intensity, power, fade_h):
    """Both producers' fade as their live preview spells it (mayatk's
    ``_expr_opacity``, blendertk's material driver), in the rig's OWN unit:
    every length here is in that unit, and so is each constant."""
    model = ShadowProjection.model(
        C, L, ground, radius, height, max_stretch=max_stretch
    )
    length = model.reach + model.base + model.top
    stretch = max(1.0, length / max(1e-4, 2.0 * radius))
    dist_opacity = intensity / max(0.001, stretch**power)
    height_fade = min(max(L[1] - C[1], 0.0), 1.0)
    rise_fade = min(max(1.0 - max(0.0, C[1] - ground) / max(0.001, fade_h), 0.0), 1.0)
    return min(max(dist_opacity * height_fade * rise_fade, 0.0), 1.0)


class TestFade(unittest.TestCase):
    """``ShadowProjection.fade``: how strongly a plane draws, from its
    placement and its terms, in metres -- the fade both producers previewed
    and baked as keys, now every reader's to evaluate each frame."""

    def test_a_grounded_target_under_a_high_source_lightens_as_it_stretches(self):
        model = ShadowProjection.model((0, 0, 0), (2.0, 4.0, 0.0), 0.0, 0.5, 1.0)
        stretch = model.length / 1.0
        fade = ShadowProjection.fade(
            (0, 0, 0), (2.0, 4.0, 0.0), 0.0, 0.5, 1.0, intensity=0.8, falloff_power=1.5
        )
        self.assertAlmostEqual(fade, 0.8 / stretch**1.5, places=12)

    def test_an_overhead_source_draws_near_its_intensity(self):
        """No reach past the footprint: the stretch is the top disk's growth
        alone, so the shadow draws all but at its intensity."""
        fade = ShadowProjection.fade(
            (0, 0, 0), (0.0, 50.0, 0.0), 0.0, 0.5, 0.2, intensity=0.7
        )
        k_top = 50.0 / 49.8
        self.assertAlmostEqual(fade, 0.7 / ((1.0 + k_top) / 2.0) ** 1.2, places=12)
        self.assertGreater(fade, 0.69)

    def test_a_source_under_the_contact_casts_nothing_and_ramps_in_over_its_ramp(self):
        """The source-height ramp spans the plane's ``source_ramp`` (metres):
        one unit of the rig that previews it -- 1 cm for a centimetre rig,
        1 m for a metre one -- whatever unit its record is in. No falloff
        (power 0), so the ramp alone sets the fade."""
        C = (0.0, 0.5, 0.0)
        flat = dict(ground=0.5, falloff_power=0.0, intensity=0.9)
        for ramp in (1.0, 0.01):
            with self.subTest(ramp=ramp):
                span = ramp
                below = ShadowProjection.fade(
                    C, (1.0, 0.4, 0.0), source_ramp=ramp, **flat
                )
                self.assertEqual(below, 0.0)
                half = ShadowProjection.fade(
                    C, (1.0, 0.5 + 0.5 * span, 0.0), source_ramp=ramp, **flat
                )
                self.assertAlmostEqual(half, 0.45, places=6)
                full = ShadowProjection.fade(
                    C, (1.0, 0.5 + 1.0 * span, 0.0), source_ramp=ramp, **flat
                )
                self.assertAlmostEqual(full, 0.9, places=9)

    def test_a_ramp_at_or_under_zero_is_floored_as_every_port_floors_it(self):
        """A file whose ``source_ramp`` is not over 0 (the record requires it
        to be) fades by a 1e-4 m ramp, as the viewer's ``shadowFade`` and the
        C# port floor it -- never a ZeroDivisionError. Added: 2026-11-08"""
        flat = dict(falloff_power=0.0, intensity=0.8)
        for ramp in (0.0, -1.0):
            with self.subTest(ramp=ramp):
                half = ShadowProjection.fade(
                    (0.0, 0.0, 0.0), (1.0, 5e-5, 0.0), source_ramp=ramp, **flat
                )
                self.assertAlmostEqual(half, 0.4, places=6)

    def test_a_target_rising_off_the_ground_fades_out_at_its_fade_height(self):
        L = (0.0, 100.0, 0.0)
        flat = dict(fade_height=2.0, falloff_power=0.0)
        self.assertEqual(ShadowProjection.fade((0, 0, 0), L, **flat), 1.0)
        self.assertAlmostEqual(
            ShadowProjection.fade((0, 1.0, 0), L, **flat), 0.5, places=12
        )
        self.assertEqual(ShadowProjection.fade((0, 2.5, 0), L, **flat), 0.0)

    def test_a_sun_fades_from_its_far_point(self):
        """A directional source is its far point (``far_point``), so it fades
        as a point that far back does: by its stretch alone above the
        horizon, to nothing below it."""
        up = ShadowProjection.fade((0, 0, 0), direction=(1.0, -1.0, 0.0), radius=0.5)
        stretch = ShadowProjection.model(
            (0, 0, 0), direction=(1.0, -1.0, 0.0), radius=0.5
        ).length
        self.assertAlmostEqual(up, 1.0 / stretch**1.2, places=9)
        down = ShadowProjection.fade((0, 0, 0), direction=(1.0, 0.1, 0.0))
        self.assertEqual(down, 0.0)

    def test_it_is_what_both_producers_baked_in_any_rig_unit(self):
        """The producers evaluate the fade in their rig's unit; the model, in
        metres from the record's lengths times ``unit_scale`` and the ramp
        one rig unit (``source_ramp``), is the same number at every unit -- a
        centimetre Maya rig and a metre Blender rig alike."""
        rng = np.random.default_rng(5)
        for _ in range(200):
            C = (rng.uniform(-2, 2), rng.uniform(0, 1.5), rng.uniform(-2, 2))
            L = (rng.uniform(-6, 6), rng.uniform(-0.5, 6), rng.uniform(-6, 6))
            terms = dict(
                ground=0.0,
                radius=rng.uniform(0.1, 1.0),
                height=rng.uniform(0.2, 2.0),
                max_stretch=rng.uniform(2, 8),
            )
            fade = dict(
                intensity=rng.uniform(0.2, 1.0),
                power=rng.uniform(0.0, 3.0),
                fade_h=rng.uniform(0.5, 4.0),
            )
            for unit in (1.0, 0.01, 0.0254):
                # The record's lengths in its unit; the model reads metres.
                rec = {k: v / unit for k, v in terms.items() if k != "max_stretch"}
                want = _producer_fade(
                    [c / unit for c in C],
                    [c / unit for c in L],
                    rec["ground"],
                    rec["radius"],
                    rec["height"],
                    terms["max_stretch"],
                    fade["intensity"],
                    fade["power"],
                    fade["fade_h"] / unit,
                )
                got = ShadowProjection.fade(
                    C,
                    L,
                    terms["ground"],
                    terms["radius"],
                    terms["height"],
                    max_stretch=terms["max_stretch"],
                    intensity=fade["intensity"],
                    falloff_power=fade["power"],
                    fade_height=fade["fade_h"],
                    source_ramp=unit,
                )
                self.assertAlmostEqual(got, want, places=12)


#: The fade cases :meth:`ShadowConformance.cases` appends after the guards.
_FADES = 8


class TestShadowConformance(unittest.TestCase):
    """The golden cases the ports run: each is the reference's own answer."""

    def setUp(self):
        self.doc = ShadowConformance.cases(seed=3, per_kind=4)

    def test_every_case_is_the_reference_model_and_placement(self):
        for case in self.doc["cases"]:
            given = case["input"]
            model = ShadowProjection.model(
                given["contact"],
                given["light"],
                given["ground"],
                given["radius"],
                given["height"],
                direction=given["direction"],
                max_stretch=given["max_stretch"],
            )
            centre, along, across = model.placement(given["canvas"])
            fade = ShadowProjection.fade(
                given["contact"],
                given["light"],
                given["ground"],
                given["radius"],
                given["height"],
                direction=given["direction"],
                max_stretch=given["max_stretch"],
                intensity=given["intensity"],
                falloff_power=given["falloff_power"],
                fade_height=given["fade_height"],
                source_ramp=given["source_ramp"],
            )
            self.assertEqual(case["fade"], fade, case["name"])
            self.assertEqual(case["model"]["anchor"], list(model.anchor), case["name"])
            self.assertEqual(case["model"]["reach"], model.reach, case["name"])
            self.assertEqual(case["placement"]["centre"], list(centre), case["name"])
            self.assertEqual(case["placement"]["along"], along, case["name"])
            self.assertEqual(case["placement"]["across"], across, case["name"])

    def test_a_sun_carries_its_unit_direction_and_far_point(self):
        sun = next(c for c in self.doc["cases"] if c["name"] == "sun")
        given = sun["input"]
        self.assertIsNone(given["light"])
        self.assertAlmostEqual(math.hypot(*given["direction"]), 1.0, places=12)
        far = ShadowProjection.far_point(
            given["contact"],
            given["direction"],
            max(given["height"], 2 * given["radius"]),
        )
        self.assertEqual(sun["source"], list(far))

    def test_the_named_cases_reach_every_clamp_and_fallback(self):
        named = {c["name"]: c["model"] for c in self.doc["cases"]}
        self.assertTrue(named["overhead"]["overhead"])
        self.assertFalse(named["point"]["overhead"])
        # A source under the top: the top disk's factor meets its cap.
        self.assertGreater(named["below_top"]["k_top"], named["below_top"]["k_base"])
        # A source under a floating contact: the base factor meets 1 + max_stretch.
        self.assertAlmostEqual(
            named["below_contact"]["k_base"], 1.0 + ShadowProjection.DEFAULT_MAX_STRETCH
        )
        self.assertGreater(named["raised"]["k_base"], 1.0)
        # A grazing sun: the reach is capped at max_stretch heights.
        self.assertAlmostEqual(named["low_sun"]["reach"], 4.0)

    def test_the_random_cases_cover_both_kinds(self):
        randoms = self.doc["cases"][9 : 9 + 2 * 4]
        kinds = {c["name"].rsplit("_", 1)[0] for c in randoms}
        self.assertEqual(kinds, {"point", "sun"})
        self.assertEqual(len(self.doc["cases"]), 9 + 2 * 4 + 6 + _FADES)

    def test_the_guard_cases_come_after_the_draw_whatever_it_is(self):
        """Appended after the random ones, fixed: every earlier case keeps its
        index, and each guard case binds at any seed."""
        guards = self.doc["cases"][-6 - _FADES : -_FADES]
        other = ShadowConformance.cases(seed=11, per_kind=1)["cases"]
        self.assertEqual(guards, other[-6 - _FADES : -_FADES])
        self.assertTrue(all(c["name"].startswith(("tiny", "near", "overhead")) for c in guards))

    def test_the_fade_cases_come_last_and_reach_every_branch(self):
        """The fade's own cases (2026-11-07), appended after the guards so
        every earlier case keeps its index: each term of the fade decides one
        -- the source under the contact, inside its ramp, the target risen
        part way and past its fade height, a falloff, a sun, a centimetre
        rig's ramp -- and they are the same at any seed."""
        fades = {c["name"]: c for c in self.doc["cases"][-_FADES:]}
        other = ShadowConformance.cases(seed=11, per_kind=1)["cases"][-_FADES:]
        self.assertEqual(list(fades.values()), other)
        self.assertTrue(all(name.startswith("fade_") for name in fades))
        self.assertEqual(fades["fade_source_below"]["fade"], 0.0)
        self.assertEqual(fades["fade_risen_away"]["fade"], 0.0)
        for name in ("fade_source_ramp", "fade_rising", "fade_ramp_cm"):
            with self.subTest(case=name):
                given = fades[name]["input"]
                self.assertTrue(0.0 < fades[name]["fade"] < given["intensity"])
        self.assertEqual(fades["fade_ramp_cm"]["input"]["source_ramp"], 0.01)
        # The ramp's 1e-4 floor, where it binds: a ramp of 0, the source half
        # a floor over the contact (appended last: every case keeps its index).
        floor = self.doc["cases"][-1]
        self.assertEqual(floor["name"], "fade_ramp_floor")
        self.assertEqual(floor["input"]["source_ramp"], 0.0)
        self.assertAlmostEqual(floor["fade"], 0.5 * floor["input"]["intensity"], places=6)
        self.assertIsNotNone(fades["fade_sun"]["input"]["direction"])

    def test_every_case_states_the_fade_terms_a_record_carries(self):
        """A case hands a port the plane's fade terms as the record names
        them (``intensity``, ``falloff_power``, ``fade_height``,
        ``source_ramp``), and the fade's tolerance."""
        self.assertIn("fade", self.doc["tolerance"])
        for case in self.doc["cases"]:
            with self.subTest(case=case["name"]):
                for key in ("intensity", "falloff_power", "fade_height", "source_ramp"):
                    self.assertIsInstance(case["input"][key], float)
                self.assertIsInstance(case["fade"], float)

    def test_each_guard_case_sits_where_its_guard_binds(self):
        """The model scales with its lengths but for its guard constants, so
        each guard case is built where one decides the answer: change the
        constant (in a port) and that case's model changes."""
        eps = ShadowProjection.EPS
        guard = {c["name"]: c for c in self.doc["cases"][-6 - _FADES : -_FADES]}

        def given(name):
            case = guard[name]
            g = case["input"]
            (cx, cy, cz), (lx, ly, lz) = g["contact"], case["source"]
            return case["model"], g, math.hypot(cx - lx, cz - lz), ly - cy, ly

        # The bearing's EPS, from both sides.
        model, _, dist, _, _ = given("near_overhead")
        self.assertTrue(eps < dist < 10 * eps, dist)
        self.assertFalse(model["overhead"])
        model, _, dist, _, _ = given("overhead_within_eps")
        self.assertTrue(0.1 * eps < dist <= eps, dist)
        self.assertTrue(model["overhead"])
        # The reach cap's max(dist, EPS): the top factor is the cap through dist.
        model, g, dist, rise, ly = given("tiny_near_overhead")
        self.assertTrue(eps < dist < 10 * eps, dist)
        cap = model["k_base"] + g["max_stretch"] * g["height"] / dist
        self.assertAlmostEqual(model["k_top"], cap, places=12)
        self.assertGreater(rise - g["height"], 1e-4)  # clear of the top's floor
        self.assertLess(cap, (ly - g["ground"]) / (rise - g["height"]))
        self.assertLess(cap, 1.0 + g["max_stretch"])
        # The top factor's 1e-4 floor, under the cap.
        model, g, _, rise, ly = given("tiny_over_top")
        self.assertLess(rise - g["height"], 1e-4)
        self.assertAlmostEqual(model["k_top"], (ly - g["ground"]) / 1e-4, places=9)
        self.assertLess(model["k_top"], 1.0 + g["max_stretch"])
        # The base factor's 1e-4 floor, under 1 + max_stretch.
        model, g, _, rise, ly = given("tiny_over_contact")
        self.assertLess(rise, 1e-4)
        self.assertAlmostEqual(model["k_base"], (ly - g["ground"]) / 1e-4, places=9)
        self.assertLess(model["k_base"], 1.0 + g["max_stretch"])
        # The far point's 1e-3 scale floor.
        sun = guard["tiny_sun"]
        g = sun["input"]
        self.assertLess(max(g["height"], 2 * g["radius"]), 1e-3)
        far = math.dist(sun["source"], g["contact"])
        self.assertAlmostEqual(far, ShadowProjection.FAR_FACTOR * 1e-3, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
