# !/usr/bin/python
# coding=utf-8
"""KeyCurve: the host-free animation curve a graph editor draws and edits.

The hosts' own evaluation pins it from their side (mayatk / blendertk
``test_sequencer`` / ``test_shot_sequencer`` compare it with ``keyframe -eval``
and ``fcurve.evaluate``); these are the analytic cases.
"""

import math
import unittest

from pythontk import CurveKey, KeyCurve, MathUtils


def _curve(weighted=False, **kw):
    """0 -> 10 over frames 0..12, slope 1 out of the first key, 0 into the last."""
    return KeyCurve(
        [
            CurveKey(0.0, 0.0, in_handle=(-1.0, -1.0), out_handle=(1.0, 1.0)),
            CurveKey(12.0, 10.0, in_handle=(-1.0, 0.0), out_handle=(1.0, 0.0)),
        ],
        weighted=weighted,
        **kw,
    )


class TestKeyCurveEvaluate(unittest.TestCase):
    def test_keys_are_sorted_on_construction(self):
        c = KeyCurve([CurveKey(5.0, 1.0), CurveKey(1.0, 2.0)])
        self.assertEqual(c.times(), [1.0, 5.0])

    def test_an_unweighted_span_is_the_hermite_of_its_slopes(self):
        """A third-of-the-span Bezier IS the cubic Hermite with those slopes:
        the vectorised ``MathUtils.evaluate_hermite`` agrees everywhere."""
        c = _curve()
        at = [0.5 * i for i in range(25)]
        ref = MathUtils.evaluate_hermite(
            [0.0, 12.0], [0.0, 10.0], [0, 1], [1.0, 0.0], [1.0, 0.0], at
        )
        for t, want in zip(at, ref):
            self.assertAlmostEqual(c.evaluate(t), float(want), places=6)

    def test_keys_evaluate_to_their_values(self):
        c = _curve()
        self.assertEqual(c.evaluate(0.0), 0.0)
        self.assertEqual(c.evaluate(12.0), 10.0)

    def test_linear_step_and_stepnext_spans(self):
        def keys(interp):
            return [CurveKey(0.0, 0.0, interpolation=interp), CurveKey(10.0, 4.0)]

        self.assertAlmostEqual(KeyCurve(keys("linear")).evaluate(2.5), 1.0)
        self.assertEqual(KeyCurve(keys("step")).evaluate(9.9), 0.0)
        self.assertEqual(KeyCurve(keys("stepnext")).evaluate(0.1), 4.0)
        self.assertEqual(KeyCurve(keys("stepnext")).evaluate(0.0), 0.0)

    def test_a_sampled_span_is_drawn_through_its_samples(self):
        k0 = CurveKey(0.0, 0.0, interpolation="linear", samples=[(5.0, 8.0)])
        c = KeyCurve([k0, CurveKey(10.0, 0.0)])
        self.assertAlmostEqual(c.evaluate(2.5), 4.0)
        self.assertEqual(c.segments()[0]["samples"], [(5.0, 8.0)])

    def test_weighted_handles_follow_their_length(self):
        c = KeyCurve(
            [
                CurveKey(0.0, 0.0, out_handle=(4.0, 4.0)),
                CurveKey(12.0, 0.0, in_handle=(-4.0, 4.0)),
            ]
        )
        p0, p1, p2, p3 = c.control_points(0)
        self.assertEqual((p1, p2), ((4.0, 4.0), (8.0, 4.0)))
        self.assertAlmostEqual(c.evaluate(6.0), 3.0)  # 3/4 of the handle height

    def test_an_overreaching_weighted_handle_is_pulled_back_to_the_span(self):
        c = KeyCurve(
            [
                CurveKey(0.0, 0.0, out_handle=(9.0, 3.0)),
                CurveKey(6.0, 0.0, in_handle=(-9.0, 3.0)),
            ]
        )
        _p0, p1, p2, _p3 = c.control_points(0)
        self.assertEqual((p1, p2), ((6.0, 2.0), (0.0, 2.0)), "each along itself")
        reach = KeyCurve(
            [
                CurveKey(0.0, 0.0, out_handle=(5.0, 1.0)),
                CurveKey(6.0, 0.0, in_handle=(-5.0, 1.0)),
            ]
        )
        _p0, p1, p2, _p3 = reach.control_points(0)
        self.assertEqual((p1, p2), ((5.0, 1.0), (1.0, 1.0)), "together past: kept")


class TestKeyCurveInfinity(unittest.TestCase):
    def _ramp(self, mode):
        return KeyCurve(
            [
                CurveKey(0.0, 0.0, interpolation="linear"),
                CurveKey(10.0, 5.0, in_handle=(-1.0, -0.5), out_handle=(1.0, 0.5)),
            ],
            pre_infinity=mode,
            post_infinity=mode,
        )

    def test_constant_holds_the_end_values(self):
        c = self._ramp("constant")
        self.assertEqual((c.evaluate(-5.0), c.evaluate(30.0)), (0.0, 5.0))

    def test_linear_continues_the_end_slopes(self):
        c = self._ramp("linear")
        self.assertAlmostEqual(c.evaluate(-4.0), -2.0)  # the linear span's slope
        self.assertAlmostEqual(c.evaluate(14.0), 7.0)  # the last key's handle

    def test_cycle_repeats_the_curve(self):
        c = self._ramp("cycle")
        self.assertAlmostEqual(c.evaluate(12.0), c.evaluate(2.0))
        self.assertAlmostEqual(c.evaluate(-8.0), c.evaluate(2.0))

    def test_cycle_offset_stacks_each_repeat(self):
        c = self._ramp("cycle_offset")
        self.assertAlmostEqual(c.evaluate(12.0), c.evaluate(2.0) + 5.0)
        self.assertAlmostEqual(c.evaluate(-8.0), c.evaluate(2.0) - 5.0)

    def test_oscillate_plays_every_other_repeat_backwards(self):
        c = self._ramp("oscillate")
        self.assertAlmostEqual(c.evaluate(12.0), c.evaluate(8.0))
        self.assertAlmostEqual(c.evaluate(22.0), c.evaluate(2.0))

    def test_linear_off_a_degenerate_end_continues_flat(self):
        """Bug (review 2026-10-12): two keys on one frame divided by zero, and
        a vertical end handle extended at 1e6 per frame (the stand-in slope a
        control point uses); Blender holds both ends flat."""
        stacked = KeyCurve(
            [
                CurveKey(0.0, 0.0, interpolation="linear"),
                CurveKey(0.0, 1.0),
                CurveKey(10.0, 5.0),
            ],
            pre_infinity="linear",
        )
        self.assertEqual(stacked.evaluate(-1.0), 0.0)
        vertical = KeyCurve(
            [CurveKey(0.0, 0.0), CurveKey(10.0, 5.0, out_handle=(0.0, 2.0))],
            post_infinity="linear",
        )
        self.assertEqual(vertical.evaluate(20.0), 5.0)

    def test_unknown_modes_are_refused(self):
        with self.assertRaises(ValueError):
            KeyCurve([], post_infinity="bounce")
        with self.assertRaises(ValueError):
            CurveKey(0.0, 0.0, interpolation="smooth")


class TestKeyCurveBoundsAndPreview(unittest.TestCase):
    def test_bounds_take_in_an_overshoot(self):
        c = KeyCurve(
            [
                CurveKey(0.0, 0.0, out_handle=(3.0, 9.0)),
                CurveKey(10.0, 0.0, in_handle=(-3.0, 9.0)),
            ]
        )
        lo, hi = c.value_bounds()
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 6.75)  # 3/4 of 9 at the midpoint
        self.assertEqual(c.value_bounds(handles=True)[1], 9.0)

    def test_bounds_past_the_keys_take_in_every_repeat(self):
        """Bug (review 2026-10-12): outside the keys the bounds read only the
        window's two ends, so a repeating curve framed past its keys (the
        graph editor's fit) read one value: a 0 -> 10 -> 0 tent cycled over
        12..18 read (4, 4) where it peaks at 10 (frame 15)."""
        keys = [
            CurveKey(0.0, 0.0, interpolation="linear"),
            CurveKey(5.0, 10.0, interpolation="linear"),
            CurveKey(10.0, 0.0),
        ]
        for mode in ("cycle", "oscillate"):
            tent = KeyCurve(keys, pre_infinity=mode, post_infinity=mode)
            self.assertEqual(tent.value_bounds(12.0, 18.0), (4.0, 10.0), mode)
            self.assertEqual(tent.value_bounds(-8.0, -2.0), (4.0, 10.0), mode)
            self.assertEqual(tent.value_bounds(103.0, 104.0), (6.0, 8.0), mode)
            self.assertEqual(tent.value_bounds(-100.0, 100.0), (0.0, 10.0), mode)
        ramp = KeyCurve(
            [CurveKey(0.0, 0.0, interpolation="linear"), CurveKey(10.0, 5.0)],
            pre_infinity="cycle_offset",
            post_infinity="cycle_offset",
        )
        self.assertEqual(ramp.value_bounds(-25.0, 32.0), (-12.5, 16.0))
        self.assertEqual(ramp.value_bounds(12.0, 18.0), (6.0, 9.0))

    def test_normalization_maps_onto_minus_one_one(self):
        c = _curve()
        offset, scale = c.normalization()
        lo, hi = c.value_bounds()
        self.assertAlmostEqual((lo - offset) * scale, -1.0)
        self.assertAlmostEqual((hi - offset) * scale, 1.0)
        flat = KeyCurve([CurveKey(0.0, 3.0), CurveKey(5.0, 3.0)])
        self.assertEqual(flat.normalization(), (3.0, 1.0))

    def test_preview_windows_the_keys_with_a_bounding_key_each_side(self):
        c = KeyCurve([CurveKey(float(t), float(t)) for t in range(0, 50, 10)])
        p = c.to_preview(15.0, 25.0, hidden_times=[20.0])
        self.assertEqual([k[0] for k in p["keys"]], [10.0, 20.0, 30.0])
        self.assertEqual(p["hidden"], [False, True, False])
        self.assertEqual(len(p["segments"]), 2)
        self.assertIsNone(c.to_preview(100.0, 120.0))
        crossing = c.to_preview(12.0, 18.0)
        self.assertEqual([k[0] for k in crossing["keys"]], [10.0, 20.0])

    def test_only_bezier_spans_carry_control_points(self):
        c = KeyCurve(
            [
                CurveKey(0.0, 0.0, interpolation="step"),
                CurveKey(5.0, 1.0),
                CurveKey(10.0, 0.0),
            ]
        )
        segs = c.segments()
        self.assertEqual(
            [(s["out_type"], s["cp1"] is None) for s in segs],
            [("step", True), ("bezier", False)],
        )


class TestKeyCurveEdits(unittest.TestCase):
    def _three(self, weighted=True):
        return KeyCurve(
            [
                CurveKey(0.0, 0.0, out_handle=(2.0, 0.0)),
                CurveKey(10.0, 5.0, in_handle=(-2.0, -1.0), out_handle=(4.0, 2.0)),
                CurveKey(20.0, 0.0, in_handle=(-2.0, 0.0)),
            ],
            weighted=weighted,
        )

    def test_a_move_returns_a_new_curve(self):
        c = self._three()
        moved = c.with_keys_moved([1], dt=2.0, dv=1.0)
        self.assertEqual((moved.keys[1].time, moved.keys[1].value), (12.0, 6.0))
        self.assertEqual(c.keys[1].time, 10.0, "the original is untouched")

    def test_a_move_never_crosses_an_unmoved_key(self):
        moved = self._three().with_keys_moved([1], dt=50.0)
        self.assertLess(moved.keys[1].time, 20.0)
        self.assertEqual(moved.times(), sorted(moved.times()))
        together = self._three().with_keys_moved([1, 2], dt=50.0)
        self.assertEqual(together.times(), [0.0, 60.0, 70.0])

    def test_a_move_against_a_crowding_key_never_reverses(self):
        """Bug (review 2026-10-12): a neighbour already nearer than the key
        tolerance turned the clamp's limit past zero, so a drag LEFT moved
        the key right (0.0005 -> 0.001)."""
        crowded = KeyCurve([CurveKey(0.0, 0.0), CurveKey(0.0005, 1.0)])
        self.assertEqual(crowded.with_keys_moved([1], dt=-1.0).times(), [0.0, 0.0005])
        self.assertEqual(crowded.with_keys_moved([0], dt=1.0).times(), [0.0, 0.0005])

    def test_a_unified_weighted_key_swings_its_partner_keeping_its_length(self):
        c = self._three().with_handle(1, "out", (3.0, 3.0))
        ix, iy = c.keys[1].in_handle
        self.assertAlmostEqual(math.hypot(ix, iy), math.hypot(2.0, 1.0))
        self.assertAlmostEqual(iy / ix, 1.0)

    def test_an_unweighted_partner_keeps_its_time_offset(self):
        c = self._three(weighted=False).with_handle(1, "out", (1.0, 3.0))
        self.assertEqual(c.keys[1].in_handle, (-2.0, -6.0))

    def test_a_broken_key_leaves_its_partner(self):
        c = self._three()
        c.keys[1].broken = True
        out = c.with_handle(1, "out", (3.0, 3.0))
        self.assertEqual(out.keys[1].in_handle, (-2.0, -1.0))
        forced = self._three().with_handle(1, "out", (3.0, 3.0), unify=False)
        self.assertEqual(forced.keys[1].in_handle, (-2.0, -1.0))

    def test_a_handle_cannot_point_across_its_key(self):
        c = self._three().with_handle(1, "out", (-3.0, 1.0), unify=False)
        self.assertEqual(c.keys[1].out_handle, (0.0, 1.0))

    def test_infinity_edit(self):
        c = self._three().with_infinity(post="cycle")
        self.assertEqual((c.pre_infinity, c.post_infinity), ("constant", "cycle"))


if __name__ == "__main__":
    unittest.main()
