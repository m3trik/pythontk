# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.math_utils.ramp_keys (RampKeys) -- the pulse and fade plans
mayatk's and blendertk's render-effect writers key, and the WebXR preview
publishes without keying. Pinned DCC-free here in its home; the DCC suites
cover the writers that compose it.
"""

import unittest

from pythontk import RampKeys


class RampKeysPulseTest(unittest.TestCase):
    def test_four_linear_keys_per_cycle_bracketed_dim(self):
        """The golden train the Maya writer's own test reads back off the curve:
        the lead-in is the cycle's own ramp (20), so every bright hold -- the
        first included -- is preceded by one ramp, and the trail-out lands dim."""
        keys = RampKeys.pulse(0, 200, 100, bright_fraction=0.6, ramp_fraction=0.2)
        self.assertEqual(
            keys,
            [
                (0.0, 0.0),
                (20.0, 1.0),
                (60.0, 1.0),
                (80.0, 0.0),
                (100.0, 0.0),
                (120.0, 1.0),
                (160.0, 1.0),
                (180.0, 0.0),
                (200.0, 0.0),
            ],
        )

    def test_the_two_gaps_can_be_set_apart(self):
        """A slow open and a hard cut: a zero gap still brackets, one whole
        frame before the end."""
        keys = RampKeys.pulse(
            0, 200, 100, bright_fraction=0.6, ramp_fraction=0.2, lead_in=40, lead_out=0
        )
        self.assertEqual(keys[:2], [(0.0, 0.0), (40.0, 1.0)])
        self.assertEqual(keys[-1], (200.0, 0.0))
        self.assertEqual(keys[-2][0], 199.0)

    def test_gaps_that_cannot_fit_are_scaled_to_the_window(self):
        keys = RampKeys.pulse(0, 100, 50, lead_in=400, lead_out=400)
        times = [t for t, _ in keys]
        self.assertEqual(times, sorted(times))
        self.assertEqual((times[0], times[-1]), (0.0, 100.0))

    def test_one_value_per_frame(self):
        keys = RampKeys.pulse(0, 400, 86)
        times = [t for t, _ in keys]
        self.assertEqual(len(times), len(set(times)))

    def test_nothing_to_plan_is_an_empty_list(self):
        self.assertEqual(RampKeys.pulse(0, 100, 0), [])
        self.assertEqual(RampKeys.pulse(50, 50, 10), [])
        # Snapped first: a window that rounds shut is empty too.
        self.assertEqual(RampKeys.pulse(10.4, 10.2, 5), [])

    def test_sub_frame_cadence_keeps_the_exact_period(self):
        """``whole_frames=False`` advances by the exact period and floors the
        brackets at PULSE_GAP_MIN, not a whole frame."""
        keys = RampKeys.pulse(0, 400, 85.8, whole_frames=False, lead_in=0, lead_out=0)
        brights = [t for t, v in keys if v == 1.0]
        self.assertAlmostEqual(brights[0], RampKeys.PULSE_GAP_MIN)
        self.assertTrue(
            any(abs(t - (RampKeys.PULSE_GAP_MIN + 85.8)) < 1e-9 for t in brights)
        )

    def test_gap_floors(self):
        self.assertEqual(RampKeys.pulse_gaps(0, 100, 10, 0, 0, gap_min=1.0), (1.0, 1.0))
        self.assertEqual(RampKeys.pulse_gaps(0, 100, 10), (10.0, 10.0))


class RampKeysFadeTest(unittest.TestCase):
    def test_in_and_out_are_two_keys(self):
        self.assertEqual(RampKeys.fade(1, 15), [(1.0, 0.0), (15.0, 1.0)])
        self.assertEqual(RampKeys.fade(1, 15, "out"), [(1.0, 1.0), (15.0, 0.0)])
        self.assertEqual(RampKeys.fade(10.4, 25.6), [(10.0, 0.0), (26.0, 1.0)])

    def test_auto_is_the_hosts_to_resolve(self):
        with self.assertRaises(ValueError):
            RampKeys.fade(0, 10, "auto")

    def test_a_loop_holds_either_side_of_the_ramp(self):
        self.assertEqual(
            RampKeys.fade_loop(15, hold=18),
            [(0.0, 0.0), (18.0, 0.0), (33.0, 1.0), (51.0, 1.0)],
        )
        self.assertEqual(
            RampKeys.fade_loop(15, hold=18, direction="out"),
            [(0.0, 1.0), (18.0, 1.0), (33.0, 0.0), (51.0, 0.0)],
        )

    def test_an_auto_loop_ramps_both_ways(self):
        self.assertEqual(
            RampKeys.fade_loop(10, hold=5, direction="auto"),
            [(0.0, 0.0), (5.0, 0.0), (15.0, 1.0), (20.0, 1.0), (30.0, 0.0)],
        )
        self.assertEqual(
            RampKeys.fade_loop(10, hold=5, direction="sideways"),
            RampKeys.fade_loop(10, hold=5, direction="auto"),
        )

    def test_no_hold_collapses_the_duplicate_frames(self):
        self.assertEqual(RampKeys.fade_loop(10), [(0.0, 0.0), (10.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
