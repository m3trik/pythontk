#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk StepToggle.

Run with:
    python -m pytest test_step_toggle.py -v
    python test_step_toggle.py
"""
import unittest

from pythontk.core_utils.step_toggle import StepToggle

from conftest import BaseTestCase


class FakeClock:
    """Injected monotonic clock — advance time without sleeping."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


class StepToggleTest(BaseTestCase):
    """StepToggle test class."""

    def setUp(self):
        self.clock = FakeClock()
        StepToggle.clear()

    def tearDown(self):
        StepToggle.clear()

    def _toggle(self, steps=2, timeout=2.0):
        return StepToggle(steps=steps, timeout=timeout, clock=self.clock)

    # ------------------------------------------------------------------ stepping
    def test_single_step_is_a_bistate_toggle(self):
        toggle = self._toggle(steps=1)
        self.assertEqual(toggle.advance(), 1)
        self.assertEqual(toggle.advance(), 0)
        self.assertEqual(toggle.advance(), 1)

    def test_two_steps_is_a_tristate_toggle(self):
        toggle = self._toggle(steps=2)
        self.assertEqual([toggle.advance() for _ in range(4)], [1, 2, 0, 1])

    def test_steps_can_be_overridden_per_press(self):
        toggle = self._toggle(steps=2)
        self.assertEqual([toggle.advance(steps=3) for _ in range(4)], [1, 2, 3, 0])

    def test_state_beyond_a_shrunken_step_count_returns_home(self):
        toggle = self._toggle(steps=3)
        toggle.advance()
        toggle.advance()
        self.assertEqual(toggle.state, 2)
        self.assertEqual(toggle.advance(steps=1), 0)

    # ------------------------------------------------------------------ timing
    def test_stale_press_from_a_step_returns_home(self):
        """A multi-step toggle collapses to a single step once the user pauses."""
        toggle = self._toggle(steps=3)
        self.assertEqual(toggle.advance(), 1)
        self.clock.tick(5.0)
        self.assertEqual(toggle.advance(), 0)

    def test_stale_press_at_home_starts_a_fresh_cycle(self):
        toggle = self._toggle(steps=2)
        self.clock.tick(5.0)
        self.assertEqual(toggle.advance(), 1)

    def test_press_inside_the_window_steps_deeper(self):
        toggle = self._toggle(steps=2, timeout=2.0)
        self.assertEqual(toggle.advance(), 1)
        self.clock.tick(1.9)
        self.assertEqual(toggle.advance(), 2)

    def test_timeout_none_never_goes_stale(self):
        toggle = self._toggle(steps=2, timeout=None)
        self.assertEqual(toggle.advance(), 1)
        self.clock.tick(1000.0)
        self.assertEqual(toggle.advance(), 2)

    def test_timeout_can_be_overridden_per_press(self):
        toggle = self._toggle(steps=2, timeout=2.0)
        self.assertEqual(toggle.advance(), 1)
        self.clock.tick(3.0)
        self.assertEqual(toggle.advance(timeout=10.0), 2)

    # ------------------------------------------------------------------ context
    def test_new_context_restarts_at_step_one(self):
        toggle = self._toggle(steps=3)
        self.assertEqual(toggle.advance(context="a"), 1)
        self.assertEqual(toggle.advance(context="a"), 2)
        self.assertEqual(toggle.advance(context="b"), 1)

    def test_same_context_steps_normally(self):
        toggle = self._toggle(steps=2)
        self.assertEqual(toggle.advance(context="a"), 1)
        self.assertEqual(toggle.advance(context="a"), 2)
        self.assertEqual(toggle.advance(context="a"), 0)

    def test_context_none_opts_out_of_context_tracking(self):
        toggle = self._toggle(steps=2)
        toggle.advance(context="a")
        self.assertEqual(toggle.advance(), 2)  # no context given -> plain step

    # ------------------------------------------------------------------ cycle starts
    def test_began_cycle_only_on_the_press_that_leaves_home(self):
        toggle = self._toggle(steps=2)
        toggle.advance()
        self.assertTrue(toggle.began_cycle)
        toggle.advance()
        self.assertFalse(toggle.began_cycle)  # stepping deeper
        toggle.advance()
        self.assertFalse(toggle.began_cycle)  # going home

    def test_began_cycle_on_a_stale_retarget(self):
        """After a pause, acting on something else is a new session — its home is
        the view being left now, not the one from the abandoned cycle."""
        toggle = self._toggle(steps=2)
        toggle.advance(context="a")
        self.clock.tick(5.0)
        self.assertEqual(toggle.advance(context="b"), 1)
        self.assertTrue(toggle.began_cycle)

    def test_no_new_cycle_when_retargeting_inside_the_window(self):
        """A quick re-aim stays in the same cycle, so it still unwinds to where it began."""
        toggle = self._toggle(steps=2)
        toggle.advance(context="a")
        self.assertEqual(toggle.advance(context="b"), 1)
        self.assertFalse(toggle.began_cycle)

    def test_began_cycle_after_a_stale_return_home(self):
        toggle = self._toggle(steps=2)
        toggle.advance()
        self.clock.tick(5.0)
        self.assertEqual(toggle.advance(), 0)  # stale -> home
        self.assertFalse(toggle.began_cycle)
        self.clock.tick(5.0)
        self.assertEqual(toggle.advance(), 1)  # ... and out again
        self.assertTrue(toggle.began_cycle)

    def test_reset_clears_began_cycle(self):
        toggle = self._toggle(steps=2)
        toggle.advance()
        toggle.reset()
        self.assertFalse(toggle.began_cycle)

    # ------------------------------------------------------------------ misc
    def test_at_home_and_reset(self):
        toggle = self._toggle(steps=2)
        self.assertTrue(toggle.at_home)
        toggle.advance()
        toggle.payload = "snapshot"
        self.assertFalse(toggle.at_home)
        toggle.reset()
        self.assertTrue(toggle.at_home)
        self.assertIsNone(toggle.payload)
        self.assertEqual(toggle.advance(), 1)

    def test_payload_is_untouched_by_advance(self):
        toggle = self._toggle(steps=2)
        toggle.advance()
        toggle.payload = {"cam": 1}
        toggle.advance()
        toggle.advance()
        self.assertEqual(toggle.payload, {"cam": 1})

    def test_get_returns_a_shared_instance(self):
        a = StepToggle.get("frame", steps=3, clock=self.clock)
        b = StepToggle.get("frame")
        self.assertIs(a, b)
        self.assertEqual(b.steps, 3)
        self.assertEqual(a.advance(), 1)
        self.assertEqual(b.state, 1)

    def test_clear_drops_shared_state(self):
        StepToggle.get("frame", clock=self.clock).advance()
        StepToggle.clear("frame")
        self.assertTrue(StepToggle.get("frame", clock=self.clock).at_home)

    def test_steps_are_floored_at_one(self):
        toggle = self._toggle(steps=0)
        self.assertEqual(toggle.steps, 1)
        self.assertEqual(toggle.advance(steps=0), 1)
        self.assertEqual(toggle.advance(steps=0), 0)

    # ------------------------------------------------------------------ scales
    def test_scales_single_step_is_unscaled(self):
        self.assertEqual(StepToggle.scales(1), [1.0])

    def test_scales_ascend_and_start_gentler_as_steps_grow(self):
        two, three = StepToggle.scales(2), StepToggle.scales(3)
        self.assertEqual(len(two), 2)
        self.assertEqual(len(three), 3)
        self.assertLess(two[0], two[1])
        self.assertLess(three[0], three[1])
        self.assertLess(three[1], three[2])
        self.assertLess(three[0], two[0])  # longer cycle -> wider first step
        self.assertGreater(three[-1], two[-1])  # ... and steps in further

    def test_scales_gain_is_per_step(self):
        scales = StepToggle.scales(3, spread=0.0, gain=2.0)
        self.assertEqual(scales, [1.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main(exit=False)
