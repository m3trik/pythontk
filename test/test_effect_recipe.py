# !/usr/bin/python
# coding=utf-8
"""Tests for the scene's effect recipe (``core_utils/engines/shots/effect_recipe.py``)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk
from pythontk.core_utils.engines.shots.effect_recipe import EffectRecipe
from pythontk.math_utils.ramp_keys import RampKeys


class TestDefaultsAndCoercion(unittest.TestCase):
    def test_public_name(self):
        self.assertIs(ptk.EffectRecipe, EffectRecipe)

    def test_the_defaults_are_the_measured_look(self):
        r = EffectRecipe()
        self.assertEqual(r.fade_frames, 15.0)
        self.assertEqual(
            (r.pulse_period, r.pulse_duty, r.pulse_ramp), (2.86, 0.59, 0.25)
        )
        self.assertEqual((r.pulse_lead_in, r.pulse_lead_out), (0.72, 0.72))
        # The linear value of #0054DD, the production blue; dim black.
        self.assertEqual(r.colors, ((0.0, 0.088656, 0.723055), (0.0, 0.0, 0.0)))
        self.assertTrue(r.audio_snap)

    def test_every_field_is_held_to_its_range(self):
        r = EffectRecipe(
            fade_frames=0,
            pulse_period=-1,
            pulse_duty=2.0,
            pulse_ramp=0.9,
            pulse_lead_in=-3,
        )
        self.assertEqual(r.fade_frames, 1.0)
        self.assertEqual(r.pulse_period, EffectRecipe.LIMITS["pulse_period"][0])
        self.assertEqual(r.pulse_duty, 0.99)
        self.assertEqual(r.pulse_ramp, 0.5)
        self.assertEqual(r.pulse_lead_in, 0.0)

    def test_a_bad_value_falls_back_to_its_default(self):
        r = EffectRecipe(
            fade_frames="lots",
            pulse_period=float("nan"),
            pulse_bright="red",
            pulse_dim=(1, 2),
        )
        self.assertEqual(r.fade_frames, 15.0)
        self.assertEqual(r.pulse_period, 2.86)
        self.assertEqual(r.pulse_bright, EffectRecipe().pulse_bright)
        self.assertEqual(r.pulse_dim, (0.0, 0.0, 0.0))

    def test_a_colour_may_exceed_one_but_never_go_negative(self):
        r = EffectRecipe(pulse_bright=(2.5, -1, 0.1234567891))
        self.assertEqual(r.pulse_bright, (2.5, 0.0, 0.123457))

    def test_frozen(self):
        with self.assertRaises(Exception):
            EffectRecipe().fade_frames = 3


class TestSerialisation(unittest.TestCase):
    def test_round_trip(self):
        r = EffectRecipe(pulse_period=2.0, pulse_dim=(0.1, 0.2, 0.3), audio_snap=False)
        self.assertEqual(EffectRecipe.from_dict(r.to_dict()), r)
        self.assertEqual(r.to_dict()["pulse_dim"], [0.1, 0.2, 0.3])

    def test_from_dict_is_tolerant(self):
        self.assertEqual(EffectRecipe.from_dict(None), EffectRecipe())
        self.assertEqual(EffectRecipe.from_dict("junk"), EffectRecipe())
        r = EffectRecipe.from_dict({"pulse_duty": 0.5, "from_the_future": 1})
        self.assertEqual(r.pulse_duty, 0.5)

    def test_replace_refuses_an_unknown_field(self):
        self.assertEqual(EffectRecipe().replace(fade_frames=20).fade_frames, 20.0)
        with self.assertRaises(TypeError):
            EffectRecipe().replace(fade_length=20)


class TestFingerprint(unittest.TestCase):
    def test_stable_and_per_effect(self):
        a, b = EffectRecipe(), EffectRecipe()
        self.assertEqual(a.fingerprint("pulse"), b.fingerprint("pulse"))
        self.assertNotEqual(a.fingerprint("pulse"), a.fingerprint("fade_in"))
        self.assertNotEqual(a.fingerprint("fade_in"), a.fingerprint("fade_out"))

    def test_a_cadence_change_changes_the_pulse_only(self):
        a = EffectRecipe()
        b = a.replace(pulse_period=2.0)
        self.assertNotEqual(a.fingerprint("pulse"), b.fingerprint("pulse"))
        self.assertEqual(a.fingerprint("fade_in"), b.fingerprint("fade_in"))

    def test_colours_never_stale_a_pulse(self):
        """A Build writes the colours only on a channel it creates, so a new
        colour is no reason to re-key."""
        a = EffectRecipe()
        b = a.replace(pulse_bright=(1, 0, 0), pulse_dim=(0.2, 0.2, 0.2))
        self.assertEqual(a.fingerprint("pulse"), b.fingerprint("pulse"))

    def test_spin_box_noise_is_not_a_change(self):
        a = EffectRecipe()
        b = a.replace(pulse_period=2.86 + 1e-9)
        self.assertEqual(a.fingerprint("pulse"), b.fingerprint("pulse"))

    def test_unknown_effect(self):
        with self.assertRaises(ValueError):
            EffectRecipe().fingerprint("glow")


class TestKeying(unittest.TestCase):
    def test_pulse_cadence_is_seconds_at_the_scene_rate(self):
        """``key_pulse(period=86)`` was frames: 2.86 s only at 30 fps."""
        cad = EffectRecipe().pulse_cadence(24.0)
        self.assertAlmostEqual(cad["period"], 2.86 * 24)
        self.assertAlmostEqual(cad["lead_in"], 0.72 * 24)
        self.assertEqual(cad["bright_fraction"], 0.59)
        self.assertEqual(cad["ramp_fraction"], 0.25)

    def test_plan_is_the_planner_with_the_recipe(self):
        r = EffectRecipe()
        self.assertEqual(r.plan("fade_in", 10, 25, 30.0), [(10.0, 0.0), (25.0, 1.0)])
        self.assertEqual(r.plan("fade_out", 10, 25, 30.0), [(10.0, 1.0), (25.0, 0.0)])
        self.assertEqual(
            r.plan("pulse", 0, 120, 24.0),
            RampKeys.pulse(0, 120, **r.pulse_cadence(24.0)),
        )
        with self.assertRaises(ValueError):
            r.plan("clip", 0, 10, 24.0)

    def test_window_places_a_fade(self):
        r = EffectRecipe()
        self.assertEqual(r.window("fade_in", 100, 300, "start"), (100.0, 115.0))
        self.assertEqual(r.window("fade_out", 100, 300, "end"), (285.0, 300.0))
        self.assertEqual(r.window("fade_in", 100, 300, 0.5), (192.5, 207.5))
        # An anchor (the manifest's spread) overrides the placement.
        self.assertEqual(r.window("fade_out", 100, 300, "end", 0.0), (100.0, 115.0))

    def test_window_spans_for_a_pulse_and_a_clip(self):
        r = EffectRecipe()
        self.assertEqual(r.window("pulse", 100, 300, "span", 0.0), (100.0, 300.0))
        self.assertEqual(r.window("clip", 100, 300), (100.0, 300.0))

    def test_envelope_states_a_fade_as_one_phase(self):
        env = EffectRecipe(fade_frames=20).envelope("fade_out", "end")
        self.assertEqual(list(env), ["visibility"])
        self.assertEqual(list(env["visibility"]), ["out"])
        block = env["visibility"]["out"]
        self.assertEqual((block["anchor"], block["duration"]), ("end", 20.0))
        self.assertEqual(block["values"], [1.0, 0.0])

    def test_envelope_states_a_pulse_as_a_span_holding_a_beat(self):
        env = EffectRecipe().envelope("pulse", "span", fps=30.0)["highlight"]
        self.assertEqual(set(env), {"in", "out"})
        self.assertAlmostEqual(env["in"]["duration"], (0.72 + 2.86) * 30)
        self.assertAlmostEqual(env["out"]["duration"], 0.72 * 30)
        self.assertEqual(EffectRecipe().envelope("clip"), {})

    def test_pulse_cycles_counts_the_train_between_the_leads(self):
        r = EffectRecipe()
        self.assertAlmostEqual(r.pulse_cycles(4.0), (4.0 - 1.44) / 2.86)
        self.assertAlmostEqual(
            r.replace(pulse_lead_in=0, pulse_lead_out=0).pulse_cycles(4.0), 4.0 / 2.86
        )
        self.assertEqual(r.pulse_cycles(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
