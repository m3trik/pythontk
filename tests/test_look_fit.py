# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.look_match.fit: weights fitted to the measures."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from pythontk.core_utils.engines.lighting.light_layers import LightLayers, LightMix
from pythontk.core_utils.engines.lighting.light_predictor import LightPredictor
from pythontk.core_utils.engines.lighting.look_match.fit import LookFit
from pythontk.core_utils.engines.lighting.look_match.target import LookRegion, LookTarget


class TestLookFit(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="look_fit_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        h, w = 30, 40
        yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
        self.store = LightLayers(os.path.join(self.dir, "layers"))
        # three lights with different footprints: left, right, a top wash
        self.store.add(
            "left",
            np.repeat(np.exp(-((xx - 0.25) ** 2) / 0.02)[..., None], 3, 2).astype(
                np.float32
            ),
        )
        self.store.add(
            "right",
            np.repeat(np.exp(-((xx - 0.75) ** 2) / 0.02)[..., None], 3, 2).astype(
                np.float32
            ),
        )
        self.store.add(
            "wash",
            np.repeat((0.3 + 0.7 * (1 - yy))[..., None], 3, 2).astype(np.float32),
        )
        self.predictor = LightPredictor(self.store)
        self.truth = LightMix(
            {
                "left": {"weight": 1.2, "kelvin": 3000},
                "right": {"weight": 0.4, "kelvin": 6500},
                "wash": {"weight": 0.15},
            }
        )
        ref = (self.predictor.frame(self.truth) * 255 + 0.5).astype(np.uint8)
        path = os.path.join(self.dir, "ref.png")
        Image.fromarray(ref).save(path)
        regions = [
            LookRegion(
                "left",
                "area",
                {"rect": [0.1, 0.2, 0.4, 0.8]},
                measures=["mean", "warmth"],
            ),
            LookRegion(
                "right",
                "area",
                {"rect": [0.6, 0.2, 0.9, 0.8]},
                measures=["mean", "warmth"],
            ),
            LookRegion(
                "top", "area", {"rect": [0.0, 0.0, 1.0, 0.15]}, measures=["mean"]
            ),
        ]
        self.target = LookTarget(path, regions)

    def test_it_recovers_the_weights_that_made_the_reference(self):
        start = self.truth.with_weights({"left": 0.4, "right": 1.5, "wash": 0.5})
        fit = LookFit(self.predictor, self.target, tolerance=0.0, regularize=0.0)
        result = fit.run(start, ["left", "right", "wash"])
        self.assertLess(result.cost_after, result.cost_before * 0.01)
        for name in ("left", "right", "wash"):
            self.assertAlmostEqual(
                result.mix.weight(name),
                self.truth.weight(name),
                delta=0.08 * self.truth.weight(name) + 0.01,
                msg=name,
            )

    def test_it_recovers_the_colour_temperatures_that_made_the_reference(self):
        """Colour temperature is a lighting control too: a hue the weights cannot
        reach is the light's colour."""
        start = self.truth.with_entries(
            {"left": {"kelvin": 5000}, "right": {"kelvin": 4000}}
        ).with_weights({"left": 0.8})
        fit = LookFit(self.predictor, self.target, tolerance=0.0, regularize=0.0)
        result = fit.run(start, ["left"], kelvin=["left", "right"])
        self.assertAlmostEqual(result.mix.kelvin("left"), 3000, delta=250)
        self.assertAlmostEqual(result.mix.kelvin("right"), 6500, delta=600)
        self.assertAlmostEqual(result.mix.weight("left"), 1.2, delta=0.1)
        self.assertEqual(result.kelvins["right"][0], 4000)

    def test_fixed_layers_keep_their_weight(self):
        start = self.truth.with_weights({"left": 0.4})
        result = LookFit(
            self.predictor, self.target, tolerance=0.0, regularize=0.0
        ).run(start, ["left"])
        self.assertEqual(result.mix.weight("right"), self.truth.weight("right"))
        self.assertAlmostEqual(result.mix.weight("left"), 1.2, delta=0.1)

    def test_bounds_hold(self):
        start = self.truth.with_weights({"left": 0.4})
        result = LookFit(
            self.predictor,
            self.target,
            tolerance=0.0,
            regularize=0.0,
            bounds=(0.01, 0.6),
        ).run(start, ["left"])
        self.assertLessEqual(result.mix.weight("left"), 0.6 + 1e-9)

    def test_regularisation_holds_a_layer_the_measures_cannot_see(self):
        self.store.add("hidden", np.zeros((30, 40, 3), np.float32))
        start = self.truth.with_entries({"hidden": {"weight": 0.7}})
        result = LookFit(
            self.predictor, self.target, tolerance=0.0, regularize=0.5
        ).run(start, ["hidden", "left"])
        self.assertAlmostEqual(result.mix.weight("hidden"), 0.7, delta=1e-3)

    def test_the_report_names_what_changed(self):
        start = self.truth.with_weights({"left": 0.4})
        result = LookFit(
            self.predictor, self.target, tolerance=0.0, regularize=0.0
        ).run(start, ["left"])
        self.assertIn("left", result.changes)
        before, after = result.changes["left"]
        self.assertAlmostEqual(before, 0.4)
        self.assertGreater(after, 1.0)
        self.assertGreater(result.evaluations, 1)

    def test_a_free_layer_the_store_lacks_is_named(self):
        with self.assertRaises(KeyError):
            LookFit(self.predictor, self.target).run(self.truth, ["nope"])


if __name__ == "__main__":
    unittest.main(exit=False)
