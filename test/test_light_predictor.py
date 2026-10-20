# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.light_predictor: the frame of a mix."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from pythontk import ImgUtils
from pythontk.core_utils.engines.lighting.light_layers import LightLayers, LightMix
from pythontk.core_utils.engines.lighting.light_predictor import (
    LightPredictor,
    ViewerTransform,
)


class TestViewerTransform(unittest.TestCase):
    def test_display_is_the_tone_map_then_srgb(self):
        lin = np.random.default_rng(0).random((3, 4, 3)).astype(np.float32) * 3
        expected = ImgUtils.linear_to_srgb(ImgUtils.tonemap(lin, "aces_filmic", 1.5))
        np.testing.assert_allclose(
            ViewerTransform(exposure=1.5).display(lin), expected, atol=1e-6
        )

    def test_round_trips_through_a_dict(self):
        t = ViewerTransform.from_dict(ViewerTransform("linear", 2.0).to_dict())
        self.assertEqual((t.operator, t.exposure), ("linear", 2.0))


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="light_predictor_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        rng = np.random.default_rng(1)
        self.store = LightLayers(self.root)
        self.a = rng.random((8, 10, 3)).astype(np.float32)
        self.b = rng.random((8, 10, 3)).astype(np.float32)
        self.e = rng.random((8, 10, 3)).astype(np.float32)
        self.store.add("a", self.a)
        self.store.add("b", self.b)
        self.store.add("e", self.e, kind="emission")
        self.predictor = LightPredictor(self.store)

    def test_the_frame_is_the_tinted_sum(self):
        mix = LightMix({"a": {"weight": 2.0, "kelvin": 3000}, "b": {"weight": 0.5}})
        expected = 2.0 * np.asarray(ImgUtils.kelvin_tint(3000), np.float32) * self.a + 0.5 * self.b
        np.testing.assert_allclose(
            self.predictor.linear(mix), expected, rtol=2e-3, atol=1e-4
        )

    def test_emission_is_not_tinted(self):
        mix = LightMix({"e": {"weight": 1.0, "kelvin": 2000}})
        np.testing.assert_allclose(
            self.predictor.linear(mix), self.e, rtol=2e-3, atol=1e-4
        )

    def test_the_display_frame_goes_through_the_viewer(self):
        mix = LightMix({"a": {"weight": 1.0}}, exposure=2.0)
        np.testing.assert_allclose(
            self.predictor.frame(mix),
            ViewerTransform(exposure=2.0).display(self.predictor.linear(mix)),
            atol=1e-6,
        )

    def test_a_region_and_a_size_pass_through(self):
        mix = LightMix({"a": {"weight": 1.0}})
        self.assertEqual(
            self.predictor.linear(mix, region=(2, 1, 7, 5)).shape, (4, 5, 3)
        )
        self.assertEqual(self.predictor.linear(mix, size=(5, 4)).shape, (4, 5, 3))

    def test_a_layer_the_store_lacks_is_named(self):
        with self.assertRaises(KeyError) as ctx:
            self.predictor.linear(LightMix({"nope": {"weight": 1.0}}))
        self.assertIn("nope", str(ctx.exception))

    def test_zero_weights_are_never_read(self):
        mix = LightMix({"a": {"weight": 1.0}, "missing": {"weight": 0.0}})
        self.predictor.linear(mix)  # no KeyError for the zero-weight layer

    def test_an_empty_mix_is_black(self):
        out = self.predictor.linear(LightMix({}))
        self.assertEqual(out.shape, (8, 10, 3))
        self.assertEqual(float(out.max()), 0.0)

    def test_contributions_split_the_light_in_a_mask(self):
        mix = LightMix({"a": {"weight": 1.0}, "b": {"weight": 3.0}})
        mask = np.zeros((8, 10), bool)
        mask[2:5, 3:7] = True
        shares = self.predictor.contributions(mix, mask)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=5)
        a_light = float(ImgUtils.luminance(self.a[mask]).sum())
        b_light = float(ImgUtils.luminance(3.0 * self.b[mask]).sum())
        self.assertAlmostEqual(shares["a"], a_light / (a_light + b_light), places=3)

    def test_contributions_for_many_regions_in_one_pass(self):
        mix = LightMix({"a": {"weight": 1.0}, "b": {"weight": 3.0}})
        left = np.zeros((8, 10), bool)
        left[:, :5] = True
        right = ~left
        both = self.predictor.region_contributions(mix, {"left": left, "right": right})
        self.assertEqual(set(both), {"left", "right"})
        for name, mask in (("left", left), ("right", right)):
            single = self.predictor.contributions(mix, mask)
            for layer, share in single.items():
                self.assertAlmostEqual(both[name][layer], share, places=6)

    def test_bakes_sum_like_frames(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV reads EXR bakes")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        rng = np.random.default_rng(5)
        atlases = {}
        for name in ("a", "b"):
            d = os.path.join(self.root, f"_bake_{name}")
            os.makedirs(d)
            img = rng.random((4, 6, 3)).astype(np.float32)
            atlases[name] = img
            path = os.path.join(d, "ROOM_Lightmap.exr")
            cv2.imwrite(path, img[..., ::-1])  # OpenCV writes BGR
            self.store.add(name, getattr(self, name), replace=True, bakes=[path])
        mix = LightMix({"a": {"weight": 2.0, "kelvin": 3000}, "b": {"weight": 0.5}})
        sums = self.predictor.bakes(mix)
        expected = 2.0 * np.asarray(ImgUtils.kelvin_tint(3000), np.float32) * atlases["a"] + 0.5 * atlases["b"]
        np.testing.assert_allclose(
            sums["ROOM_Lightmap.exr"], expected, rtol=1e-5, atol=1e-6
        )

    def test_a_viewer_only_layer_is_no_part_of_a_bake(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV reads EXR bakes")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        d = os.path.join(self.root, "_bake_a")
        os.makedirs(d)
        path = os.path.join(d, "ROOM_Lightmap.exr")
        cv2.imwrite(path, np.ones((2, 2, 3), np.float32))
        self.store.add("a", self.a, replace=True, bakes=[path])
        self.store.add("p", self.b, kind="panel")  # a preview: nothing ships it
        sums = self.predictor.bakes(
            LightMix({"a": {"weight": 1.0}, "p": {"weight": 5.0}})
        )
        np.testing.assert_allclose(sums["ROOM_Lightmap.exr"], 1.0)

    def test_a_bake_reads_as_rgb_linear_float(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV reads EXR bakes")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        path = os.path.join(self.root, "warm.exr")
        cv2.imwrite(path, np.array([[[0.25, 0.5, 2.0]]], np.float32))  # BGR
        image = LightPredictor.read_bake(path)
        self.assertEqual((image.dtype, image.shape), (np.float32, (1, 1, 3)))
        np.testing.assert_allclose(image[0, 0], [2.0, 0.5, 0.25])

    def test_a_mix_ships_its_layers_lights_at_their_weight_and_tint(self):
        """What a deliverable carries for the highlights: each light layer's own
        lights (baked white) at the layer's weight, tinted by its temperature."""
        lamp = {
            "name": "K",
            "type": "point",
            "position": [0.0, 1.0, 0.0],
            "color": [1.0, 1.0, 1.0],
            "intensity": 3.0,
            "unit_scale": 1.0,
        }
        self.store.add("a", self.a, replace=True, lights=[lamp])
        mix = LightMix({"a": {"weight": 2.0, "kelvin": 3000}, "b": {"weight": 1.0}})
        (light,) = self.predictor.lights(mix)
        self.assertAlmostEqual(light["intensity"], 6.0)
        np.testing.assert_allclose(light["color"], np.asarray(ImgUtils.kelvin_tint(3000), np.float32), rtol=1e-5)
        self.assertEqual(lamp["intensity"], 3.0)  # the layer's own record untouched
        self.assertEqual(self.predictor.lights(LightMix({"b": {"weight": 1.0}})), [])

    def test_a_layers_specular_scale_dims_only_what_it_ships_for_highlights(self):
        lamp = {
            "name": "K",
            "type": "point",
            "position": [0.0, 1.0, 0.0],
            "color": [1.0, 1.0, 1.0],
            "intensity": 3.0,
            "unit_scale": 1.0,
        }
        self.store.add("a", self.a, replace=True, lights=[lamp])
        mix = LightMix({"a": {"weight": 2.0, "specular": 0.25}})
        (light,) = self.predictor.lights(mix)
        self.assertAlmostEqual(light["intensity"], 1.5)  # 3 x 2 x 0.25
        np.testing.assert_allclose(  # the light itself -- its bake -- unchanged
            self.predictor.linear(mix),
            self.predictor.linear(LightMix({"a": {"weight": 2.0}})),
        )


if __name__ == "__main__":
    unittest.main(exit=False)
