# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.light_layers: the layer store and the mix."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from pythontk.core_utils.engines.lighting.light_layers import LightLayers, LightMix


class TestLightMix(unittest.TestCase):
    def test_entries_round_trip_through_json(self):
        mix = LightMix(
            {"key": {"weight": 2.0, "kelvin": 3200}, "fill": {"weight": 0.5}},
            notes="first",
        )
        again = LightMix.from_dict(json.loads(json.dumps(mix.to_dict())))
        self.assertEqual(again.weight("key"), 2.0)
        self.assertEqual(again.kelvin("key"), 3200)
        self.assertIsNone(again.kelvin("fill"))
        self.assertEqual(again.notes, "first")
        self.assertEqual(again.to_dict()["version"], LightMix.VERSION)

    def test_a_specular_scale_rides_with_an_entry(self):
        """A layer's highlight level, apart from its light: Arnold's aiSpecular."""
        mix = LightMix({"a": {"weight": 1.0, "specular": 0.4}, "b": {"weight": 1.0}})
        self.assertEqual(mix.specular("a"), 0.4)
        self.assertEqual(mix.specular("b"), 1.0)  # the default: the light's own
        self.assertNotIn("specular", mix.to_dict()["layers"]["b"])  # files unchanged
        again = LightMix.from_dict(json.loads(json.dumps(mix.to_dict())))
        self.assertEqual(again.specular("a"), 0.4)
        self.assertEqual(again.with_entries({"a": {"weight": 2.0}}).specular("a"), 0.4)
        with self.assertRaises(ValueError):
            LightMix({"a": {"weight": 1.0, "specular": -1.0}})

    def test_active_names_skip_zero_weights(self):
        mix = LightMix({"a": {"weight": 1.0}, "b": {"weight": 0.0}})
        self.assertEqual(mix.active(), ["a"])

    def test_set_changes_one_field_and_returns_a_copy(self):
        mix = LightMix({"a": {"weight": 1.0, "kelvin": 4000}})
        other = mix.with_entries(
            {"a": {"weight": 3.0}, "b": {"weight": 0.2, "kelvin": 2700}}
        )
        self.assertEqual(mix.weight("a"), 1.0)  # unchanged
        self.assertEqual(other.weight("a"), 3.0)
        self.assertEqual(other.kelvin("a"), 4000)  # kept
        self.assertEqual(other.kelvin("b"), 2700)

    def test_a_newer_mix_is_refused(self):
        with self.assertRaises(ValueError):
            LightMix.from_dict({"version": 99, "layers": {}})

    def test_save_and_load(self):
        d = tempfile.mkdtemp(prefix="light_mix_")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = os.path.join(d, "m.json")
        LightMix({"a": {"weight": 1.5}}).save(path)
        self.assertEqual(LightMix.load(path).weight("a"), 1.5)


class TestLightLayers(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="light_layers_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.store = LightLayers(self.root)
        self.frame = np.random.default_rng(0).random((12, 20, 3)).astype(np.float32)

    def test_a_registered_kind_reaches_the_store_the_prediction_and_the_rig(self):
        """The seam: a kind is data -- tint, ship, lights, rig -- read wherever a
        kind decides; a new one is one registration."""
        from pythontk.core_utils.engines.lighting.light_layers import LayerKind
        from pythontk.core_utils.engines.lighting.light_predictor import LightPredictor

        LightLayers.register_kind("probe", LayerKind(shippable=False))
        self.addCleanup(LightLayers.KINDS.pop, "probe")
        self.store.add("p", self.frame, kind="probe")
        self.assertFalse(self.store.info("p")["shippable"])
        self.assertFalse(self.store.kind("p").tinted)
        mix = LightMix({"p": {"weight": 2.0, "kelvin": 2700}})
        np.testing.assert_allclose(  # untinted: only weighted
            LightPredictor(self.store).linear(mix), 2.0 * self.store.load("p"), rtol=1e-5
        )
        with self.assertRaises(ValueError):
            self.store.add("q", self.frame, kind="sparkle")
        LightLayers.KINDS.pop("probe")  # a store a newer tool wrote: read, inert
        self.addCleanup(LightLayers.register_kind, "probe", LayerKind(shippable=False))
        self.assertIs(self.store.kind("p"), LightLayers.UNKNOWN_KIND)

    def test_a_layer_round_trips_at_half_precision(self):
        self.store.add("key", self.frame, kind="light", spec={"type": "spot"})
        np.testing.assert_allclose(
            self.store.load("key"), self.frame, rtol=1e-3, atol=1e-4
        )
        self.assertEqual(self.store.info("key")["kind"], "light")
        self.assertEqual(self.store.info("key")["spec"], {"type": "spot"})
        self.assertEqual(self.store.size, (20, 12))

    def test_the_manifest_survives_a_new_instance(self):
        self.store.add("key", self.frame, shippable=False, kind="panel")
        again = LightLayers(self.root)
        self.assertEqual(again.names(), ["key"])
        self.assertFalse(again.info("key")["shippable"])

    def test_every_layer_has_the_store_size(self):
        self.store.add("key", self.frame)
        with self.assertRaises(ValueError):
            self.store.add("fill", np.zeros((6, 10, 3), np.float32))

    def test_a_smaller_size_is_the_area_mean(self):
        self.store.add("key", self.frame)
        small = self.store.load("key", size=(10, 6))
        self.assertEqual(small.shape, (6, 10, 3))
        np.testing.assert_allclose(small.mean(), self.frame.mean(), rtol=2e-3)

    def test_a_region_reads_only_its_window(self):
        self.store.add("key", self.frame)
        win = self.store.load("key", region=(4, 2, 9, 7))
        np.testing.assert_allclose(win, self.frame[2:7, 4:9], rtol=1e-3, atol=1e-4)

    def test_an_unknown_layer_names_itself(self):
        with self.assertRaises(KeyError) as ctx:
            self.store.load("nope")
        self.assertIn("nope", str(ctx.exception))

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.add("x", self.frame, kind="sparkle")

    def test_replacing_a_layer_drops_its_cached_reads(self):
        self.store.add("key", self.frame)
        self.store.load("key", size=(10, 6))
        self.store.add("key", self.frame * 2.0, replace=True)
        np.testing.assert_allclose(
            self.store.load("key", size=(10, 6)).mean(),
            2 * self.frame.mean(),
            rtol=2e-3,
        )

    def test_replacing_needs_saying_so(self):
        self.store.add("key", self.frame)
        with self.assertRaises(FileExistsError):
            self.store.add("key", self.frame)

    def test_remove(self):
        self.store.add("key", self.frame)
        self.store.remove("key")
        self.assertEqual(LightLayers(self.root).names(), [])

    def test_a_layer_keeps_its_bake_files(self):
        bake_dir = os.path.join(self.root, "_incoming")
        os.makedirs(bake_dir)
        for name in ("ROOM_Lightmap.exr", "SCENE_Probe.exr"):
            with open(os.path.join(bake_dir, name), "wb") as f:
                f.write(b"exr")
        self.store.add(
            "key",
            self.frame,
            bakes=[os.path.join(bake_dir, n) for n in os.listdir(bake_dir)],
        )
        files = LightLayers(self.root).bake_files("key")
        self.assertEqual(sorted(files), ["ROOM_Lightmap.exr", "SCENE_Probe.exr"])
        self.assertTrue(all(os.path.isfile(p) for p in files.values()))

    def test_a_layer_without_a_bake_has_no_files(self):
        self.store.add("key", self.frame)
        self.assertEqual(self.store.bake_files("key"), {})

    def test_adding_a_layer_with_a_bake_returns_its_frame_file(self):
        """The bake copy's loop variable shadowed the frame's path, so add
        returned the last bake file instead of the ``.npy`` it wrote."""
        bake = os.path.join(self.root, "_incoming", "ROOM_Lightmap.exr")
        os.makedirs(os.path.dirname(bake))
        with open(bake, "wb") as f:
            f.write(b"exr")
        path = self.store.add("key", self.frame, bakes=[bake])
        self.assertEqual(path, os.path.join(self.root, "key.npy"))

    def test_two_writers_on_one_store_keep_each_others_changes(self):
        """A capture service and a loop each hold the store: a write from a
        stale copy must not drop what the other wrote (a stamp, a layer)."""
        service = LightLayers(self.root)
        loop = LightLayers(self.root)
        loop.set_stamp(materials={"MAT_ROOM_ai": {"emission": 0.0}})
        service.add("x", self.frame)
        loop.add("y", self.frame)
        again = LightLayers(self.root)
        self.assertEqual(again.stamp["materials"], {"MAT_ROOM_ai": {"emission": 0.0}})
        self.assertEqual(again.names(), ["x", "y"])

    def test_validate_names_a_missing_capture(self):
        self.store.add("key", self.frame)
        os.remove(os.path.join(self.root, "key.npy"))
        self.assertTrue(
            any("key" in problem for problem in LightLayers(self.root).validate())
        )

    def test_validate_names_a_light_layer_whose_highlights_were_not_recorded(self):
        """Its lights are in its spec but not recorded from its bake: an exact
        frame ships none of their highlights, so it shows less than the rig."""
        spec = {"lights": [{"name": "K", "type": "spot", "pos": [0, 2, 0]}]}
        self.store.add("old", self.frame, spec=spec)
        self.store.add(
            "new", self.frame, spec=spec, lights=[{"type": "spot", "intensity": 1.0}]
        )
        problems = LightLayers(self.root).validate()
        self.assertTrue(
            any(p.startswith("old:") and "re-bake" in p for p in problems), problems
        )
        self.assertFalse(any(p.startswith("new:") for p in problems), problems)


if __name__ == "__main__":
    unittest.main(exit=False)
