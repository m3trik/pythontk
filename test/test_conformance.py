# !/usr/bin/python
# coding=utf-8
"""Conformance -- the registry of golden-case documents the ports are held to.

What every registered provider must give a port's runner: one JSON-able
document of one shape, the same for the same seed. The cases' CONTENT is
each model's own suite's business (test_articulation, test_shadow_projection).
"""

import json
import unittest

from pythontk import Conformance


class TestConformance(unittest.TestCase):
    def test_the_ported_models_are_registered(self):
        self.assertEqual(Conformance.names(), ["articulation", "shadow_projection"])

    def test_every_provider_writes_one_document_shape(self):
        for name in Conformance.names():
            with self.subTest(name):
                doc = Conformance.cases(name, seed=4)
                self.assertIsInstance(doc["tolerance"], dict)
                self.assertTrue(doc["tolerance"])
                self.assertTrue(
                    all(
                        isinstance(v, float) and v > 0
                        for v in doc["tolerance"].values()
                    )
                )
                self.assertIsInstance(doc["cases"], list)
                self.assertTrue(doc["cases"])
                # Plain values, what a browser or a Unity runner is handed.
                self.assertEqual(json.loads(json.dumps(doc)), doc)

    def test_the_same_seed_gives_the_same_cases(self):
        for name in Conformance.names():
            with self.subTest(name):
                self.assertEqual(
                    Conformance.cases(name, seed=9), Conformance.cases(name, seed=9)
                )
                self.assertNotEqual(
                    Conformance.cases(name, seed=9), Conformance.cases(name, seed=10)
                )

    def test_a_providers_own_sizing_passes_through(self):
        small = Conformance.cases("articulation", seed=1, per_rig=1)
        large = Conformance.cases("articulation", seed=1, per_rig=2)
        self.assertEqual(2 * len(small["cases"]), len(large["cases"]))

    def test_an_unknown_model_names_the_registered_ones(self):
        with self.assertRaises(KeyError) as caught:
            Conformance.cases("teapot")
        self.assertIn("articulation", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
