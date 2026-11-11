# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.light_rig: the light rig engine, written once.

The engine runs over :class:`PureLightRigHost` (a scene in memory); each host
library's adapter is tested in that library.
"""

import shutil
import tempfile
import unittest

from pythontk.core_utils.engines.lighting.light_rig import (
    LightRig,
    LightRigHost,
    PureLightRigHost,
)
from pythontk.core_utils.engines.lighting.light_rig_payload import LightRigLight
from pythontk.core_utils.presets.store import PresetStore


class _FollowingHost(PureLightRigHost):
    """A pure host where ``Bridge.emission`` follows ``Mat.emission``."""

    def plug(self, node, attr):
        plug = f"{node}.{attr}"
        return plug, ("Mat.emission" if plug == "Bridge.emission" else plug)


class _RigCase(unittest.TestCase):
    host_class = PureLightRigHost

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ptk_light_rig_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.host = self.host_class()

        class Bound(LightRig):
            HOST = self.host

        self.Rig = Bound
        self.rig = Bound(store=PresetStore("light_rigs", user_dir=self.root))

    def light(self, name, group="LIGHT_RIG", **values):
        entry = {"name": name, "type": "spot", "pos": [0, 0, 0], **values}
        self.host.groups.setdefault(group, {})[name] = entry
        return entry


class TestBinding(unittest.TestCase):
    def test_pure_host_is_a_light_rig_host(self):
        self.assertIsInstance(PureLightRigHost(), LightRigHost)

    def test_the_neutral_tables_are_built_from_the_bindings(self):
        class Bound(LightRig):
            HOST_TYPES = {t: t.upper() for t in LightRigLight.TYPES}

        self.assertEqual(Bound.TYPES["spot"], "SPOT")
        self.assertEqual(LightRig.TYPES["spot"], "spot")  # the base keeps its own

    def test_a_type_the_binding_does_not_map_fails_it(self):
        with self.assertRaises(KeyError):

            class Bound(LightRig):
                HOST_TYPES = {"spot": "SPOT"}

    def test_the_kelvin_and_fixture_tables_are_the_bindings(self):
        class Bound(LightRig):
            KELVIN_ATTRS = ("useTemp", "temp")
            DIFFUSE_VISIBILITY = ("seen",)
            HOST_SETTINGS = dict(LightRig.HOST_SETTINGS, intensity="energy")

        light = Bound.normalize({"lights": [{"name": "a", "type": "spot", "kelvin": 3000}]})
        self.assertEqual(light["lights"][0]["attrs"], {"useTemp": 1, "temp": 3000})
        overrides = Bound.fixture_overrides(
            [{"name": "f", "mode": "lamp", "meshes": ["m"], "lamps": {"l": 2}}]
        )
        self.assertEqual(overrides, {"m": {"seen": False}, "l": {"energy": 2.0}})

    def test_a_newer_payload_names_the_host_package(self):
        with self.assertRaisesRegex(ValueError, "update pythontk"):
            LightRig.normalize({"version": LightRig.PAYLOAD_VERSION + 1})


class TestLibraryAndRecord(_RigCase):
    def test_save_records_the_scene_rig_and_edits_read_as_edited(self):
        light = self.light("Key", intensity=2)
        self.rig.save("one", description="first")
        self.assertEqual(self.rig.applied(), ("one", False))
        self.assertEqual(self.rig.store.info("one")["lights"], 1)
        light["pos"] = [1, 0, 0]
        self.assertEqual(self.rig.applied(), ("one", True))
        self.assertEqual(self.rig.scene_rigs(), [("LIGHT_RIG", "one")])

    def test_no_rig_applied(self):
        self.assertEqual(self.rig.applied(), (None, False))
        self.assertFalse(self.rig.can_swap_back())
        self.assertEqual(self.rig.remove(), [])

    def test_rename_and_delete(self):
        self.light("Key")
        self.rig.save("one")
        self.assertTrue(self.rig.rename("one", "two"))
        self.assertEqual(self.rig.names(), ["two"])
        self.assertTrue(self.rig.delete("two"))
        self.assertEqual(self.rig.names(), [])


class TestSwap(_RigCase):
    def test_apply_then_swap_back_toggles(self):
        self.light("Key")
        self.host.nodes["Lamp"] = {"intensity": 1.0}
        self.rig.save("one")
        report = self.rig.apply(
            {
                "version": 1,
                "lights": [{"name": "Fill", "type": "point"}],
                "existing": {"Lamp": {"intensity": 5.0}},
            },
            name="two",
        )
        self.assertEqual(report["lights"], ["Fill"])
        self.assertEqual(self.host.nodes["Lamp"]["intensity"], 5.0)
        self.assertEqual(self.rig.applied(), ("two", False))
        self.assertTrue(self.rig.can_swap_back("LIGHT_RIG"))
        back = self.rig.swap_back()
        self.assertEqual(back["lights"], ["Key"])
        self.assertEqual(self.host.nodes["Lamp"]["intensity"], 1.0)
        self.assertEqual(self.rig.applied(), ("one", False))
        self.assertEqual(self.rig.swap_back()["lights"], ["Fill"])  # and again

    def test_swap_back_with_nothing_replaced(self):
        with self.assertRaises(LookupError):
            self.rig.swap_back()

    def test_swap_back_with_several_groups_names_them(self):
        for group in ("A", "B"):
            self.light("L" + group, group=group)
            self.rig.apply({"version": 1, "lights": []}, group=group)
        with self.assertRaisesRegex(LookupError, "A, B"):
            self.rig.swap_back()

    def test_a_light_the_host_cannot_build_is_reported(self):
        class Refusing(PureLightRigHost):
            def build_light(self, light):
                return None, [f"{light['name']}: not here"]

        class Bound(LightRig):
            HOST = Refusing()

        report = Bound().apply({"version": 1, "lights": [{"name": "X", "type": "spot"}]})
        self.assertEqual((report["lights"], report["skipped"]), ([], ["X: not here"]))

    def test_the_group_keeps_its_parent(self):
        self.light("Key")
        self.host.parents["LIGHT_RIG"] = "RIGS"
        self.rig.apply({"version": 1, "lights": []})
        self.assertEqual(self.host.parents["LIGHT_RIG"], "RIGS")

    def test_carried_sections_ride_the_record(self):
        self.rig.apply({"version": 1, "lights": [], "viewer": {"fov": 40}})
        self.assertEqual(self.rig._entry("LIGHT_RIG")["extras"]["viewer"], {"fov": 40})


class TestCompare(_RigCase):
    def test_compare_fills_defaults_from_the_host(self):
        class Defaults(PureLightRigHost):
            def defaults(self, kind):
                return {"intensity": 1.0}, {}

        class Bound(LightRig):
            HOST = Defaults()

        Bound.HOST.groups["LIGHT_RIG"] = {"Key": {"name": "Key", "type": "spot"}}
        rig = Bound()
        target = {"version": 1, "lights": [{"name": "Key", "type": "spot", "intensity": 1.0}]}
        self.assertEqual(rig.compare(target), [])
        target["lights"][0]["intensity"] = 3.0
        self.assertEqual(rig.compare(target), ["Key: intensity 1 -> 3"])

    def test_compare_counts_only_the_plugs_the_rig_overrides(self):
        self.host.nodes["Lamp"] = {"intensity": 1.0, "color": [1, 1, 1]}
        changes = self.rig.compare({"version": 1, "lights": [], "existing": {"Lamp": {"intensity": 2.0}}})
        self.assertEqual(changes, ["existing Lamp.intensity 1 -> 2"])


class TestLandedOverrides(_RigCase):
    host_class = _FollowingHost

    def test_a_follower_lands_on_what_it_follows_and_yields_to_it(self):
        self.host.nodes.update({"Bridge": {}, "Mat": {}})
        landed, yielded = self.Rig.landed_overrides(
            {"existing": {}, "materials": {"Bridge": {"emission": 2}, "Mat": {"emission": 1}}}
        )
        self.assertEqual(landed, {"materials": {"Mat": {"emission": 1}}})
        self.assertEqual(len(yielded), 1)
        self.assertIn("follows Mat.emission", yielded[0])


if __name__ == "__main__":
    unittest.main()
