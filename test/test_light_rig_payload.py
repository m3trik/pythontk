# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.light_rig_payload: the rig payload, declared once."""

import json
import os
import shutil
import time
import unittest
import uuid
from pathlib import Path

from pythontk.core_utils.engines.lighting.light_layers import (
    LightLayerInfo,
    LightLayerScene,
)
from pythontk.core_utils.engines.lighting.light_rig_payload import (
    LightRigFixture,
    LightRigLight,
    LightRigPayload,
)
from pythontk.core_utils.user_config import CONFIG_ROOT_ENV_VAR

TEMP = os.path.join(os.path.dirname(__file__), "temp_tests")


def _scratch(case) -> str:
    """A folder of this test's own under ``test/temp_tests/``, removed after it."""
    root = os.path.join(TEMP, f"rig_payload_{case._testMethodName}_{uuid.uuid4().hex[:6]}")
    os.makedirs(root)
    case.addCleanup(shutil.rmtree, root, ignore_errors=True)
    return root

RIG = {
    "version": 1,
    "group": "MOODY_RIG",
    "lights": [
        {
            "name": "KEY",
            "type": "spot",
            "pos": [0, 200, 100],
            "rot": [-40, 0, 0],
            "intensity": 2.0,
            "cone": 40,
            "color": [1.0, 0.9, 0.8],
            "specular": 0.5,
        },
        {
            "name": "FILL",
            "type": "area",
            "pos": [0, 300, 0],
            "scale": [100, 100, 1],
            "kelvin": 4300,
            "normalize": False,
        },
    ],
    "existing": {"LIGHT_A": {"intensity": 0.0}},
    "materials": {"MAT_PANELS": {"emissive_intensity": 0.75}},
    "bake_gain": 3.14159,
    "source": {"tool": "look-match"},
}


class TestLightRigPayload(unittest.TestCase):
    def test_a_rig_in_the_neutral_spelling_fits(self):
        checked = LightRigPayload.validate(RIG)
        self.assertEqual((checked.errors, checked.warnings), ([], []))

    def test_a_light_outside_the_contract_is_an_error(self):
        bad = dict(
            RIG,
            lights=[
                {"name": "X", "type": "laser"},
                {"name": "Y", "type": "spot", "color": "warm"},
            ],
        )
        errors = LightRigPayload.validate(bad).errors
        self.assertTrue(any("laser" in e for e in errors), errors)
        self.assertTrue(any("color" in e for e in errors), errors)
        self.assertTrue(
            LightRigPayload.validate({"version": 1}).errors
        )  # no lights key

    def test_fixtures_name_one_mode_each(self):
        rig = dict(
            RIG,
            fixtures=[
                {"name": "PANELS", "mode": "lamp", "meshes": ["LIGHT_D"], "lamps": {"LIGHT_A": 2.5}},
                {"name": "DESK", "mode": "glow", "meshes": ["LAMP_MESH"]},
            ],
        )
        self.assertEqual(LightRigPayload.validate(rig).errors, [])
        self.assertEqual(LightRigFixture.MODES, ("glow", "emissive", "lamp"))
        bad = dict(rig, fixtures=[{"name": "PANELS", "mode": "lit", "meshes": []}])
        self.assertTrue(any("lit" in e for e in LightRigPayload.validate(bad).errors))
        no_meshes = dict(rig, fixtures=[{"name": "PANELS", "mode": "glow"}])
        self.assertTrue(
            any("meshes" in e for e in LightRigPayload.validate(no_meshes).errors)
        )

    def test_lengths_are_centimetres_and_angles_degrees_whatever_the_scene(self):
        """A rig saved in one scene lights another, whatever units either works
        in: the contract fixes them, and its published schema says which
        (2026-10-19: positions rode the scene's working unit, radii cm)."""
        doc = LightRigPayload.json_schema()
        props = doc["$defs"]["LightRigLight"]["properties"]
        for name in ("pos", "aim", "radius", "scale"):
            self.assertIn("centimetres", props[name]["description"], name)
        for name in ("rot", "cone", "penumbra", "gobo_rotate"):
            self.assertIn("degrees", props[name]["description"], name)
        for summary in (doc["description"], doc["$defs"]["LightRigLight"]["description"]):
            self.assertIn("centimetres", summary)
            self.assertIn("degrees", summary)

    def test_settings_and_notes_are_declared_fields(self):
        """A host maps each setting onto its own attribute: they must be real
        fields, and the structural keys are everything else."""
        names = {f for f in LightRigLight.__dataclass_fields__}
        self.assertLessEqual(
            set(LightRigLight.SETTINGS) | set(LightRigLight.NOTES), names
        )


class TestTheLibrary(unittest.TestCase):
    def setUp(self):
        self.root = _scratch(self)
        prev = os.environ.get(CONFIG_ROOT_ENV_VAR)
        os.environ[CONFIG_ROOT_ENV_VAR] = self.root
        self.addCleanup(
            lambda: os.environ.pop(CONFIG_ROOT_ENV_VAR, None)
            if prev is None
            else os.environ.__setitem__(CONFIG_ROOT_ENV_VAR, prev)
        )

    def test_one_library_for_every_host_with_the_old_one_taken_in(self):
        """Rigs saved where mayatk kept them before the library was shared are
        still there after the move -- they are user data."""
        legacy = Path(self.root) / "mayatk" / "light_rigs"
        legacy.mkdir(parents=True)
        (legacy / "moody.json").write_text(json.dumps(RIG), encoding="utf-8")
        store = LightRigPayload.library()
        self.assertEqual(store.user_dir, Path(self.root) / "light_rigs")
        self.assertEqual(store.load("moody")["group"], "MOODY_RIG")
        self.assertTrue((legacy / "moody.json").is_file())


class TestLightLayerScene(unittest.TestCase):
    def test_a_layer_from_an_older_save_of_its_scene_is_stale(self):
        scene = os.path.join(_scratch(self), "room.ma")
        Path(scene).write_text("//Maya ASCII", encoding="utf-8")
        record = LightLayerScene.of(scene)
        self.assertTrue(LightLayerScene.validate(record).ok)
        self.assertTrue(LightLayerScene.matches(record, scene))
        self.assertFalse(LightLayerScene.matches(None, scene))
        other = os.path.join(os.path.dirname(scene), "hall.ma")
        shutil.copy2(scene, other)
        self.assertFalse(LightLayerScene.matches(record, other))  # another scene
        later = time.time() + 5
        os.utime(scene, (later, later))  # saved since
        self.assertFalse(LightLayerScene.matches(record, scene))
        os.remove(scene)
        self.assertFalse(LightLayerScene.matches(record, scene))


class TestLightLayerInfo(unittest.TestCase):
    def test_a_bake_folders_layer_json_fits_and_its_target_is_checked(self):
        info = {
            "name": "glow",
            "kind": "emission",
            "spec": {"materials": {"MAT_PANELS": {"emissive_intensity": 3.14}}},
            "gain": 1.0,
            "settings": {"resolution": 256},
            "files": ["ROOM_Lightmap.exr", "scene_Probe.exr"],
            "lights": [],
            "target": {
                "node": "MAT_PANELS",
                "attr": "emissive_intensity",
                "scale": 3.14,
            },
            "scene": {"file": "room.ma", "mtime": 1760000000.5, "bytes": 2048},
            "seconds": 12.5,
        }
        self.assertTrue(LightLayerInfo.validate(info).ok)
        self.assertTrue(
            LightLayerInfo.validate(dict(info, scene={"file": "room.ma"})).errors
        )
        broken = dict(info, target={"node": "MAT_PANELS"})  # no attr
        self.assertTrue(LightLayerInfo.validate(broken).errors)
        self.assertTrue(LightLayerInfo.validate(dict(info, files="a.exr")).errors)
        self.assertEqual(LightLayerInfo.FILE, "layer.json")


class TestTheContractsPublish(unittest.TestCase):
    def test_each_contract_is_a_json_schema_other_languages_read(self):
        """A typed contract publishes its shape (``json_schema``); one whose
        annotation and ``nested=`` disagreed raised instead."""
        for spec in (LightRigPayload, LightLayerInfo):
            with self.subTest(spec=spec.__name__):
                doc = spec.json_schema()
                json.dumps(doc)
                self.assertEqual(doc["title"], spec.__name__)


if __name__ == "__main__":
    unittest.main()
