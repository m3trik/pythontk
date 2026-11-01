# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.look_match.session: the session file, declared once."""

import json
import os
import shutil
import tempfile
import unittest

from pythontk.core_utils.engines.lighting.light_layers import LightMix
from pythontk.core_utils.engines.lighting.look_match.session import LookSession
from pythontk.core_utils.user_config import UserConfig

SESSION = {
    "version": 1,
    "id": "room_moody",
    "layers": "cache:layers",
    "target": "target.json",
    "mix": "mixes/h4.json",
    "template": "cache:template/ROOM.glb",
    "size": [820, 480],
    "transform": {"operator": "aces_filmic", "exposure": 1.0},
    "capture": {"camera": {"vfov": 48.3}, "size": [1638, 960], "black": ["PROPS"]},
    "loop": {
        "queue": "cache:bake_queue",
        "service": "http://127.0.0.1:8770/",
        "worker": {
            "module": "mayatk.light_utils.light_layer_baker",
            "scene": "../scenes/ROOM.ma",
        },
    },
}


class TestLookSession(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="look_session_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.cache = os.path.join(self.dir, "cache")
        prev = os.environ.get(UserConfig.CACHE_ROOT_ENV_VAR)
        os.environ[UserConfig.CACHE_ROOT_ENV_VAR] = self.cache  # never the real cache
        self.addCleanup(
            lambda: os.environ.pop(UserConfig.CACHE_ROOT_ENV_VAR, None)
            if prev is None
            else os.environ.__setitem__(UserConfig.CACHE_ROOT_ENV_VAR, prev)
        )
        self.path = os.path.join(self.dir, "project", "look", "session.json")
        os.makedirs(os.path.dirname(self.path))

    def write(self, data) -> str:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return self.path

    def test_a_session_reads_and_its_paths_resolve_beside_it_or_in_its_cache(self):
        data = LookSession.read(self.write(SESSION))
        here = os.path.dirname(self.path)
        self.assertEqual(
            LookSession.resolve(self.path, data, data["target"]),
            os.path.join(here, "target.json"),
        )
        self.assertEqual(
            LookSession.resolve(self.path, data, data["loop"]["worker"]["scene"]),
            os.path.normpath(os.path.join(here, "..", "scenes", "ROOM.ma")),
        )
        self.assertEqual(
            LookSession.resolve(self.path, data, data["layers"]),
            os.path.join(self.cache, "look_match", "room_moody", "layers"),
        )
        self.assertIsNone(LookSession.resolve(self.path, data, None))

    def test_a_cache_path_needs_an_id(self):
        data = LookSession.read(self.write(dict(SESSION, id=None)))
        with self.assertRaises(ValueError):
            LookSession.resolve(self.path, data, "cache:layers")

    def test_what_does_not_fit_is_refused_with_why(self):
        for bad, why in (
            (dict(SESSION, version=2), "newer"),
            ({k: v for k, v in SESSION.items() if k != "layers"}, "layers"),
            (dict(SESSION, id="../elsewhere"), "folder name"),
            (dict(SESSION, loop={"worker": {"scene": "room.ma"}}), "module"),
            (["layers", "target"], "object"),  # not a mapping at all
        ):
            with self.subTest(why=why):
                with self.assertRaises(ValueError) as ctx:
                    LookSession.read(self.write(bad))
                self.assertIn(why, str(ctx.exception))

    def test_a_session_from_before_the_contract_still_reads(self):
        """No version, no id, plain relative paths: the MOODY session's shape."""
        old = {"layers": "layers", "target": "target.json", "loop": {"queue": "q"}}
        data = LookSession.read(self.write(old))
        self.assertEqual(
            LookSession.resolve(self.path, data, data["layers"]),
            os.path.join(os.path.dirname(self.path), "layers"),
        )

    def test_the_contract_publishes_its_shape(self):
        doc = LookSession.json_schema()
        json.dumps(doc)
        self.assertEqual(
            sorted(doc["$defs"]), ["LookCaptureSettings", "LookLoopSettings", "LookWorker"]
        )

    def test_the_mixes_live_beside_the_sessions_mix_one_file_each(self):
        data = LookSession.read(self.write(SESSION))
        folder = os.path.join(os.path.dirname(self.path), "mixes")
        self.assertEqual(LookSession.mix_dir(self.path, data), folder)
        os.makedirs(folder)
        h4 = LightMix({"key": {"weight": 2.0, "kelvin": 3200}}, notes="warm")
        h4.save(os.path.join(folder, "h4.json"))  # a mix saved before the store

        mixes = LookSession.mixes(self.path, data)
        self.assertEqual(mixes.list(), ["h4"])
        self.assertEqual(str(mixes.path("h4")), os.path.join(folder, "h4.json"))
        self.assertEqual(LightMix.from_dict(mixes.load("h4")).weight("key"), 2.0)

        mixes.save("h5", h4.to_dict())  # the file LightMix.save writes, byte for byte
        h4.save(os.path.join(self.dir, "reference.json"))
        with open(os.path.join(folder, "h5.json"), encoding="utf-8") as saved, open(
            os.path.join(self.dir, "reference.json"), encoding="utf-8"
        ) as reference:
            self.assertEqual(saved.read(), reference.read())
        self.assertEqual(mixes.list(), ["h4", "h5"])
        self.assertTrue(mixes.rename("h5", "h6"))
        # The project's folder, not a preset collection: the mixes alone -- no
        # library marker, no per-mix sidecar (it held the OS login, synced).
        self.assertEqual(sorted(os.listdir(folder)), ["h4.json", "h6.json"])

    def test_a_session_with_no_mix_keeps_its_mixes_in_a_mixes_folder(self):
        data = LookSession.read(self.write(dict(SESSION, mix=None)))
        folder = os.path.join(os.path.dirname(self.path), "mixes")
        self.assertEqual(LookSession.mix_dir(self.path, data), folder)
        self.assertEqual(LookSession.mixes(self.path, data).list(), [])  # none yet


if __name__ == "__main__":
    unittest.main()
