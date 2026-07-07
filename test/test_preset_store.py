# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.PresetStore — Qt-free built-in + user named-preset store."""
import json
import os
import shutil
import tempfile
import unittest

from pythontk.core_utils.preset_store import PresetStore, sanitize_preset_name
from pythontk.core_utils.user_config import user_config_root, CONFIG_ROOT_ENV_VAR


class PresetStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.builtin = os.path.join(self.tmp, "builtin")
        self.user = os.path.join(self.tmp, "user")
        os.makedirs(self.builtin)
        self._write(self.builtin, "specular_metal", {"depth_filter": "moderate", "align_downscale": 2})
        self._write(self.builtin, "studio", {"depth_filter": "mild"})
        self.store = PresetStore("photog", "extapps", builtin_dir=self.builtin, user_dir=self.user)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, d, name, data):
        with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    # --- discovery -----------------------------------------------------------
    def test_list_union_with_user_shadowing(self):
        self.store.save("specular_metal", {"depth_filter": "aggressive"})  # shadows builtin
        self.store.save("custom", {"x": 1})
        # Each name once; builtin + user unioned.
        self.assertEqual(self.store.list(), ["custom", "specular_metal", "studio"])
        self.assertEqual(self.store.list("builtin"), ["specular_metal", "studio"])
        self.assertEqual(self.store.list("user"), ["custom", "specular_metal"])

    def test_source_and_exists(self):
        self.assertEqual(self.store.source("studio"), "builtin")
        self.assertIsNone(self.store.source("nope"))
        self.assertFalse(self.store.exists("nope"))
        self.store.save("studio", {"depth_filter": "x"})
        self.assertEqual(self.store.source("studio"), "user")  # user now shadows

    # --- io ------------------------------------------------------------------
    def test_load_prefers_user_over_builtin(self):
        self.assertEqual(self.store.load("specular_metal")["align_downscale"], 2)  # builtin
        self.store.save("specular_metal", {"align_downscale": 99})
        self.assertEqual(self.store.load("specular_metal")["align_downscale"], 99)  # user wins

    def test_load_missing_raises_keyerror_with_available(self):
        with self.assertRaises(KeyError) as ctx:
            self.store.load("ghost")
        self.assertIn("specular_metal", str(ctx.exception))

    def test_save_writes_user_tier_only(self):
        self.store.save("studio", {"depth_filter": "x"})
        # built-in file untouched on disk.
        with open(os.path.join(self.builtin, "studio.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["depth_filter"], "mild")

    def test_delete_user_only_never_builtin(self):
        # Deleting a builtin-only name is a no-op (built-ins are read-only).
        self.assertFalse(self.store.delete("studio"))
        self.assertTrue(self.store.exists("studio"))  # builtin survives
        self.store.save("studio", {"x": 1})
        self.assertTrue(self.store.delete("studio"))   # removes the user shadow
        self.assertEqual(self.store.source("studio"), "builtin")  # falls back to builtin

    def test_rename_user_preset(self):
        self.store.save("draft", {"x": 1})
        self.assertTrue(self.store.rename("draft", "final"))
        self.assertEqual(self.store.load("final")["x"], 1)
        self.assertFalse(self.store.exists("draft"))

    def test_rename_refuses_to_shadow_builtin(self):
        self.store.save("draft", {"x": 1})
        self.assertFalse(self.store.rename("draft", "studio"))  # 'studio' is a builtin
        self.assertTrue(self.store.exists("draft"))

    def test_name_sanitized_consistently(self):
        self.store.save("a/b:c", {"x": 1})
        # The same sanitized stem is used for save + load + path.
        self.assertEqual(self.store.load("a/b:c")["x"], 1)
        self.assertEqual(self.store.path("a/b:c", "user").name, sanitize_preset_name("a/b:c") + ".json")

    # --- active pointer ------------------------------------------------------
    def test_active_round_trips_and_clears(self):
        self.assertIsNone(self.store.active)  # unset by default
        self.store.active = "studio"
        self.assertEqual(self.store.active, "studio")
        # A fresh store over the same dirs reads the same pointer (cross-session).
        other = PresetStore("photog", "extapps", builtin_dir=self.builtin, user_dir=self.user)
        self.assertEqual(other.active, "studio")
        self.store.active = None
        self.assertIsNone(self.store.active)

    def test_active_sidecar_excluded_from_listing(self):
        self.store.save("custom", {"x": 1})
        self.store.active = "custom"
        # The .active dotfile must not surface as a preset.
        self.assertEqual(self.store.list("user"), ["custom"])
        self.assertFalse(self.store.exists(".active"))

    def test_delete_clears_dangling_active(self):
        self.store.save("draft", {"x": 1})
        self.store.active = "draft"
        self.store.delete("draft")
        self.assertIsNone(self.store.active)  # pointed-at preset is gone

    def test_delete_keeps_active_when_builtin_remains(self):
        # A user shadow deleted but the built-in of the same name survives.
        self.store.save("studio", {"x": 1})
        self.store.active = "studio"
        self.store.delete("studio")  # removes user shadow, builtin remains
        self.assertEqual(self.store.active, "studio")

    def test_rename_follows_active(self):
        self.store.save("draft", {"x": 1})
        self.store.active = "draft"
        self.store.rename("draft", "final")
        self.assertEqual(self.store.active, "final")

    def test_no_builtin_dir_is_user_only(self):
        store = PresetStore("p", "extapps", user_dir=self.user)
        self.assertIsNone(store.builtin_dir)
        self.assertEqual(store.list("builtin"), [])
        store.save("only", {"x": 1})
        self.assertEqual(store.list(), ["only"])


class PresetStoreDefaultLocationTest(unittest.TestCase):
    """With no explicit user_dir, the user tier lands under user_config_root."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev = os.environ.get(CONFIG_ROOT_ENV_VAR)
        os.environ[CONFIG_ROOT_ENV_VAR] = self.tmp  # never touch the real user dir

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(CONFIG_ROOT_ENV_VAR, None)
        else:
            os.environ[CONFIG_ROOT_ENV_VAR] = self._prev
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_user_dir_under_package_and_name(self):
        store = PresetStore("photog_presets", "extapps")
        self.assertEqual(store.user_dir, user_config_root() / "extapps" / "photog_presets")
        # Lazily created on save, not on read.
        self.assertFalse(store.user_dir.exists())
        store.save("p", {"x": 1})
        self.assertTrue(store.user_dir.is_dir())


class PresetStoreCodecTest(unittest.TestCase):
    """A pluggable codec changes the on-disk format and extension."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_codec_is_json(self):
        store = PresetStore("p", user_dir=self.tmp)
        self.assertEqual(store.ext, ".json")

    def test_custom_codec_writes_its_extension_and_round_trips(self):
        from pythontk.core_utils.preset_store import Codec

        # A trivial non-JSON codec (here still JSON-encoded text, but a distinct
        # extension) proves discovery + IO route through the codec, not hardcoded.
        codec = Codec(".yaml", json.loads, lambda d: json.dumps(d))
        store = PresetStore("p", user_dir=self.tmp, codec=codec)
        path = store.save("cfg", {"a": 1})
        self.assertEqual(path.suffix, ".yaml")
        self.assertEqual(store.list(), ["cfg"])  # glob uses the codec ext
        self.assertEqual(store.load("cfg"), {"a": 1})

    def test_json_files_are_invisible_to_a_yaml_store(self):
        from pythontk.core_utils.preset_store import Codec

        json_store = PresetStore("p", user_dir=self.tmp)
        json_store.save("only_json", {"a": 1})
        yaml_store = PresetStore(
            "p", user_dir=self.tmp, codec=Codec(".yaml", json.loads, json.dumps)
        )
        self.assertEqual(yaml_store.list(), [])  # different extension, not found

    def test_codec_ext_is_normalized_with_leading_dot(self):
        from pythontk.core_utils.preset_store import Codec

        # A dotless ext is a natural public-API mistake; it must not produce
        # 'cfgyaml' filenames / '*yaml' globs. __post_init__ prepends the dot.
        store = PresetStore(
            "p", user_dir=self.tmp, codec=Codec("yaml", json.loads, json.dumps)
        )
        self.assertEqual(store.ext, ".yaml")
        self.assertEqual(store.save("cfg", {"a": 1}).suffix, ".yaml")
        self.assertEqual(store.list(), ["cfg"])


if __name__ == "__main__":
    unittest.main()
