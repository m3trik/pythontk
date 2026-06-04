# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.UserConfig — Qt-free user-config resolution."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from pythontk.core_utils.user_config import (
    UserConfig,
    user_config_root,
    CONFIG_ROOT_ENV_VAR,
)


class UserConfigRootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev = os.environ.get(CONFIG_ROOT_ENV_VAR)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(CONFIG_ROOT_ENV_VAR, None)
        else:
            os.environ[CONFIG_ROOT_ENV_VAR] = self._prev
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_override_env_used_as_given(self):
        os.environ[CONFIG_ROOT_ENV_VAR] = self.tmp
        self.assertEqual(user_config_root(), Path(self.tmp))

    def test_default_root_is_under_uitk_wrapper(self):
        os.environ.pop(CONFIG_ROOT_ENV_VAR, None)
        # Whatever the platform base, the ecosystem wrapper folder is the leaf.
        self.assertEqual(user_config_root().name, "uitk")
        self.assertTrue(user_config_root().is_absolute())


class UserConfigResolveTest(unittest.TestCase):
    DEFAULT = {
        "graphics_root": "${TEMP}/photogrammetry",
        "curate": {"sharpness_percentile": 10, "hash_threshold": 5},
        "equalize": {"strength": 0.5, "reference": "median"},
        "exts": [".jpg", ".png"],
    }

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev_root = os.environ.get(CONFIG_ROOT_ENV_VAR)
        self._prev_env = os.environ.get("PHOTOG_TEST_PROFILE")
        os.environ[CONFIG_ROOT_ENV_VAR] = self.tmp  # never touch the real user dir

    def tearDown(self):
        for var, prev in (
            (CONFIG_ROOT_ENV_VAR, self._prev_root),
            ("PHOTOG_TEST_PROFILE", self._prev_env),
        ):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def test_returns_default_when_no_file(self):
        cfg = UserConfig.resolve("photogrammetry", package="extapps", default=self.DEFAULT)
        self.assertEqual(cfg["curate"]["sharpness_percentile"], 10)
        self.assertEqual(cfg["equalize"]["reference"], "median")

    def test_default_location_file_deep_merges(self):
        # A *partial* user doc overrides only one nested key.
        loc = UserConfig.path_for("photogrammetry", "extapps")
        self._write(str(loc), {"curate": {"hash_threshold": 12}, "graphics_root": "X:/g"})
        cfg = UserConfig.resolve("photogrammetry", package="extapps", default=self.DEFAULT)
        self.assertEqual(cfg["graphics_root"], "X:/g")            # overridden
        self.assertEqual(cfg["curate"]["hash_threshold"], 12)      # overridden
        self.assertEqual(cfg["curate"]["sharpness_percentile"], 10)  # preserved from default
        self.assertEqual(cfg["equalize"]["strength"], 0.5)         # preserved branch

    def test_explicit_path_wins(self):
        explicit = os.path.join(self.tmp, "custom.json")
        self._write(explicit, {"graphics_root": "P:/explicit"})
        cfg = UserConfig.resolve(
            "photogrammetry", package="extapps", default=self.DEFAULT, path=explicit
        )
        self.assertEqual(cfg["graphics_root"], "P:/explicit")

    def test_env_pointer_used(self):
        env_file = os.path.join(self.tmp, "viaenv.json")
        self._write(env_file, {"graphics_root": "E:/env"})
        os.environ["PHOTOG_TEST_PROFILE"] = env_file
        cfg = UserConfig.resolve(
            "photogrammetry", package="extapps", env="PHOTOG_TEST_PROFILE",
            default=self.DEFAULT,
        )
        self.assertEqual(cfg["graphics_root"], "E:/env")

    def test_malformed_file_falls_back_to_default(self):
        loc = UserConfig.path_for("photogrammetry", "extapps")
        os.makedirs(os.path.dirname(str(loc)), exist_ok=True)
        with open(str(loc), "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        cfg = UserConfig.resolve("photogrammetry", package="extapps", default=self.DEFAULT)
        self.assertEqual(cfg["curate"]["hash_threshold"], 5)  # default intact


class DeepMergeExpandTest(unittest.TestCase):
    def test_deep_merge_nested_and_replace(self):
        base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
        over = {"a": {"y": 9, "z": 3}, "b": [9]}
        out = UserConfig.deep_merge(base, over)
        self.assertEqual(out["a"], {"x": 1, "y": 9, "z": 3})  # nested merge
        self.assertEqual(out["b"], [9])                        # list replaces
        self.assertEqual(out["c"], 3)                          # untouched
        self.assertEqual(base["a"], {"x": 1, "y": 2})          # base not mutated

    def test_expand_env_and_tilde_recursive(self):
        os.environ["UC_TEST_VAR"] = "VALUE"
        try:
            out = UserConfig.expand(
                {"p": "${UC_TEST_VAR}/x", "home": "~", "n": 5, "l": ["${UC_TEST_VAR}"]}
            )
        finally:
            os.environ.pop("UC_TEST_VAR", None)
        self.assertEqual(out["p"], "VALUE/x")
        self.assertNotIn("~", out["home"])
        self.assertEqual(out["n"], 5)
        self.assertEqual(out["l"], ["VALUE"])


if __name__ == "__main__":
    unittest.main()
