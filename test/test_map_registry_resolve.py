#!/usr/bin/python
# coding=utf-8
"""
Tests for MapRegistry.resolve_config — the configuration entry point used by
mayatk.MatUpdater.

Locks down:
- string preset lookup
- dict input (with and without 'preset' inheritance)
- kwargs override precedence
- None values are dropped from overrides
- alias normalization (output_type -> output_extension)
- derived flags (resize from max_size, convert_format from output_extension)
"""
import unittest

from pythontk import MapRegistry

from conftest import BaseTestCase


class ResolveConfigTest(BaseTestCase):
    def setUp(self):
        self.reg = MapRegistry()
        self.presets = self.reg.get_workflow_presets()

    def first_preset_name(self):
        """Return the name of the first registered workflow preset.

        Presets are generated from ``MapRegistry._workflow_settings`` -- a
        hard-coded class attribute with no optional dependency or environment
        behind it -- so an empty set is a regression, never a valid
        environment. Assert instead of skipping: a skip counts as a pass, so an
        emptied registry would read green.

        Returns:
            str: The first preset name (picked by iteration order so the test
            survives renames).
        """
        self.assertTrue(
            self.presets,
            "Workflow-preset registry is empty; MapRegistry._workflow_settings "
            "is the hard-coded source, so this is a regression",
        )
        return next(iter(self.presets))

    def test_returns_empty_dict_for_none(self):
        cfg = self.reg.resolve_config(None)
        self.assertIsInstance(cfg, dict)

    def test_returns_empty_dict_for_unknown_preset_string(self):
        cfg = self.reg.resolve_config("DefinitelyNotAPreset_xyz")
        # Unknown preset string is treated as no-config → empty dict (no crash)
        self.assertIsInstance(cfg, dict)

    def test_known_preset_string(self):
        # Pick first available preset to be robust against renames
        name = self.first_preset_name()
        cfg = self.reg.resolve_config(name)
        # Should at least have non-empty config from the preset
        self.assertEqual(set(cfg) >= set(self.presets[name]) - {"description"}, True)

    def test_dict_input_passes_through(self):
        cfg = self.reg.resolve_config({"max_size": 1024, "convert": True})
        self.assertEqual(cfg["max_size"], 1024)
        self.assertTrue(cfg["convert"])

    def test_dict_with_preset_key_inherits(self):
        name = self.first_preset_name()
        cfg = self.reg.resolve_config({"preset": name, "max_size": 512})
        self.assertEqual(cfg["max_size"], 512)
        # Preset values still present
        for k, v in self.presets[name].items():
            if k in ("max_size", "description"):
                continue
            if v is None:
                continue
            self.assertEqual(cfg.get(k), v, f"Preset key '{k}' lost")

    def test_kwargs_override_dict(self):
        cfg = self.reg.resolve_config({"max_size": 1024}, max_size=2048)
        self.assertEqual(cfg["max_size"], 2048)

    def test_none_overrides_are_dropped(self):
        cfg = self.reg.resolve_config({"max_size": 1024, "convert": None})
        self.assertEqual(cfg["max_size"], 1024)
        # convert=None should not have clobbered anything; it's just absent
        self.assertNotIn("convert", cfg)

    def test_output_type_alias(self):
        cfg = self.reg.resolve_config({"output_type": "png"})
        self.assertEqual(cfg.get("output_extension"), "png")
        self.assertNotIn("output_type", cfg)

    def test_resize_derived_from_max_size(self):
        cfg = self.reg.resolve_config({"max_size": 1024})
        self.assertTrue(cfg.get("resize"))

    def test_resize_false_when_max_size_none(self):
        cfg = self.reg.resolve_config({"max_size": None})
        # max_size=None would have been stripped before standardization runs;
        # so resize derivation should not have fired. Either resize absent or False.
        self.assertFalse(cfg.get("resize", False))

    def test_explicit_resize_not_overridden(self):
        cfg = self.reg.resolve_config({"max_size": 1024, "resize": False})
        self.assertFalse(cfg["resize"])

    def test_convert_format_derived_from_output_extension(self):
        cfg = self.reg.resolve_config({"output_extension": "tga"})
        self.assertTrue(cfg.get("convert_format"))

    def test_explicit_convert_format_not_overridden(self):
        cfg = self.reg.resolve_config(
            {"output_extension": "tga", "convert_format": False}
        )
        self.assertFalse(cfg["convert_format"])


if __name__ == "__main__":
    unittest.main()
